"""
Export CausalVGGT predictions to COLMAP sparse reconstruction format.

Usage:
    python demo/demo_colmap.py --scene_dir /path/to/scene
    python demo/demo_colmap.py --scene_dir /path/to/scene --output_dir output/colmap --base_model streamvggt

The scene directory should contain an `images/` subfolder with .png or .jpg files.
Output: <output_dir>/sparse/{cameras.bin, images.bin, points3D.bin, points.ply}
"""

import random
import re
import numpy as np
import glob
import os
import sys
import copy
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import trimesh
import pycolmap

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from model_wrapper import load_model, run_model
from causalvggt.utils.load_fn import load_and_preprocess_images_square
from causalvggt.utils.pose_enc import pose_encoding_to_extri_intri
from causalvggt.utils.geometry import unproject_depth_map_to_point_map


def parse_args():
    parser = argparse.ArgumentParser(description="COLMAP export demo")
    parser.add_argument("--scene_dir", type=str, required=True,
                        help="Directory containing images/ subfolder")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save outputs (default: scene_dir)")
    parser.add_argument("--base_model", type=str, default="stream3r",
                        choices=["stream3r", "streamvggt"])
    parser.add_argument("--size", type=int, default=512, choices=[224, 512, 518])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shared_camera", action="store_true", default=False,
                        help="Use shared camera for all images")
    parser.add_argument("--camera_type", type=str, default="PINHOLE",
                        help="Camera type for reconstruction")
    parser.add_argument("--conf_thres", type=float, default=5.0,
                        help="Confidence threshold for depth filtering")
    parser.add_argument("--max_points", type=int, default=100000,
                        help="Max 3D points to write into reconstruction")
    return parser.parse_args()


# ---- Utility functions (previously from vggt.utils.helper) ----

def create_pixel_coordinate_grid(num_frames, height, width):
    """Create (S, H, W, 3) grid with (x, y, frame_index) per pixel."""
    u = np.arange(width, dtype=np.float32)
    v = np.arange(height, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)
    grid = np.stack([uu, vv], axis=-1)                     # (H, W, 2)
    grid = np.tile(grid[None], (num_frames, 1, 1, 1))      # (S, H, W, 2)
    frame_idx = np.arange(num_frames, dtype=np.float32)
    frame_idx = frame_idx[:, None, None, None].repeat(height, axis=1).repeat(width, axis=2)
    return np.concatenate([grid, frame_idx], axis=-1)       # (S, H, W, 3)


def randomly_limit_trues(mask, max_trues):
    """Randomly keep at most `max_trues` True values in a boolean array."""
    true_indices = np.where(mask.ravel())[0]
    if len(true_indices) <= max_trues:
        return mask
    keep = np.random.choice(true_indices, size=max_trues, replace=False)
    new_mask = np.zeros_like(mask.ravel())
    new_mask[keep] = True
    return new_mask.reshape(mask.shape).astype(bool)


# ---- pycolmap conversion (previously from vggt.dependency.np_to_pycolmap) ----

def build_pycolmap_reconstruction(
    points_3d,
    points_xyf,
    points_rgb,
    extrinsics,
    intrinsics,
    image_size,
    image_names=None,
    shared_camera=False,
    camera_type="PINHOLE",
):
    """
    Build a pycolmap.Reconstruction from numpy arrays (no tracks / no BA).

    Args:
        points_3d:  (N, 3) world coordinates
        points_xyf: (N, 3) pixel x, y and frame index per point
        points_rgb:  (N, 3) uint8 colour per point
        extrinsics: (S, 3, 4) camera-from-world
        intrinsics: (S, 3, 3) camera intrinsic matrices
        image_size: (2,) array [width, height] of the input resolution
        image_names: optional list of S image file names
        shared_camera: if True all frames share one camera
        camera_type: pycolmap camera model string
    """
    reconstruction = pycolmap.Reconstruction()
    S = extrinsics.shape[0]
    w, h = int(image_size[0]), int(image_size[1])

    # --- Cameras ---
    num_cameras = 1 if shared_camera else S
    for cam_id in range(num_cameras):
        idx = 0 if shared_camera else cam_id
        fx, fy = intrinsics[idx, 0, 0], intrinsics[idx, 1, 1]
        cx, cy = intrinsics[idx, 0, 2], intrinsics[idx, 1, 2]
        cam_model = pycolmap.CameraModelId(camera_type)
        if camera_type == "SIMPLE_PINHOLE":
            params = np.array([fx, cx, cy])
        elif camera_type == "PINHOLE":
            params = np.array([fx, fy, cx, cy])
        else:
            params = np.array([fx, fy, cx, cy])
        camera = pycolmap.Camera(
            model=cam_model,
            width=w,
            height=h,
            params=params,
            camera_id=cam_id,
        )
        reconstruction.add_camera(camera)

    # --- Images (poses) ---
    for img_id in range(S):
        R = extrinsics[img_id, :3, :3]
        t = extrinsics[img_id, :3, 3]
        qvec = pycolmap.rotmat_to_qvec(R)
        cam_id = 0 if shared_camera else img_id
        name = image_names[img_id] if image_names is not None else f"frame_{img_id:04d}.png"
        image = pycolmap.Image(
            name=name,
            camera_id=cam_id,
            cam_from_world=pycolmap.Rigid3d(pycolmap.Rotation3d(qvec), t),
        )
        image.image_id = img_id + 1
        reconstruction.add_image(image)
        reconstruction.register_image(image.image_id)

    # --- 3D Points ---
    for i in range(len(points_3d)):
        point = pycolmap.Point3D()
        point.xyz = points_3d[i]
        point.color = points_rgb[i].astype(np.uint8)
        reconstruction.add_point3D(point.xyz, pycolmap.Track(), point.color)

    return reconstruction


def rescale_colmap_cameras(
    reconstruction, image_names, original_coords, img_size,
    shift_point2d=False, shared_camera=False,
):
    """Rescale pycolmap camera params back to original image resolution."""
    rescale = True
    for pyimageid in reconstruction.images:
        pyimage = reconstruction.images[pyimageid]
        pycamera = reconstruction.cameras[pyimage.camera_id]
        pyimage.name = image_names[pyimageid - 1]

        if rescale:
            pred_params = copy.deepcopy(pycamera.params)
            real_image_size = original_coords[pyimageid - 1, -2:]
            resize_ratio = max(real_image_size) / img_size
            pred_params = pred_params * resize_ratio
            pred_params[-2:] = real_image_size / 2  # principal point at center
            pycamera.params = pred_params
            pycamera.width = int(real_image_size[0])
            pycamera.height = int(real_image_size[1])

        if shift_point2d:
            top_left = original_coords[pyimageid - 1, :2]
            for pt2d in pyimage.points2D:
                pt2d.xy = (pt2d.xy - top_left) * resize_ratio

        if shared_camera:
            rescale = False
    return reconstruction


# ---- Main ----

def demo_fn(args):
    print("Arguments:", vars(args))

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    # 1. Load model
    model = load_model("causalvggt", base_model=args.base_model, device=device)

    # 2. Load images (square-padded at 1024 for original-resolution rescaling)
    image_dir = os.path.join(args.scene_dir, "images")
    image_path_list = sorted(glob.glob(os.path.join(image_dir, "*")))
    if not image_path_list:
        raise FileNotFoundError(f"No images found in {image_dir}")
    base_image_names = [os.path.basename(p) for p in image_path_list]

    img_load_resolution = 1024
    images, original_coords = load_and_preprocess_images_square(image_path_list, img_load_resolution)
    images = images.to(device)
    print(f"Loaded {len(images)} images from {image_dir}")

    # 3. Run inference at target resolution
    if args.size == 512:
        resolution = (384, 512)
    elif args.size == 518:
        resolution = (336, 518)
    elif args.size == 224:
        resolution = (224, 224)
    else:
        raise ValueError(f"Unsupported size: {args.size}")

    images_resized = F.interpolate(images, size=resolution, mode="bilinear", align_corners=False)
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=dtype):
        predictions = run_model(model, images_resized, "causalvggt",
                                mode="full", streaming=False, dtype=dtype, device=device)

    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"], images_resized.shape[-2:]
    )

    extrinsic = extrinsic.squeeze(0).cpu().numpy()
    intrinsic = intrinsic.squeeze(0).cpu().numpy()
    depth_map = predictions["depth"].squeeze(0).cpu().numpy()
    depth_conf = predictions["depth_conf"].squeeze(0).cpu().numpy()

    # 4. Unproject depth → 3D points
    points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)

    S, H, W, _ = points_3d.shape
    infer_size = resolution  # (H, W)

    points_rgb = F.interpolate(images, size=infer_size, mode="bilinear", align_corners=False)
    points_rgb = (points_rgb.cpu().numpy() * 255).astype(np.uint8).transpose(0, 2, 3, 1)

    points_xyf = create_pixel_coordinate_grid(S, H, W)

    conf_mask = depth_conf >= args.conf_thres
    conf_mask = randomly_limit_trues(conf_mask, args.max_points)

    pts = points_3d[conf_mask]
    xyf = points_xyf[conf_mask]
    rgb = points_rgb[conf_mask]

    # 5. Build pycolmap Reconstruction
    print("Converting to COLMAP format...")
    image_size = np.array([infer_size[1], infer_size[0]])  # (W, H)
    reconstruction = build_pycolmap_reconstruction(
        pts, xyf, rgb, extrinsic, intrinsic, image_size,
        image_names=base_image_names,
        shared_camera=args.shared_camera,
        camera_type=args.camera_type,
    )

    # Rescale cameras to original image resolution
    reconstruction = rescale_colmap_cameras(
        reconstruction, base_image_names,
        original_coords.cpu().numpy(),
        img_size=img_load_resolution,
        shift_point2d=True,
        shared_camera=args.shared_camera,
    )

    # 6. Save
    output_dir = args.output_dir if args.output_dir else args.scene_dir
    sparse_dir = os.path.join(output_dir, "sparse")
    os.makedirs(sparse_dir, exist_ok=True)
    reconstruction.write(sparse_dir)
    trimesh.PointCloud(pts, colors=rgb).export(os.path.join(sparse_dir, "points.ply"))
    print(f"Saved COLMAP reconstruction to {sparse_dir}")


if __name__ == "__main__":
    args = parse_args()
    with torch.no_grad():
        demo_fn(args)
