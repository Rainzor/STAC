"""
Minimal example: load a CausalVGGT model and run inference on a folder of images.

Usage:
    python main.py --scene_dir /path/to/scene
    python main.py --scene_dir /path/to/scene --base_model streamvggt --streaming --mode window_chunk_merge -win 4 -ck 4 -hh 2 -ret_sz 2 -ret_buf

The scene directory should contain an `images/` subfolder with .png or .jpg files.
Checkpoints should be placed under ckpt/{stream3r,streamvggt}/ (see README.md).
"""

import re
import argparse
import logging
from pathlib import Path

import torch
import torch.nn.functional as F

from model_wrapper import load_model, run_model
from causalvggt.utils.load_fn import load_and_preprocess_images
from causalvggt.utils.pose_enc import pose_encoding_to_extri_intri
from causalvggt.utils.geometry import unproject_depth_map_to_point_map

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger("main")


def parse_args():
    parser = argparse.ArgumentParser(description="STAC — minimal inference example")
    parser.add_argument("--scene_dir", type=str, required=True,
                        help="Directory containing images/ subfolder")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save outputs (default: scene_dir)")
    parser.add_argument("--base_model", type=str, default="stream3r",
                        choices=["stream3r", "streamvggt"],
                        help="Backbone weights to use")
    parser.add_argument("--size", type=int, default=518, choices=[224, 512, 518],
                        help="Input resolution")
    parser.add_argument("--mode", type=str, default="full",
                        help="Attention mode (full, causal, window_kv, window_chunk_merge, ...)")
    parser.add_argument("--streaming", action="store_true",
                        help="Enable frame-by-frame streaming via StreamSession")
    parser.add_argument("--window_size", "-win", type=int, default=0)
    parser.add_argument("--chunk_size", "-ck", type=int, default=1)
    parser.add_argument("--hh_size", "-hh", type=int, default=0)
    parser.add_argument("--retrieval_size", "-ret_sz", type=int, default=0)
    parser.add_argument("--retrieve_buf", "-ret_buf", action="store_true")
    return parser.parse_args()


def load_scene_images(scene_dir, size=518):
    """Load images from scene_dir/images/ and resize to the target resolution."""
    image_dir = Path(scene_dir) / "images"
    exts = ("*.png", "*.jpg", "*.jpeg")
    image_paths = []
    for ext in exts:
        image_paths.extend(image_dir.glob(ext))

    def numerical_sort(p: Path):
        m = re.search(r"(\d+)", p.stem)
        return int(m.group(1)) if m else -1

    image_paths = sorted(image_paths, key=numerical_sort)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir}")

    images = load_and_preprocess_images([str(p) for p in image_paths])

    # resolution as (H, W) for F.interpolate; matches eval (W,H): 512->(512,384), 518->(518,336)
    if size == 512:
        resolution = (384, 512)
    elif size == 518:
        resolution = (392, 518)
    elif size == 224:
        resolution = (224, 224)
    else:
        raise ValueError(f"Unsupported size: {size}")
    images = F.interpolate(images, size=resolution, mode="bilinear", align_corners=False)
    return images


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    # 1. Load model
    model = load_model("causalvggt", base_model=args.base_model, device=device)

    # 2. Load images  — (S, 3, H, W) tensor in [0, 1]
    images = load_scene_images(args.scene_dir, size=args.size).to(device)
    logger.info(f"Loaded {images.shape[0]} frames, shape {tuple(images.shape)}")

    # 3. Run inference
    model_kwargs = {
        "window_size": args.window_size,
        "chunk_size": args.chunk_size,
        "hh_size": args.hh_size,
        "retrieval_size": args.retrieval_size,
        "return_buf": args.retrieve_buf,
    }
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=dtype):
        predictions = run_model(
            model, images, "causalvggt",
            mode=args.mode,
            streaming=args.streaming,
            dtype=dtype, device=device,
            **model_kwargs,
        )

    # 4. Decode predictions
    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"], images.shape[-2:]
    )
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    depth_map = predictions["depth"]
    if isinstance(depth_map, torch.Tensor):
        depth_map = depth_map.cpu().numpy().squeeze(0)
        extrinsic_np = extrinsic.cpu().numpy().squeeze(0) if isinstance(extrinsic, torch.Tensor) else extrinsic
        intrinsic_np = intrinsic.cpu().numpy().squeeze(0) if isinstance(intrinsic, torch.Tensor) else intrinsic
    else:
        extrinsic_np, intrinsic_np = extrinsic, intrinsic
    world_points = unproject_depth_map_to_point_map(depth_map, extrinsic_np, intrinsic_np)

    logger.info(f"Extrinsic shape: {extrinsic_np.shape}")
    logger.info(f"Depth shape:     {depth_map.shape}")
    logger.info(f"World pts shape: {world_points.shape}")
    logger.info("Done. Predictions keys: %s", list(predictions.keys()))


if __name__ == "__main__":
    main()
