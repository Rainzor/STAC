"""
Minimal example: load a CausalVGGT model and run inference on a folder of images.

Usage:
    python main.py --scene_dir /path/to/scene
    python main.py --scene_dir /path/to/scene --base_model streamvggt --streaming --mode window_chunk_merge -win 4 -ck 4 -hh 2 -ret_sz 2 -ret_buf

The scene directory should contain an `images/` subfolder with .png or .jpg files.
Checkpoints should be placed under ckpt/{stream3r,streamvggt}/ (see README.md).
"""

import argparse
import logging
from contextlib import nullcontext

import torch

from model_wrapper import load_model, run_model
from eval.utils.image import load_scene_images
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
    parser.add_argument("--kf_every", type=int, default=10,
                        help="Sample every k frames for limited memory inference")
    parser.add_argument("--mode", type=str, default="stac",
                        help="Attention mode (stac, full, causal, window_kv, window_chunk_merge, ...)")
    parser.add_argument("--streaming", action="store_true",
                        help="Enable frame-by-frame streaming via StreamSession")
    parser.add_argument("--window_size", "-win", type=int, default=0)
    parser.add_argument("--chunk_size", "-ck", type=int, default=1)
    parser.add_argument("--hh_size", "-hh", type=int, default=0)
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "fp16", "bf16"])
    parser.add_argument("--attn_backend", type=str, default="cuda", choices=["cuda", "triton"])
    parser.add_argument("--subsample", type=float, default=1.0)
    parser.add_argument("--pinned", type=int, default=[0], nargs="+")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--retrieval_size", "-ret_sz", type=int, default=0)
    parser.add_argument("--retrieve_buf", "-ret_buf", action="store_true")
    parser.add_argument("--voxel_size", type=float, default=0.05)
    parser.add_argument("--voxel_num", type=int, default=4096)
    parser.add_argument("--voxel_conf", type=float, default=2.0)
    parser.add_argument("--voxel_buf_cap", type=int, default=8)
    parser.add_argument("--voxel_piv_cap", type=int, default=4)
    parser.add_argument("--voxel_backend", type=str, default="cuda", choices=["cuda", "python"])
    parser.add_argument("--allocator", "-alloc", type=str, default="segment", choices=["static", "slab", "segment"])

    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.dtype == "bf16":
        dtype = torch.bfloat16
    elif args.dtype == "fp16":
        dtype = torch.float16
    else:
        if device == "cuda":
            dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        else:
            dtype = torch.float32

    # 1. Load model
    model = load_model("causalvggt", base_model=args.base_model, device=device)

    # 2. Load images  — (S, 3, H, W) tensor in [0, 1]
    images = load_scene_images(args.scene_dir, size=args.size).to(device)

    # 3. Sample images for limited memory inference
    images = images[::args.kf_every] if args.kf_every > 1 else images

    logger.info(f"Loaded {images.shape[0]} frames, shape {tuple(images.shape)}")

    # 3. Run inference
    model_kwargs = {
        "window_size": args.window_size,
        "chunk_size": args.chunk_size,
        "hh_size": args.hh_size,
        "retrieval_size": args.retrieval_size,
        "return_buf": args.retrieve_buf,
        "temperature": args.temperature,
        "attn_backend": args.attn_backend,
        "subsample_ratio": args.subsample,
        "pinned_frame_indices": args.pinned,
        "voxel_size": args.voxel_size,
        "voxel_num": args.voxel_num,
        "conf_threshold": args.voxel_conf,
        "voxel_buf_cap": args.voxel_buf_cap,
        "voxel_piv_cap": args.voxel_piv_cap,
        "voxel_backend": args.voxel_backend,
        "allocator": args.allocator,
    }
    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=dtype)
        if device == "cuda"
        else nullcontext()
    )
    with torch.no_grad(), autocast_ctx:
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
