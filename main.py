import logging
import random
import numpy as np
import os
import sys
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

# Configure CUDA settings
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(root_dir, "src"))

from vggt.utils.load_fn import load_and_preprocess_images_square

from stream3r.stream_session import StreamSession

from vggt.models.vggt import VGGT
from sparsevggt.models.sparsevggt import SparseVGGT

from streamvggt.models.streamvggt import StreamVGGT
from stream3r.models.stream3r import STream3R

model_paths = {
    "vggt": 'facebook/VGGT-1B',
    "stream3r": 'yslan/STream3R',
    "sparsevggt": os.path.join(root_dir, 'ckpt', 'vggt', 'model.pt'),
    "streamvggt": 'lch01/StreamVGGT',
}

model_wrappers = {
    "vggt": VGGT,
    "stream3r": STream3R,
    "sparsevggt": SparseVGGT,
    "streamvggt": StreamVGGT
}


def load_model(model_name, device):

    # Run VGGT for camera and depth estimation
    if model_name == "sparsevggt":
        model = model_wrappers[model_name]()
        ckpt = torch.load(os.path.join(root_dir, model_paths[model_name]), map_location="cpu")
        model.load_state_dict(ckpt, strict=True)
    else:
        model = model_wrappers[model_name].from_pretrained(model_paths[model_name])
    model.eval()
    model = model.to(device)

    return model


def run_model(model, images, model_name, mode='full', streaming=False, kv_cache_list=None):
    if model_name == "vggt":
        if streaming or mode != "full":
            logger.warning(
                "Warning: VGGT only supports 'full' attention mode without streaming. Switching to 'full' mode."
            )
        streaming = False
        mode = "full"
        predictions = model(images)
    elif model_name == "sparsevggt":
        if streaming or mode != "full":
            logger.warning(
                "Warning: SparseVGGT only supports 'full' attention mode without streaming. Switching to 'full' mode."
            )
        streaming = False
        mode = "full"
        predictions = model(images, mode=mode)
    elif model_name == "stream3r":
        if streaming:
        # Use StreamSession for sequential processing
            if mode == "full":
                logger.warning(
                                "Warning: Streaming mode does not support 'full' attention mode. Switching to 'causal' mode."
                            )
                mode = "causal"

            session = StreamSession(model, mode=mode)

            # Process images one by one to simulate streaming inference
            for i in range(images.shape[0]):
                image = images[i : i + 1]
                predictions = session.forward_stream(image)

            session.clear()
        else:
            # Use batch processing (original behavior)
            predictions = model(images, mode=mode)
    else:
        raise NotImplementedError(f"Model {model_name} not implemented")
    
    predictions["mode"] = mode
    predictions["streaming"] = streaming
    predictions["kv_cache_list"] = kv_cache_list

    return predictions

vggt_fixed_resolution = 518
img_load_resolution = 1024


logger = logging.getLogger("BasicLogger")
logging.basicConfig(level=logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser(description="VGGT")
    parser.add_argument("--scene_dir", type=str, required=True, help="Directory containing the scene images")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save outputs")
    parser.add_argument("--model_name", type=str, default="vggt", choices=["vggt", "stream3r", "streamvggt", "sparsevggt"], help="Model to use")
    parser.add_argument("--mode", type=str, default="full", choices=["causal", "window", "full"],
                        help="Processing mode")
    parser.add_argument("--streaming", action="store_true",
                        help="Use streaming mode (sequential processing)")
    parser.add_argument("--size", type=int, default=518, choices=[224, 512, 518],
                        help="Input image size for the model (VGGT uses 518, Stream3R uses 224 or 512)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    return args


def run(model, images, device, dtype, args):
    # images: [S, 3, H, W]
    if args.size == 518:
        resolution = (336, 518)
    elif args.size == 512:
        resolution = (384, 512)
    elif args.size == 224:
        resolution = (224, 224)
    else:
        raise NotImplementedError
    assert len(images.shape) == 4
    assert images.shape[1] == 3

    # hard-coded to use 518 for VGGT
    images = F.interpolate(images, size=resolution, mode="bilinear", align_corners=False)
    logger.info(f"Input images shape: {images.shape}")
    with torch.no_grad():
        with torch.amp.autocast(
            device_type="cuda", dtype=dtype
        ):
            predictions = run_model(model, images, args.model_name, mode=args.mode, streaming=args.streaming)
            args.mode = predictions.get("mode", args.mode)
            args.streaming = predictions.get("streaming", args.streaming)
            predictions.pop("mode", None)
            predictions.pop("streaming", None)
    return predictions

def main(args):
    # Set seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)  # for multi-GPU
    logger.info(f"Setting seed as: {args.seed}")

    # Determine device and dtype for model inference
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    # Initialize the model and load the pretrained weights.
    # This will automatically download the model weights the first time it's run, which may take a while.
    model = load_model(args.model_name, device)

    # save model
    # os.makedirs(os.path.join(root_dir, 'ckpt', f'{args.model_name}'), exist_ok=True)
    # torch.save(model.state_dict(), os.path.join(root_dir, 'ckpt', f'{args.model_name}', 'model.pt'))

    # Load and preprocess example images (replace with your own image paths)
    image_dir = Path(args.scene_dir)/"images"
    import re

    def numerical_sort(path: Path):
        # path 是 Path 对象
        match = re.search(r'(\d+)', path.stem)  # path.stem = 文件名去掉扩展名
        return int(match.group(1)) if match else -1

    image_dir = Path(args.scene_dir) / "images"
    image_paths = sorted(image_dir.glob("*.png"), key=numerical_sort)
    
    if len(image_paths) == 0:
        raise ValueError(f"No images found in {image_dir}")

    # Load images and original coordinates
    # Load Image in 1024x1024, while running VGGT with 518

    images, original_coords = load_and_preprocess_images_square(image_paths, img_load_resolution)
    images = images.to(device)
    # original_coords = original_coords.to(device)

    # Run VGGT to estimate camera and depth
    # Run with 518x518 images
    prediction = run(model, images, device, dtype, args)

    # TODO: What you want to do with the prediction

if __name__ == "__main__":
    args = parse_args()
    main(args)