#!/usr/bin/env python3
"""
Test script for STream3R model performance evaluation.
Based on app.py run_model function, this script processes images and saves results as NPY files.
"""

import os
import argparse
import time
import glob
import torch
import numpy as np
from datetime import datetime

from stream3r.models.stream3r import STream3R
from stream3r.stream_session import StreamSession
from stream3r.models.components.utils.load_fn import load_and_preprocess_images
from stream3r.models.components.utils.pose_enc import pose_encoding_to_extri_intri
from stream3r.models.components.utils.geometry import unproject_depth_map_to_point_map


def run_model_test(target_dir: str, model: STream3R, mode: str = "causal", streaming: bool = False) -> dict:
    """
    Run the STream3R model on images in the 'target_dir/images' folder and return predictions.
    This is adapted from app.py run_model function for performance testing.

    Args:
        target_dir: Directory containing the images subfolder
        model: STream3R model instance
        mode: Processing mode ("causal", "window", or "full")
        streaming: If True, use StreamSession for sequential processing; if False, use batch processing
    """
    print(f"Processing images from {target_dir}")

    # Device check
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available():
        print("Warning: CUDA is not available. Using CPU.")
        device = "cpu"

    # Move model to device
    model = model.to(device)
    model.eval()

    # Load and preprocess images
    # image_names = glob.glob(os.path.join(target_dir, "images", "*"))
    image_names = glob.glob(os.path.join(target_dir, "*"))
    image_names = sorted(image_names)
    print(f"Found {len(image_names)} images")
    if len(image_names) == 0:
        raise ValueError("No images found. Check your input directory.")

    images = load_and_preprocess_images(image_names).to(device)
    print(f"Preprocessed images shape: {images.shape}")

    # Run inference with timing
    print(f"Running inference in {'streaming' if streaming else 'batch'} mode with {mode} attention...")
    
    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    else:
        dtype = torch.float32

    inference_start = time.time()
    
    with torch.no_grad():
        with torch.amp.autocast(dtype=dtype, device_type=device):
            if streaming:
                # Use StreamSession for sequential processing
                if mode == "full":
                    print("Warning: Streaming mode does not support 'full' attention mode. Switching to 'causal' mode.")
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

    inference_time = time.time() - inference_start
    print(f"Inference completed in {inference_time:.3f} seconds")

    # Convert pose encoding to extrinsic and intrinsic matrices
    print("Converting pose encoding to extrinsic and intrinsic matrices...")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic # (S, 3, 4)
    predictions["intrinsic"] = intrinsic # (S, 3, 3)

    # Convert tensors to numpy
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)  # remove batch dimension
    predictions['pose_enc_list'] = None  # remove pose_enc_list

    # Generate world points from depth map
    print("Computing world points from depth map...")
    depth_map = predictions["depth"]  # (S, H, W, 1)
    world_points = unproject_depth_map_to_point_map(depth_map, predictions["extrinsic"], predictions["intrinsic"])
    predictions["world_points_from_depth"] = world_points

    # Add timing information
    predictions["inference_time"] = inference_time
    predictions["num_images"] = len(image_names)
    predictions["mode"] = mode
    predictions["streaming"] = streaming

    # Clean up
    if device == "cuda":
        torch.cuda.empty_cache()
    
    return predictions


def main():
    parser = argparse.ArgumentParser(description="Test STream3R model performance")
    parser.add_argument("--input_dir", type=str, required=True, 
                        help="Input directory containing 'images' subfolder")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Output directory for saving NPY results")
    parser.add_argument("--mode", type=str, default="causal", choices=["causal", "window", "full"],
                        help="Processing mode")
    parser.add_argument("--streaming", action="store_true",
                        help="Use streaming mode (sequential processing)")
    parser.add_argument("--model_name", type=str, default="yslan/STream3R",
                        help="Model name/path for STream3R")
    
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model: {args.model_name}")
    model = STream3R.from_pretrained(args.model_name)
    
    # Run model test
    start_time = time.time()
    predictions = run_model_test(
        target_dir=args.input_dir,
        model=model,
        mode=args.mode,
        streaming=args.streaming
    )
    total_time = time.time() - start_time
    
    # Generate output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    streaming_suffix = "streaming" if args.streaming else "batch"
    output_filename = f"predictions_{args.mode}_{streaming_suffix}_{timestamp}.npy"
    output_path = os.path.join(args.output_dir, output_filename)
    
    # Save results
    print(f"Saving results to {output_path}")
    np.save(output_path, predictions)
    
    # Print performance summary
    print("\n" + "="*50)
    print("PERFORMANCE SUMMARY")
    print("="*50)
    print(f"Input directory: {args.input_dir}")
    print(f"Number of images: {predictions['num_images']}")
    print(f"Processing mode: {args.mode}")
    print(f"Streaming mode: {args.streaming}")
    print(f"Inference time: {predictions['inference_time']:.3f} seconds")
    print(f"Total time: {total_time:.3f} seconds")
    print(f"Time per image: {predictions['inference_time']/predictions['num_images']:.3f} seconds")
    print(f"Results saved to: {output_path}")
    
    # Save performance log
    log_filename = f"performance_log_{timestamp}.txt"
    log_path = os.path.join(args.output_dir, log_filename)
    with open(log_path, 'w') as f:
        f.write(f"STream3R Performance Test Results\n")
        f.write(f"Timestamp: {datetime.now()}\n")
        f.write(f"Input directory: {args.input_dir}\n")
        f.write(f"Number of images: {predictions['num_images']}\n")
        f.write(f"Processing mode: {args.mode}\n")
        f.write(f"Streaming mode: {args.streaming}\n")
        f.write(f"Inference time: {predictions['inference_time']:.3f} seconds\n")
        f.write(f"Total time: {total_time:.3f} seconds\n")
        f.write(f"Time per image: {predictions['inference_time']/predictions['num_images']:.3f} seconds\n")
        f.write(f"Results file: {output_filename}\n")
    
    print(f"Performance log saved to: {log_path}")


if __name__ == "__main__":
    main()