#!/usr/bin/env python3
"""
Convert NPY prediction results to GLB 3D visualization files.
Based on app.py visualization functions, this script loads NPY files and generates GLB outputs.
"""

import os
import argparse
import numpy as np
from datetime import datetime

from stream3r.utils.visual_utils import predictions_to_glb


def load_predictions(npy_path: str) -> dict:
    """
    Load predictions from NPY file.
    
    Args:
        npy_path: Path to the NPY file containing predictions
        
    Returns:
        Dictionary containing predictions
    """
    print(f"Loading predictions from {npy_path}")
    
    # Load the NPY file
    predictions_data = np.load(npy_path, allow_pickle=True).item()
    
    # Key list from app.py update_visualization function
    key_list = [
        "pose_enc",
        "depth",
        "depth_conf",
        "world_points",
        "world_points_conf",
        "images",
        "extrinsic",
        "intrinsic",
        "world_points_from_depth",
    ]
    
    # Extract only the required keys for GLB generation
    predictions = {}
    for key in key_list:
        if key in predictions_data:
            predictions[key] = predictions_data[key]
        else:
            print(f"Warning: Key '{key}' not found in predictions")
    
    # Print information about loaded data
    if "num_images" in predictions_data:
        print(f"Number of images: {predictions_data['num_images']}")
    if "inference_time" in predictions_data:
        print(f"Original inference time: {predictions_data['inference_time']:.3f} seconds")
    if "mode" in predictions_data:
        print(f"Processing mode: {predictions_data['mode']}")
    if "streaming" in predictions_data:
        print(f"Streaming mode: {predictions_data['streaming']}")
    
    return predictions


def convert_to_glb(predictions: dict, output_path: str, **kwargs) -> None:
    """
    Convert predictions to GLB format using the same function as app.py.
    
    Args:
        predictions: Dictionary containing model predictions
        output_path: Path where GLB file will be saved
        **kwargs: Additional parameters for GLB generation
    """
    print(f"Converting predictions to GLB format...")
    
    # Extract parameters with defaults (same as app.py)
    conf_thres = kwargs.get("conf_thres", 50.0)
    frame_filter = kwargs.get("frame_filter", "All")
    mask_black_bg = kwargs.get("mask_black_bg", False)
    mask_white_bg = kwargs.get("mask_white_bg", False)
    show_cam = kwargs.get("show_cam", True)
    mask_sky = kwargs.get("mask_sky", False)
    prediction_mode = kwargs.get("prediction_mode", "Depthmap and Camera Branch")
    
    print(f"GLB generation parameters:")
    print(f"  Confidence threshold: {conf_thres}")
    print(f"  Frame filter: {frame_filter}")
    print(f"  Mask black background: {mask_black_bg}")
    print(f"  Mask white background: {mask_white_bg}")
    print(f"  Show camera: {show_cam}")
    print(f"  Mask sky: {mask_sky}")
    print(f"  Prediction mode: {prediction_mode}")
    
    # Create target directory for compatibility (used by predictions_to_glb)
    target_dir = os.path.dirname(output_path)
    
    # Generate GLB filename using same pattern as app.py
    # Extract mode from predictions if available, otherwise use default
    mode = "causal"  # default mode
    if "mode" in kwargs and kwargs["mode"]:
        mode = kwargs["mode"]
    
    # Build GLB file name following app.py pattern (lines 238-241)
    glb_filename = f"glbscene_{conf_thres}_{frame_filter.replace('.', '_').replace(':', '').replace(' ', '_')}_maskb{mask_black_bg}_maskw{mask_white_bg}_cam{show_cam}_sky{mask_sky}_pred{prediction_mode.replace(' ', '_')}_mode{mode}.glb"
    
    # Use the generated filename but keep the user's output directory
    if target_dir:
        final_output_path = os.path.join(target_dir, glb_filename)
    else:
        final_output_path = glb_filename
    
    print(f"Generated GLB filename: {glb_filename}")
    
    # Generate GLB scene
    glbscene = predictions_to_glb(
        predictions,
        conf_thres=conf_thres,
        filter_by_frames=frame_filter,
        mask_black_bg=mask_black_bg,
        mask_white_bg=mask_white_bg,
        show_cam=show_cam,
        mask_sky=mask_sky,
        target_dir=target_dir,
        prediction_mode=prediction_mode,
    )
    
    # Export to GLB file
    print(f"Exporting GLB to {final_output_path}")
    glbscene.export(file_obj=final_output_path)
    print(f"GLB file saved successfully!")
    
    return final_output_path


def main():
    parser = argparse.ArgumentParser(description="Convert NPY predictions to GLB visualization")
    parser.add_argument("--input_npy", type=str, required=True,
                        help="Input NPY file containing predictions")
    parser.add_argument("--output_glb", type=str, default=None,
                        help="Output GLB file path (auto-generated if not specified)")
    parser.add_argument("--conf_thres", type=float, default=50.0,
                        help="Confidence threshold for point filtering (0-100)")
    parser.add_argument("--frame_filter", type=str, default="All",
                        help="Frame filter ('All' or specific frame)")
    parser.add_argument("--mask_black_bg", action="store_true",
                        help="Filter black background points")
    parser.add_argument("--mask_white_bg", action="store_true",
                        help="Filter white background points")
    parser.add_argument("--no_show_cam", action="store_true",
                        help="Don't show camera positions")
    parser.add_argument("--mask_sky", action="store_true",
                        help="Filter sky points")
    parser.add_argument("--prediction_mode", type=str, 
                        default="Depthmap and Camera Branch",
                        choices=["Depthmap and Camera Branch", "Pointmap Branch"],
                        help="Prediction mode for visualization")
    parser.add_argument("--mode", type=str, default="causal",
                        choices=["causal", "window", "full"],
                        help="Processing mode used during inference (affects GLB filename)")
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_npy):
        raise FileNotFoundError(f"Input NPY file not found: {args.input_npy}")
    
    # Generate output path if not specified
    if args.output_glb is None:
        base_name = os.path.splitext(os.path.basename(args.input_npy))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_glb = f"{base_name}_converted_{timestamp}.glb"
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_glb)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load predictions
        predictions = load_predictions(args.input_npy)
        
        # Convert to GLB
        final_glb_path = convert_to_glb(
            predictions=predictions,
            output_path=args.output_glb,
            conf_thres=args.conf_thres,
            frame_filter=args.frame_filter,
            mask_black_bg=args.mask_black_bg,
            mask_white_bg=args.mask_white_bg,
            show_cam=not args.no_show_cam,
            mask_sky=args.mask_sky,
            prediction_mode=args.prediction_mode,
            mode=args.mode
        )
        
        print("\n" + "="*50)
        print("CONVERSION COMPLETE")
        print("="*50)
        print(f"Input NPY: {args.input_npy}")
        print(f"Output GLB: {final_glb_path}")
        print(f"GLB file size: {os.path.getsize(final_glb_path) / 1024 / 1024:.2f} MB")
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        raise


def batch_convert(input_dir: str, output_dir: str, **kwargs):
    """
    Batch convert all NPY files in a directory to GLB format.
    
    Args:
        input_dir: Directory containing NPY files
        output_dir: Directory to save GLB files
        **kwargs: Additional parameters for GLB generation
    """
    import glob
    
    os.makedirs(output_dir, exist_ok=True)
    
    npy_files = glob.glob(os.path.join(input_dir, "*.npy"))
    
    if not npy_files:
        print(f"No NPY files found in {input_dir}")
        return
    
    print(f"Found {len(npy_files)} NPY files for batch conversion")
    
    for npy_file in npy_files:
        try:
            base_name = os.path.splitext(os.path.basename(npy_file))[0]
            glb_file = os.path.join(output_dir, f"{base_name}.glb")
            
            print(f"\nProcessing: {npy_file}")
            predictions = load_predictions(npy_file)
            convert_to_glb(predictions, glb_file, **kwargs)
            
        except Exception as e:
            print(f"Error converting {npy_file}: {e}")
            continue
    
    print(f"\nBatch conversion complete. GLB files saved to {output_dir}")


if __name__ == "__main__":
    main()