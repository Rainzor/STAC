import os
import sys
from copy import deepcopy

import torch
import logging
from safetensors.torch import load_file as load_safetensors
# Add project root to Python path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
logger = logging.getLogger("model wrapper")


# Core imports (always required)
from causalvggt.stream_session import StreamSession
from causalvggt.models.vggt import CausalVGGT

# Optional imports for other model types
try:
    from vggt.models.vggt import VGGT
except ImportError:
    VGGT = None
    logger.debug("vggt.models.vggt not available")

ckpt_root = os.path.join(root_dir, 'ckpt')

model_paths = {
    "vggt": os.path.join(ckpt_root, 'vggt'),
    "stream3r": os.path.join(ckpt_root, 'stream3r'),
    "streamvggt": os.path.join(ckpt_root, 'streamvggt'),
}

model_wrappers = {
    "vggt": VGGT,
    "causalvggt": CausalVGGT
}

stream_sessions = {
    "causalvggt": StreamSession
}

def _load_checkpoint(ckpt_dir):
    """Load checkpoint from a directory, preferring safetensors over pt."""
    safetensors_path = os.path.join(ckpt_dir, 'model.safetensors')
    pt_path = os.path.join(ckpt_dir, 'model.pt')
    if os.path.isfile(safetensors_path):
        logger.info(f"Loading checkpoint from {safetensors_path}")
        return load_safetensors(safetensors_path)
    elif os.path.isfile(pt_path):
        logger.info(f"Loading checkpoint from {pt_path}")
        return torch.load(pt_path, map_location="cpu")
    else:
        raise FileNotFoundError(
            f"No checkpoint found in {ckpt_dir}. "
            f"Expected 'model.safetensors' or 'model.pt'."
        )

def _safe_load_state_dict(model, ckpt):
    """Load state dict allowing extra keys (unused heads) but rejecting missing ones."""
    result = model.load_state_dict(ckpt, strict=False)
    if result.missing_keys:
        raise RuntimeError(f"Missing keys in checkpoint: {result.missing_keys}")
    if result.unexpected_keys:
        logger.info(f"Skipped {len(result.unexpected_keys)} extra checkpoint keys "
                     f"(unused heads): {result.unexpected_keys[:5]}{'...' if len(result.unexpected_keys) > 5 else ''}")

def load_model(model_name, base_model='vggt', device='cuda'):

    if model_name == "causalvggt":
        model = model_wrappers[model_name](base_model=base_model)
        ckpt_dir = model_paths[base_model]
        ckpt = _load_checkpoint(ckpt_dir)
        _safe_load_state_dict(model, ckpt)
    else:
        model = model_wrappers[model_name]()
        ckpt_dir = model_paths[model_name]
        ckpt = _load_checkpoint(ckpt_dir)
        _safe_load_state_dict(model, ckpt)
    model.eval()
    model = model.to(device)

    return model

def run_model(model, images, model_name, mode='full',
              streaming=False, dtype=torch.bfloat16, device='cuda', 
              **kwargs
              ):
    if model_name == "vggt":
        if streaming or mode != "full":
            logger.warning(
                "Warning: VGGT only supports 'full' attention mode without streaming. Switching to 'full' mode."
            )
        streaming = False
        mode = "full"
        predictions = model(images)
    elif model_name == "causalvggt":
        processed_frames = images.shape[0]
        if streaming:
            logger.info("Using streaming mode for CausalVGGT.")
            if mode == "full":
                logger.warning("Warning: you are trying to use 'full' attention mode with streaming, which will cause high memory usage.")
            max_frames = kwargs.get("max_frames",50)
            cam_cache_update = kwargs.get("cam_cache_update", True)
            kwargs.pop("max_frames", None)
            kwargs.pop("cam_cache_update", None)
            session:StreamSession = stream_sessions[model_name](
                                                        model, 
                                                        device=device,
                                                        cam_cache_update=cam_cache_update,
                                                        max_frames=max_frames)
            
            session.pipeline(images, mode=mode,
                             dtype=dtype, device=device,
                             **kwargs)
            predictions = session.get_all_predictions()
            benchmark_metrics = session.get_benchmark()
            total_time = 0
            for k in benchmark_metrics:
                benchmark_metrics[k] = benchmark_metrics[k] / processed_frames
                total_time += benchmark_metrics[k]
                logger.info(f" Average {k} time per frame: {benchmark_metrics[k]:.2f}ms")
            logger.info(f"🧭 Total average time per frame: {total_time:.2f}ms, FPS: {1000/total_time:.1f} ")
            benchmark_metrics["infer_fps"] = 1000.0 / total_time if total_time > 0 else 0
            predictions["timing"] = benchmark_metrics

            predictions["merger"] = session.get_stats()

            session.clear()
        else:
            # Use batch processing (non-streaming inference)
            predictions = model(images, 
                                mode=mode, 
                                streaming=False, 
                                **kwargs)
            benchmark_metrics = predictions.get("timing", {})
            total_time = 0
            for k in benchmark_metrics:
                benchmark_metrics[k] = benchmark_metrics[k] / processed_frames
                total_time += benchmark_metrics[k]
                logger.info(f" Average {k} time per frame: {benchmark_metrics[k]:.2f}ms")
            logger.info(f"🧭 Total average time per frame: {total_time:.2f}ms, FPS: {1000/total_time:.1f} ")
            benchmark_metrics["infer_fps"] = 1000.0 / total_time if total_time > 0 else 0
            predictions["timing"] = benchmark_metrics
    else:
        raise NotImplementedError(f"Model {model_name} not implemented")
    
    predictions["mode"] = mode
    predictions["streaming"] = streaming

    return predictions