import os

import torch
import logging
from safetensors.torch import load_file as load_safetensors

root_dir = os.path.dirname(os.path.abspath(__file__))
logger = logging.getLogger("model wrapper")


# Core imports (always required)
from stream_session import StreamSession
from causalvggt.models.vggt import CausalVGGT

ckpt_root = os.path.join(root_dir, 'ckpt')

model_paths = {
    "stream3r": os.path.join(ckpt_root, 'stream3r'),
    "streamvggt": os.path.join(ckpt_root, 'streamvggt'),
}

model_wrappers = {
    "causalvggt": CausalVGGT,
}

stream_sessions = {
    "causalvggt": StreamSession
}

def _load_checkpoint(ckpt_path):
    """Load checkpoint from a file path or from a directory."""
    if os.path.isfile(ckpt_path):
        if ckpt_path.endswith(".safetensors"):
            logger.info(f"Loading checkpoint from {ckpt_path}")
            return load_safetensors(ckpt_path)
        if ckpt_path.endswith(".pt") or ckpt_path.endswith(".pth"):
            logger.info(f"Loading checkpoint from {ckpt_path}")
            return torch.load(ckpt_path, map_location="cpu")
        raise ValueError(
            f"Unsupported checkpoint file: {ckpt_path}. "
            "Expected '.safetensors', '.pt', or '.pth'."
        )

    safetensors_path = os.path.join(ckpt_path, 'model.safetensors')
    pt_path = os.path.join(ckpt_path, 'model.pt')
    pth_path = os.path.join(ckpt_path, 'model.pth')
    if os.path.isfile(safetensors_path):
        logger.info(f"Loading checkpoint from {safetensors_path}")
        return load_safetensors(safetensors_path)
    if os.path.isfile(pt_path):
        logger.info(f"Loading checkpoint from {pt_path}")
        return torch.load(pt_path, map_location="cpu")
    if os.path.isfile(pth_path):
        logger.info(f"Loading checkpoint from {pth_path}")
        return torch.load(pth_path, map_location="cpu")
    raise FileNotFoundError(
        f"No checkpoint found at {ckpt_path}. "
        "Expected a file path (*.safetensors, *.pt, or *.pth), "
        "or a directory containing 'model.safetensors', 'model.pt', or 'model.pth'."
    )

def _safe_load_state_dict(model, ckpt):
    """Load state dict allowing extra keys (unused heads) but rejecting missing ones."""
    result = model.load_state_dict(ckpt, strict=False)
    if result.missing_keys:
        raise RuntimeError(f"Missing keys in checkpoint: {result.missing_keys}")
    if result.unexpected_keys:
        logger.info(f"Skipped {len(result.unexpected_keys)} extra checkpoint keys "
                     f"(unused heads): {result.unexpected_keys[:5]}{'...' if len(result.unexpected_keys) > 5 else ''}")

def load_model(model_name, base_model='stream3r', device='cuda', model_path=None):
    if model_name != "causalvggt":
        raise ValueError(f"Unsupported model_name '{model_name}'. Only 'causalvggt' is supported.")
    if base_model not in model_paths:
        raise ValueError(f"Unsupported base_model '{base_model}'. Choose from: {list(model_paths.keys())}")

    model = model_wrappers[model_name](base_model=base_model)
    ckpt_source = model_path if model_path is not None else model_paths[base_model]
    ckpt = _load_checkpoint(ckpt_source)
    _safe_load_state_dict(model, ckpt)
    model.eval()
    model = model.to(device)

    return model

STAC_DEFAULTS = {
    "window_size": (4, 0),
    "chunk_size": (4, 1),
    "hh_size": (2, 0),
    "retrieval_size": (2, 0),
    "return_buf": (True, False),
    "voxel_backend": ("cuda", "python"),
    "allocator": ("segment", "slab"),
}

def run_model(model, images, model_name, mode='full',
              streaming=False, dtype=torch.bfloat16, device='cuda', 
              **kwargs
              ):
    if model_name != "causalvggt":
        raise NotImplementedError(f"Model '{model_name}' not supported. Only 'causalvggt' is supported.")

    # Keep user-facing mode so multi-scene eval (launch.py) does not overwrite args.mode
    # with the expanded internal mode and break the next scene (e.g. window_size would stay 0).
    user_mode = mode
    if mode == "stac":
        mode = "window_chunk_merge"
        streaming = True
        for k, (stac_val, argparse_default) in STAC_DEFAULTS.items():
            if kwargs.get(k, argparse_default) == argparse_default:
                kwargs[k] = stac_val
        logger.info(f"Mode 'stac' expanded: mode=window_chunk_merge, streaming=True, "
                     f"win={kwargs.get('window_size')}, ck={kwargs.get('chunk_size')}, "
                     f"hh={kwargs.get('hh_size')}, ret_sz={kwargs.get('retrieval_size')}, "
                     f"ret_buf={kwargs.get('return_buf')}")

    processed_frames = images.shape[0]
    if streaming:
        logger.info("Using streaming mode for CausalVGGT.")
        if mode == "full":
            logger.warning("Warning: you are trying to use 'full' attention mode with streaming, which will cause high memory usage.")
        cam_cache_update = kwargs.get("cam_cache_update", False)
        kwargs.pop("cam_cache_update", None)
        session: StreamSession = stream_sessions[model_name](
            model, device=device, cam_cache_update=cam_cache_update)

        session.pipeline(images, mode=mode,
                         dtype=dtype, device=device,
                         **kwargs)
        predictions = session.get_all_predictions()
        benchmark_metrics = session.get_benchmark()
        total_time = 0
        for k in benchmark_metrics:
            benchmark_metrics[k] = benchmark_metrics[k] / processed_frames
            total_time += benchmark_metrics[k]
            logger.info(f" Average {k} time per frame: {benchmark_metrics[k]:.2f} ms")
        logger.info(f"Total average time per frame: {total_time:.2f} ms, FPS: {1000/total_time:.1f} ")
        benchmark_metrics["infer_fps"] = 1000.0 / total_time if total_time > 0 else 0
        predictions["timing"] = benchmark_metrics
        predictions["merger"] = session.get_stats()
        session.clear()
    else:
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
        logger.info(f"Total average time per frame: {total_time:.2f}  ms, FPS: {1000/total_time:.1f} ")
        benchmark_metrics["infer_fps"] = 1000.0 / total_time if total_time > 0 else 0
        predictions["timing"] = benchmark_metrics

    predictions["mode"] = user_mode
    predictions["streaming"] = streaming
    # Effective config actually used (e.g. after stac expansion) for accurate metrics
    predictions["effective_config"] = {
        "mode": mode,
        "streaming": streaming,
        "window_size": kwargs.get("window_size"),
        "chunk_size": kwargs.get("chunk_size"),
        "hh_size": kwargs.get("hh_size"),
        "retrieval_size": kwargs.get("retrieval_size"),
        "return_buf": kwargs.get("return_buf"),
        "temperature": kwargs.get("temperature"),
        "attn_backend": kwargs.get("attn_backend"),
        "subsample_ratio": kwargs.get("subsample_ratio"),
        "voxel_size": kwargs.get("voxel_size"),
        "voxel_num": kwargs.get("voxel_num"),
        "conf_threshold": kwargs.get("conf_threshold"),
        "voxel_buf_cap": kwargs.get("voxel_buf_cap"),
        "voxel_piv_cap": kwargs.get("voxel_piv_cap"),
        "voxel_backend": kwargs.get("voxel_backend"),
        "allocator": kwargs.get("allocator"),
        "pinned_frame_indices": kwargs.get("pinned_frame_indices"),
    }

    return predictions