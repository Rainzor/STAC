# Copyright (c) 2024-2026 CausalVGGT Authors
# SPDX-License-Identifier: Apache-2.0
"""
merger-cuda: Stateful C++ Merger Classes for CausalVGGT.

This module provides the pure-CUDA stateful merger implementation:
- MergerWrapper: Tensor-owning wrapper (recommended)

Usage:
    from merger_cuda import cuda_available, has_merger_wrapper, create_merger_wrapper
    
    if cuda_available() and has_merger_wrapper():
        wrapper = create_merger_wrapper(num_heads=16, head_dim=64)
        wrapper.insert_and_merge(K, V, S, VX, num_voxels=1000)
        K_out, V_out, M_out, bias = wrapper.retrieve(voxel_ids)
"""

# ============================================================================
# Extension Import
# ============================================================================

_C = None
_cuda_available = False

try:
    import torch  # loads libc10.so / libtorch.so so the dynamic linker can resolve _ext's deps
    from . import _ext as _C
    _cuda_available = _C.is_available() if hasattr(_C, 'is_available') else True
except ImportError:
    _C = None
    _cuda_available = False


# ============================================================================
# Public API - Availability Checks
# ============================================================================

def cuda_available() -> bool:
    """Check if CUDA extension is available and functional."""
    return _cuda_available


def get_version() -> str:
    """Get the version of the CUDA extension."""
    if _C is not None and hasattr(_C, 'get_version'):
        return _C.get_version()
    return "0.0.0"


def get_device_info() -> dict:
    """Get CUDA device information."""
    if _C is not None and hasattr(_C, 'get_device_info'):
        return _C.get_device_info()
    return {"available": "false"}


# ============================================================================
# MergerWrapper: Tensor-owning wrapper (recommended)
# ============================================================================

def has_merger_wrapper() -> bool:
    """Check if MergerWrapper class is available."""
    return _C is not None and hasattr(_C, 'MergerWrapper')


# Lazy-load MergerWrapper class to avoid import errors if not built
MergerWrapper = None
if _C is not None and hasattr(_C, 'MergerWrapper'):
    MergerWrapper = _C.MergerWrapper


def create_merger_wrapper(
    num_heads: int,
    head_dim: int,
    pivot_cap: int = 4,
    budget_cap: int = 8,
    init_voxels: int = 1024,
    dtype=None,
    device=None,
    seg_mode: bool = False,
):
    """
    Create a MergerWrapper instance (recommended).
    
    MergerWrapper follows the diff-gaussian-rasterization pattern:
    - Wrapper layer owns all torch::Tensor storage
    - Backend layer operates on raw pointers only
    - Pointer views are automatically refreshed after tensor reallocation
    
    This is the recommended interface for the KV merger pipeline.
    
    Args:
        num_heads: Number of attention heads (H)
        head_dim: Head dimension (D)
        pivot_cap: Pivot capacity per voxel (P), default 4
        budget_cap: Buffer capacity per voxel (B), default 8
        init_voxels: Initial voxel capacity, default 1024
        dtype: Data type (default: torch.float16)
        device: CUDA device (default: current CUDA device)
        seg_mode: If True, use segmented pool storage (1 token = 1 segment).
                  Skips contiguous pool allocation entirely.
    
    Returns:
        MergerWrapper instance
    
    Raises:
        RuntimeError: If MergerWrapper is not available
        
    Example:
        >>> wrapper = create_merger_wrapper(num_heads=16, head_dim=64)
        >>> wrapper.insert_and_merge(K, V, S, VX, num_voxels=1000)
        >>> K_out, V_out, M_out, bias = wrapper.retrieve(voxel_ids)
    """
    import torch
    
    if not has_merger_wrapper():
        raise RuntimeError(
            "MergerWrapper not available. Rebuild merger_cuda with CUDA support: "
            "pip install --force-reinstall ."
        )
    
    if dtype is None:
        dtype = torch.float16
    if device is None:
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
    elif isinstance(device, str):
        device = torch.device(device)
    
    return MergerWrapper(
        num_heads, head_dim, pivot_cap, budget_cap, init_voxels,
        dtype, device, seg_mode
    )


def get_pipeline_status() -> dict:
    """
    Get the availability status of merger-cuda components.
    
    Returns:
        Dictionary mapping component names to availability status
    """
    return {
        "cuda_available": cuda_available(),
        "merger_wrapper": has_merger_wrapper(),
    }


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    # Availability checks
    "cuda_available",
    "get_version",
    "get_device_info",
    "get_pipeline_status",
    # MergerWrapper (recommended)
    "MergerWrapper",
    "has_merger_wrapper",
    "create_merger_wrapper",
]
