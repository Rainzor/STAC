#!/usr/bin/env python
"""
Setup script for attn-cuda (STAC Flash Attention with bias + colsum).

Provides a C++/CUDA flash attention kernel using CUTLASS cute MMA atoms
and cp_async for SM80 (A100), forward-only, D=64, fp16/bf16.

Usage:
    cd attn-cuda && pip install -e . --no-build-isolation
"""

import os
from setuptools import setup, find_packages
import torch
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

_project_root = os.path.dirname(os.path.abspath(__file__))

_VENDORED_CUTLASS_DIR = os.path.join(
    _project_root, "third_party", "cutlass", "include"
)
_LEGACY_CUTLASS_DIR = os.path.join(
    _project_root, "..", "attn-flash", "flash-attention", "csrc", "cutlass", "include"
)

CUTLASS_DIR = os.environ.get("CUTLASS_DIR", _VENDORED_CUTLASS_DIR)
if not os.path.exists(os.path.join(CUTLASS_DIR, "cute", "tensor.hpp")):
    if CUTLASS_DIR == _VENDORED_CUTLASS_DIR and os.path.exists(
        os.path.join(_LEGACY_CUTLASS_DIR, "cute", "tensor.hpp")
    ):
        CUTLASS_DIR = _LEGACY_CUTLASS_DIR
    else:
        raise RuntimeError(
            "CUTLASS headers not found. Expected 'cute/tensor.hpp' under "
            f"{CUTLASS_DIR}. Set CUTLASS_DIR explicitly or vendor headers into "
            "attn-cuda/third_party/cutlass/include."
        )

CSRC_DIR = "csrc"
INCLUDE_DIR_ABS = os.path.join(_project_root, "csrc", "include")
CSRC_DIR_ABS = os.path.join(_project_root, "csrc")

sources = [
    os.path.join(CSRC_DIR, "bindings.cpp"),
    os.path.join(CSRC_DIR, "launch.cu"),
]


def _choose_arch_list() -> str:
    # 1) Project-specific override
    arch = os.environ.get("STAC_CUDA_ARCHS")
    if arch:
        return arch

    # 2) Standard PyTorch override
    arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if arch:
        return arch

    # 3) Profile fallback
    profile = os.environ.get("STAC_BUILD_PROFILE", "dev").strip().lower()
    if profile == "release":
        return "8.0;8.6;8.9;9.0+PTX"

    # 4) Dev fallback: current GPU only
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        return f"{major}.{minor}"

    # 5) Last fallback for GPU-less build hosts
    return "8.0;8.6"


arch_list = _choose_arch_list()
os.environ["TORCH_CUDA_ARCH_LIST"] = arch_list
print(f"[attn-cuda] TORCH_CUDA_ARCH_LIST={arch_list}")

extra_compile_args = {
    "cxx": ["-O3", "-std=c++17"],
    "nvcc": [
        "-O3",
        "--fmad=true",
        "-std=c++17",
        "--expt-relaxed-constexpr",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "--threads=4",
    ],
}

setup(
    name="attn-cuda",
    version="0.1.0",
    packages=find_packages(),
    description="STAC Flash Attention with vector bias and column-sum scoring",
    ext_modules=[
        CUDAExtension(
            name="attn_cuda._ext",
            sources=sources,
            include_dirs=[INCLUDE_DIR_ABS, CSRC_DIR_ABS, CUTLASS_DIR],
            extra_compile_args=extra_compile_args,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
