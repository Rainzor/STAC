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
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

_project_root = os.path.dirname(os.path.abspath(__file__))

CUTLASS_DIR = os.environ.get(
    "CUTLASS_DIR",
    os.path.join(_project_root, "..", "attn-flash",
                 "flash-attention", "csrc", "cutlass", "include"))

FA2_DIR = os.path.join(_project_root, "..", "attn-flash",
                       "flash-attention", "hopper")

CSRC_DIR = "csrc"
INCLUDE_DIR_ABS = os.path.join(_project_root, "csrc", "include")
CSRC_DIR_ABS = os.path.join(_project_root, "csrc")

sources = [
    os.path.join(CSRC_DIR, "bindings.cpp"),
    os.path.join(CSRC_DIR, "launch.cu"),
]

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
        "-gencode=arch=compute_80,code=sm_80",
        "-gencode=arch=compute_86,code=sm_86",
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
            include_dirs=[INCLUDE_DIR_ABS, CSRC_DIR_ABS, CUTLASS_DIR, FA2_DIR],
            extra_compile_args=extra_compile_args,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
