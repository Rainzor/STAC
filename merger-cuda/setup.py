#!/usr/bin/env python
# Copyright (c) 2024-2026 CausalVGGT Authors
# SPDX-License-Identifier: Apache-2.0
"""
Setup script for merger-cuda (CausalVGGT Stateful CUDA Merger).

This package provides stateful C++ merger class:
- MergerWrapper: Tensor-owning wrapper (recommended)

Usage:
    # Install from the merger-cuda directory
    cd merger-cuda && pip install .
    
    # Development install (editable mode)
    cd merger-cuda && pip install -e .[dev]
    
    # Install from project root (non-editable only)
    pip install merger-cuda

Environment Requirements:
    export CUDA_HOME=/usr/local/cuda-11.8
    export PATH=$CUDA_HOME/bin:$PATH

After installation:
    from merger_cuda import has_merger_wrapper, create_merger_wrapper, MergerWrapper
"""

import os
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

# Get absolute path for this setup.py's directory
_project_root = os.path.dirname(os.path.abspath(__file__))

# Source paths: relative (required for editable installs)
CSRC_DIR = "csrc"
KERNELS_DIR = os.path.join(CSRC_DIR, "kernels")

# Include paths: MUST be absolute (compilation happens in temp directories)
INCLUDE_DIR_ABS = os.path.join(_project_root, "csrc", "include")
CSRC_DIR_ABS = os.path.join(_project_root, "csrc")

# Collect source files (relative paths)
sources = [
    os.path.join(CSRC_DIR, "bindings.cpp"),
    os.path.join(CSRC_DIR, "stub_ops.cu"),
    os.path.join(CSRC_DIR, "merger_wrapper.cu"),  # MergerWrapper (tensor-owning wrapper)
    os.path.join(KERNELS_DIR, "merger_pipeline.cu"),  # Merger pipeline orchestration
    os.path.join(KERNELS_DIR, "merger_kernels.cu"),  # Merger kernel implementations
]

# Compiler flags
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
    ],
}

setup(
    name="merger-cuda",
    packages=['merger_cuda'],
    version='0.1.0',
    description='Stateful C++ merger class for CUDA',
    ext_modules=[
        CUDAExtension(
            name="merger_cuda._ext",
            sources=sources,
            include_dirs=[INCLUDE_DIR_ABS, CSRC_DIR_ABS],
            extra_compile_args=extra_compile_args,
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
