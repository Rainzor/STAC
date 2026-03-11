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
    cd merger-cuda && pip install -e .
    
    # Install from project root
    pip install -e merger-cuda

Environment Requirements:
    export CUDA_HOME=/usr/local/cuda-12.8
    export PATH=$CUDA_HOME/bin:$PATH

After installation:
    from merger_cuda import has_merger_wrapper, create_merger_wrapper, MergerWrapper
"""

import os
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

# Get absolute path for this setup.py's directory
_project_root = os.path.dirname(os.path.abspath(__file__))

# Sources: must be relative to setup.py dir (setuptools requirement)
# Includes: must be absolute (compilation happens in temp directories)
CSRC_REL = "csrc"
KERNELS_REL = os.path.join(CSRC_REL, "kernels")
INCLUDE_DIR_ABS = os.path.join(_project_root, "csrc", "include")
CSRC_DIR_ABS = os.path.join(_project_root, "csrc")

sources = [
    os.path.join(CSRC_REL, "bindings.cpp"),
    os.path.join(CSRC_REL, "stub_ops.cu"),
    os.path.join(CSRC_REL, "merger_wrapper.cu"),
    os.path.join(KERNELS_REL, "merger_pipeline.cu"),
    os.path.join(KERNELS_REL, "merger_kernels.cu"),
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
    version="0.1.0",
    description="merger-cuda: CUDA kernels for CausalVGGT KV Merger Pipeline",
    author="CausalVGGT Authors",
    license="Apache-2.0",
    python_requires=">=3.8",
    install_requires=["torch>=2.0.0"],
    packages=['merger_cuda'],
    package_data={"merger_cuda": ["*.pyi"]},
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
