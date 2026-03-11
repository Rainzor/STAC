// Copyright (c) 2024-2026 CausalVGGT Authors
// SPDX-License-Identifier: Apache-2.0
//
// Common header for CausalVGGT CUDA extension

#pragma once

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <vector>
#include <string>
#include <stdexcept>

// Version info
#define SPARSEVGGT_CUDA_VERSION_MAJOR 0
#define SPARSEVGGT_CUDA_VERSION_MINOR 1
#define SPARSEVGGT_CUDA_VERSION_PATCH 0

namespace causalvggt {

// Utility macros for CUDA error checking
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            throw std::runtime_error(                                         \
                std::string("CUDA error at ") + __FILE__ + ":" +              \
                std::to_string(__LINE__) + ": " + cudaGetErrorString(err));   \
        }                                                                     \
    } while (0)

// Check tensor properties
inline void check_cuda_tensor(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

// Get CUDA stream from PyTorch
inline cudaStream_t get_cuda_stream() {
    return c10::cuda::getCurrentCUDAStream().stream();
}

// Common data structures for KV merger
struct MergerConfig {
    int num_heads;
    int head_dim;
    int pivot_cap;
    int budget_cap;
    int init_voxels;
    float sim_thresh;
    float replace_thresh;
    float score_thresh;
    
    MergerConfig() 
        : num_heads(16), head_dim(64), pivot_cap(4), budget_cap(8),
          init_voxels(1024), sim_thresh(0.75f), replace_thresh(0.5f),
          score_thresh(0.2f) {}
};

// Timing result structure
struct TimingResult {
    float init_time_ms;
    float append_time_ms;
    float merge_all2one_time_ms;
    float merge_one2one_time_ms;
    float remerge_time_ms;
    
    TimingResult() 
        : init_time_ms(0.0f), append_time_ms(0.0f), 
          merge_all2one_time_ms(0.0f), merge_one2one_time_ms(0.0f),
          remerge_time_ms(0.0f) {}
};

}  // namespace causalvggt
