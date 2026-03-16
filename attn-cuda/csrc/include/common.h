#pragma once

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#include <cuda_bf16.h>
#endif

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>

#include <cmath>
#include <vector>
#include <string>
#include <stdexcept>

#define ATTN_CUDA_VERSION_MAJOR 0
#define ATTN_CUDA_VERSION_MINOR 1
#define ATTN_CUDA_VERSION_PATCH 0

namespace stac_attn {

#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            throw std::runtime_error(                                           \
                std::string("CUDA error at ") + __FILE__ + ":" +                \
                std::to_string(__LINE__) + ": " + cudaGetErrorString(err));     \
        }                                                                       \
    } while (0)

inline void check_cuda_tensor(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

inline void check_half_tensor(const torch::Tensor& t, const std::string& name) {
    check_cuda_tensor(t, name);
    TORCH_CHECK(t.dtype() == torch::kHalf || t.dtype() == torch::kBFloat16,
                name, " must be fp16 or bf16");
}

inline cudaStream_t get_cuda_stream() {
    return c10::cuda::getCurrentCUDAStream().stream();
}

struct StacFlashParams {
    void* __restrict__ q_ptr;
    void* __restrict__ k_ptr;
    void* __restrict__ v_ptr;
    void* __restrict__ o_ptr;
    float* __restrict__ lse_ptr;
    float* __restrict__ bias_ptr;
    float* __restrict__ colsum_ptr;

    int batch_size;
    int seqlen_q;
    int seqlen_k;
    int num_heads;
    int head_dim;

    int seqlen_q_sub;   // M_sub: number of subsampled Q rows for colsum
    int q_m_stride;     // physical row stride for Q/LSE in colsum (1 = no subsampling)

    int q_row_stride;
    int k_row_stride;
    int v_row_stride;
    int o_row_stride;
    int q_head_stride;
    int k_head_stride;
    int v_head_stride;
    int o_head_stride;
    int q_batch_stride;
    int k_batch_stride;
    int v_batch_stride;
    int o_batch_stride;

    float softmax_scale;
    float softmax_scale_log2;

    bool has_bias;
    bool has_colsum;
};

}  // namespace stac_attn
