// Copyright (c) 2024-2026 CausalVGGT Authors
// SPDX-License-Identifier: Apache-2.0
//
// Test CUDA operations for CausalVGGT
// Contains simple test kernels for verifying CUDA is working

#include "include/common.h"
#include <cuda_runtime.h>

namespace causalvggt {
namespace cuda {

// Simple test kernel to verify CUDA is working
__global__ void test_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f;
    }
}

// Test CUDA functionality - doubles all values in the input tensor
torch::Tensor test_cuda_op(torch::Tensor input) {
    check_cuda_tensor(input, "input");
    
    auto output = input.clone();
    int n = output.numel();
    
    if (n > 0 && output.scalar_type() == torch::kFloat32) {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        test_kernel<<<blocks, threads, 0, get_cuda_stream()>>>(
            output.data_ptr<float>(), n
        );
    }
    
    return output;
}

}  // namespace cuda
}  // namespace causalvggt
