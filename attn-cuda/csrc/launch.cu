#include "include/common.h"
#include "include/kernel_traits.h"
#include "include/softmax.h"
#include "kernels/fwd_kernel.h"
#include "kernels/colsum_kernel.h"

namespace stac_attn {

template <typename Traits, bool HasBias>
void launch_fwd_kernel(StacFlashParams& params, cudaStream_t stream) {
    constexpr int kBlockM = Traits::kBlockM;

    int const grid_m = (params.seqlen_q + kBlockM - 1) / kBlockM;
    int const grid_bh = params.batch_size * params.num_heads;
    dim3 grid(grid_m, grid_bh);
    dim3 block(Traits::kNumThreads);

    int smem_size = Traits::kSmemSize;
    auto kernel = fwd_bias_colsum_kernel<Traits, HasBias, false>;

    if (smem_size > 48 * 1024) {
        CUDA_CHECK(cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }

    kernel<<<grid, block, smem_size, stream>>>(params);
    CUDA_CHECK(cudaGetLastError());
}

template <typename Traits, bool HasBias>
void launch_colsum_kernel(StacFlashParams& params, cudaStream_t stream) {
    constexpr int kBlockN = Traits::kBlockN;
    constexpr int kBlockM = Traits::kBlockM;
    constexpr int kHeadDim = Traits::kHeadDim;
    constexpr int kNWarps = Traits::kNWarps;

    int const grid_n = (params.seqlen_k + kBlockN - 1) / kBlockN;
    int const grid_bh = params.batch_size * params.num_heads;
    dim3 grid(grid_n, grid_bh);
    dim3 block(Traits::kNumThreads);

    using SmemLayoutAtom = typename Traits::SmemLayoutAtom;
    using SmemK1 = decltype(tile_to_shape(SmemLayoutAtom{},
                                           Shape<Int<kBlockN>, Int<kHeadDim>>{}));
    using SmemQ1 = decltype(tile_to_shape(SmemLayoutAtom{},
                                           Shape<Int<kBlockM>, Int<kHeadDim>>{}));
    // Main loop: K tile + Q tile + double-buffered LSE [2*kBlockM] + bias buf
    int smem_main = (int(cute::cosize(SmemK1{})) + int(cute::cosize(SmemQ1{})))
                    * sizeof(typename Traits::Element)
                    + 2 * kBlockM * sizeof(float)
                    + (HasBias ? kBlockN * sizeof(float) : 0);
    // Reduction tail reuses smem as per-warp buffer [kNWarps][kBlockN]
    int smem_reduce = kNWarps * kBlockN * sizeof(float);
    int smem_size = smem_main > smem_reduce ? smem_main : smem_reduce;

    auto kernel = colsum_n_major_kernel<Traits, HasBias>;

    if (smem_size > 48 * 1024) {
        CUDA_CHECK(cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }

    kernel<<<grid, block, smem_size, stream>>>(params);
    CUDA_CHECK(cudaGetLastError());
}

static constexpr int kBlockN_Large_Threshold = 2048;

template <typename Element, bool HasBias>
void run_fwd_kernel(StacFlashParams& params, cudaStream_t stream) {
    if (params.seqlen_k >= kBlockN_Large_Threshold) {
        launch_fwd_kernel<FwdKernelTraits<Element, 128, 128, 64, 8>, HasBias>(params, stream);
    } else {
        launch_fwd_kernel<FwdKernelTraits<Element>, HasBias>(params, stream);
    }
}

template <typename Element, bool HasBias>
void run_colsum_kernel(StacFlashParams& params, cudaStream_t stream) {
    if (params.seqlen_k >= kBlockN_Large_Threshold) {
        launch_colsum_kernel<FwdKernelTraits<Element, 128, 128>, HasBias>(params, stream);
    } else {
        launch_colsum_kernel<FwdKernelTraits<Element>, HasBias>(params, stream);
    }
}

void stac_flash_fwd(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor out, torch::Tensor lse,
    float softmax_scale,
    c10::optional<torch::Tensor> bias,
    c10::optional<torch::Tensor> colsum,
    int seqlen_q_sub,
    int q_m_stride)
{
    StacFlashParams params;
    params.q_ptr = q.data_ptr();
    params.k_ptr = k.data_ptr();
    params.v_ptr = v.data_ptr();
    params.o_ptr = out.data_ptr();
    params.lse_ptr = lse.data_ptr<float>();
    params.bias_ptr = bias.has_value() ? bias.value().data_ptr<float>() : nullptr;
    params.colsum_ptr = colsum.has_value() ? colsum.value().data_ptr<float>() : nullptr;

    params.batch_size = q.size(0);
    params.seqlen_q = q.size(1);
    params.seqlen_k = k.size(1);
    params.num_heads = q.size(2);
    params.head_dim = q.size(3);

    params.q_row_stride = q.stride(1);
    params.k_row_stride = k.stride(1);
    params.v_row_stride = v.stride(1);
    params.o_row_stride = out.stride(1);
    params.q_head_stride = q.stride(2);
    params.k_head_stride = k.stride(2);
    params.v_head_stride = v.stride(2);
    params.o_head_stride = out.stride(2);
    params.q_batch_stride = q.stride(0);
    params.k_batch_stride = k.stride(0);
    params.v_batch_stride = v.stride(0);
    params.o_batch_stride = out.stride(0);

    params.softmax_scale = softmax_scale;
    params.softmax_scale_log2 = softmax_scale * float(M_LOG2E);

    params.seqlen_q_sub = seqlen_q_sub;
    params.q_m_stride = q_m_stride;

    params.has_bias = bias.has_value();
    params.has_colsum = colsum.has_value();

    cudaStream_t stream = get_cuda_stream();

    auto dispatch = [&](auto element_type) {
        using Element = decltype(element_type);

        if (params.has_bias) {
            run_fwd_kernel<Element, true>(params, stream);
        } else {
            run_fwd_kernel<Element, false>(params, stream);
        }

        if (params.has_colsum) {
            if (params.has_bias) {
                run_colsum_kernel<Element, true>(params, stream);
            } else {
                run_colsum_kernel<Element, false>(params, stream);
            }
        }
    };

    if (q.dtype() == torch::kHalf) {
        dispatch(cutlass::half_t{});
    } else {
        dispatch(cutlass::bfloat16_t{});
    }
}

}  // namespace stac_attn
