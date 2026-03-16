#pragma once

#include "include/common.h"
#include "include/kernel_traits.h"
#include "include/softmax.h"

namespace stac_attn {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////
// SM80 GEMM helpers (adapted from FA2 utils.h)
////////////////////////////////////////////////////////////////////////////////////////////////////

template<bool A_in_regs=false,
         typename Tensor0, typename Tensor1, typename Tensor2,
         typename Tensor3, typename Tensor4,
         typename TiledMma, typename TiledCopyA, typename TiledCopyB,
         typename ThrCopyA, typename ThrCopyB>
CUTLASS_DEVICE void gemm_sm80(
    Tensor0& acc, Tensor1& tCrA, Tensor2& tCrB,
    Tensor3 const& tCsA, Tensor4 const& tCsB,
    TiledMma tiled_mma,
    TiledCopyA smem_tiled_copy_A, TiledCopyB smem_tiled_copy_B,
    ThrCopyA smem_thr_copy_A, ThrCopyB smem_thr_copy_B)
{
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));
    Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    if (!A_in_regs) { cute::copy(smem_tiled_copy_A, tCsA(_, _, _0{}), tCrA_copy_view(_, _, _0{})); }
    cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
    #pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {
            if (!A_in_regs) { cute::copy(smem_tiled_copy_A, tCsA(_, _, i + 1), tCrA_copy_view(_, _, i + 1)); }
            cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
        }
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
    }
}

template<typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3,
         typename TiledMma, typename TiledCopy, typename ThrCopy>
CUTLASS_DEVICE void gemm_rs_sm80(
    Tensor0& acc, Tensor1& tCrA, Tensor2& tCrB, Tensor3 const& tCsB,
    TiledMma tiled_mma, TiledCopy smem_tiled_copy_B, ThrCopy smem_thr_copy_B)
{
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
    #pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {
            cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
        }
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Traits>
struct FwdNRows {
    static constexpr int value = 2 * Traits::kBlockM / (16 * Traits::kNWarps);
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Main forward kernel — computes O + LSE only.
// ColSum is handled by the separate N-major colsum_kernel.h.
//
// Target workload: M=4096, N=16384, H=16, D=64, B=1.
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, bool HasBias, bool HasColSum = false>
__global__ void __launch_bounds__(Traits::kNumThreads)
fwd_bias_colsum_kernel(StacFlashParams params)
{
    using Element = typename Traits::Element;
    using TiledMma = typename Traits::TiledMma;
    using SmemLayoutQ = typename Traits::SmemLayoutQ;
    using SmemLayoutK = typename Traits::SmemLayoutK;
    using SmemCopyAtom = typename Traits::SmemCopyAtom;
    using SmemCopyAtomTransposed = typename Traits::SmemCopyAtomTransposed;
    using GmemTiledCopy = typename Traits::GmemTiledCopy;
    using SmemLayoutAtom = typename Traits::SmemLayoutAtom;

    static constexpr int kBlockM = Traits::kBlockM;
    static constexpr int kBlockN = Traits::kBlockN;
    static constexpr int kHeadDim = Traits::kHeadDim;
    static constexpr int kStages = Traits::kStages;
    static constexpr int kNRows = FwdNRows<Traits>::value;

    int const thread_idx = threadIdx.x;
    int const m_block = blockIdx.x;
    int const bh_idx = blockIdx.y;
    int const head_idx = bh_idx % params.num_heads;
    int const batch_idx = bh_idx / params.num_heads;

    int const seqlen_q = params.seqlen_q;
    int const seqlen_k = params.seqlen_k;

    if (m_block * kBlockM >= seqlen_q) { return; }

    float const softmax_scale_log2 = params.softmax_scale_log2;
    int const n_block_max = (seqlen_k + kBlockN - 1) / kBlockN;

    // --- Shared memory layout: Q + K (2-stage) + V (2-stage) + bias buffer ---
    using SmemLayoutV = typename Traits::SmemLayoutV;
    using SmemLayoutVt = typename Traits::SmemLayoutVt;

    extern __shared__ char smem_[];
    Element* smem_q = reinterpret_cast<Element*>(smem_);
    Element* smem_k = smem_q + cute::cosize(SmemLayoutQ{});
    Element* smem_v = smem_k + cute::cosize(SmemLayoutK{});
    float*   smem_bias = reinterpret_cast<float*>(smem_v + cute::cosize(SmemLayoutV{}));

    Tensor sQ = make_tensor(make_smem_ptr(smem_q), SmemLayoutQ{});
    Tensor sK = make_tensor(make_smem_ptr(smem_k), SmemLayoutK{});
    Tensor sV = make_tensor(make_smem_ptr(smem_v), SmemLayoutV{});
    Tensor sVt = make_tensor(make_smem_ptr(smem_v), SmemLayoutVt{});

    // Global memory pointers
    Element* q_ptr = reinterpret_cast<Element*>(params.q_ptr)
        + batch_idx * params.q_batch_stride + head_idx * params.q_head_stride;
    Element* k_ptr = reinterpret_cast<Element*>(params.k_ptr)
        + batch_idx * params.k_batch_stride + head_idx * params.k_head_stride;
    Element* v_ptr = reinterpret_cast<Element*>(params.v_ptr)
        + batch_idx * params.v_batch_stride + head_idx * params.v_head_stride;

    Tensor mQ = make_tensor(make_gmem_ptr(q_ptr),
                            make_shape(seqlen_q, Int<kHeadDim>{}),
                            make_stride(params.q_row_stride, _1{}));
    Tensor mK = make_tensor(make_gmem_ptr(k_ptr),
                            make_shape(seqlen_k, Int<kHeadDim>{}),
                            make_stride(params.k_row_stride, _1{}));
    Tensor mV = make_tensor(make_gmem_ptr(v_ptr),
                            make_shape(seqlen_k, Int<kHeadDim>{}),
                            make_stride(params.v_row_stride, _1{}));

    Tensor gQ = local_tile(mQ, Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_coord(m_block, _0{}));
    Tensor gK = local_tile(mK, Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_coord(_, _0{}));
    Tensor gV = local_tile(mV, Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_coord(_, _0{}));

    GmemTiledCopy gmem_tiled_copy;
    auto gmem_thr_copy = gmem_tiled_copy.get_thread_slice(thread_idx);

    Tensor tQgQ = gmem_thr_copy.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy.partition_D(sQ);
    Tensor tKgK = gmem_thr_copy.partition_S(gK);
    Tensor tKsK = gmem_thr_copy.partition_D(sK);
    Tensor tVgV = gmem_thr_copy.partition_S(gV);
    Tensor tVsV = gmem_thr_copy.partition_D(sV);

    // Predicates for Q rows
    Tensor cQ = cute::make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});
    Tensor tQcQ = gmem_thr_copy.partition_S(cQ);
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
    #pragma unroll
    for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(_0{}, _0{}, k)) < kHeadDim; }

    // Predicates for KV rows
    Tensor cKV = cute::make_identity_tensor(Shape<Int<kBlockN>, Int<kHeadDim>>{});
    Tensor tKVcKV = gmem_thr_copy.partition_S(cKV);
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tVsV)));
    #pragma unroll
    for (int k = 0; k < size(tKVpKV); ++k) { tKVpKV(k) = get<1>(tKVcKV(_0{}, _0{}, k)) < kHeadDim; }

    // Load Q to smem
    {
        int const q_row_limit = seqlen_q - m_block * kBlockM;
        bool const even_q = (seqlen_q % kBlockM) == 0;
        int const q_block_max = (seqlen_q + kBlockM - 1) / kBlockM;
        bool const is_q_tail_block = (!even_q) && (m_block == (q_block_max - 1));
        #pragma unroll
        for (int m = 0; m < size<1>(tQgQ); ++m) {
            bool pred_m = get<0>(tQcQ(_0{}, m, _0{})) < q_row_limit;
            #pragma unroll
            for (int k = 0; k < size<2>(tQgQ); ++k) {
                if (!is_q_tail_block) {
                    cute::copy(gmem_tiled_copy.with(pred_m && tQpQ(k)),
                               tQgQ(_, m, k), tQsQ(_, m, k));
                } else {
                    // Tail-safe path for Q: require all lanes in the copy atom in-bounds.
                    bool pred_all = pred_m && tQpQ(k);
                    #pragma unroll
                    for (int v = 0; v < size<0>(tQcQ); ++v) {
                        pred_all = pred_all && (get<0>(tQcQ(v, m, k)) < q_row_limit);
                    }
                    if (pred_all) {
                        cute::copy(gmem_tiled_copy.with(true),
                                   tQgQ(_, m, k), tQsQ(_, m, k));
                    } else {
                        cute::fill(tQsQ(_, m, k), Element(0));
                    }
                }
            }
        }
    }
    cute::cp_async_fence();
    cp_async_wait<0>();
    __syncthreads();

    // MMA setup
    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_slice(thread_idx);

    auto smem_tiled_copy_Q = make_tiled_copy_A(SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(thread_idx);
    auto smem_tiled_copy_K = make_tiled_copy_B(SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(thread_idx);
    auto smem_tiled_copy_V = make_tiled_copy_B(SmemCopyAtomTransposed{}, tiled_mma);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(thread_idx);
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

    // Accumulators
    Tensor tOrO = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});
    clear(tOrO);

    Softmax<kNRows> softmax(softmax_scale_log2);

    // Score identity tensor for index calculation
    Tensor cS = cute::make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});
    Tensor tScS = thr_mma.partition_C(cS);

    // Load tile helpers
    auto load_k_tile = [&](int n_block, int stage) {
        int const row_limit = seqlen_k - n_block * kBlockN;
        bool const even_n = (seqlen_k % kBlockN) == 0;
        bool const is_tail_block = (!even_n) && (n_block == (n_block_max - 1));
        #pragma unroll
        for (int m = 0; m < size<1>(tKgK); ++m) {
            bool pred_m = get<0>(tKVcKV(_0{}, m, _0{})) < row_limit;
            #pragma unroll
            for (int k = 0; k < size<2>(tKgK); ++k) {
                if (!is_tail_block) {
                    cute::copy(gmem_tiled_copy.with(pred_m && tKVpKV(k)),
                               tKgK(_, m, k, n_block), tKsK(_, m, k, stage));
                } else {
                    // Tail-safe path for K: only copy when all lanes are in-bounds.
                    bool pred_all = pred_m && tKVpKV(k);
                    #pragma unroll
                    for (int v = 0; v < size<0>(tKVcKV); ++v) {
                        pred_all = pred_all && (get<0>(tKVcKV(v, m, k)) < row_limit);
                    }
                    if (pred_all) {
                        cute::copy(gmem_tiled_copy.with(true),
                                   tKgK(_, m, k, n_block), tKsK(_, m, k, stage));
                    } else {
                        cute::fill(tKsK(_, m, k, stage), Element(0));
                    }
                }
            }
        }
    };

    auto load_v_tile = [&](int n_block, int stage) {
        int const row_limit = seqlen_k - n_block * kBlockN;
        bool const even_n = (seqlen_k % kBlockN) == 0;
        bool const is_tail_block = (!even_n) && (n_block == (n_block_max - 1));
        #pragma unroll
        for (int m = 0; m < size<1>(tVgV); ++m) {
            bool pred_m = get<0>(tKVcKV(_0{}, m, _0{})) < row_limit;
            #pragma unroll
            for (int k = 0; k < size<2>(tVgV); ++k) {
                if (!is_tail_block) {
                    cute::copy(gmem_tiled_copy.with(pred_m && tKVpKV(k)),
                               tVgV(_, m, k, n_block), tVsV(_, m, k, stage));
                } else {
                    // Tail-safe path for V mirrors K to avoid partial-lane stale values.
                    bool pred_all = pred_m && tKVpKV(k);
                    #pragma unroll
                    for (int v = 0; v < size<0>(tKVcKV); ++v) {
                        pred_all = pred_all && (get<0>(tKVcKV(v, m, k)) < row_limit);
                    }
                    if (pred_all) {
                        cute::copy(gmem_tiled_copy.with(true),
                                   tVgV(_, m, k, n_block), tVsV(_, m, k, stage));
                    } else {
                        cute::fill(tVsV(_, m, k, stage), Element(0));
                    }
                }
            }
        }
    };

    // Bias base pointer (computed once, used per N-block if HasBias)
    float const* bias_base = nullptr;
    float bias_inv_scale = 0.0f;
    if constexpr (HasBias) {
        bias_base = params.bias_ptr
            + batch_idx * params.num_heads * seqlen_k
            + head_idx * seqlen_k;
        bias_inv_scale = 1.0f / params.softmax_scale;
    }

    // --- Prologue: prefetch kStages N-blocks for true double-buffering ---
    int const n_stages_prefetch = n_block_max < kStages ? n_block_max : kStages;
    #pragma unroll
    for (int s = 0; s < kStages; ++s) {
        if (s < n_stages_prefetch) {
            load_k_tile(n_block_max - 1 - s, s);
            load_v_tile(n_block_max - 1 - s, s);
            cute::cp_async_fence();
        }
    }

    // Pre-load bias for the first iteration into smem
    if constexpr (HasBias) {
        int const first_n_offset = (n_block_max - 1) * kBlockN;
        for (int i = thread_idx; i < kBlockN; i += Traits::kNumThreads) {
            int n_idx = first_n_offset + i;
            smem_bias[i] = (n_idx < seqlen_k) ? bias_base[n_idx] : 0.0f;
        }
    }

    // === Main loop over N blocks (2-stage K+V pipeline) ===
    #pragma unroll 1
    for (int ni = 0; ni < n_block_max; ++ni) {
        int cur_n_block = n_block_max - 1 - ni;
        int cur_stage = ni % kStages;
        bool is_first = (ni == 0);

        // Wait: allow up to kStages-1 groups in flight; drain on last iterations
        if (ni >= n_block_max - (kStages - 1)) {
            cp_async_wait<0>();
        } else {
            cp_async_wait<kStages - 1>();
        }
        __syncthreads();
        // KV data for cur_stage is ready; bias smem from previous iter/prologue visible

        // S = Q @ K^T (using current K stage)
        auto sK_cur = sK(_, _, cur_stage);
        auto tSsK_cur = smem_thr_copy_K.partition_S(sK_cur);

        Tensor tSrS = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
        clear(tSrS);
        Tensor tSrQ_cur = thr_mma.partition_fragment_A(sQ);
        Tensor tSrK = thr_mma.partition_fragment_B(sK_cur);

        gemm_sm80<false>(
            tSrS, tSrQ_cur, tSrK, tSsQ, tSsK_cur,
            tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K);

        // Apply bias from smem + mask OOB in one pass
        {
            int const n_offset = cur_n_block * kBlockN;
            int const m_offset = m_block * kBlockM;
            #pragma unroll
            for (int i = 0; i < size(tSrS); ++i) {
                int n_idx = get<1>(tScS(i)) + n_offset;
                int m_idx = get<0>(tScS(i)) + m_offset;
                if (n_idx >= seqlen_k || m_idx >= seqlen_q) {
                    tSrS(i) = -INFINITY;
                } else if constexpr (HasBias) {
                    tSrS(i) += smem_bias[get<1>(tScS(i))] * bias_inv_scale;
                }
            }
        }

        // Online softmax
        if (is_first) {
            auto scores_scale = softmax.template max_get_scale<true, true>(tSrS);
            softmax.template online_softmax<true, true>(tSrS);
        } else {
            auto scores_scale = softmax.template max_get_scale<false, true>(tSrS);
            softmax.template online_softmax<false, true>(tSrS);
            softmax.rescale_o(tOrO, scores_scale);
        }

        // Convert P to Element type
        Tensor tOrP_acc = make_tensor(tSrS.data(),
            convert_layout_acc_Aregs<TiledMma>(tSrS.layout()));
        Tensor tOrP = make_tensor_like<Element>(tOrP_acc);
        {
            constexpr int kNumElems = decltype(size(tOrP_acc))::value;
            cutlass::NumericArrayConverter<Element, float, kNumElems> converter;
            *reinterpret_cast<cutlass::Array<Element, kNumElems>*>(tOrP.data()) =
                converter(*reinterpret_cast<const cutlass::Array<float, kNumElems>*>(tOrP_acc.data()));
        }

        // O += P @ V^T (reads V[cur_stage])
        auto sVt_cur = sVt(_, _, cur_stage);
        auto tOsVt_cur = smem_thr_copy_V.partition_S(sVt_cur);
        Tensor tOrV = thr_mma.partition_fragment_B(sVt_cur);
        gemm_rs_sm80(tOrO, tOrP, tOrV, tOsVt_cur,
                     tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);

        // Warp sync: ensure all warps finished reading K+V of cur_stage
        // before we overwrite cur_stage smem with prefetch + next bias
        __syncthreads();

        // Prefetch KV for iter ni+kStages into cur_stage (now safe)
        int pf_ni = ni + kStages;
        if (pf_ni < n_block_max) {
            load_k_tile(n_block_max - 1 - pf_ni, cur_stage);
            load_v_tile(n_block_max - 1 - pf_ni, cur_stage);
            cute::cp_async_fence();
        }

        // Load bias for next iteration into smem (will be visible after sync at loop top)
        if constexpr (HasBias) {
            int next_ni = ni + 1;
            if (next_ni < n_block_max) {
                int next_n_block = n_block_max - 1 - next_ni;
                int const next_n_offset = next_n_block * kBlockN;
                for (int i = thread_idx; i < kBlockN; i += Traits::kNumThreads) {
                    int n_idx = next_n_offset + i;
                    smem_bias[i] = (n_idx < seqlen_k) ? bias_base[n_idx] : 0.0f;
                }
            }
        }
    }

    // === Epilogue ===
    auto final_scale = softmax.finalize();
    softmax.rescale_o(tOrO, final_scale);

    Element* o_ptr = reinterpret_cast<Element*>(params.o_ptr)
        + batch_idx * params.o_batch_stride + head_idx * params.o_head_stride;

    // Vectorized O store via smem staging: MMA regs → smem → 128-bit gmem stores
    // Reuse Q's smem as staging buffer (Q is no longer needed).
    {
        __syncthreads();
        Element* smem_o_flat = smem_q;

        Tensor cO = cute::make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});
        Tensor tOcO = thr_mma.partition_C(cO);

        // Scatter-write fp16 O values from MMA regs to flat smem [kBlockM * kHeadDim]
        #pragma unroll
        for (int i = 0; i < size(tOrO); ++i) {
            int m_idx = get<0>(tOcO(i));
            int d_idx = get<1>(tOcO(i));
            if (m_idx < kBlockM && d_idx < kHeadDim) {
                smem_o_flat[m_idx * kHeadDim + d_idx] = Element(tOrO(i));
            }
        }
        __syncthreads();

        // Cooperative 128-bit stores: each thread handles ceil(kBlockM/kNumThreads) rows,
        // each row = kHeadDim fp16 = kHeadDim*2 bytes
        static constexpr int kElemsPerStore = sizeof(cute::uint128_t) / sizeof(Element);
        static constexpr int kStoresPerRow = kHeadDim / kElemsPerStore;
        int const o_row_stride = params.o_row_stride;

        for (int row = thread_idx; row < kBlockM; row += Traits::kNumThreads) {
            int global_m = m_block * kBlockM + row;
            if (global_m < seqlen_q) {
                Element const* src_row = smem_o_flat + row * kHeadDim;
                Element* dst_row = o_ptr + global_m * o_row_stride;
                #pragma unroll
                for (int s = 0; s < kStoresPerRow; ++s) {
                    *reinterpret_cast<cute::uint128_t*>(dst_row + s * kElemsPerStore) =
                        *reinterpret_cast<cute::uint128_t const*>(src_row + s * kElemsPerStore);
                }
            }
        }
    }

    // Store LSE
    float* lse_ptr = params.lse_ptr + batch_idx * params.num_heads * seqlen_q
                     + head_idx * seqlen_q;
    Tensor cLSE = cute::make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});
    Tensor tLSEcLSE = thr_mma.partition_C(cLSE);

    #pragma unroll
    for (int mi = 0; mi < kNRows; ++mi) {
        Tensor acc_rowcol_idx = make_tensor(tLSEcLSE.data(),
                                            convert_layout_acc_rowcol(tLSEcLSE.layout()));
        int m_idx = get<0>(acc_rowcol_idx(mi, 0));
        int global_m = m_block * kBlockM + m_idx;
        if (global_m < seqlen_q) {
            lse_ptr[global_m] = softmax.row_sum(mi);
        }
    }


}

}  // namespace stac_attn
