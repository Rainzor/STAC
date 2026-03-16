#pragma once

#include "include/common.h"
#include "include/kernel_traits.h"
#include "include/softmax.h"

namespace stac_attn {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////
// N-major ColSum kernel: each CTA owns a block of N columns, scans all M rows.
// Grid: (ceil(N/kBlockN), B*H).  No cross-CTA atomics.
//
// Optimizations over naive version:
//   - Q prefetch overlaps with epilogue (issue cp_async for Q[mi+1] after GEMM)
//   - Double-buffered LSE eliminates end-of-loop __syncthreads
//   - Bias cached in registers (avoid repeated smem reads)
//   - Deterministic cross-warp reduction (no shared-memory atomics)
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits, bool HasBias>
__global__ void __launch_bounds__(Traits::kNumThreads)
colsum_n_major_kernel(StacFlashParams params)
{
    using Element = typename Traits::Element;
    using TiledMma = typename Traits::TiledMma;
    using SmemLayoutAtom = typename Traits::SmemLayoutAtom;
    using SmemCopyAtom = typename Traits::SmemCopyAtom;
    using GmemTiledCopy = typename Traits::GmemTiledCopy;

    static constexpr int kBlockM = Traits::kBlockM;
    static constexpr int kBlockN = Traits::kBlockN;
    static constexpr int kHeadDim = Traits::kHeadDim;
    static constexpr int kNWarps = Traits::kNWarps;

    int const thread_idx = threadIdx.x;
    int const n_block = blockIdx.x;
    int const bh_idx = blockIdx.y;
    int const head_idx = bh_idx % params.num_heads;
    int const batch_idx = bh_idx / params.num_heads;

    int const seqlen_q = params.seqlen_q;
    int const seqlen_q_eff = params.seqlen_q_sub;
    int const q_stride = params.q_m_stride;
    int const seqlen_k = params.seqlen_k;
    int const n_off = n_block * kBlockN;

    if (n_off >= seqlen_k) { return; }

    float const softmax_scale = params.softmax_scale;
    float const scale_log2 = params.softmax_scale_log2;
    static constexpr float kLog2e = 1.4426950408889634074f;

    int const m_block_max = (seqlen_q_eff + kBlockM - 1) / kBlockM;

    // --- Shared memory layout ---
    // K tile [kBlockN, kHeadDim]  — loaded once
    // Q tile [kBlockM, kHeadDim]  — single buffer, prefetched after GEMM
    // LSE double-buffer [2][kBlockM] — eliminates end-of-loop sync
    // bias [kBlockN] (if HasBias) — loaded once
    using SmemLayoutK1 = decltype(tile_to_shape(SmemLayoutAtom{},
                                                 Shape<Int<kBlockN>, Int<kHeadDim>>{}));
    using SmemLayoutQ1 = decltype(tile_to_shape(SmemLayoutAtom{},
                                                 Shape<Int<kBlockM>, Int<kHeadDim>>{}));

    extern __shared__ char smem_[];
    Element* smem_k = reinterpret_cast<Element*>(smem_);
    Element* smem_q = smem_k + cute::cosize(SmemLayoutK1{});
    float*   smem_lse_base = reinterpret_cast<float*>(smem_q + cute::cosize(SmemLayoutQ1{}));
    float*   smem_lse_0 = smem_lse_base;
    float*   smem_lse_1 = smem_lse_base + kBlockM;
    float*   smem_bias = smem_lse_1 + kBlockM;

    Tensor sK = make_tensor(make_smem_ptr(smem_k), SmemLayoutK1{});
    Tensor sQ = make_tensor(make_smem_ptr(smem_q), SmemLayoutQ1{});

    // --- Global memory tensors ---
    Element* q_ptr = reinterpret_cast<Element*>(params.q_ptr)
        + batch_idx * params.q_batch_stride + head_idx * params.q_head_stride;
    Element* k_ptr = reinterpret_cast<Element*>(params.k_ptr)
        + batch_idx * params.k_batch_stride + head_idx * params.k_head_stride;

    Tensor mK = make_tensor(make_gmem_ptr(k_ptr),
                            make_shape(seqlen_k, Int<kHeadDim>{}),
                            make_stride(params.k_row_stride, _1{}));
    Tensor mQ = make_tensor(make_gmem_ptr(q_ptr),
                            make_shape(seqlen_q_eff, Int<kHeadDim>{}),
                            make_stride(params.q_row_stride * q_stride, _1{}));

    Tensor gK = local_tile(mK, Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_coord(n_block, _0{}));
    Tensor gQ = local_tile(mQ, Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_coord(_, _0{}));

    GmemTiledCopy gmem_tiled_copy;
    auto gmem_thr_copy = gmem_tiled_copy.get_thread_slice(thread_idx);

    // --- Load K tile to smem (once) ---
    {
        Tensor tKgK = gmem_thr_copy.partition_S(gK);
        Tensor tKsK = gmem_thr_copy.partition_D(sK);

        Tensor cKV = cute::make_identity_tensor(Shape<Int<kBlockN>, Int<kHeadDim>>{});
        Tensor tcKV = gmem_thr_copy.partition_S(cKV);
        Tensor tpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));
        #pragma unroll
        for (int d = 0; d < size(tpKV); ++d) {
            tpKV(d) = get<1>(tcKV(_0{}, _0{}, d)) < kHeadDim;
        }

        int const k_row_limit = seqlen_k - n_off;
        #pragma unroll
        for (int m = 0; m < size<1>(tKgK); ++m) {
            bool pred_m = get<0>(tcKV(_0{}, m, _0{})) < k_row_limit;
            #pragma unroll
            for (int d = 0; d < size<2>(tKgK); ++d) {
                cute::copy(gmem_tiled_copy.with(pred_m && tpKV(d)),
                           tKgK(_, m, d), tKsK(_, m, d));
            }
        }
        cute::cp_async_fence();
        cp_async_wait<0>();
        __syncthreads();
    }

    // --- MMA setup ---
    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_slice(thread_idx);

    auto smem_tiled_copy_Q = make_tiled_copy_A(SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(thread_idx);
    auto smem_tiled_copy_K = make_tiled_copy_B(SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(thread_idx);
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    // Score identity tensor for index extraction
    Tensor cS = cute::make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});
    Tensor tScS = thr_mma.partition_C(cS);
    Tensor idx_rowcol = make_tensor(tScS.data(),
                                    convert_layout_acc_rowcol(tScS.layout()));

    static constexpr int kNRows_cs = 2 * kBlockM / (16 * kNWarps);
    static constexpr int kNCols_per_thread = decltype(size<1>(idx_rowcol))::value;

    float col_acc[kNCols_per_thread];
    #pragma unroll
    for (int j = 0; j < kNCols_per_thread; ++j) { col_acc[j] = 0.0f; }

    // Load bias to smem once, then cache in registers (pre-scaled to log2 space)
    float bias_reg[kNCols_per_thread];
    if constexpr (HasBias) {
        float const* bias_ptr = params.bias_ptr
            + batch_idx * params.num_heads * seqlen_k
            + head_idx * seqlen_k;
        for (int i = thread_idx; i < kBlockN; i += Traits::kNumThreads) {
            int n_idx = n_off + i;
            smem_bias[i] = (n_idx < seqlen_k) ? bias_ptr[n_idx] * kLog2e : 0.0f;
        }
        __syncthreads();
        #pragma unroll
        for (int j = 0; j < kNCols_per_thread; ++j) {
            bias_reg[j] = smem_bias[get<1>(idx_rowcol(_0{}, j))];
        }
    }

    // LSE pointer
    float const* lse_ptr = params.lse_ptr
        + batch_idx * params.num_heads * seqlen_q
        + head_idx * seqlen_q;

    // Q load predicates
    Tensor cQ_id = cute::make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});
    Tensor tQcQ = gmem_thr_copy.partition_S(cQ_id);
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(gmem_thr_copy.partition_D(sQ))));
    #pragma unroll
    for (int d = 0; d < size(tQpQ); ++d) {
        tQpQ(d) = get<1>(tQcQ(_0{}, _0{}, d)) < kHeadDim;
    }

    // Lambda: issue cp_async for Q[mi]
    auto load_q_async = [&](int mi) {
        Tensor tQgQ = gmem_thr_copy.partition_S(gQ);
        Tensor tQsQ = gmem_thr_copy.partition_D(sQ);
        int const q_row_limit = seqlen_q_eff - mi * kBlockM;
        bool const even_q = (seqlen_q_eff % kBlockM) == 0;
        bool const is_q_tail_block = (!even_q) && (mi == (m_block_max - 1));
        #pragma unroll
        for (int r = 0; r < size<1>(tQgQ); ++r) {
            bool pred_r = get<0>(tQcQ(_0{}, r, _0{})) < q_row_limit;
            #pragma unroll
            for (int d = 0; d < size<2>(tQgQ); ++d) {
                if (!is_q_tail_block) {
                    cute::copy(gmem_tiled_copy.with(pred_r && tQpQ(d)),
                               tQgQ(_, r, d, mi), tQsQ(_, r, d));
                } else {
                    // Tail-safe path for Q in colsum kernel.
                    bool pred_all = pred_r && tQpQ(d);
                    #pragma unroll
                    for (int v = 0; v < size<0>(tQcQ); ++v) {
                        pred_all = pred_all && (get<0>(tQcQ(v, r, d)) < q_row_limit);
                    }
                    if (pred_all) {
                        cute::copy(gmem_tiled_copy.with(true),
                                   tQgQ(_, r, d, mi), tQsQ(_, r, d));
                    } else {
                        cute::fill(tQsQ(_, r, d), Element(0));
                    }
                }
            }
        }
    };

    // Cooperative LSE load, pre-scaled to log2 space for exp2f
    auto load_lse = [&](int mi, float* lse_buf) {
        int const m_off = mi * kBlockM;
        for (int i = thread_idx; i < kBlockM; i += Traits::kNumThreads) {
            int m_idx = m_off + i;
            int m_phys = m_idx * q_stride;
            if (m_idx < seqlen_q_eff && m_phys < seqlen_q) {
                float v = lse_ptr[m_phys];
                lse_buf[i] = (v == -INFINITY) ? -INFINITY : v * kLog2e;
            } else {
                lse_buf[i] = -INFINITY;
            }
        }
    };

    // === Prologue: issue Q[0] cp_async ===
    load_q_async(0);
    cute::cp_async_fence();

    // === Main loop ===
    #pragma unroll 1
    for (int mi = 0; mi < m_block_max; ++mi) {
        int const m_off = mi * kBlockM;
        float* lse_cur = (mi % 2 == 0) ? smem_lse_0 : smem_lse_1;

        // Load LSE[mi] into current double-buffer slot
        load_lse(mi, lse_cur);

        // Wait for Q[mi] cp_async (issued in prologue or previous iter's post-GEMM)
        cp_async_wait<0>();
        __syncthreads();

        // S = Q @ K^T
        Tensor tSrS = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
        clear(tSrS);
        Tensor tSrQ = thr_mma.partition_fragment_A(sQ);
        Tensor tSrK = thr_mma.partition_fragment_B(sK);

        gemm_sm80<false>(
            tSrS, tSrQ, tSrK, tSsQ, tSsK,
            tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K);

        // After GEMM, smem_q is no longer read — start Q[mi+1] prefetch
        // to overlap DMA with the epilogue computation below
        if (mi + 1 < m_block_max) {
            load_q_async(mi + 1);
            cute::cp_async_fence();
        }

        // Epilogue: accumulate exp2(S * scale_log2 + bias_log2 - lse_log2) into col_acc
        // Row-major traversal: hoist per-row lse load + validity check outside inner loop.
        // All values pre-scaled to log2 space so we use exp2f (single HW instruction).
        Tensor scores_rc = make_tensor(tSrS.data(),
                                       convert_layout_acc_rowcol(tSrS.layout()));

        #pragma unroll
        for (int i = 0; i < kNRows_cs; ++i) {
            int m_idx = get<0>(idx_rowcol(i, _0{})) + m_off;
            if (m_idx >= seqlen_q_eff) continue;
            float lse_log2 = lse_cur[get<0>(idx_rowcol(i, _0{}))];
            if (lse_log2 == -INFINITY) continue;
            #pragma unroll
            for (int j = 0; j < kNCols_per_thread; ++j) {
                float exp_arg = scores_rc(i, j) * scale_log2 - lse_log2;
                if constexpr (HasBias) {
                    exp_arg += bias_reg[j];
                }
                col_acc[j] += exp2f(exp_arg);
            }
        }
        // No end-of-loop sync: LSE double-buffer prevents write hazard,
        // Q[mi+1] cp_async writes to smem_q which we no longer read.
    }

    // --- Cross-thread reduction (deterministic, no atomics) ---
    int const lane_id = thread_idx % 32;
    int const warp_id = thread_idx / 32;

    #pragma unroll
    for (int j = 0; j < kNCols_per_thread; ++j) {
        col_acc[j] += __shfl_xor_sync(0xffffffff, col_acc[j], 4);
        col_acc[j] += __shfl_xor_sync(0xffffffff, col_acc[j], 8);
        col_acc[j] += __shfl_xor_sync(0xffffffff, col_acc[j], 16);
    }

    __syncthreads();
    float* warp_buf = reinterpret_cast<float*>(smem_);

    if (lane_id < 4) {
        #pragma unroll
        for (int j = 0; j < kNCols_per_thread; ++j) {
            int n_local = get<1>(idx_rowcol(_0{}, j));
            warp_buf[warp_id * kBlockN + n_local] = col_acc[j];
        }
    }
    __syncthreads();

    float* colsum_ptr = params.colsum_ptr
        + batch_idx * params.num_heads * seqlen_k
        + head_idx * seqlen_k;

    for (int i = thread_idx; i < kBlockN; i += Traits::kNumThreads) {
        if (n_off + i < seqlen_k) {
            float sum = warp_buf[i];
            #pragma unroll
            for (int w = 1; w < kNWarps; ++w) {
                sum += warp_buf[w * kBlockN + i];
            }
            colsum_ptr[n_off + i] = sum;
        }
    }
}

}  // namespace stac_attn
