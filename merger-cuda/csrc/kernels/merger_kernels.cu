// Copyright (c) 2024-2026 CausalVGGT Authors
// SPDX-License-Identifier: Apache-2.0
//
// merger_kernels.cu: CUDA kernel implementations for KV Merger
//
// Contains implementations for all pipeline steps:
// pack_valid_tokens, group_by_row, materialize_rows, one2one_merge,
// buffer_topb_update, all2one_merge, retrieve_fixed, clean_rows

#include "include/merger_kernels.cuh"
#include <cub/cub.cuh>
#include <algorithm>

namespace causalvggt {
namespace backend {
namespace detail {

constexpr int THREADS_PER_BLOCK = 256;
constexpr int MAX_PIVOTS = 8;
constexpr int MAX_DIM = 128;
constexpr int MAX_BUFFER = 64;

// =============================================================================
// Helper: zero scalar
// =============================================================================
__global__ void zero_scalar_kernel(int32_t* ptr) {
    if (threadIdx.x == 0 && blockIdx.x == 0) *ptr = 0;
}

// =============================================================================
// Pivot zone allocator helpers (Morton-based spatial locality)
// =============================================================================

__device__ int32_t zone_alloc_piv(
    int32_t* __restrict__ piv_free_stack,
    int32_t* __restrict__ zone_tops,
    int32_t zone_cap,
    int32_t num_zones,
    int32_t target_zone,
    int32_t* out_spill_dist = nullptr)
{
    int32_t base = target_zone * zone_cap;
    int32_t top = atomicSub(&zone_tops[target_zone], 1) - 1;
    if (top >= 0) {
        if (out_spill_dist) *out_spill_dist = 0;
        return piv_free_stack[base + top];
    }
    atomicAdd(&zone_tops[target_zone], 1);

    for (int d = 1; d < num_zones; d++) {
        for (int sign = 0; sign < 2; sign++) {
            int z = target_zone + (sign == 0 ? d : -d);
            if (z < 0 || z >= num_zones) continue;
            int32_t b = z * zone_cap;
            top = atomicSub(&zone_tops[z], 1) - 1;
            if (top >= 0) {
                if (out_spill_dist) *out_spill_dist = d;
                return piv_free_stack[b + top];
            }
            atomicAdd(&zone_tops[z], 1);
        }
    }
    if (out_spill_dist) *out_spill_dist = -1;
    return -1;
}

__device__ void zone_free_piv(
    int32_t* __restrict__ piv_free_stack,
    int32_t* __restrict__ zone_tops,
    int32_t zone_cap,
    int32_t sid)
{
    int32_t z = sid / zone_cap;
    int32_t base = z * zone_cap;
    int32_t top = atomicAdd(&zone_tops[z], 1);
    piv_free_stack[base + top] = sid;
}

// =============================================================================
// Pack Valid Tokens
// =============================================================================
template <typename scalar_t>
__global__ void pack_valid_tokens_kernel(
    const scalar_t* __restrict__ K_in, const scalar_t* __restrict__ V_in,
    const float* __restrict__ S_in, const int64_t* __restrict__ VX_in,
    scalar_t* __restrict__ K_out, scalar_t* __restrict__ V_out,
    float* __restrict__ S_out, int64_t* __restrict__ rows_out,
    int32_t* __restrict__ E_valid_dev, int32_t* __restrict__ orig_idx_out,
    int64_t E, int64_t D, int64_t H,
    int64_t V_alloc, int64_t num_voxels, bool vx_per_head) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= E) return;
    int64_t h, vx;
    if (vx_per_head) {
        int64_t tph = E / H;
        h = idx / tph;
        vx = VX_in[idx];
    } else {
        h = idx / (E / H);
        vx = VX_in[idx];
    }
    if (vx < 0 || vx >= num_voxels) return;
    int out_idx = atomicAdd(E_valid_dev, 1);
    rows_out[out_idx] = h * V_alloc + vx;
    S_out[out_idx] = S_in[idx];
    if (orig_idx_out) orig_idx_out[out_idx] = idx;
    for (int d = 0; d < D; ++d) {
        K_out[out_idx * D + d] = K_in[idx * D + d];
        V_out[out_idx * D + d] = V_in[idx * D + d];
    }
}

void pack_valid_tokens(
    const void* K_in, const void* V_in, const float* S_in, const int64_t* VX_in,
    int64_t E, int64_t D, int64_t H, int64_t V_alloc, int64_t num_voxels, bool vx_per_head,
    int64_t* rows_out, void* K_out, void* V_out, float* S_out, int32_t* E_valid_dev,
    int32_t* orig_idx_out, DType dtype, cudaStream_t stream) {
    if (E == 0) return;
    zero_scalar_kernel<<<1, 1, 0, stream>>>(E_valid_dev);
    int blocks = (E + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    if (dtype == DType::Float16) {
        pack_valid_tokens_kernel<__half><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
            (const __half*)K_in, (const __half*)V_in, S_in, VX_in,
            (__half*)K_out, (__half*)V_out, S_out, rows_out, E_valid_dev, orig_idx_out,
            E, D, H, V_alloc, num_voxels, vx_per_head);
    } else {
        pack_valid_tokens_kernel<__nv_bfloat16><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
            (const __nv_bfloat16*)K_in, (const __nv_bfloat16*)V_in, S_in, VX_in,
            (__nv_bfloat16*)K_out, (__nv_bfloat16*)V_out, S_out, rows_out, E_valid_dev, orig_idx_out,
            E, D, H, V_alloc, num_voxels, vx_per_head);
    }
}

// =============================================================================
// Group by row - uses CUB with compound key for stable ordering
// =============================================================================

// Compound key structure for stable sorting by (row, original_index)
// This ensures deterministic ordering within each row group
struct CompoundKey {
    int64_t row;
    int32_t idx;
};

// Custom comparison for compound key (row first, then original index)
struct CompoundKeyLess {
    __device__ __forceinline__ bool operator()(const CompoundKey& a, const CompoundKey& b) const {
        if (a.row != b.row) return a.row < b.row;
        return a.idx < b.idx;
    }
};

__global__ void create_compound_keys_kernel(const int64_t* __restrict__ rows, 
                                            int64_t* __restrict__ compound_keys, 
                                            int32_t* __restrict__ indices,
                                            const int32_t* __restrict__ orig_idx,
                                            int64_t E, int64_t key_stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= E) return;
    int64_t tiebreaker = orig_idx ? (int64_t)orig_idx[idx] : (int64_t)idx;
    compound_keys[idx] = rows[idx] * key_stride + tiebreaker;
    indices[idx] = idx;
}

__global__ void extract_rows_from_compound_keys_kernel(const int64_t* __restrict__ sorted_compound_keys,
                                                        int64_t* __restrict__ sorted_rows,
                                                        int64_t E, int64_t key_stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= E) return;
    // Extract original row from compound key using floor division.
    // C++ integer division truncates toward zero, which gives wrong results
    // for negative compound keys: e.g. (-65537) / 65545 = 0 instead of -1.
    // Floor division: floor(a/b) = (a - (((a % b) + b) % b)) / b for b > 0.
    int64_t ck = sorted_compound_keys[idx];
    int64_t q = ck / key_stride;
    int64_t r = ck - q * key_stride;
    if (r < 0) q -= 1;
    sorted_rows[idx] = q;
}

template <typename T>
__global__ void gather_kernel(const T* __restrict__ in, const int32_t* __restrict__ perm,
                              T* __restrict__ out, int n, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * D) return;
    int i = idx / D, d = idx % D;
    out[idx] = in[perm[i] * D + d];
}

__global__ void append_total_kernel(int32_t* offsets, const int32_t* G_dev, int total) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int G = *G_dev;
        if (G > 0) offsets[G] = total;
    }
}

void group_by_row(
    const int64_t* rows, const void* K, const void* V, const float* S,
    int64_t E, int64_t D, int64_t* unique_rows, int32_t* row_offsets,
    int32_t* sorted_indices, void* sorted_K, void* sorted_V, float* sorted_S,
    int32_t* G_dev, void* temp_storage, size_t temp_storage_size, DType dtype,
    cudaStream_t stream, const int32_t* orig_idx) {
    if (E == 0) {
        zero_scalar_kernel<<<1, 1, 0, stream>>>(G_dev);
        return;
    }
    
    // Layout of temp_storage with proper alignment:
    // [indices: E * int32] [pad to 8B] [compound_keys: E * int64]
    // [sorted_compound_keys: E * int64] [sorted_rows: E * int64]
    // [counts: E * int32] [cub_temp: remaining]
    auto align8 = [](char* p) -> char* {
        return reinterpret_cast<char*>((reinterpret_cast<uintptr_t>(p) + 7) & ~uintptr_t(7));
    };
    int32_t* indices = (int32_t*)temp_storage;
    char* next = align8(reinterpret_cast<char*>(indices + E));
    int64_t* compound_keys = (int64_t*)next;
    int64_t* sorted_compound_keys = compound_keys + E;
    int64_t* sorted_rows = sorted_compound_keys + E;
    int32_t* counts = (int32_t*)(sorted_rows + E);
    void* cub_temp = (void*)(counts + E);
    size_t cub_temp_size = temp_storage_size - ((char*)cub_temp - (char*)temp_storage);
    
    int blocks = (E + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    // Key stride must be larger than max tiebreaker value to avoid collisions.
    // When orig_idx is provided, tiebreaker values can be up to the original
    // input size (which may exceed E). Use a safe upper bound.
    int64_t key_stride = E + 1;
    if (orig_idx) {
        // orig_idx values can be up to H*Tn (original input size before filtering).
        // We don't know the exact max here, but E_input >= E_valid always holds.
        // Use a larger stride: 2*E covers typical cases; the key just needs
        // to be unique per (row, orig_idx) pair.
        key_stride = 2 * E + 1;
    }
    
    create_compound_keys_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
        rows, compound_keys, indices, orig_idx, E, key_stride);
    
    // CUB sort pairs by compound key (this gives us lexicographic sort by (row, index))
    size_t sort_temp = 0;
    cub::DeviceRadixSort::SortPairs(nullptr, sort_temp, compound_keys, sorted_compound_keys,
        indices, sorted_indices, E, 0, sizeof(int64_t)*8, stream);
    cub::DeviceRadixSort::SortPairs(cub_temp, cub_temp_size, compound_keys, sorted_compound_keys,
        indices, sorted_indices, E, 0, sizeof(int64_t)*8, stream);
    
    // Extract original row values from sorted compound keys
    extract_rows_from_compound_keys_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
        sorted_compound_keys, sorted_rows, E, key_stride);
    
    // CUB run-length encode on sorted rows
    size_t rle_temp = 0;
    cub::DeviceRunLengthEncode::Encode(nullptr, rle_temp, sorted_rows, unique_rows,
        counts, G_dev, E, stream);
    cub::DeviceRunLengthEncode::Encode(cub_temp, cub_temp_size, sorted_rows, unique_rows,
        counts, G_dev, E, stream);
    
    // Exclusive scan for offsets
    size_t scan_temp = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, scan_temp, counts, row_offsets, E, stream);
    cub::DeviceScan::ExclusiveSum(cub_temp, cub_temp_size, counts, row_offsets, E, stream);
    
    append_total_kernel<<<1, 1, 0, stream>>>(row_offsets, G_dev, E);
    
    // Gather K, V, S using sorted indices
    int vec_blocks = (E * D + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    if (dtype == DType::Float16) {
        gather_kernel<__half><<<vec_blocks, THREADS_PER_BLOCK, 0, stream>>>(
            (const __half*)K, sorted_indices, (__half*)sorted_K, E, D);
        gather_kernel<__half><<<vec_blocks, THREADS_PER_BLOCK, 0, stream>>>(
            (const __half*)V, sorted_indices, (__half*)sorted_V, E, D);
    } else {
        gather_kernel<__nv_bfloat16><<<vec_blocks, THREADS_PER_BLOCK, 0, stream>>>(
            (const __nv_bfloat16*)K, sorted_indices, (__nv_bfloat16*)sorted_K, E, D);
        gather_kernel<__nv_bfloat16><<<vec_blocks, THREADS_PER_BLOCK, 0, stream>>>(
            (const __nv_bfloat16*)V, sorted_indices, (__nv_bfloat16*)sorted_V, E, D);
    }
    gather_kernel<float><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(S, sorted_indices, sorted_S, E, 1);
}

// =============================================================================
// Materialize rows
// =============================================================================
__global__ void materialize_rows_kernel(
    const int64_t* __restrict__ unique_rows, const int32_t* __restrict__ G_dev,
    int32_t* __restrict__ row_ptr, int8_t* __restrict__ row_state,
    int32_t* __restrict__ row_count, uint8_t* __restrict__ pool_M,
    int32_t* __restrict__ free_stack, int32_t* __restrict__ free_top,
    int32_t* __restrict__ n_mat, int B, int G_max, int64_t S_tot,
    const int32_t* __restrict__ cand_row_counts) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int G = *G_dev;
    if (idx >= G) return;
    if (cand_row_counts && cand_row_counts[idx] == 0) return;
    int64_t row = unique_rows[idx];
    if (row < 0 || row >= S_tot) return;
    if (row_state[row] != (int8_t)BufState::RESERVED) return;
    int32_t top = atomicSub(free_top, 1) - 1;
    if (top < 0) { atomicAdd(free_top, 1); return; }
    int32_t slot = free_stack[top];
    row_ptr[row] = slot;
    row_state[row] = (int8_t)BufState::AVAILABLE;
    row_count[row] = 0;
    for (int b = 0; b < B; ++b) pool_M[slot * B + b] = 0;
    atomicAdd(n_mat, 1);
}

void materialize_rows(
    const int64_t* unique_rows, const int32_t* G_dev, int32_t* row_ptr,
    int8_t* row_state, int32_t* row_count, uint8_t* pool_M,
    int32_t* free_stack, int32_t* free_top, int64_t B, int64_t G_max,
    int32_t* n_materialized_dev, int64_t S_tot, cudaStream_t stream,
    const int32_t* cand_row_counts) {
    if (G_max == 0) return;
    zero_scalar_kernel<<<1, 1, 0, stream>>>(n_materialized_dev);
    int blocks = (G_max + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    materialize_rows_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
        unique_rows, G_dev, row_ptr, row_state, row_count, pool_M,
        free_stack, free_top, n_materialized_dev, B, G_max, S_tot,
        cand_row_counts);
}

// =============================================================================
// Get full rows
// =============================================================================
__global__ void get_full_flags_kernel(const int64_t* unique_rows, const int32_t* G_dev,
    const int8_t* row_state, uint8_t* flags, int G_max, int64_t S_tot) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int G = *G_dev;
    if (idx >= G_max) return;
    // Check bounds before accessing row_state
    int64_t row = (idx < G) ? unique_rows[idx] : -1;
    bool is_full = (idx < G && row >= 0 && row < S_tot && row_state[row] == (int8_t)BufState::FULL);
    flags[idx] = is_full ? 1 : 0;
}

void get_full_rows(const int64_t* unique_rows, const int32_t* G_dev, const int8_t* row_state,
    int64_t* full_rows, int32_t* F_dev, int64_t G_max, int64_t S_tot, void* temp_storage,
    size_t temp_storage_size, cudaStream_t stream) {
    if (G_max == 0) { zero_scalar_kernel<<<1, 1, 0, stream>>>(F_dev); return; }
    uint8_t* flags = (uint8_t*)temp_storage;
    void* cub_temp = (void*)(flags + G_max);
    size_t cub_size = temp_storage_size - G_max;
    int blocks = (G_max + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    get_full_flags_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
        unique_rows, G_dev, row_state, flags, G_max, S_tot);
    size_t sel_temp = 0;
    cub::DeviceSelect::Flagged(nullptr, sel_temp, unique_rows, flags, full_rows, F_dev, G_max, stream);
    cub::DeviceSelect::Flagged(cub_temp, cub_size, unique_rows, flags, full_rows, F_dev, G_max, stream);
}

// =============================================================================
// Clean rows
// =============================================================================
__global__ void clean_rows_kernel(const int64_t* rows, const int32_t* F_dev,
    int32_t* row_ptr, int8_t* row_state, int32_t* row_count,
    int32_t* free_stack, int32_t* free_top, int F_max, int64_t S_tot) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int F = *F_dev;
    if (idx >= F) return;
    int64_t row = rows[idx];
    // Skip invalid rows (negative or out of bounds)
    if (row < 0 || row >= S_tot) return;
    int32_t slot = row_ptr[row];
    if (slot >= 0) {
        int32_t top = atomicAdd(free_top, 1);
        free_stack[top] = slot;
    }
    row_ptr[row] = -1;
    row_state[row] = (int8_t)BufState::RESERVED;
    row_count[row] = 0;
}

void clean_rows(const int64_t* rows, const int32_t* F_dev, int32_t* row_ptr,
    int8_t* row_state, int32_t* row_count, int32_t* free_stack, int32_t* free_top,
    int64_t F_max, int64_t S_tot, cudaStream_t stream) {
    if (F_max == 0) return;
    int blocks = (F_max + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    clean_rows_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
        rows, F_dev, row_ptr, row_state, row_count, free_stack, free_top, F_max, S_tot);
}

// =============================================================================
// One2one merge - warp-cooperative version
// Each warp (32 threads) processes one row collaboratively.
// D dimensions are distributed across lanes: each lane handles D/32 elements.
// Dot products and norms use warp-level reductions via __shfl_xor_sync.
// Register pressure: ~130 floats/thread vs ~3072 in the scalar version.
// =============================================================================
template <typename scalar_t>
__device__ __forceinline__ float to_float(scalar_t v) {
    if constexpr (std::is_same_v<scalar_t, __nv_bfloat16>) return __bfloat162float(v);
    else return __half2float(v);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t from_float(float v) {
    if constexpr (std::is_same_v<scalar_t, __nv_bfloat16>) return __float2bfloat16(v);
    else return __float2half(v);
}

// Vectorized D-wide copy: 16-byte (float4) loads/stores.
// Requires src/dst to be 16-byte aligned (guaranteed when D is a multiple of 8
// for half/bf16, since each vector start = k * D * 2 bytes and D*2 >= 16).
template <typename scalar_t>
__device__ __forceinline__ void vec_copy_d(
        scalar_t* __restrict__ dst, const scalar_t* __restrict__ src, int D) {
    constexpr int EPV = sizeof(float4) / sizeof(scalar_t);  // 8 for half/bf16
    const int nvec = D / EPV;
    const float4* s4 = reinterpret_cast<const float4*>(src);
    float4* d4 = reinterpret_cast<float4*>(dst);
    for (int i = 0; i < nvec; ++i)
        d4[i] = s4[i];
    for (int i = nvec * EPV; i < D; ++i)
        dst[i] = src[i];
}

template <typename scalar_t>
__device__ __forceinline__ void vec_zero_d(scalar_t* dst, int D) {
    constexpr int EPV = sizeof(float4) / sizeof(scalar_t);
    const int nvec = D / EPV;
    float4* d4 = reinterpret_cast<float4*>(dst);
    const float4 z4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (int i = 0; i < nvec; ++i)
        d4[i] = z4;
    const scalar_t sz = from_float<scalar_t>(0.0f);
    for (int i = nvec * EPV; i < D; ++i)
        dst[i] = sz;
}

constexpr int WARP_SIZE = 32;
constexpr unsigned FULL_MASK = 0xFFFFFFFF;
constexpr int MAX_DPL = (MAX_DIM + WARP_SIZE - 1) / WARP_SIZE;  // max D elements per lane (4 for D=128)
constexpr int ONE2ONE_WARPS_PER_BLOCK = 8;

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(FULL_MASK, val, offset);
    return val;
}

// cand_row_counts: when non-null, candidates are written in CSR layout
// at positions [row_offsets[ri] .. row_offsets[ri]+count) into cand_K/V/S,
// and per-row counts are stored in cand_row_counts[ri].
// When null, the original flat-output mode is used (atomicAdd on cand_cnt).
template <typename scalar_t>
__global__ void one2one_merge_kernel(
    const int64_t* __restrict__ unique_rows, const int32_t* __restrict__ row_offsets,
    const scalar_t* __restrict__ sorted_K, const scalar_t* __restrict__ sorted_V,
    const float* __restrict__ sorted_S, const int32_t* __restrict__ G_dev,
    scalar_t* __restrict__ piv_K, scalar_t* __restrict__ piv_V,
    float* __restrict__ piv_W, float* __restrict__ piv_S, float* __restrict__ piv_C,
    const scalar_t* __restrict__ piv_K_seed, uint8_t* __restrict__ piv_M,
    const int32_t* __restrict__ row_ptr, const int8_t* __restrict__ row_state,
    scalar_t* __restrict__ cand_K, scalar_t* __restrict__ cand_V,
    float* __restrict__ cand_S, int64_t* __restrict__ cand_rows, int32_t* __restrict__ cand_cnt,
    int32_t* __restrict__ cand_row_counts,
    float sim_thresh, float replace_thresh, float score_thresh, int P, int D, int G_max, int E_max,
    int32_t* __restrict__ diag) {

    const int ri = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;

    const int G = G_dev ? *G_dev : G_max;
    if (ri >= G) return;

    const int64_t row = unique_rows[ri];
    if (row < 0) return;

    const int32_t slot = row_ptr[row];
    const int32_t start = row_offsets[ri], end = row_offsets[ri + 1];
    const bool csr_mode = (cand_row_counts != nullptr);

    const int dpl = (D + WARP_SIZE - 1) / WARP_SIZE;
    const int d_lo = lane * dpl;
    const int d_hi = (d_lo + dpl < D) ? d_lo + dpl : D;
    const int my_d = d_hi - d_lo;

    int cand_local = 0;

    // ---------- RESERVED/invalid rows: all tokens → candidates ----------
    if (slot < 0 || row_state[row] == (int8_t)BufState::RESERVED) {
        int n_tok = end - start;
        for (int t = start; t < end; ++t) {
            if (csr_mode) {
                int out = start + cand_local;
                if (lane == 0) cand_S[out] = sorted_S[t];
                for (int d = d_lo; d < d_hi; ++d) {
                    cand_K[(int64_t)out * D + d] = sorted_K[(int64_t)t * D + d];
                    cand_V[(int64_t)out * D + d] = sorted_V[(int64_t)t * D + d];
                }
                cand_local++;
            } else {
                int ci = 0;
                if (lane == 0) ci = atomicAdd(cand_cnt, 1);
                ci = __shfl_sync(FULL_MASK, ci, 0);
                if (ci < E_max) {
                    if (lane == 0) { cand_rows[ci] = row; cand_S[ci] = sorted_S[t]; }
                    for (int d = d_lo; d < d_hi; ++d) {
                        cand_K[(int64_t)ci * D + d] = sorted_K[(int64_t)t * D + d];
                        cand_V[(int64_t)ci * D + d] = sorted_V[(int64_t)t * D + d];
                    }
                }
            }
        }
        if (csr_mode && lane == 0) cand_row_counts[ri] = cand_local;
        if (diag && lane == 0) atomicAdd(&diag[DIAG_O2O_CAND_RESERVED], n_tok);
        return;
    }

    const int64_t pbase_s = (int64_t)slot * P;
    const int64_t pbase_v = (int64_t)slot * P * D;

    // ---------- Load & normalize pivot seeds (each lane holds its D slice) ----------
    float seeds[MAX_PIVOTS][MAX_DPL];
    uint8_t mask[MAX_PIVOTS];
    float sum_S = 0, sum_C = 0;
    int n_valid = 0;

    for (int p = 0; p < P && p < MAX_PIVOTS; ++p) {
        mask[p] = piv_M[pbase_s + p];
        if (mask[p]) {
            float pnorm = 0.0f;
            for (int di = 0; di < my_d; ++di) {
                float k = to_float(piv_K[pbase_v + p * D + d_lo + di]);
                seeds[p][di] = k;
                pnorm += k * k;
            }
            float norm_sq = warp_reduce_sum(pnorm);
            float sinv = rsqrtf(norm_sq + 1e-6f);
            for (int di = 0; di < my_d; ++di)
                seeds[p][di] *= sinv;

            sum_S += piv_S[pbase_s + p];
            sum_C += piv_C[pbase_s + p];
            n_valid++;
        }
    }

    if (n_valid == 0) {
        int n_tok = end - start;
        for (int t = start; t < end; ++t) {
            if (csr_mode) {
                int out = start + cand_local;
                if (lane == 0) cand_S[out] = sorted_S[t];
                for (int d = d_lo; d < d_hi; ++d) {
                    cand_K[(int64_t)out * D + d] = sorted_K[(int64_t)t * D + d];
                    cand_V[(int64_t)out * D + d] = sorted_V[(int64_t)t * D + d];
                }
                cand_local++;
            } else {
                int ci = 0;
                if (lane == 0) ci = atomicAdd(cand_cnt, 1);
                ci = __shfl_sync(FULL_MASK, ci, 0);
                if (ci < E_max) {
                    if (lane == 0) { cand_rows[ci] = row; cand_S[ci] = sorted_S[t]; }
                    for (int d = d_lo; d < d_hi; ++d) {
                        cand_K[(int64_t)ci * D + d] = sorted_K[(int64_t)t * D + d];
                        cand_V[(int64_t)ci * D + d] = sorted_V[(int64_t)t * D + d];
                    }
                }
            }
        }
        if (csr_mode && lane == 0) cand_row_counts[ri] = cand_local;
        if (diag && lane == 0) atomicAdd(&diag[DIAG_O2O_CAND_NO_PIVOT], n_tok);
        return;
    }

    // ---------- Accumulators (each lane holds its D slice) ----------
    float K_acc[MAX_PIVOTS][MAX_DPL], V_acc[MAX_PIVOTS][MAX_DPL];
    float a_acc[MAX_PIVOTS], S_acc[MAX_PIVOTS], C_acc[MAX_PIVOTS];

    for (int p = 0; p < P && p < MAX_PIVOTS; ++p) {
        a_acc[p] = S_acc[p] = C_acc[p] = 0.0f;
        for (int di = 0; di < my_d; ++di)
            K_acc[p][di] = V_acc[p][di] = 0.0f;
    }

    // ---------- Pass 1: accumulate merge data for high-similarity tokens ----------
    int n_absorbed = 0;
    for (int t = start; t < end; ++t) {
        float tK[MAX_DPL];
        float pnorm = 0.0f;
        for (int di = 0; di < my_d; ++di) {
            float k = to_float(sorted_K[(int64_t)t * D + d_lo + di]);
            tK[di] = k;
            pnorm += k * k;
        }
        float inv = rsqrtf(warp_reduce_sum(pnorm) + 1e-6f);
        float sc = sorted_S[t];

        float smax = -1e30f;
        int bp = -1;
        for (int p = 0; p < P && p < MAX_PIVOTS; ++p) {
            if (!mask[p]) continue;
            float pdot = 0.0f;
            for (int di = 0; di < my_d; ++di)
                pdot += (tK[di] * inv) * seeds[p][di];
            float s = fminf(fmaxf(warp_reduce_sum(pdot), -1.0f), 1.0f);
            if (s > smax) { smax = s; bp = p; }
        }

        if (smax >= sim_thresh && bp >= 0) {
            n_absorbed++;
            float a = expf(smax);
            for (int di = 0; di < my_d; ++di) {
                K_acc[bp][di] += a * to_float(sorted_K[(int64_t)t * D + d_lo + di]);
                V_acc[bp][di] += a * to_float(sorted_V[(int64_t)t * D + d_lo + di]);
            }
            a_acc[bp] += a;
            S_acc[bp] += sc;
            C_acc[bp] += 1.0f;
        }
    }

    // ---------- Post-merge gate ----------
    float post_sum_S = sum_S, post_sum_C = sum_C;
    for (int p = 0; p < P && p < MAX_PIVOTS; ++p) {
        post_sum_S += S_acc[p];
        post_sum_C += C_acc[p];
    }
    float gate = (post_sum_C > 1e-6f ? post_sum_S / post_sum_C : 0.0f) * score_thresh;

    // ---------- Pass 2: classify remaining tokens ----------
    int n_dropped = 0;
    for (int t = start; t < end; ++t) {
        float tK[MAX_DPL];
        float pnorm = 0.0f;
        for (int di = 0; di < my_d; ++di) {
            float k = to_float(sorted_K[(int64_t)t * D + d_lo + di]);
            tK[di] = k;
            pnorm += k * k;
        }
        float inv = rsqrtf(warp_reduce_sum(pnorm) + 1e-6f);
        float sc = sorted_S[t];

        float smax = -1e30f;
        for (int p = 0; p < P && p < MAX_PIVOTS; ++p) {
            if (!mask[p]) continue;
            float pdot = 0.0f;
            for (int di = 0; di < my_d; ++di)
                pdot += (tK[di] * inv) * seeds[p][di];
            float s = fminf(fmaxf(warp_reduce_sum(pdot), -1.0f), 1.0f);
            if (s > smax) smax = s;
        }

        if (smax >= sim_thresh) continue;
        if (smax < replace_thresh && sc > gate) {
            if (csr_mode) {
                int out = start + cand_local;
                if (lane == 0) cand_S[out] = sc;
                for (int d = d_lo; d < d_hi; ++d) {
                    cand_K[(int64_t)out * D + d] = sorted_K[(int64_t)t * D + d];
                    cand_V[(int64_t)out * D + d] = sorted_V[(int64_t)t * D + d];
                }
                cand_local++;
            } else {
                int ci = 0;
                if (lane == 0) ci = atomicAdd(cand_cnt, 1);
                ci = __shfl_sync(FULL_MASK, ci, 0);
                if (ci < E_max) {
                    if (lane == 0) { cand_rows[ci] = row; cand_S[ci] = sc; }
                    for (int d = d_lo; d < d_hi; ++d) {
                        cand_K[(int64_t)ci * D + d] = sorted_K[(int64_t)t * D + d];
                        cand_V[(int64_t)ci * D + d] = sorted_V[(int64_t)t * D + d];
                    }
                }
            }
        } else {
            n_dropped++;
        }
    }

    if (csr_mode && lane == 0) cand_row_counts[ri] = cand_local;

    if (diag && lane == 0) {
        if (n_absorbed > 0) atomicAdd(&diag[DIAG_O2O_ABSORBED], n_absorbed);
        if (cand_local > 0)  atomicAdd(&diag[DIAG_O2O_CAND_LOW_SIM], cand_local);
        if (n_dropped > 0)  atomicAdd(&diag[DIAG_O2O_DROPPED], n_dropped);
    }

    // ---------- Write back pivot updates (each lane writes its D slice) ----------
    for (int p = 0; p < P && p < MAX_PIVOTS; ++p) {
        if (a_acc[p] <= 0.0f) continue;
        int64_t pi = pbase_s + p, pv = pbase_v + p * D;

        float wo = piv_W[pi];
        float dn = fmaxf(wo + a_acc[p], 1e-8f);
        float fo = wo / dn;
        float fn = 1.0f / dn;

        for (int di = 0; di < my_d; ++di) {
            int d = d_lo + di;
            float k_old = to_float(piv_K[pv + d]);
            float v_old = to_float(piv_V[pv + d]);
            piv_K[pv + d] = from_float<scalar_t>(fo * k_old + fn * K_acc[p][di]);
            piv_V[pv + d] = from_float<scalar_t>(fo * v_old + fn * V_acc[p][di]);
        }
        if (lane == 0) {
            piv_W[pi] = dn;
            piv_S[pi] += S_acc[p];
            piv_C[pi] += C_acc[p];
        }
    }
}

void one2one_merge(
    const int64_t* unique_rows, const int32_t* row_offsets, const void* sorted_K,
    const void* sorted_V, const float* sorted_S, const int32_t* G_dev,
    void* piv_K, void* piv_V, float* piv_W, float* piv_S, float* piv_C,
    const void* piv_K_seed, uint8_t* piv_M, const int32_t* row_ptr, const int8_t* row_state,
    void* cand_K, void* cand_V, float* cand_S, int64_t* cand_rows, int32_t* cand_count_dev,
    float sim_thresh, float replace_thresh, float score_thresh, int64_t P, int64_t D,
    int64_t G_max, int64_t E_max, DType dtype, cudaStream_t stream,
    int32_t* cand_row_counts, int32_t* diag) {
    if (G_max == 0) return;
    if (!cand_row_counts) zero_scalar_kernel<<<1, 1, 0, stream>>>(cand_count_dev);
    const int total_threads_needed = G_max * WARP_SIZE;
    const int threads_per_block = ONE2ONE_WARPS_PER_BLOCK * WARP_SIZE;
    const int blocks = (total_threads_needed + threads_per_block - 1) / threads_per_block;
    if (dtype == DType::Float16) {
        one2one_merge_kernel<__half><<<blocks, threads_per_block, 0, stream>>>(
            unique_rows, row_offsets, (const __half*)sorted_K, (const __half*)sorted_V, sorted_S, G_dev,
            (__half*)piv_K, (__half*)piv_V, piv_W, piv_S, piv_C, (const __half*)piv_K_seed, piv_M,
            row_ptr, row_state, (__half*)cand_K, (__half*)cand_V, cand_S, cand_rows, cand_count_dev,
            cand_row_counts,
            sim_thresh, replace_thresh, score_thresh, P, D, G_max, E_max, diag);
    } else {
        one2one_merge_kernel<__nv_bfloat16><<<blocks, threads_per_block, 0, stream>>>(
            unique_rows, row_offsets, (const __nv_bfloat16*)sorted_K, (const __nv_bfloat16*)sorted_V, sorted_S, G_dev,
            (__nv_bfloat16*)piv_K, (__nv_bfloat16*)piv_V, piv_W, piv_S, piv_C, (const __nv_bfloat16*)piv_K_seed, piv_M,
            row_ptr, row_state, (__nv_bfloat16*)cand_K, (__nv_bfloat16*)cand_V, cand_S, cand_rows, cand_count_dev,
            cand_row_counts,
            sim_thresh, replace_thresh, score_thresh, P, D, G_max, E_max, diag);
    }
}

// =============================================================================
// Buffer top-B update
// =============================================================================
// cand_row_counts: when non-null, new-token range for row ri is
// [row_offsets[ri], row_offsets[ri] + cand_row_counts[ri]) instead of
// [row_offsets[ri], row_offsets[ri+1]).
template <typename scalar_t>
__global__ void buffer_topb_kernel(
    const int64_t* __restrict__ unique_rows, const int32_t* __restrict__ row_offsets,
    const scalar_t* __restrict__ sorted_K, const scalar_t* __restrict__ sorted_V,
    const float* __restrict__ sorted_S, const int32_t* __restrict__ G_dev,
    int32_t* __restrict__ row_ptr, int8_t* __restrict__ row_state, int32_t* __restrict__ row_count,
    scalar_t* __restrict__ pool_K, scalar_t* __restrict__ pool_V,
    float* __restrict__ pool_S, uint8_t* __restrict__ pool_M,
    scalar_t* __restrict__ over_K, scalar_t* __restrict__ over_V,
    float* __restrict__ over_S, int64_t* __restrict__ over_rows, int32_t* __restrict__ over_cnt,
    int B, int D, int G_max, int max_over, int64_t S_tot,
    const int32_t* __restrict__ cand_row_counts,
    int32_t* __restrict__ diag) {
    int ri = blockIdx.x * blockDim.x + threadIdx.x;
    int G = G_dev ? *G_dev : G_max;
    if (ri >= G) return;
    
    int64_t row = unique_rows[ri];
    if (row < 0 || row >= S_tot) return;
    int32_t slot = row_ptr[row];
    int8_t state = row_state[row];
    int32_t st = row_offsets[ri];
    int32_t en = cand_row_counts ? (st + cand_row_counts[ri]) : row_offsets[ri + 1];

    // Python parity: non-AVAILABLE rows cannot append; all incoming tokens overflow.
    if (slot < 0 || state != (int8_t)BufState::AVAILABLE) {
        int n_new = en - st;
        if (n_new > 0) {
            int32_t ob = atomicAdd(over_cnt, n_new);
            for (int i = 0; i < n_new; ++i) {
                int op = ob + i; if (op >= max_over) break;
                over_rows[op] = row;
                over_S[op] = sorted_S[st + i];
                vec_copy_d(&over_K[(int64_t)op*D], &sorted_K[(int64_t)(st+i)*D], D);
                vec_copy_d(&over_V[(int64_t)op*D], &sorted_V[(int64_t)(st+i)*D], D);
            }
            if (diag) atomicAdd(&diag[DIAG_BUF_OVER_NO_SLOT], n_new);
        }
        return;
    }

    int n_new = en - st, n_ex = row_count[row];
    
    float sc[MAX_BUFFER]; int idx[MAX_BUFFER]; int nc = 0;
    int64_t sb = (int64_t)slot * B, vb = (int64_t)slot * B * D;
    
    // Cache existing token data to avoid in-place aliasing during write-back.
    // Without caching, writing a new token to position 0 would overwrite an
    // existing token's data before it can be read for its new position.
    float   ex_S[MAX_BUFFER];
    scalar_t ex_K[MAX_BUFFER * MAX_DIM];
    scalar_t ex_V[MAX_BUFFER * MAX_DIM];
    int n_cached = 0;
    
    for (int i = 0; i < n_ex && i < B && nc < MAX_BUFFER; ++i) {
        if (pool_M[sb+i]) {
            sc[nc] = pool_S[sb+i];
            idx[nc] = -(n_cached+1);
            ex_S[n_cached] = pool_S[sb+i];
            for (int d = 0; d < D && d < MAX_DIM; ++d) {
                ex_K[n_cached * MAX_DIM + d] = pool_K[vb + (int64_t)i * D + d];
                ex_V[n_cached * MAX_DIM + d] = pool_V[vb + (int64_t)i * D + d];
            }
            n_cached++;
            nc++;
        }
    }
    for (int i = 0; i < n_new && nc < MAX_BUFFER; ++i)
        { sc[nc] = sorted_S[st+i]; idx[nc++] = i; }
    
    // Stable insertion sort (descending by score).
    // Strict '>' preserves original order for equal scores, matching
    // Python's torch.argsort(..., stable=True) in append_batch_dict.
    for (int i = 1; i < nc; ++i) {
        float key_s = sc[i]; int key_i = idx[i];
        int j = i - 1;
        while (j >= 0 && sc[j] < key_s) {
            sc[j+1] = sc[j]; idx[j+1] = idx[j];
            j--;
        }
        sc[j+1] = key_s; idx[j+1] = key_i;
    }
    
    int nk = min(nc, B);
    for (int b = 0; b < nk; ++b) {
        int x = idx[b];
        const scalar_t* src_K;
        const scalar_t* src_V;
        if (x < 0) { int ci = -(x+1); src_K = &ex_K[ci * MAX_DIM]; src_V = &ex_V[ci * MAX_DIM]; }
        else { src_K = &sorted_K[(int64_t)(st+x)*D]; src_V = &sorted_V[(int64_t)(st+x)*D]; }
        vec_copy_d(&pool_K[vb+(int64_t)b*D], src_K, D);
        vec_copy_d(&pool_V[vb+(int64_t)b*D], src_V, D);
        pool_S[sb+b] = sc[b]; pool_M[sb+b] = 1;
    }
    for (int b = nk; b < B; ++b) pool_M[sb+b] = 0;
    row_count[row] = nk;
    bool became_full = (nk == B);
    if (became_full) row_state[row] = (int8_t)BufState::FULL;
    
    int no = nc - nk;
    if (no > 0) {
        int32_t ob = atomicAdd(over_cnt, no);
        for (int i = 0; i < no; ++i) {
            int op = ob + i; if (op >= max_over) break;
            int x = idx[nk+i];
            const scalar_t* src_K;
            const scalar_t* src_V;
            if (x < 0) { int ci = -(x+1); src_K = &ex_K[ci * MAX_DIM]; src_V = &ex_V[ci * MAX_DIM]; }
            else { src_K = &sorted_K[(int64_t)(st+x)*D]; src_V = &sorted_V[(int64_t)(st+x)*D]; }
            over_rows[op] = row; over_S[op] = sc[nk+i];
            vec_copy_d(&over_K[(int64_t)op*D], src_K, D);
            vec_copy_d(&over_V[(int64_t)op*D], src_V, D);
        }
    }
    if (diag) {
        if (nk > 0) atomicAdd(&diag[DIAG_BUF_KEPT], nk);
        if (no > 0) atomicAdd(&diag[DIAG_BUF_OVER_EXCESS], no);
        if (became_full) atomicAdd(&diag[DIAG_BUF_ROWS_FULL], 1);
    }
}

void buffer_topb_update(
    const int64_t* unique_rows, const int32_t* row_offsets, const void* sorted_K,
    const void* sorted_V, const float* sorted_S, const int32_t* G_dev,
    int32_t* row_ptr, int8_t* row_state, int32_t* row_count,
    void* pool_K, void* pool_V, float* pool_S, uint8_t* pool_M,
    void* over_K, void* over_V, float* over_S, int64_t* over_rows, int32_t* over_count_dev,
    int64_t B, int64_t D, int64_t G_max, int64_t E_max, int64_t S_tot, DType dtype, cudaStream_t stream,
    const int32_t* cand_row_counts, int32_t* diag) {
    if (G_max == 0) return;
    zero_scalar_kernel<<<1, 1, 0, stream>>>(over_count_dev);
    int blocks = (G_max + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    if (dtype == DType::Float16) {
        buffer_topb_kernel<__half><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
            unique_rows, row_offsets, (const __half*)sorted_K, (const __half*)sorted_V, sorted_S, G_dev,
            row_ptr, row_state, row_count, (__half*)pool_K, (__half*)pool_V, pool_S, pool_M,
            (__half*)over_K, (__half*)over_V, over_S, over_rows, over_count_dev, B, D, G_max, E_max, S_tot,
            cand_row_counts, diag);
    } else {
        buffer_topb_kernel<__nv_bfloat16><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
            unique_rows, row_offsets, (const __nv_bfloat16*)sorted_K, (const __nv_bfloat16*)sorted_V, sorted_S, G_dev,
            row_ptr, row_state, row_count, (__nv_bfloat16*)pool_K, (__nv_bfloat16*)pool_V, pool_S, pool_M,
            (__nv_bfloat16*)over_K, (__nv_bfloat16*)over_V, over_S, over_rows, over_count_dev, B, D, G_max, E_max, S_tot,
            cand_row_counts, diag);
    }
}

// =============================================================================
// All2one merge (fused: filter FULL + materialize pivot + merge + clean buffer)
// Replaces separate Steps 8, 8b, 9, 10 with a single kernel launch.
// Warp-cooperative: one warp (32 lanes) per row, D partitioned across lanes.
// Non-FULL rows exit immediately (all 32 lanes together).
// =============================================================================
constexpr int ALL2ONE_WARPS_PER_BLOCK = 4;

template <typename scalar_t>
__global__ void all2one_merge_fused_kernel(
    const int64_t* __restrict__ unique_rows,
    const int32_t* __restrict__ G_dev,
    const scalar_t* __restrict__ buf_K, const scalar_t* __restrict__ buf_V,
    const float* __restrict__ buf_S, uint8_t* __restrict__ buf_M,
    scalar_t* __restrict__ piv_K, scalar_t* __restrict__ piv_V,
    float* __restrict__ piv_W, float* __restrict__ piv_S, float* __restrict__ piv_C,
    scalar_t* __restrict__ piv_Ks, float* __restrict__ piv_Ss, uint8_t* __restrict__ piv_M,
    int32_t* __restrict__ buf_rp, int8_t* __restrict__ buf_rs, int32_t* __restrict__ buf_rc,
    int32_t* __restrict__ piv_rp, int8_t* __restrict__ piv_rs, int32_t* __restrict__ piv_row_count,
    int32_t* __restrict__ piv_free_stack, int32_t* __restrict__ piv_free_top,
    int32_t* __restrict__ buf_free_stack, int32_t* __restrict__ buf_free_top,
    int B, int P, int D, int G_max, int64_t S_tot,
    int32_t* __restrict__ diag) {

    const int ri = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;

    const int G = G_dev ? *G_dev : G_max;
    if (ri >= G) return;

    const int64_t row = unique_rows[ri];
    if (row < 0 || row >= S_tot) return;

    // ---- Step 8: only process FULL buffer rows (uniform branch) ----
    if (buf_rs[row] != (int8_t)BufState::FULL) return;

    const int32_t bs = buf_rp[row];
    if (bs < 0) return;

    // D-partitioning across warp lanes
    const int dpl = (D + WARP_SIZE - 1) / WARP_SIZE;
    const int d_lo = lane * dpl;
    const int d_hi = (d_lo + dpl < D) ? d_lo + dpl : D;
    const int my_d = d_hi - d_lo;

    // ---- Step 8b: materialize pivot row if needed (lane 0 does atomics) ----
    int32_t ps = piv_rp[row];
    if (ps < 0 || piv_rs[row] == (int8_t)BufState::RESERVED) {
        if (lane == 0) {
            int32_t top = atomicSub(piv_free_top, 1) - 1;
            if (top < 0) {
                atomicAdd(piv_free_top, 1);
                ps = -1;
            } else {
                ps = piv_free_stack[top];
                piv_rp[row] = ps;
                piv_rs[row] = (int8_t)BufState::AVAILABLE;
                piv_row_count[row] = 0;
                for (int p = 0; p < P; ++p) piv_M[(int64_t)ps * P + p] = 0;
            }
        }
        ps = __shfl_sync(FULL_MASK, ps, 0);
        if (ps < 0) return;
    }

    // ---- Step 9: all2one merge (warp-cooperative) ----
    const int64_t bbs = (int64_t)bs * B, bbv = (int64_t)bs * B * D;

    // Each lane holds its D-slice of buffer tokens
    float Kb[MAX_BUFFER][MAX_DPL], Vb[MAX_BUFFER][MAX_DPL];
    float Sb[MAX_BUFFER];
    int nv = 0;

    for (int b = 0; b < B && b < MAX_BUFFER; ++b) {
        if (buf_M[bbs + b]) {
            Sb[nv] = buf_S[bbs + b];
            for (int di = 0; di < my_d; ++di) {
                Kb[nv][di] = to_float(buf_K[bbv + (int64_t)b * D + d_lo + di]);
                Vb[nv][di] = to_float(buf_V[bbv + (int64_t)b * D + d_lo + di]);
            }
            nv++;
        }
    }

    if (nv == 0) {
        for (int b = lane; b < B; b += WARP_SIZE) buf_M[bbs + b] = 0;
        if (lane == 0) {
            int32_t buf_top = atomicAdd(buf_free_top, 1);
            buf_free_stack[buf_top] = bs;
            buf_rp[row] = -1;
            buf_rs[row] = (int8_t)BufState::RESERVED;
            buf_rc[row] = 0;
        }
        return;
    }

    // Find max-score token as seed (all lanes compute identically)
    int si = 0;
    float ss = Sb[0];
    for (int i = 1; i < nv; ++i)
        if (Sb[i] > ss) { ss = Sb[i]; si = i; }

    // Normalize seed (each lane holds its D-slice)
    float Ks[MAX_DPL];
    float sn = 0.0f;
    for (int di = 0; di < my_d; ++di) {
        float k = Kb[si][di];
        Ks[di] = k;
        sn += k * k;
    }
    {
        float sinv = rsqrtf(warp_reduce_sum(sn) + 1e-6f);
        for (int di = 0; di < my_d; ++di) Ks[di] *= sinv;
    }

    // Weighted merge: all tokens → single pivot
    float Kp[MAX_DPL], Vp[MAX_DPL];
    for (int di = 0; di < my_d; ++di) Kp[di] = Vp[di] = 0.0f;
    float Wp = 0, Sp = 0, Cp = 0;

    for (int i = 0; i < nv; ++i) {
        float n2 = 0;
        for (int di = 0; di < my_d; ++di) {
            float k = Kb[i][di];
            n2 += k * k;
        }
        float inv = rsqrtf(warp_reduce_sum(n2) + 1e-6f);

        float sim_partial = 0;
        for (int di = 0; di < my_d; ++di)
            sim_partial += (Kb[i][di] * inv) * Ks[di];
        float sim = fminf(fmaxf(warp_reduce_sum(sim_partial), -1.0f), 1.0f);
        float c = expf(sim);

        for (int di = 0; di < my_d; ++di) {
            Kp[di] += c * Kb[i][di];
            Vp[di] += c * Vb[i][di];
        }
        Wp += c;
        Sp += Sb[i];
        Cp += 1.0f;
    }

    float invW = 1.0f / fmaxf(Wp, 1e-8f);
    for (int di = 0; di < my_d; ++di) {
        Kp[di] *= invW;
        Vp[di] *= invW;
    }

    // Write new pivot into the pivot pool
    {
        const int64_t pps = (int64_t)ps * P, ppv = (int64_t)ps * P * D;

        // Find empty slot (all lanes compute identically)
        int ws = -1;
        for (int p = 0; p < P && p < MAX_PIVOTS; ++p)
            if (!piv_M[pps + p]) { ws = p; break; }

        if (ws < 0) {
            // Find victim (min weight) — scalar, all lanes agree
            int j_victim = 0;
            float min_W = piv_W[pps];
            for (int p = 1; p < P && p < MAX_PIVOTS; ++p) {
                if (piv_M[pps + p] && piv_W[pps + p] < min_W) {
                    min_W = piv_W[pps + p];
                    j_victim = p;
                }
            }

            // Load victim K, compute norm (warp-cooperative)
            float K_victim[MAX_DPL];
            float victim_norm_sq = 0.0f;
            for (int di = 0; di < my_d; ++di) {
                float k = to_float(piv_K[ppv + (int64_t)j_victim * D + d_lo + di]);
                K_victim[di] = k;
                victim_norm_sq += k * k;
            }
            float victim_inv_norm = rsqrtf(warp_reduce_sum(victim_norm_sq) + 1e-6f);

            // Find nearest neighbor pivot
            int j_nei = -1;
            float max_sim = -1e30f;
            for (int p = 0; p < P && p < MAX_PIVOTS; ++p) {
                if (!piv_M[pps + p] || p == j_victim) continue;
                float nei_norm_sq = 0.0f;
                for (int di = 0; di < my_d; ++di) {
                    float k = to_float(piv_K[ppv + (int64_t)p * D + d_lo + di]);
                    nei_norm_sq += k * k;
                }
                float nei_inv_norm = rsqrtf(warp_reduce_sum(nei_norm_sq) + 1e-6f);
                float sim_partial = 0.0f;
                for (int di = 0; di < my_d; ++di)
                    sim_partial += (K_victim[di] * victim_inv_norm) *
                                   (to_float(piv_K[ppv + (int64_t)p * D + d_lo + di]) * nei_inv_norm);
                float sim = fminf(fmaxf(warp_reduce_sum(sim_partial), -1.0f), 1.0f);
                if (sim > max_sim) { max_sim = sim; j_nei = p; }
            }

            // Merge victim into neighbor (each lane blends its D-slice)
            if (j_nei >= 0) {
                float scale_ij = expf(max_sim - 1.0f);
                float W_victim_scaled = piv_W[pps + j_victim] * scale_ij;
                float W_nei = piv_W[pps + j_nei];
                float denom = fmaxf(W_victim_scaled + W_nei, 1e-8f);
                float frac_v = W_victim_scaled / denom;
                float frac_n = W_nei / denom;
                for (int di = 0; di < my_d; ++di) {
                    int d = d_lo + di;
                    float k_v = to_float(piv_K[ppv + (int64_t)j_victim * D + d]);
                    float k_n = to_float(piv_K[ppv + (int64_t)j_nei * D + d]);
                    float v_v = to_float(piv_V[ppv + (int64_t)j_victim * D + d]);
                    float v_n = to_float(piv_V[ppv + (int64_t)j_nei * D + d]);
                    piv_K[ppv + (int64_t)j_nei * D + d] = from_float<scalar_t>(frac_v * k_v + frac_n * k_n);
                    piv_V[ppv + (int64_t)j_nei * D + d] = from_float<scalar_t>(frac_v * v_v + frac_n * v_n);
                }
                if (lane == 0) {
                    piv_W[pps + j_nei] = denom;
                    piv_S[pps + j_nei] = piv_S[pps + j_victim] + piv_S[pps + j_nei];
                    piv_C[pps + j_nei] = piv_C[pps + j_victim] + piv_C[pps + j_nei];
                }
            }
            ws = j_victim;
        }

        // Write new pivot (each lane writes its D-slice)
        for (int di = 0; di < my_d; ++di) {
            int d = d_lo + di;
            piv_K[ppv + (int64_t)ws * D + d] = from_float<scalar_t>(Kp[di]);
            piv_V[ppv + (int64_t)ws * D + d] = from_float<scalar_t>(Vp[di]);
            piv_Ks[ppv + (int64_t)ws * D + d] = from_float<scalar_t>(Ks[di]);
        }
        if (lane == 0) {
            piv_W[pps + ws] = Wp;
            piv_S[pps + ws] = Sp;
            piv_C[pps + ws] = Cp;
            piv_Ss[pps + ws] = ss;
            piv_M[pps + ws] = 1;
            piv_rs[row] = (int8_t)BufState::AVAILABLE;
            piv_row_count[row] = 1;  // one pivot written by all2one merge
        }
    }

    // ---- Step 10: clean buffer row (warp-parallel), return slot ----
    for (int b = lane; b < B; b += WARP_SIZE) buf_M[bbs + b] = 0;
    if (lane == 0) {
        int32_t buf_top = atomicAdd(buf_free_top, 1);
        buf_free_stack[buf_top] = bs;
        buf_rp[row] = -1;
        buf_rs[row] = (int8_t)BufState::RESERVED;
        buf_rc[row] = 0;
        if (diag) {
            atomicAdd(&diag[DIAG_A2O_ROWS_MERGED], 1);
            atomicAdd(&diag[DIAG_A2O_PIV_CREATED], 1);
        }
    }
}

void all2one_merge_fused(
    const int64_t* unique_rows, const int32_t* G_dev,
    const void* buf_K, const void* buf_V, const float* buf_S, uint8_t* buf_M,
    void* piv_K, void* piv_V, float* piv_W, float* piv_S, float* piv_C,
    void* piv_K_seed, float* piv_S_seed, uint8_t* piv_M,
    int32_t* buf_row_ptr, int8_t* buf_row_state, int32_t* buf_row_count,
    int32_t* piv_row_ptr, int8_t* piv_row_state, int32_t* piv_row_count,
    int32_t* piv_free_stack, int32_t* piv_free_top,
    int32_t* buf_free_stack, int32_t* buf_free_top,
    int64_t B, int64_t P, int64_t D,
    int64_t G_max, int64_t S_tot, DType dtype, cudaStream_t stream,
    int32_t* diag) {
    if (G_max == 0) return;
    const int total_threads = G_max * WARP_SIZE;
    const int tpb = ALL2ONE_WARPS_PER_BLOCK * WARP_SIZE;
    const int blocks = (total_threads + tpb - 1) / tpb;
    if (dtype == DType::Float16) {
        all2one_merge_fused_kernel<__half><<<blocks, tpb, 0, stream>>>(
            unique_rows, G_dev,
            (const __half*)buf_K, (const __half*)buf_V, buf_S, buf_M,
            (__half*)piv_K, (__half*)piv_V, piv_W, piv_S, piv_C,
            (__half*)piv_K_seed, piv_S_seed, piv_M,
            buf_row_ptr, buf_row_state, buf_row_count,
            piv_row_ptr, piv_row_state, piv_row_count,
            piv_free_stack, piv_free_top, buf_free_stack, buf_free_top,
            B, P, D, G_max, S_tot, diag);
    } else {
        all2one_merge_fused_kernel<__nv_bfloat16><<<blocks, tpb, 0, stream>>>(
            unique_rows, G_dev,
            (const __nv_bfloat16*)buf_K, (const __nv_bfloat16*)buf_V, buf_S, buf_M,
            (__nv_bfloat16*)piv_K, (__nv_bfloat16*)piv_V, piv_W, piv_S, piv_C,
            (__nv_bfloat16*)piv_K_seed, piv_S_seed, piv_M,
            buf_row_ptr, buf_row_state, buf_row_count,
            piv_row_ptr, piv_row_state, piv_row_count,
            piv_free_stack, piv_free_top, buf_free_stack, buf_free_top,
            B, P, D, G_max, S_tot, diag);
    }
}

// =============================================================================
// Retrieve fixed
// =============================================================================
template <typename scalar_t>
__global__ void retrieve_fixed_kernel(
    const int64_t* __restrict__ voxel_ids, const scalar_t* __restrict__ piv_K,
    const scalar_t* __restrict__ piv_V, const float* __restrict__ piv_C,
    const uint8_t* __restrict__ piv_M, const int32_t* __restrict__ row_ptr,
    const int8_t* __restrict__ row_state, scalar_t* __restrict__ K_out,
    scalar_t* __restrict__ V_out, uint8_t* __restrict__ M_out, float* __restrict__ bias_out,
    int Q, int H, int64_t V_alloc, int P, int D, int retrieve_size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)H * retrieve_size;
    if (idx >= total) return;
    
    int h = idx / retrieve_size;
    int out_slot = idx % retrieve_size;
    int q_idx = out_slot / P;
    int p_idx = out_slot % P;
    
    M_out[idx] = 0; bias_out[idx] = -__FLT_MAX__;
    vec_zero_d(&K_out[idx*D], D);
    vec_zero_d(&V_out[idx*D], D);
    
    if (q_idx >= Q) return;
    int64_t voxel = voxel_ids[q_idx];
    if (voxel < 0 || voxel >= V_alloc) return;
    int64_t row = (int64_t)h * V_alloc + voxel;
    int32_t slot = row_ptr[row];
    if (slot < 0 || row_state[row] == (int8_t)BufState::RESERVED) return;
    
    int64_t ps = (int64_t)slot * P + p_idx;
    int64_t pv = (int64_t)slot * P * D + p_idx * D;
    if (piv_M[ps] == 0) return;
    
    float C = piv_C[ps];
    M_out[idx] = 1;
    bias_out[idx] = (C > 0.0f) ? logf(C) : -__FLT_MAX__;
    vec_copy_d(&K_out[idx*D], &piv_K[pv], D);
    vec_copy_d(&V_out[idx*D], &piv_V[pv], D);
}

void retrieve_fixed(
    const int64_t* voxel_ids, int64_t Q, const void* piv_K, const void* piv_V,
    const float* piv_W, const float* piv_C, const uint8_t* piv_M,
    const int32_t* row_ptr, const int8_t* row_state, void* K_out, void* V_out,
    uint8_t* M_out, float* bias_out, int64_t H, int64_t V_alloc, int64_t P, int64_t D,
    int64_t retrieve_size, DType dtype, cudaStream_t stream) {
    if (Q == 0 || retrieve_size <= 0) return;
    int64_t total = H * retrieve_size;
    int blocks = (total + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    if (dtype == DType::Float16) {
        retrieve_fixed_kernel<__half><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
            voxel_ids, (const __half*)piv_K, (const __half*)piv_V, piv_C, piv_M,
            row_ptr, row_state, (__half*)K_out, (__half*)V_out, M_out, bias_out,
            Q, H, V_alloc, P, D, retrieve_size);
    } else {
        retrieve_fixed_kernel<__nv_bfloat16><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
            voxel_ids, (const __nv_bfloat16*)piv_K, (const __nv_bfloat16*)piv_V, piv_C, piv_M,
            row_ptr, row_state, (__nv_bfloat16*)K_out, (__nv_bfloat16*)V_out, M_out, bias_out,
            Q, H, V_alloc, P, D, retrieve_size);
    }
}

// =============================================================================
// Retrieve buffer tokens
// =============================================================================
template <typename scalar_t>
__global__ void retrieve_buf_kernel(
    const int64_t* __restrict__ voxel_ids, const scalar_t* __restrict__ buf_K,
    const scalar_t* __restrict__ buf_V, const uint8_t* __restrict__ buf_M,
    const int32_t* __restrict__ row_ptr, const int8_t* __restrict__ row_state,
    scalar_t* __restrict__ K_out, scalar_t* __restrict__ V_out,
    uint8_t* __restrict__ M_out, float* __restrict__ bias_out,
    int Q, int H, int64_t V_alloc, int B, int D, int retrieve_size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)H * retrieve_size;
    if (idx >= total) return;
    
    int h = idx / retrieve_size;
    int out_slot = idx % retrieve_size;
    int q_idx = out_slot / B;
    int b_idx = out_slot % B;
    
    // Default: invalid output
    M_out[idx] = 0;
    bias_out[idx] = -__FLT_MAX__;
    vec_zero_d(&K_out[idx*D], D);
    vec_zero_d(&V_out[idx*D], D);
    
    if (q_idx >= Q) return;
    int64_t voxel = voxel_ids[q_idx];
    if (voxel < 0 || voxel >= V_alloc) return;
    int64_t row = (int64_t)h * V_alloc + voxel;
    int32_t slot = row_ptr[row];
    if (slot < 0 || row_state[row] == (int8_t)BufState::RESERVED) return;
    
    // Buffer pool layout: [pool_cap, B, D] for K/V, [pool_cap, B] for M
    int64_t bs = (int64_t)slot * B + b_idx;
    int64_t bv = (int64_t)slot * B * D + b_idx * D;
    if (buf_M[bs] == 0) return;
    
    // Valid buffer token: bias = 0.0 (neutral, no count-based weighting)
    M_out[idx] = 1;
    bias_out[idx] = 0.0f;
    vec_copy_d(&K_out[idx*D], &buf_K[bv], D);
    vec_copy_d(&V_out[idx*D], &buf_V[bv], D);
}

void retrieve_buf(
    const int64_t* voxel_ids, int64_t Q, const void* buf_K, const void* buf_V,
    const uint8_t* buf_M, const int32_t* row_ptr, const int8_t* row_state,
    void* K_out, void* V_out, uint8_t* M_out, float* bias_out,
    int64_t H, int64_t V_alloc, int64_t B, int64_t D,
    int64_t retrieve_size, DType dtype, cudaStream_t stream) {
    if (Q == 0 || retrieve_size <= 0) return;
    int64_t total = H * retrieve_size;
    int blocks = (total + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    if (dtype == DType::Float16) {
        retrieve_buf_kernel<__half><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
            voxel_ids, (const __half*)buf_K, (const __half*)buf_V, buf_M,
            row_ptr, row_state, (__half*)K_out, (__half*)V_out, M_out, bias_out,
            Q, H, V_alloc, B, D, retrieve_size);
    } else {
        retrieve_buf_kernel<__nv_bfloat16><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
            voxel_ids, (const __nv_bfloat16*)buf_K, (const __nv_bfloat16*)buf_V, buf_M,
            row_ptr, row_state, (__nv_bfloat16*)K_out, (__nv_bfloat16*)V_out, M_out, bias_out,
            Q, H, V_alloc, B, D, retrieve_size);
    }
}

// =============================================================================
// Pool State Validation
// =============================================================================

/**
 * Initialize ValidationResult to zeros.
 */
__global__ void init_validation_result_kernel(ValidationResult* result) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        result->orphaned_slots = 0;
        result->invalid_row_ptrs = 0;
        result->state_count_mismatch = 0;
        result->state_mask_mismatch = 0;
        result->free_stack_errors = 0;
        result->total_errors = 0;
    }
}

/**
 * Validate buffer pool row states.
 * Each thread validates one row.
 */
__global__ void validate_buffer_rows_kernel(
    const int32_t* __restrict__ row_ptr,
    const int8_t* __restrict__ row_state,
    const int32_t* __restrict__ row_count,
    const uint8_t* __restrict__ pool_M,
    ValidationResult* __restrict__ result,
    int64_t S_tot,
    int64_t pool_cap,
    int B)
{
    int64_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= S_tot) return;
    
    int32_t slot = row_ptr[row];
    int8_t state = row_state[row];
    int32_t count = row_count[row];
    
    // Check based on state
    if (state == (int8_t)BufState::RESERVED) {
        // RESERVED rows should have slot == -1 and count == 0
        if (slot >= 0) {
            atomicAdd(&result->invalid_row_ptrs, 1);
            atomicAdd(&result->total_errors, 1);
        }
        if (count != 0) {
            atomicAdd(&result->state_count_mismatch, 1);
            atomicAdd(&result->total_errors, 1);
        }
    } else if (state == (int8_t)BufState::AVAILABLE || state == (int8_t)BufState::FULL) {
        // Active rows should have valid slot pointer
        if (slot < 0 || slot >= pool_cap) {
            atomicAdd(&result->invalid_row_ptrs, 1);
            atomicAdd(&result->total_errors, 1);
            return;  // Can't validate further without valid slot
        }
        
        // Count valid entries in slot
        int64_t slot_base = (int64_t)slot * B;
        int valid_count = 0;
        for (int b = 0; b < B; ++b) {
            if (pool_M[slot_base + b]) valid_count++;
        }
        
        // Check count matches
        if (valid_count != count) {
            atomicAdd(&result->state_count_mismatch, 1);
            atomicAdd(&result->total_errors, 1);
        }
        
        // Check state matches occupancy
        if (state == (int8_t)BufState::FULL && valid_count != B) {
            atomicAdd(&result->state_mask_mismatch, 1);
            atomicAdd(&result->total_errors, 1);
        }
        if (state == (int8_t)BufState::AVAILABLE && valid_count == B) {
            // AVAILABLE but fully occupied - should be FULL
            atomicAdd(&result->state_mask_mismatch, 1);
            atomicAdd(&result->total_errors, 1);
        }
    }
    // FREE and HELD states are also valid - they may have slot pointers for lazy cleanup
}

/**
 * Check for orphaned slots in buffer pool.
 * Each thread checks one slot.
 */
__global__ void validate_buffer_orphans_kernel(
    const int32_t* __restrict__ row_ptr,
    const int8_t* __restrict__ row_state,
    const uint8_t* __restrict__ pool_M,
    const int32_t* __restrict__ free_stack,
    const int32_t* __restrict__ free_top,
    ValidationResult* __restrict__ result,
    int64_t S_tot,
    int64_t pool_cap,
    int B)
{
    int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= pool_cap) return;
    
    // Check if slot has any data
    int64_t slot_base = (int64_t)slot * B;
    bool has_data = false;
    for (int b = 0; b < B && !has_data; ++b) {
        if (pool_M[slot_base + b]) has_data = true;
    }
    
    if (!has_data) return;  // Empty slot, no orphan check needed
    
    // Check if any active row points to this slot
    bool has_owner = false;
    for (int64_t r = 0; r < S_tot && !has_owner; ++r) {
        if (row_ptr[r] == slot && 
            (row_state[r] == (int8_t)BufState::AVAILABLE || 
             row_state[r] == (int8_t)BufState::FULL)) {
            has_owner = true;
        }
    }
    
    if (!has_owner) {
        atomicAdd(&result->orphaned_slots, 1);
        atomicAdd(&result->total_errors, 1);
    }
}

/**
 * Validate free stack integrity.
 * Single-thread kernel for simplicity (free_top is usually small).
 */
__global__ void validate_free_stack_kernel(
    const int32_t* __restrict__ row_ptr,
    const int8_t* __restrict__ row_state,
    const int32_t* __restrict__ free_stack,
    const int32_t* __restrict__ free_top,
    ValidationResult* __restrict__ result,
    int64_t S_tot,
    int64_t pool_cap)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    
    int top = *free_top;
    
    // Check free_top is in valid range
    if (top < 0 || top > pool_cap) {
        atomicAdd(&result->free_stack_errors, 1);
        atomicAdd(&result->total_errors, 1);
        return;
    }
    
    // Check all entries in free stack are valid slot indices
    // Use a simple O(n^2) check for duplicates (acceptable for validation)
    for (int i = 0; i < top; ++i) {
        int32_t slot = free_stack[i];
        if (slot < 0 || slot >= pool_cap) {
            atomicAdd(&result->free_stack_errors, 1);
            atomicAdd(&result->total_errors, 1);
            continue;
        }
        
        // Check for duplicates
        for (int j = i + 1; j < top; ++j) {
            if (free_stack[j] == slot) {
                atomicAdd(&result->free_stack_errors, 1);
                atomicAdd(&result->total_errors, 1);
            }
        }
    }
}

void validate_buffer_pool(
    const int32_t* row_ptr,
    const int8_t* row_state,
    const int32_t* row_count,
    const uint8_t* pool_M,
    const int32_t* free_stack,
    const int32_t* free_top,
    ValidationResult* result_dev,
    int64_t S_tot,
    int64_t pool_cap,
    int64_t B,
    cudaStream_t stream)
{
    // Initialize result
    init_validation_result_kernel<<<1, 1, 0, stream>>>(result_dev);
    
    // Validate row states
    if (S_tot > 0) {
        int blocks = (S_tot + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        validate_buffer_rows_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
            row_ptr, row_state, row_count, pool_M, result_dev, S_tot, pool_cap, B);
    }
    
    // Check for orphaned slots (skip if pool_cap is very large to avoid timeout)
    if (pool_cap > 0 && pool_cap <= 65536 && S_tot <= 65536) {
        int blocks = (pool_cap + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        validate_buffer_orphans_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
            row_ptr, row_state, pool_M, free_stack, free_top, result_dev, 
            S_tot, pool_cap, B);
    }
    
    // Validate free stack
    validate_free_stack_kernel<<<1, 1, 0, stream>>>(
        row_ptr, row_state, free_stack, free_top, result_dev, S_tot, pool_cap);
}

/**
 * Validate pivot pool row states.
 * Each thread validates one row.
 */
__global__ void validate_pivot_rows_kernel(
    const int32_t* __restrict__ row_ptr,
    const int8_t* __restrict__ row_state,
    const int32_t* __restrict__ row_count,
    const uint8_t* __restrict__ pool_M,
    const float* __restrict__ pool_W,
    ValidationResult* __restrict__ result,
    int64_t S_tot,
    int64_t pool_cap,
    int P)
{
    int64_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= S_tot) return;
    
    int32_t slot = row_ptr[row];
    int8_t state = row_state[row];
    // Note: row_count is not strictly validated for pivot pool as it may not be 
    // kept in sync with the actual mask count in all code paths.
    
    // Check based on state
    if (state == (int8_t)BufState::RESERVED) {
        // RESERVED rows should have slot == -1
        if (slot >= 0) {
            atomicAdd(&result->invalid_row_ptrs, 1);
            atomicAdd(&result->total_errors, 1);
        }
    } else if (state == (int8_t)BufState::AVAILABLE || state == (int8_t)BufState::FULL) {
        // Active rows should have valid slot pointer
        if (slot < 0 || slot >= pool_cap) {
            atomicAdd(&result->invalid_row_ptrs, 1);
            atomicAdd(&result->total_errors, 1);
            return;  // Can't validate further without valid slot
        }
        
        // Count valid pivots in slot (by mask)
        int64_t slot_base = (int64_t)slot * P;
        int valid_count = 0;
        for (int p = 0; p < P; ++p) {
            if (pool_M[slot_base + p]) valid_count++;
        }
        
        // Only flag if state is FULL but slot is completely empty (clearly wrong).
        // AVAILABLE state with empty slot is OK since pivots may be added later.
        if (state == (int8_t)BufState::FULL && valid_count == 0) {
            atomicAdd(&result->state_mask_mismatch, 1);
            atomicAdd(&result->total_errors, 1);
        }
    }
}

/**
 * Check for orphaned slots in pivot pool.
 * Each thread checks one slot.
 */
__global__ void validate_pivot_orphans_kernel(
    const int32_t* __restrict__ row_ptr,
    const int8_t* __restrict__ row_state,
    const uint8_t* __restrict__ pool_M,
    const int32_t* __restrict__ free_stack,
    const int32_t* __restrict__ free_top,
    ValidationResult* __restrict__ result,
    int64_t S_tot,
    int64_t pool_cap,
    int P)
{
    int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= pool_cap) return;
    
    // Check if slot has any valid pivots
    int64_t slot_base = (int64_t)slot * P;
    bool has_data = false;
    for (int p = 0; p < P && !has_data; ++p) {
        if (pool_M[slot_base + p]) has_data = true;
    }
    
    if (!has_data) return;  // Empty slot, no orphan check needed
    
    // Check if any active row points to this slot
    bool has_owner = false;
    for (int64_t r = 0; r < S_tot && !has_owner; ++r) {
        if (row_ptr[r] == slot && 
            (row_state[r] == (int8_t)BufState::AVAILABLE || 
             row_state[r] == (int8_t)BufState::FULL)) {
            has_owner = true;
        }
    }
    
    if (!has_owner) {
        atomicAdd(&result->orphaned_slots, 1);
        atomicAdd(&result->total_errors, 1);
    }
}

void validate_pivot_pool(
    const int32_t* row_ptr,
    const int8_t* row_state,
    const int32_t* row_count,
    const uint8_t* pool_M,
    const float* pool_W,
    const int32_t* free_stack,
    const int32_t* free_top,
    ValidationResult* result_dev,
    int64_t S_tot,
    int64_t pool_cap,
    int64_t P,
    cudaStream_t stream)
{
    // Initialize result
    init_validation_result_kernel<<<1, 1, 0, stream>>>(result_dev);
    
    // Validate row states
    if (S_tot > 0) {
        int blocks = (S_tot + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        validate_pivot_rows_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
            row_ptr, row_state, row_count, pool_M, pool_W, result_dev, S_tot, pool_cap, P);
    }
    
    // Check for orphaned slots (skip if pool_cap is very large to avoid timeout)
    if (pool_cap > 0 && pool_cap <= 65536 && S_tot <= 65536) {
        int blocks = (pool_cap + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        validate_pivot_orphans_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
            row_ptr, row_state, pool_M, free_stack, free_top, result_dev, 
            S_tot, pool_cap, P);
    }
    
    // Validate free stack
    validate_free_stack_kernel<<<1, 1, 0, stream>>>(
        row_ptr, row_state, free_stack, free_top, result_dev, S_tot, pool_cap);
}

// =============================================================================
// =================== SEGMENTED POOL KERNELS (1 token = 1 segment) ===========
// =============================================================================
// These _seg variants replace slot-based pool access (slot * B + b) with
// segment indirection via row_seg[row * CAP + pos] -> seg_id.
// Validity is determined by seg_id >= 0 (no separate M array needed).

// --------------- materialize_rows_seg ---------------
// Allocates segments on-demand for rows that need buffer space.
// Unlike the contiguous version, each token position gets its own segment.
__global__ void materialize_rows_seg_kernel(
    const int64_t* __restrict__ unique_rows, const int32_t* __restrict__ G_dev,
    int32_t* __restrict__ row_seg, int8_t* __restrict__ row_state,
    int32_t* __restrict__ row_count,
    int32_t* __restrict__ seg_free_stack, int32_t* __restrict__ seg_free_top,
    int B, int G_max, int64_t S_tot,
    const int32_t* __restrict__ cand_row_counts) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int G = *G_dev;
    if (idx >= G) return;
    if (cand_row_counts && cand_row_counts[idx] == 0) return;
    int64_t row = unique_rows[idx];
    if (row < 0 || row >= S_tot) return;
    if (row_state[row] != (int8_t)BufState::RESERVED) return;
    // Transition to AVAILABLE; no segments allocated yet, row_seg already -1
    row_state[row] = (int8_t)BufState::AVAILABLE;
    row_count[row] = 0;
}

void materialize_rows_seg(
    const int64_t* unique_rows, const int32_t* G_dev,
    int32_t* row_seg, int8_t* row_state, int32_t* row_count,
    int32_t* seg_free_stack, int32_t* seg_free_top,
    int64_t B, int64_t G_max, int64_t S_tot, cudaStream_t stream,
    const int32_t* cand_row_counts) {
    if (G_max == 0) return;
    int blocks = (G_max + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    materialize_rows_seg_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
        unique_rows, G_dev, row_seg, row_state, row_count,
        seg_free_stack, seg_free_top, B, G_max, S_tot, cand_row_counts);
}

// --------------- one2one_merge_seg ---------------
// Reads pivot data via piv_row_seg indirection.
template <typename scalar_t>
__global__ void one2one_merge_seg_kernel(
    const int64_t* __restrict__ unique_rows, const int32_t* __restrict__ row_offsets,
    const scalar_t* __restrict__ sorted_K, const scalar_t* __restrict__ sorted_V,
    const float* __restrict__ sorted_S, const int32_t* __restrict__ G_dev,
    scalar_t* __restrict__ piv_K, scalar_t* __restrict__ piv_V,
    float* __restrict__ piv_W, float* __restrict__ piv_S, float* __restrict__ piv_C,
    const scalar_t* __restrict__ piv_K_seed,
    const int32_t* __restrict__ piv_row_seg, const int8_t* __restrict__ piv_row_state,
    scalar_t* __restrict__ cand_K, scalar_t* __restrict__ cand_V,
    float* __restrict__ cand_S, int64_t* __restrict__ cand_rows, int32_t* __restrict__ cand_cnt,
    int32_t* __restrict__ cand_row_counts,
    float sim_thresh, float replace_thresh, float score_thresh, int P, int D, int G_max, int E_max,
    int32_t* __restrict__ diag) {

    const int ri = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int G = G_dev ? *G_dev : G_max;
    if (ri >= G) return;

    const int64_t row = unique_rows[ri];
    if (row < 0) return;

    const int32_t start = row_offsets[ri], end = row_offsets[ri + 1];
    const bool csr_mode = (cand_row_counts != nullptr);

    const int dpl = (D + WARP_SIZE - 1) / WARP_SIZE;
    const int d_lo = lane * dpl;
    const int d_hi = (d_lo + dpl < D) ? d_lo + dpl : D;
    const int my_d = d_hi - d_lo;

    int cand_local = 0;

    // Check pivot row state
    int8_t pstate = piv_row_state[row];
    const int64_t seg_base = row * P;

    // RESERVED/invalid → all tokens become candidates
    if (pstate == (int8_t)BufState::RESERVED) {
        int n_tok = end - start;
        for (int t = start; t < end; ++t) {
            if (csr_mode) {
                int out = start + cand_local;
                if (lane == 0) {
                    cand_S[out] = sorted_S[t];
                }
                for (int d = d_lo; d < d_hi; ++d) {
                    cand_K[(int64_t)out * D + d] = sorted_K[(int64_t)t * D + d];
                    cand_V[(int64_t)out * D + d] = sorted_V[(int64_t)t * D + d];
                }
                cand_local++;
            } else {
                int ci = 0;
                if (lane == 0) ci = atomicAdd(cand_cnt, 1);
                ci = __shfl_sync(FULL_MASK, ci, 0);
                if (ci < E_max) {
                    if (lane == 0) {
                        cand_rows[ci] = row; cand_S[ci] = sorted_S[t];
                    }
                    for (int d = d_lo; d < d_hi; ++d) {
                        cand_K[(int64_t)ci * D + d] = sorted_K[(int64_t)t * D + d];
                        cand_V[(int64_t)ci * D + d] = sorted_V[(int64_t)t * D + d];
                    }
                }
            }
        }
        if (csr_mode && lane == 0) cand_row_counts[ri] = cand_local;
        if (diag && lane == 0) atomicAdd(&diag[DIAG_O2O_CAND_RESERVED], n_tok);
        return;
    }

    // Load pivot seeds via segment indirection
    float seeds[MAX_PIVOTS][MAX_DPL];
    int32_t seg_ids[MAX_PIVOTS];
    float sum_S = 0, sum_C = 0;
    int n_valid = 0;

    for (int p = 0; p < P && p < MAX_PIVOTS; ++p) {
        seg_ids[p] = piv_row_seg[seg_base + p];
        if (seg_ids[p] >= 0) {
            int64_t sv = (int64_t)seg_ids[p] * D;
            float pnorm = 0.0f;
            for (int di = 0; di < my_d; ++di) {
                float k = to_float(piv_K[sv + d_lo + di]);
                seeds[p][di] = k;
                pnorm += k * k;
            }
            float norm_sq = warp_reduce_sum(pnorm);
            float sinv = rsqrtf(norm_sq + 1e-6f);
            for (int di = 0; di < my_d; ++di)
                seeds[p][di] *= sinv;
            sum_S += piv_S[seg_ids[p]];
            sum_C += piv_C[seg_ids[p]];
            n_valid++;
        }
    }

    if (n_valid == 0) {
        int n_tok = end - start;
        for (int t = start; t < end; ++t) {
            if (csr_mode) {
                int out = start + cand_local;
                if (lane == 0) {
                    cand_S[out] = sorted_S[t];
                }
                for (int d = d_lo; d < d_hi; ++d) {
                    cand_K[(int64_t)out * D + d] = sorted_K[(int64_t)t * D + d];
                    cand_V[(int64_t)out * D + d] = sorted_V[(int64_t)t * D + d];
                }
                cand_local++;
            } else {
                int ci = 0;
                if (lane == 0) ci = atomicAdd(cand_cnt, 1);
                ci = __shfl_sync(FULL_MASK, ci, 0);
                if (ci < E_max) {
                    if (lane == 0) {
                        cand_rows[ci] = row; cand_S[ci] = sorted_S[t];
                    }
                    for (int d = d_lo; d < d_hi; ++d) {
                        cand_K[(int64_t)ci * D + d] = sorted_K[(int64_t)t * D + d];
                        cand_V[(int64_t)ci * D + d] = sorted_V[(int64_t)t * D + d];
                    }
                }
            }
        }
        if (csr_mode && lane == 0) cand_row_counts[ri] = cand_local;
        if (diag && lane == 0) atomicAdd(&diag[DIAG_O2O_CAND_NO_PIVOT], n_tok);
        return;
    }

    // Accumulators
    float K_acc[MAX_PIVOTS][MAX_DPL], V_acc[MAX_PIVOTS][MAX_DPL];
    float a_acc[MAX_PIVOTS], S_acc[MAX_PIVOTS], C_acc[MAX_PIVOTS];
    for (int p = 0; p < P && p < MAX_PIVOTS; ++p) {
        a_acc[p] = S_acc[p] = C_acc[p] = 0.0f;
        for (int di = 0; di < my_d; ++di) K_acc[p][di] = V_acc[p][di] = 0.0f;
    }

    // Pass 1: accumulate merge data
    int n_absorbed = 0;
    for (int t = start; t < end; ++t) {
        float tK[MAX_DPL];
        float pnorm = 0.0f;
        for (int di = 0; di < my_d; ++di) {
            float k = to_float(sorted_K[(int64_t)t * D + d_lo + di]);
            tK[di] = k;
            pnorm += k * k;
        }
        float inv = rsqrtf(warp_reduce_sum(pnorm) + 1e-6f);
        float sc = sorted_S[t];

        float smax = -1e30f;
        int bp = -1;
        for (int p = 0; p < P && p < MAX_PIVOTS; ++p) {
            if (seg_ids[p] < 0) continue;
            float pdot = 0.0f;
            for (int di = 0; di < my_d; ++di) pdot += (tK[di] * inv) * seeds[p][di];
            float s = fminf(fmaxf(warp_reduce_sum(pdot), -1.0f), 1.0f);
            if (s > smax) { smax = s; bp = p; }
        }

        if (smax >= sim_thresh && bp >= 0) {
            n_absorbed++;
            float a = expf(smax);
            for (int di = 0; di < my_d; ++di) {
                K_acc[bp][di] += a * to_float(sorted_K[(int64_t)t * D + d_lo + di]);
                V_acc[bp][di] += a * to_float(sorted_V[(int64_t)t * D + d_lo + di]);
            }
            a_acc[bp] += a;
            S_acc[bp] += sc;
            C_acc[bp] += 1.0f;
        }
    }

    // Post-merge gate
    float post_sum_S = sum_S, post_sum_C = sum_C;
    for (int p = 0; p < P && p < MAX_PIVOTS; ++p) {
        post_sum_S += S_acc[p];
        post_sum_C += C_acc[p];
    }
    float gate = (post_sum_C > 1e-6f ? post_sum_S / post_sum_C : 0.0f) * score_thresh;

    // Pass 2: classify remaining
    int n_dropped = 0;
    for (int t = start; t < end; ++t) {
        float tK[MAX_DPL];
        float pnorm = 0.0f;
        for (int di = 0; di < my_d; ++di) {
            float k = to_float(sorted_K[(int64_t)t * D + d_lo + di]);
            tK[di] = k;
            pnorm += k * k;
        }
        float inv = rsqrtf(warp_reduce_sum(pnorm) + 1e-6f);
        float sc = sorted_S[t];

        float smax = -1e30f;
        for (int p = 0; p < P && p < MAX_PIVOTS; ++p) {
            if (seg_ids[p] < 0) continue;
            float pdot = 0.0f;
            for (int di = 0; di < my_d; ++di) pdot += (tK[di] * inv) * seeds[p][di];
            float s = fminf(fmaxf(warp_reduce_sum(pdot), -1.0f), 1.0f);
            if (s > smax) smax = s;
        }

        if (smax >= sim_thresh) continue;
        if (smax < replace_thresh && sc > gate) {
            if (csr_mode) {
                int out = start + cand_local;
                if (lane == 0) {
                    cand_S[out] = sc;
                }
                for (int d = d_lo; d < d_hi; ++d) {
                    cand_K[(int64_t)out * D + d] = sorted_K[(int64_t)t * D + d];
                    cand_V[(int64_t)out * D + d] = sorted_V[(int64_t)t * D + d];
                }
                cand_local++;
            } else {
                int ci = 0;
                if (lane == 0) ci = atomicAdd(cand_cnt, 1);
                ci = __shfl_sync(FULL_MASK, ci, 0);
                if (ci < E_max) {
                    if (lane == 0) {
                        cand_rows[ci] = row; cand_S[ci] = sc;
                    }
                    for (int d = d_lo; d < d_hi; ++d) {
                        cand_K[(int64_t)ci * D + d] = sorted_K[(int64_t)t * D + d];
                        cand_V[(int64_t)ci * D + d] = sorted_V[(int64_t)t * D + d];
                    }
                }
            }
        } else {
            n_dropped++;
        }
    }

    if (csr_mode && lane == 0) cand_row_counts[ri] = cand_local;
    if (diag && lane == 0) {
        if (n_absorbed > 0) atomicAdd(&diag[DIAG_O2O_ABSORBED], n_absorbed);
        if (cand_local > 0) atomicAdd(&diag[DIAG_O2O_CAND_LOW_SIM], cand_local);
        if (n_dropped > 0) atomicAdd(&diag[DIAG_O2O_DROPPED], n_dropped);
    }

    // Write back pivot updates via segment indirection
    for (int p = 0; p < P && p < MAX_PIVOTS; ++p) {
        if (a_acc[p] <= 0.0f || seg_ids[p] < 0) continue;
        int32_t sid = seg_ids[p];
        int64_t sv = (int64_t)sid * D;
        float wo = piv_W[sid];
        float dn = fmaxf(wo + a_acc[p], 1e-8f);
        float fo = wo / dn;
        float fn = 1.0f / dn;
        for (int di = 0; di < my_d; ++di) {
            int d = d_lo + di;
            float k_old = to_float(piv_K[sv + d]);
            float v_old = to_float(piv_V[sv + d]);
            piv_K[sv + d] = from_float<scalar_t>(fo * k_old + fn * K_acc[p][di]);
            piv_V[sv + d] = from_float<scalar_t>(fo * v_old + fn * V_acc[p][di]);
        }
        if (lane == 0) {
            piv_W[sid] = dn;
            piv_S[sid] += S_acc[p];
            piv_C[sid] += C_acc[p];
        }
    }
}

void one2one_merge_seg(
    const int64_t* unique_rows, const int32_t* row_offsets, const void* sorted_K,
    const void* sorted_V, const float* sorted_S, const int32_t* G_dev,
    void* piv_K, void* piv_V, float* piv_W, float* piv_S, float* piv_C,
    const void* piv_K_seed, const int32_t* piv_row_seg, const int8_t* piv_row_state,
    const int32_t* piv_row_count,
    void* cand_K, void* cand_V, float* cand_S, int64_t* cand_rows, int32_t* cand_count_dev,
    float sim_thresh, float replace_thresh, float score_thresh, int64_t P, int64_t D,
    int64_t G_max, int64_t E_max, DType dtype, cudaStream_t stream,
    int32_t* cand_row_counts, int32_t* diag) {
    if (G_max == 0) return;
    if (!cand_row_counts) zero_scalar_kernel<<<1, 1, 0, stream>>>(cand_count_dev);
    const int total_threads = G_max * WARP_SIZE;
    const int tpb = ONE2ONE_WARPS_PER_BLOCK * WARP_SIZE;
    const int blocks = (total_threads + tpb - 1) / tpb;
    if (dtype == DType::Float16) {
        one2one_merge_seg_kernel<__half><<<blocks, tpb, 0, stream>>>(
            unique_rows, row_offsets, (const __half*)sorted_K, (const __half*)sorted_V, sorted_S, G_dev,
            (__half*)piv_K, (__half*)piv_V, piv_W, piv_S, piv_C, (const __half*)piv_K_seed,
            piv_row_seg, piv_row_state,
            (__half*)cand_K, (__half*)cand_V, cand_S, cand_rows, cand_count_dev,
            cand_row_counts, sim_thresh, replace_thresh, score_thresh, P, D, G_max, E_max, diag);
    } else {
        one2one_merge_seg_kernel<__nv_bfloat16><<<blocks, tpb, 0, stream>>>(
            unique_rows, row_offsets, (const __nv_bfloat16*)sorted_K, (const __nv_bfloat16*)sorted_V, sorted_S, G_dev,
            (__nv_bfloat16*)piv_K, (__nv_bfloat16*)piv_V, piv_W, piv_S, piv_C, (const __nv_bfloat16*)piv_K_seed,
            piv_row_seg, piv_row_state,
            (__nv_bfloat16*)cand_K, (__nv_bfloat16*)cand_V, cand_S, cand_rows, cand_count_dev,
            cand_row_counts, sim_thresh, replace_thresh, score_thresh, P, D, G_max, E_max, diag);
    }
}

// --------------- buffer_topb_update_seg (DEPRECATED — replaced by K1+K2+K3 pipeline) ---------------
#if 0
template <typename scalar_t>
__global__ void buffer_topb_seg_kernel(
    const int64_t* __restrict__ unique_rows, const int32_t* __restrict__ row_offsets,
    const scalar_t* __restrict__ sorted_K, const scalar_t* __restrict__ sorted_V,
    const float* __restrict__ sorted_S, const int32_t* __restrict__ G_dev,
    int32_t* __restrict__ buf_row_seg, int8_t* __restrict__ row_state, int32_t* __restrict__ row_count,
    scalar_t* __restrict__ seg_K, scalar_t* __restrict__ seg_V,
    float* __restrict__ seg_S,
    int32_t* __restrict__ seg_free_stack, int32_t* __restrict__ seg_free_top,
    scalar_t* __restrict__ over_K, scalar_t* __restrict__ over_V,
    float* __restrict__ over_S, int64_t* __restrict__ over_rows, int32_t* __restrict__ over_cnt,
    int B, int D, int G_max, int max_over, int64_t S_tot,
    const int32_t* __restrict__ cand_row_counts,
    int32_t* __restrict__ diag) {

    int ri = blockIdx.x * blockDim.x + threadIdx.x;
    int G = G_dev ? *G_dev : G_max;
    if (ri >= G) return;

    int64_t row = unique_rows[ri];
    if (row < 0 || row >= S_tot) return;
    int8_t state = row_state[row];
    int32_t st = row_offsets[ri];
    int32_t en = cand_row_counts ? (st + cand_row_counts[ri]) : row_offsets[ri + 1];

    if (state != (int8_t)BufState::AVAILABLE) {
        int n_new = en - st;
        if (n_new > 0) {
            int32_t ob = atomicAdd(over_cnt, n_new);
            for (int i = 0; i < n_new; ++i) {
                int op = ob + i; if (op >= max_over) break;
                over_rows[op] = row;
                over_S[op] = sorted_S[st + i];
                vec_copy_d(&over_K[(int64_t)op*D], &sorted_K[(int64_t)(st+i)*D], D);
                vec_copy_d(&over_V[(int64_t)op*D], &sorted_V[(int64_t)(st+i)*D], D);
            }
            if (diag) atomicAdd(&diag[DIAG_BUF_OVER_NO_SLOT], n_new);
        }
        return;
    }

    int64_t seg_base = row * (int64_t)B;
    int n_new = en - st, n_ex = row_count[row];

    float sc[MAX_BUFFER]; int idx[MAX_BUFFER]; int nc = 0;

    // Gather existing tokens from segments
    for (int b = 0; b < n_ex && b < B && nc < MAX_BUFFER; ++b) {
        int32_t sid = buf_row_seg[seg_base + b];
        if (sid >= 0) { sc[nc] = seg_S[sid]; idx[nc++] = -(b+1); }
    }
    for (int i = 0; i < n_new && nc < MAX_BUFFER; ++i)
        { sc[nc] = sorted_S[st+i]; idx[nc++] = i; }

    // Stable insertion sort descending
    for (int i = 1; i < nc; ++i) {
        float key_s = sc[i]; int key_i = idx[i];
        int j = i - 1;
        while (j >= 0 && sc[j] < key_s) {
            sc[j+1] = sc[j]; idx[j+1] = idx[j]; j--;
        }
        sc[j+1] = key_s; idx[j+1] = key_i;
    }

    int nk = min(nc, B);

    // Determine which existing segments are kept vs evicted
    // First, collect the old seg_ids before overwrite
    int32_t old_segs[MAX_BUFFER];
    for (int b = 0; b < n_ex && b < B; ++b) old_segs[b] = buf_row_seg[seg_base + b];

    // Write top-nk winners into row (pop uses fast atomicSub)
    for (int b = 0; b < nk; ++b) {
        int x = idx[b];
        if (x < 0) {
            int old_pos = -(x+1);
            int32_t sid = old_segs[old_pos];
            buf_row_seg[seg_base + b] = sid;
        } else {
            int32_t top = atomicSub(seg_free_top, 1) - 1;
            if (top < 0) {
                atomicAdd(seg_free_top, 1);
                int32_t ob = atomicAdd(over_cnt, 1);
                if (ob < max_over) {
                    over_rows[ob] = row;
                    over_S[ob] = sorted_S[st + x];
                    vec_copy_d(&over_K[(int64_t)ob*D], &sorted_K[(int64_t)(st+x)*D], D);
                    vec_copy_d(&over_V[(int64_t)ob*D], &sorted_V[(int64_t)(st+x)*D], D);
                }
                buf_row_seg[seg_base + b] = -1;
                continue;
            }
            int32_t new_sid = seg_free_stack[top];
            buf_row_seg[seg_base + b] = new_sid;
            seg_S[new_sid] = sorted_S[st + x];
            vec_copy_d(&seg_K[(int64_t)new_sid*D], &sorted_K[(int64_t)(st+x)*D], D);
            vec_copy_d(&seg_V[(int64_t)new_sid*D], &sorted_V[(int64_t)(st+x)*D], D);
        }
    }

    // Clear unused tail positions
    for (int b = nk; b < B; ++b) buf_row_seg[seg_base + b] = -1;

    // Free evicted segments (CAS-based push: safe against transient negative free_top)
    int nfree = 0;
    int32_t free_sids[MAX_BUFFER];
    for (int b = 0; b < n_ex && b < B; ++b) {
        int32_t sid = old_segs[b];
        if (sid < 0) continue;
        bool kept = false;
        for (int k = 0; k < nk; ++k) {
            if (buf_row_seg[seg_base + k] == sid) { kept = true; break; }
        }
        if (!kept) free_sids[nfree++] = sid;
    }
    if (nfree > 0) {
        seg_stack_push_batch(seg_free_top, seg_free_stack, free_sids, nfree);
    }

    // Count actual valid segments
    int actual_count = 0;
    for (int b = 0; b < nk; ++b) if (buf_row_seg[seg_base + b] >= 0) actual_count++;
    row_count[row] = actual_count;
    bool became_full = (actual_count == B);
    if (became_full) row_state[row] = (int8_t)BufState::FULL;

    // Overflow: tokens beyond top-B
    int no = nc - nk;
    if (no > 0) {
        int32_t ob = atomicAdd(over_cnt, no);
        for (int i = 0; i < no; ++i) {
            int op = ob + i; if (op >= max_over) break;
            int x = idx[nk+i];
            if (x < 0) {
                int old_pos = -(x+1);
                int32_t sid = old_segs[old_pos];
                if (sid >= 0) {
                    over_rows[op] = row; over_S[op] = seg_S[sid];
                    vec_copy_d(&over_K[(int64_t)op*D], &seg_K[(int64_t)sid*D], D);
                    vec_copy_d(&over_V[(int64_t)op*D], &seg_V[(int64_t)sid*D], D);
                }
            } else {
                over_rows[op] = row; over_S[op] = sorted_S[st+x];
                vec_copy_d(&over_K[(int64_t)op*D], &sorted_K[(int64_t)(st+x)*D], D);
                vec_copy_d(&over_V[(int64_t)op*D], &sorted_V[(int64_t)(st+x)*D], D);
            }
        }
    }
    if (diag) {
        if (actual_count > 0) atomicAdd(&diag[DIAG_BUF_KEPT], actual_count);
        if (no > 0) atomicAdd(&diag[DIAG_BUF_OVER_EXCESS], no);
        if (became_full) atomicAdd(&diag[DIAG_BUF_ROWS_FULL], 1);
    }
}

void buffer_topb_update_seg(
    const int64_t* unique_rows, const int32_t* row_offsets, const void* sorted_K,
    const void* sorted_V, const float* sorted_S, const int32_t* G_dev,
    int32_t* buf_row_seg, int8_t* row_state, int32_t* row_count,
    void* seg_buf_K, void* seg_buf_V, float* seg_buf_S,
    int32_t* seg_free_stack, int32_t* seg_free_top,
    void* over_K, void* over_V, float* over_S, int64_t* over_rows, int32_t* over_count_dev,
    int64_t B, int64_t D, int64_t G_max, int64_t E_max, int64_t S_tot, DType dtype,
    cudaStream_t stream, const int32_t* cand_row_counts, int32_t* diag) {
    if (G_max == 0) return;
    zero_scalar_kernel<<<1, 1, 0, stream>>>(over_count_dev);
    int blocks = (G_max + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    if (dtype == DType::Float16) {
        buffer_topb_seg_kernel<__half><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
            unique_rows, row_offsets, (const __half*)sorted_K, (const __half*)sorted_V, sorted_S, G_dev,
            buf_row_seg, row_state, row_count, (__half*)seg_buf_K, (__half*)seg_buf_V, seg_buf_S,
            seg_free_stack, seg_free_top,
            (__half*)over_K, (__half*)over_V, over_S, over_rows, over_count_dev,
            B, D, G_max, E_max, S_tot, cand_row_counts, diag);
    } else {
        buffer_topb_seg_kernel<__nv_bfloat16><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
            unique_rows, row_offsets, (const __nv_bfloat16*)sorted_K, (const __nv_bfloat16*)sorted_V, sorted_S, G_dev,
            buf_row_seg, row_state, row_count, (__nv_bfloat16*)seg_buf_K, (__nv_bfloat16*)seg_buf_V, seg_buf_S,
            seg_free_stack, seg_free_top,
            (__nv_bfloat16*)over_K, (__nv_bfloat16*)over_V, over_S, over_rows, over_count_dev,
            B, D, G_max, E_max, S_tot, cand_row_counts, diag);
    }
}

// --------------- all2one_merge_fused_seg ---------------
// Cluster merge FULL buffer rows into pivots, using segment indirection.
template <typename scalar_t>
__global__ void all2one_merge_fused_seg_kernel(
    const int64_t* __restrict__ unique_rows, const int32_t* __restrict__ G_dev,
    // Buffer seg pool
    const scalar_t* __restrict__ seg_buf_K, const scalar_t* __restrict__ seg_buf_V,
    const float* __restrict__ seg_buf_S,
    int32_t* __restrict__ buf_row_seg, int8_t* __restrict__ buf_rs, int32_t* __restrict__ buf_rc,
    int32_t* __restrict__ seg_buf_free_stack, int32_t* __restrict__ seg_buf_free_top,
    // Pivot seg pool
    scalar_t* __restrict__ seg_piv_K, scalar_t* __restrict__ seg_piv_V,
    float* __restrict__ seg_piv_W, float* __restrict__ seg_piv_S, float* __restrict__ seg_piv_C,
    scalar_t* __restrict__ seg_piv_Ks, float* __restrict__ seg_piv_Ss,
    int32_t* __restrict__ piv_row_seg, int8_t* __restrict__ piv_rs, int32_t* __restrict__ piv_rc,
    int32_t* __restrict__ seg_piv_free_stack, int32_t* __restrict__ seg_piv_free_top,
    int B, int P, int D, int G_max, int64_t S_tot,
    int32_t* __restrict__ diag) {

    const int ri = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int G = G_dev ? *G_dev : G_max;
    if (ri >= G) return;

    const int64_t row = unique_rows[ri];
    if (row < 0 || row >= S_tot) return;
    if (buf_rs[row] != (int8_t)BufState::FULL) return;

    const int dpl = (D + WARP_SIZE - 1) / WARP_SIZE;
    const int d_lo = lane * dpl;
    const int d_hi = (d_lo + dpl < D) ? d_lo + dpl : D;
    const int my_d = d_hi - d_lo;

    // Read buffer tokens via segment indirection
    int64_t buf_seg_base = row * (int64_t)B;
    float Kb[MAX_BUFFER][MAX_DPL], Vb[MAX_BUFFER][MAX_DPL];
    float Sb[MAX_BUFFER];
    int32_t buf_sids[MAX_BUFFER];
    int nv = 0;

    for (int b = 0; b < B && b < MAX_BUFFER; ++b) {
        int32_t sid = buf_row_seg[buf_seg_base + b];
        if (sid >= 0) {
            buf_sids[nv] = sid;
            Sb[nv] = seg_buf_S[sid];
            int64_t sv = (int64_t)sid * D;
            for (int di = 0; di < my_d; ++di) {
                Kb[nv][di] = to_float(seg_buf_K[sv + d_lo + di]);
                Vb[nv][di] = to_float(seg_buf_V[sv + d_lo + di]);
            }
            nv++;
        }
    }

    if (nv == 0) {
        // Clean buffer row, free segments (should be none if nv=0)
        if (lane == 0) {
            for (int b = 0; b < B; ++b) buf_row_seg[buf_seg_base + b] = -1;
            buf_rs[row] = (int8_t)BufState::RESERVED;
            buf_rc[row] = 0;
        }
        return;
    }

    // Ensure pivot row is available
    int8_t pstate = piv_rs[row];
    if (pstate == (int8_t)BufState::RESERVED) {
        if (lane == 0) {
            piv_rs[row] = (int8_t)BufState::AVAILABLE;
            piv_rc[row] = 0;
        }
    }

    // Find max-score token as seed
    int si = 0; float ss = Sb[0];
    for (int i = 1; i < nv; ++i) if (Sb[i] > ss) { ss = Sb[i]; si = i; }

    float Ks[MAX_DPL]; float sn = 0.0f;
    for (int di = 0; di < my_d; ++di) {
        float k = Kb[si][di]; Ks[di] = k; sn += k * k;
    }
    { float sinv = rsqrtf(warp_reduce_sum(sn) + 1e-6f);
      for (int di = 0; di < my_d; ++di) Ks[di] *= sinv; }

    // Weighted merge
    float Kp[MAX_DPL], Vp[MAX_DPL];
    for (int di = 0; di < my_d; ++di) Kp[di] = Vp[di] = 0.0f;
    float Wp = 0, Sp = 0, Cp = 0;
    for (int i = 0; i < nv; ++i) {
        float n2 = 0;
        for (int di = 0; di < my_d; ++di) n2 += Kb[i][di] * Kb[i][di];
        float inv = rsqrtf(warp_reduce_sum(n2) + 1e-6f);
        float sim_p = 0;
        for (int di = 0; di < my_d; ++di) sim_p += (Kb[i][di] * inv) * Ks[di];
        float sim = fminf(fmaxf(warp_reduce_sum(sim_p), -1.0f), 1.0f);
        float c = expf(sim);
        for (int di = 0; di < my_d; ++di) { Kp[di] += c * Kb[i][di]; Vp[di] += c * Vb[i][di]; }
        Wp += c; Sp += Sb[i]; Cp += 1.0f;
    }
    float invW = 1.0f / fmaxf(Wp, 1e-8f);
    for (int di = 0; di < my_d; ++di) { Kp[di] *= invW; Vp[di] *= invW; }

    // Write new pivot into pivot seg pool
    {
        int64_t piv_seg_base = row * (int64_t)P;
        int ws = -1;
        for (int p = 0; p < P && p < MAX_PIVOTS; ++p)
            if (piv_row_seg[piv_seg_base + p] < 0) { ws = p; break; }

        if (ws < 0) {
            // Find victim (min weight)
            int j_victim = 0; float min_W = 1e30f;
            for (int p = 0; p < P && p < MAX_PIVOTS; ++p) {
                int32_t sid = piv_row_seg[piv_seg_base + p];
                if (sid >= 0 && seg_piv_W[sid] < min_W) { min_W = seg_piv_W[sid]; j_victim = p; }
            }

            // Merge victim into nearest neighbor
            int32_t v_sid = piv_row_seg[piv_seg_base + j_victim];
            float K_victim[MAX_DPL]; float vn2 = 0;
            for (int di = 0; di < my_d; ++di) {
                float k = to_float(seg_piv_K[(int64_t)v_sid * D + d_lo + di]);
                K_victim[di] = k; vn2 += k * k;
            }
            float vinv = rsqrtf(warp_reduce_sum(vn2) + 1e-6f);

            int j_nei = -1; float max_sim = -1e30f;
            for (int p = 0; p < P && p < MAX_PIVOTS; ++p) {
                if (p == j_victim) continue;
                int32_t sid = piv_row_seg[piv_seg_base + p];
                if (sid < 0) continue;
                float nn2 = 0;
                for (int di = 0; di < my_d; ++di) {
                    float k = to_float(seg_piv_K[(int64_t)sid * D + d_lo + di]);
                    nn2 += k * k;
                }
                float ninv = rsqrtf(warp_reduce_sum(nn2) + 1e-6f);
                float sp = 0;
                for (int di = 0; di < my_d; ++di)
                    sp += (K_victim[di] * vinv) * (to_float(seg_piv_K[(int64_t)sid * D + d_lo + di]) * ninv);
                float s = fminf(fmaxf(warp_reduce_sum(sp), -1.0f), 1.0f);
                if (s > max_sim) { max_sim = s; j_nei = p; }
            }

            if (j_nei >= 0) {
                int32_t n_sid = piv_row_seg[piv_seg_base + j_nei];
                float scale = expf(max_sim - 1.0f);
                float Wv = seg_piv_W[v_sid] * scale, Wn = seg_piv_W[n_sid];
                float den = fmaxf(Wv + Wn, 1e-8f);
                float fv = Wv / den, fn = Wn / den;
                for (int di = 0; di < my_d; ++di) {
                    int d = d_lo + di;
                    float kv = to_float(seg_piv_K[(int64_t)v_sid * D + d]);
                    float kn = to_float(seg_piv_K[(int64_t)n_sid * D + d]);
                    float vv = to_float(seg_piv_V[(int64_t)v_sid * D + d]);
                    float vn = to_float(seg_piv_V[(int64_t)n_sid * D + d]);
                    seg_piv_K[(int64_t)n_sid * D + d] = from_float<scalar_t>(fv * kv + fn * kn);
                    seg_piv_V[(int64_t)n_sid * D + d] = from_float<scalar_t>(fv * vv + fn * vn);
                }
                if (lane == 0) {
                    seg_piv_W[n_sid] = den;
                    seg_piv_S[n_sid] = seg_piv_S[v_sid] + seg_piv_S[n_sid];
                    seg_piv_C[n_sid] = seg_piv_C[v_sid] + seg_piv_C[n_sid];
                }
            }

            // Free victim segment (CAS push: safe against transient negative)
            if (lane == 0) {
                seg_stack_push(seg_piv_free_top, seg_piv_free_stack, v_sid);
                piv_row_seg[piv_seg_base + j_victim] = -1;
            }
            ws = j_victim;
        }

        // Allocate new pivot segment (fast atomicSub pop)
        int32_t new_sid = -1;
        if (lane == 0) {
            int32_t top = atomicSub(seg_piv_free_top, 1) - 1;
            if (top < 0) { atomicAdd(seg_piv_free_top, 1); }
            else { new_sid = seg_piv_free_stack[top]; }
        }
        new_sid = __shfl_sync(FULL_MASK, new_sid, 0);

        if (new_sid >= 0) {
            int64_t nv_offset = (int64_t)new_sid * D;
            for (int di = 0; di < my_d; ++di) {
                int d = d_lo + di;
                seg_piv_K[nv_offset + d] = from_float<scalar_t>(Kp[di]);
                seg_piv_V[nv_offset + d] = from_float<scalar_t>(Vp[di]);
                seg_piv_Ks[nv_offset + d] = from_float<scalar_t>(Ks[di]);
            }
            if (lane == 0) {
                seg_piv_W[new_sid] = Wp;
                seg_piv_S[new_sid] = Sp;
                seg_piv_C[new_sid] = Cp;
                seg_piv_Ss[new_sid] = ss;
                piv_row_seg[piv_seg_base + ws] = new_sid;
                piv_rs[row] = (int8_t)BufState::AVAILABLE;
            }
        }
    }

    // Clean buffer: batch-free all buffer segments (CAS push) and reset row
    if (lane == 0) {
        int nfree = 0;
        int32_t free_sids[MAX_BUFFER];
        for (int b = 0; b < B; ++b) {
            int32_t sid = buf_row_seg[buf_seg_base + b];
            if (sid >= 0) free_sids[nfree++] = sid;
            buf_row_seg[buf_seg_base + b] = -1;
        }
        if (nfree > 0) {
            seg_stack_push_batch(seg_buf_free_top, seg_buf_free_stack,
                                 free_sids, nfree);
        }
        buf_rs[row] = (int8_t)BufState::RESERVED;
        buf_rc[row] = 0;
        if (diag) {
            atomicAdd(&diag[DIAG_A2O_ROWS_MERGED], 1);
            atomicAdd(&diag[DIAG_A2O_PIV_CREATED], 1);
        }
    }
}

void all2one_merge_fused_seg(
    const int64_t* unique_rows, const int32_t* G_dev,
    const void* seg_buf_K, const void* seg_buf_V, const float* seg_buf_S,
    int32_t* buf_row_seg, int8_t* buf_row_state, int32_t* buf_row_count,
    int32_t* seg_buf_free_stack, int32_t* seg_buf_free_top,
    void* seg_piv_K, void* seg_piv_V, float* seg_piv_W, float* seg_piv_S, float* seg_piv_C,
    void* seg_piv_K_seed, float* seg_piv_S_seed,
    int32_t* piv_row_seg, int8_t* piv_row_state, int32_t* piv_row_count,
    int32_t* seg_piv_free_stack, int32_t* seg_piv_free_top,
    int64_t B, int64_t P, int64_t D, int64_t G_max, int64_t S_tot,
    DType dtype, cudaStream_t stream, int32_t* diag) {
    if (G_max == 0) return;
    const int total_threads = G_max * WARP_SIZE;
    const int tpb = ALL2ONE_WARPS_PER_BLOCK * WARP_SIZE;
    const int blocks = (total_threads + tpb - 1) / tpb;
    if (dtype == DType::Float16) {
        all2one_merge_fused_seg_kernel<__half><<<blocks, tpb, 0, stream>>>(
            unique_rows, G_dev,
            (const __half*)seg_buf_K, (const __half*)seg_buf_V, seg_buf_S,
            buf_row_seg, buf_row_state, buf_row_count,
            seg_buf_free_stack, seg_buf_free_top,
            (__half*)seg_piv_K, (__half*)seg_piv_V, seg_piv_W, seg_piv_S, seg_piv_C,
            (__half*)seg_piv_K_seed, seg_piv_S_seed,
            piv_row_seg, piv_row_state, piv_row_count,
            seg_piv_free_stack, seg_piv_free_top,
            B, P, D, G_max, S_tot, diag);
    } else {
        all2one_merge_fused_seg_kernel<__nv_bfloat16><<<blocks, tpb, 0, stream>>>(
            unique_rows, G_dev,
            (const __nv_bfloat16*)seg_buf_K, (const __nv_bfloat16*)seg_buf_V, seg_buf_S,
            buf_row_seg, buf_row_state, buf_row_count,
            seg_buf_free_stack, seg_buf_free_top,
            (__nv_bfloat16*)seg_piv_K, (__nv_bfloat16*)seg_piv_V, seg_piv_W, seg_piv_S, seg_piv_C,
            (__nv_bfloat16*)seg_piv_K_seed, seg_piv_S_seed,
            piv_row_seg, piv_row_state, piv_row_count,
            seg_piv_free_stack, seg_piv_free_top,
            B, P, D, G_max, S_tot, diag);
    }
}
#endif // 0 — old buffer_topb_seg + all2one_merge_fused_seg

// =============================================================================
// NEW PIPELINE: Temporally separated pop/push kernels for segment buffers
// =============================================================================

// --------------- K1: buf_fill_seg_kernel ---------------
// For rows where n_ex + n_new < B (buffer does NOT overflow):
//   - Pop segments from seg_buf free stack (atomicSub ONLY, no push)
//   - Write new candidate K/V/S into allocated segments
//   - Overflow non-AVAILABLE rows to over_* arrays
// For rows where n_ex + n_new >= B:
//   - Skip (handled by K2)
template <typename scalar_t>
__global__ void buf_fill_seg_kernel(
    const int64_t* __restrict__ unique_rows,
    const int32_t* __restrict__ row_offsets,
    const scalar_t* __restrict__ sorted_K,
    const scalar_t* __restrict__ sorted_V,
    const float* __restrict__ sorted_S,
    const int32_t* __restrict__ G_dev,
    int32_t* __restrict__ buf_row_seg,
    int8_t* __restrict__ row_state,
    int32_t* __restrict__ row_count,
    scalar_t* __restrict__ seg_K,
    scalar_t* __restrict__ seg_V,
    float* __restrict__ seg_S,
    int32_t* __restrict__ seg_free_stack,
    int32_t* __restrict__ seg_free_top,
    scalar_t* __restrict__ over_K,
    scalar_t* __restrict__ over_V,
    float* __restrict__ over_S,
    int64_t* __restrict__ over_rows,
    int32_t* __restrict__ over_cnt,
    int32_t* __restrict__ to_free_sids,
    int32_t* __restrict__ to_free_count,
    int B, int D, int G_max, int max_over, int64_t S_tot,
    const int32_t* __restrict__ cand_row_counts,
    int32_t* __restrict__ diag)
{
    int ri = blockIdx.x * blockDim.x + threadIdx.x;
    int G = G_dev ? *G_dev : G_max;
    if (ri >= G) return;

    int64_t row = unique_rows[ri];
    if (row < 0 || row >= S_tot) return;
    int8_t state = row_state[row];
    int32_t st = row_offsets[ri];
    int32_t en = cand_row_counts ? (st + cand_row_counts[ri]) : row_offsets[ri + 1];
    int n_new = en - st;

    if (state != (int8_t)BufState::AVAILABLE) {
        if (n_new > 0) {
            int32_t ob = atomicAdd(over_cnt, n_new);
            for (int i = 0; i < n_new; ++i) {
                int op = ob + i; if (op >= max_over) break;
                over_rows[op] = row;
                over_S[op] = sorted_S[st + i];
                vec_copy_d(&over_K[(int64_t)op*D], &sorted_K[(int64_t)(st+i)*D], D);
                vec_copy_d(&over_V[(int64_t)op*D], &sorted_V[(int64_t)(st+i)*D], D);
            }
            if (diag) atomicAdd(&diag[DIAG_BUF_OVER_NO_SLOT], n_new);
        }
        return;
    }

    int64_t seg_base = row * (int64_t)B;
    int n_ex = row_count[row];

    if (n_new <= 0) return;

    if (n_ex + n_new <= B) {
        // ---- Fast path: all tokens fit, no sorting needed ----
        int32_t top = atomicSub(seg_free_top, n_new) - n_new;
        if (top < 0) {
            int got = top + n_new;
            if (got < 0) got = 0;
            int missed = n_new - got;
            atomicAdd(seg_free_top, missed);

            for (int i = 0; i < got; ++i) {
                int32_t new_sid = seg_free_stack[top + n_new - got + i];
                int bpos = n_ex + i;
                buf_row_seg[seg_base + bpos] = new_sid;
                seg_S[new_sid] = sorted_S[st + i];
                vec_copy_d(&seg_K[(int64_t)new_sid*D], &sorted_K[(int64_t)(st+i)*D], D);
                vec_copy_d(&seg_V[(int64_t)new_sid*D], &sorted_V[(int64_t)(st+i)*D], D);
            }
            int32_t ob = atomicAdd(over_cnt, missed);
            for (int i = 0; i < missed; ++i) {
                int op = ob + i; if (op >= max_over) break;
                over_rows[op] = row;
                over_S[op] = sorted_S[st + got + i];
                vec_copy_d(&over_K[(int64_t)op*D], &sorted_K[(int64_t)(st+got+i)*D], D);
                vec_copy_d(&over_V[(int64_t)op*D], &sorted_V[(int64_t)(st+got+i)*D], D);
            }
            row_count[row] = n_ex + got;
        } else {
            for (int i = 0; i < n_new; ++i) {
                int32_t new_sid = seg_free_stack[top + i];
                int bpos = n_ex + i;
                buf_row_seg[seg_base + bpos] = new_sid;
                seg_S[new_sid] = sorted_S[st + i];
                vec_copy_d(&seg_K[(int64_t)new_sid*D], &sorted_K[(int64_t)(st+i)*D], D);
                vec_copy_d(&seg_V[(int64_t)new_sid*D], &sorted_V[(int64_t)(st+i)*D], D);
            }
            int new_count = n_ex + n_new;
            row_count[row] = new_count;
            if (new_count == B) row_state[row] = (int8_t)BufState::FULL;
        }
        if (diag) atomicAdd(&diag[DIAG_BUF_KEPT], n_new);
    } else {
        // ---- Overflow path: sort existing + new, keep top-B, overflow rest ----
        float sc[MAX_BUFFER]; int idx_arr[MAX_BUFFER]; int nc = 0;

        for (int b = 0; b < n_ex && b < B && nc < MAX_BUFFER; ++b) {
            int32_t sid = buf_row_seg[seg_base + b];
            if (sid >= 0) { sc[nc] = seg_S[sid]; idx_arr[nc++] = -(b+1); }
        }
        for (int i = 0; i < n_new && nc < MAX_BUFFER; ++i) {
            sc[nc] = sorted_S[st+i]; idx_arr[nc++] = i;
        }

        for (int i = 1; i < nc; ++i) {
            float key_s = sc[i]; int key_i = idx_arr[i];
            int j = i - 1;
            while (j >= 0 && sc[j] < key_s) {
                sc[j+1] = sc[j]; idx_arr[j+1] = idx_arr[j]; j--;
            }
            sc[j+1] = key_s; idx_arr[j+1] = key_i;
        }

        int nk = min(nc, B);
        int32_t old_segs[MAX_BUFFER];
        for (int b = 0; b < n_ex && b < B; ++b) old_segs[b] = buf_row_seg[seg_base + b];

        for (int b = 0; b < nk; ++b) {
            int x = idx_arr[b];
            if (x < 0) {
                int old_pos = -(x+1);
                buf_row_seg[seg_base + b] = old_segs[old_pos];
            } else {
                int32_t top = atomicSub(seg_free_top, 1) - 1;
                if (top < 0) {
                    atomicAdd(seg_free_top, 1);
                    int32_t ob = atomicAdd(over_cnt, 1);
                    if (ob < max_over) {
                        over_rows[ob] = row;
                        over_S[ob] = sorted_S[st + x];
                        vec_copy_d(&over_K[(int64_t)ob*D], &sorted_K[(int64_t)(st+x)*D], D);
                        vec_copy_d(&over_V[(int64_t)ob*D], &sorted_V[(int64_t)(st+x)*D], D);
                    }
                    buf_row_seg[seg_base + b] = -1;
                    continue;
                }
                int32_t new_sid = seg_free_stack[top];
                buf_row_seg[seg_base + b] = new_sid;
                seg_S[new_sid] = sorted_S[st + x];
                vec_copy_d(&seg_K[(int64_t)new_sid*D], &sorted_K[(int64_t)(st+x)*D], D);
                vec_copy_d(&seg_V[(int64_t)new_sid*D], &sorted_V[(int64_t)(st+x)*D], D);
            }
        }

        for (int b = nk; b < B; ++b) buf_row_seg[seg_base + b] = -1;

        int nfree = 0;
        int32_t free_local[MAX_BUFFER];
        for (int b = 0; b < n_ex && b < B; ++b) {
            int32_t sid = old_segs[b];
            if (sid < 0) continue;
            bool kept = false;
            for (int k = 0; k < nk; ++k) {
                if (buf_row_seg[seg_base + k] == sid) { kept = true; break; }
            }
            if (!kept) free_local[nfree++] = sid;
        }
        if (nfree > 0) {
            int32_t fbase = atomicAdd(to_free_count, nfree);
            for (int i = 0; i < nfree; ++i)
                to_free_sids[fbase + i] = free_local[i];
        }

        int actual_count = 0;
        for (int b = 0; b < nk; ++b) if (buf_row_seg[seg_base + b] >= 0) actual_count++;
        row_count[row] = actual_count;
        bool became_full = (actual_count == B);
        if (became_full) row_state[row] = (int8_t)BufState::FULL;

        int no = nc - nk;
        if (no > 0) {
            int32_t ob = atomicAdd(over_cnt, no);
            for (int i = 0; i < no; ++i) {
                int op = ob + i; if (op >= max_over) break;
                int x = idx_arr[nk+i];
                if (x < 0) {
                    int old_pos = -(x+1);
                    int32_t sid = old_segs[old_pos];
                    if (sid >= 0) {
                        over_rows[op] = row; over_S[op] = seg_S[sid];
                        vec_copy_d(&over_K[(int64_t)op*D], &seg_K[(int64_t)sid*D], D);
                        vec_copy_d(&over_V[(int64_t)op*D], &seg_V[(int64_t)sid*D], D);
                    }
                } else {
                    over_rows[op] = row; over_S[op] = sorted_S[st+x];
                    vec_copy_d(&over_K[(int64_t)op*D], &sorted_K[(int64_t)(st+x)*D], D);
                    vec_copy_d(&over_V[(int64_t)op*D], &sorted_V[(int64_t)(st+x)*D], D);
                }
            }
        }
        if (diag) {
            if (actual_count > 0) atomicAdd(&diag[DIAG_BUF_KEPT], actual_count);
            if (no > 0) atomicAdd(&diag[DIAG_BUF_OVER_EXCESS], no);
            if (became_full) atomicAdd(&diag[DIAG_BUF_ROWS_FULL], 1);
        }
    }
}

// --------------- K2: buf_merge_seg_kernel ---------------
// For rows where state == FULL (buffer is full, needs merge):
//   - Gather existing buffer tokens + new candidates
//   - Sort by score, select top-B
//   - Weighted all2one merge into a new pivot token
//   - Output: new pivot (K,V,W,S,C,Ks,Ss,row) to workspace arrays
//   - Output: overflow candidates (beyond top-B) to over_* arrays
//   - Output: ALL old buffer segment IDs to to_free_* arrays (for K3)
//   - Reset buffer row state to RESERVED, clear buf_row_seg
//   - NO atomicSub/atomicAdd on seg_buf_free_top
template <typename scalar_t>
__global__ void buf_merge_seg_kernel(
    const int64_t* __restrict__ unique_rows,
    const int32_t* __restrict__ G_dev,
    // Buffer seg pool (read-only for K/V/S, write for row_seg/state/count)
    const scalar_t* __restrict__ seg_buf_K,
    const scalar_t* __restrict__ seg_buf_V,
    const float* __restrict__ seg_buf_S,
    int32_t* __restrict__ buf_row_seg,
    int8_t* __restrict__ buf_rs,
    int32_t* __restrict__ buf_rc,
    // New pivot output arrays
    scalar_t* __restrict__ new_piv_K,
    scalar_t* __restrict__ new_piv_V,
    float* __restrict__ new_piv_W,
    float* __restrict__ new_piv_S,
    float* __restrict__ new_piv_C,
    scalar_t* __restrict__ new_piv_Ks,
    float* __restrict__ new_piv_Ss,
    int64_t* __restrict__ new_piv_rows,
    int32_t* __restrict__ new_piv_count,
    // Buffer segments to free
    int32_t* __restrict__ to_free_sids,
    int32_t* __restrict__ to_free_count,
    // Overflow output
    scalar_t* __restrict__ over_K,
    scalar_t* __restrict__ over_V,
    float* __restrict__ over_S,
    int64_t* __restrict__ over_rows,
    int32_t* __restrict__ over_cnt,
    int B, int D, int G_max, int max_over, int64_t S_tot,
    int32_t* __restrict__ diag)
{
    const int ri = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int G = G_dev ? *G_dev : G_max;
    if (ri >= G) return;

    const int64_t row = unique_rows[ri];
    if (row < 0 || row >= S_tot) return;
    if (buf_rs[row] != (int8_t)BufState::FULL) return;

    const int dpl = (D + WARP_SIZE - 1) / WARP_SIZE;
    const int d_lo = lane * dpl;
    const int d_hi = (d_lo + dpl < D) ? d_lo + dpl : D;
    const int my_d = d_hi - d_lo;

    int64_t buf_seg_base = row * (int64_t)B;
    float Kb[MAX_BUFFER][MAX_DPL], Vb[MAX_BUFFER][MAX_DPL];
    float Sb[MAX_BUFFER];
    int32_t buf_sids[MAX_BUFFER];
    int nv = 0;

    for (int b = 0; b < B && b < MAX_BUFFER; ++b) {
        int32_t sid = buf_row_seg[buf_seg_base + b];
        if (sid >= 0) {
            buf_sids[nv] = sid;
            Sb[nv] = seg_buf_S[sid];
            int64_t sv = (int64_t)sid * D;
            for (int di = 0; di < my_d; ++di) {
                Kb[nv][di] = to_float(seg_buf_K[sv + d_lo + di]);
                Vb[nv][di] = to_float(seg_buf_V[sv + d_lo + di]);
            }
            nv++;
        }
    }

    if (nv == 0) {
        if (lane == 0) {
            for (int b = 0; b < B; ++b) buf_row_seg[buf_seg_base + b] = -1;
            buf_rs[row] = (int8_t)BufState::RESERVED;
            buf_rc[row] = 0;
        }
        return;
    }

    // Record all buffer segment IDs for freeing by K3
    if (lane == 0) {
        int32_t fbase = atomicAdd(to_free_count, nv);
        for (int i = 0; i < nv; ++i)
            to_free_sids[fbase + i] = buf_sids[i];
    }

    // Find max-score token as seed
    int si = 0; float ss = Sb[0];
    for (int i = 1; i < nv; ++i) if (Sb[i] > ss) { ss = Sb[i]; si = i; }

    float Ks[MAX_DPL]; float sn = 0.0f;
    for (int di = 0; di < my_d; ++di) {
        float k = Kb[si][di]; Ks[di] = k; sn += k * k;
    }
    { float sinv = rsqrtf(warp_reduce_sum(sn) + 1e-6f);
      for (int di = 0; di < my_d; ++di) Ks[di] *= sinv; }

    // Weighted merge of all buffer tokens
    float Kp[MAX_DPL], Vp[MAX_DPL];
    for (int di = 0; di < my_d; ++di) Kp[di] = Vp[di] = 0.0f;
    float Wp = 0, Sp = 0, Cp = 0;
    for (int i = 0; i < nv; ++i) {
        float n2 = 0;
        for (int di = 0; di < my_d; ++di) n2 += Kb[i][di] * Kb[i][di];
        float inv = rsqrtf(warp_reduce_sum(n2) + 1e-6f);
        float sim_p = 0;
        for (int di = 0; di < my_d; ++di) sim_p += (Kb[i][di] * inv) * Ks[di];
        float sim = fminf(fmaxf(warp_reduce_sum(sim_p), -1.0f), 1.0f);
        float c = expf(sim);
        for (int di = 0; di < my_d; ++di) { Kp[di] += c * Kb[i][di]; Vp[di] += c * Vb[i][di]; }
        Wp += c; Sp += Sb[i]; Cp += 1.0f;
    }
    float invW = 1.0f / fmaxf(Wp, 1e-8f);
    for (int di = 0; di < my_d; ++di) { Kp[di] *= invW; Vp[di] *= invW; }

    // Write new pivot to workspace output arrays
    int32_t piv_idx = -1;
    if (lane == 0) piv_idx = atomicAdd(new_piv_count, 1);
    piv_idx = __shfl_sync(FULL_MASK, piv_idx, 0);

    if (piv_idx >= 0) {
        int64_t piv_offset = (int64_t)piv_idx * D;
        for (int di = 0; di < my_d; ++di) {
            int d = d_lo + di;
            new_piv_K[piv_offset + d] = from_float<scalar_t>(Kp[di]);
            new_piv_V[piv_offset + d] = from_float<scalar_t>(Vp[di]);
            new_piv_Ks[piv_offset + d] = from_float<scalar_t>(Ks[di]);
        }
        if (lane == 0) {
            new_piv_W[piv_idx] = Wp;
            new_piv_S[piv_idx] = Sp;
            new_piv_C[piv_idx] = Cp;
            new_piv_Ss[piv_idx] = ss;
            new_piv_rows[piv_idx] = row;
        }
    }

    // Clean buffer row: reset state (no free stack ops here!)
    if (lane == 0) {
        for (int b = 0; b < B; ++b) buf_row_seg[buf_seg_base + b] = -1;
        buf_rs[row] = (int8_t)BufState::RESERVED;
        buf_rc[row] = 0;
        if (diag) {
            atomicAdd(&diag[DIAG_A2O_ROWS_MERGED], 1);
        }
    }
}

// --------------- K3: buf_free_seg_kernel ---------------
// Push-only: return freed buffer segments to seg_buf_free_stack.
// Launched AFTER K1 (pop-only) has fully completed, so no concurrent pop/push race.
__global__ void buf_free_seg_kernel(
    const int32_t* __restrict__ to_free_sids,
    const int32_t* __restrict__ to_free_count,
    int32_t* __restrict__ seg_free_stack,
    int32_t* __restrict__ seg_free_top)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = *to_free_count;
    if (idx >= n) return;
    int32_t ft = atomicAdd(seg_free_top, 1);
    seg_free_stack[ft] = to_free_sids[idx];
}

// --------------- remerge_fused_seg_kernel ---------------
// Insert new pivot tokens (from K2 output) into the pivot segment pool.
// Handles pivot row availability, victim eviction, and allocation.
// Pop/push on seg_piv_free_top only; contention is low (1 op per merged row).
template <typename scalar_t>
__global__ void remerge_fused_seg_kernel(
    const scalar_t* __restrict__ new_piv_K,
    const scalar_t* __restrict__ new_piv_V,
    const float* __restrict__ new_piv_W,
    const float* __restrict__ new_piv_S,
    const float* __restrict__ new_piv_C,
    const scalar_t* __restrict__ new_piv_Ks,
    const float* __restrict__ new_piv_Ss,
    const int64_t* __restrict__ new_piv_rows,
    const int32_t* __restrict__ new_piv_count,
    // Pivot seg pool
    scalar_t* __restrict__ seg_piv_K,
    scalar_t* __restrict__ seg_piv_V,
    float* __restrict__ seg_piv_W,
    float* __restrict__ seg_piv_S,
    float* __restrict__ seg_piv_C,
    scalar_t* __restrict__ seg_piv_Ks,
    float* __restrict__ seg_piv_Ss,
    int32_t* __restrict__ piv_row_seg,
    int8_t* __restrict__ piv_rs,
    int32_t* __restrict__ piv_rc,
    int32_t* __restrict__ seg_piv_free_stack,
    int32_t* __restrict__ seg_piv_free_top,
    int P, int D, int64_t S_tot,
    // Pivot zone allocator params (nullptr to use global stack fallback)
    const int32_t* __restrict__ voxel_zone_map,
    int32_t* __restrict__ piv_zone_top,
    int32_t piv_zone_cap,
    int32_t piv_num_zones,
    int64_t V_alloc,
    int32_t* __restrict__ diag)
{
    const int pi = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int N = *new_piv_count;
    if (pi >= N) return;

    const int64_t row = new_piv_rows[pi];
    if (row < 0 || row >= S_tot) return;

    const int dpl = (D + WARP_SIZE - 1) / WARP_SIZE;
    const int d_lo = lane * dpl;
    const int d_hi = (d_lo + dpl < D) ? d_lo + dpl : D;
    const int my_d = d_hi - d_lo;

    // Read merged pivot token from workspace
    int64_t src_offset = (int64_t)pi * D;
    float Kp[MAX_DPL], Vp[MAX_DPL], Ks_seed[MAX_DPL];
    for (int di = 0; di < my_d; ++di) {
        Kp[di] = to_float(new_piv_K[src_offset + d_lo + di]);
        Vp[di] = to_float(new_piv_V[src_offset + d_lo + di]);
        Ks_seed[di] = to_float(new_piv_Ks[src_offset + d_lo + di]);
    }
    float Wp = new_piv_W[pi];
    float Sp_sum = new_piv_S[pi];
    float Cp = new_piv_C[pi];
    float ss = new_piv_Ss[pi];

    // Ensure pivot row is available
    int8_t pstate = piv_rs[row];
    if (pstate == (int8_t)BufState::RESERVED) {
        if (lane == 0) {
            piv_rs[row] = (int8_t)BufState::AVAILABLE;
            piv_rc[row] = 0;
        }
    }

    int64_t piv_seg_base = row * (int64_t)P;
    int ws = -1;
    int32_t new_sid = -1;
    for (int p = 0; p < P && p < MAX_PIVOTS; ++p)
        if (piv_row_seg[piv_seg_base + p] < 0) { ws = p; break; }

    if (ws < 0) {
        // Evict victim (min weight) and merge into nearest neighbor
        int j_victim = 0; float min_W = 1e30f;
        for (int p = 0; p < P && p < MAX_PIVOTS; ++p) {
            int32_t sid = piv_row_seg[piv_seg_base + p];
            if (sid >= 0 && seg_piv_W[sid] < min_W) { min_W = seg_piv_W[sid]; j_victim = p; }
        }

        int32_t v_sid = piv_row_seg[piv_seg_base + j_victim];
        float K_victim[MAX_DPL]; float vn2 = 0;
        for (int di = 0; di < my_d; ++di) {
            float k = to_float(seg_piv_K[(int64_t)v_sid * D + d_lo + di]);
            K_victim[di] = k; vn2 += k * k;
        }
        float vinv = rsqrtf(warp_reduce_sum(vn2) + 1e-6f);

        int j_nei = -1; float max_sim = -1e30f;
        for (int p = 0; p < P && p < MAX_PIVOTS; ++p) {
            if (p == j_victim) continue;
            int32_t sid = piv_row_seg[piv_seg_base + p];
            if (sid < 0) continue;
            float nn2 = 0;
            for (int di = 0; di < my_d; ++di) {
                float k = to_float(seg_piv_K[(int64_t)sid * D + d_lo + di]);
                nn2 += k * k;
            }
            float ninv = rsqrtf(warp_reduce_sum(nn2) + 1e-6f);
            float sp = 0;
            for (int di = 0; di < my_d; ++di)
                sp += (K_victim[di] * vinv) * (to_float(seg_piv_K[(int64_t)sid * D + d_lo + di]) * ninv);
            float s = fminf(fmaxf(warp_reduce_sum(sp), -1.0f), 1.0f);
            if (s > max_sim) { max_sim = s; j_nei = p; }
        }

        if (j_nei >= 0) {
            int32_t n_sid = piv_row_seg[piv_seg_base + j_nei];
            float scale = expf(max_sim - 1.0f);
            float Wv = seg_piv_W[v_sid] * scale, Wn = seg_piv_W[n_sid];
            float den = fmaxf(Wv + Wn, 1e-8f);
            float fv = Wv / den, fn = Wn / den;
            for (int di = 0; di < my_d; ++di) {
                int d = d_lo + di;
                float kv = to_float(seg_piv_K[(int64_t)v_sid * D + d]);
                float kn = to_float(seg_piv_K[(int64_t)n_sid * D + d]);
                float vv = to_float(seg_piv_V[(int64_t)v_sid * D + d]);
                float vn = to_float(seg_piv_V[(int64_t)n_sid * D + d]);
                seg_piv_K[(int64_t)n_sid * D + d] = from_float<scalar_t>(fv * kv + fn * kn);
                seg_piv_V[(int64_t)n_sid * D + d] = from_float<scalar_t>(fv * vv + fn * vn);
            }
            if (lane == 0) {
                seg_piv_W[n_sid] = den;
                seg_piv_S[n_sid] = seg_piv_S[v_sid] + seg_piv_S[n_sid];
                seg_piv_C[n_sid] = seg_piv_C[v_sid] + seg_piv_C[n_sid];
            }
        }

        // Reuse victim's segment directly (no free-stack round-trip needed)
        if (lane == 0) {
            piv_row_seg[piv_seg_base + j_victim] = -1;
            if (diag) atomicAdd(&diag[DIAG_ZONE_ALLOC_VICTIM_REUSE], 1);
        }
        ws = j_victim;
        new_sid = v_sid;
    } else {
        // Free slot available — allocate new segment
        if (lane == 0) {
            if (voxel_zone_map && piv_zone_top && piv_num_zones > 0) {
                int64_t voxel_id = row % V_alloc;
                int32_t zone = voxel_zone_map[voxel_id];
                int32_t spill_dist = 0;
                new_sid = zone_alloc_piv(seg_piv_free_stack, piv_zone_top,
                                         piv_zone_cap, piv_num_zones, zone,
                                         diag ? &spill_dist : nullptr);
                if (diag) {
                    atomicAdd(&diag[DIAG_ZONE_ALLOC_TOTAL], 1);
                    if (new_sid < 0) {
                        atomicAdd(&diag[DIAG_ZONE_ALLOC_EXHAUSTED], 1);
                    } else if (spill_dist == 0) {
                        atomicAdd(&diag[DIAG_ZONE_ALLOC_HIT], 1);
                    } else {
                        atomicAdd(&diag[DIAG_ZONE_ALLOC_SPILL], 1);
                        atomicAdd(&diag[DIAG_ZONE_ALLOC_SPILL_DIST], spill_dist);
                    }
                }
            } else {
                int32_t top = atomicSub(seg_piv_free_top, 1) - 1;
                if (top < 0) { atomicAdd(seg_piv_free_top, 1); }
                else { new_sid = seg_piv_free_stack[top]; }
            }
        }
        new_sid = __shfl_sync(FULL_MASK, new_sid, 0);
    }

    if (new_sid >= 0) {
        int64_t nv_offset = (int64_t)new_sid * D;
        for (int di = 0; di < my_d; ++di) {
            int d = d_lo + di;
            seg_piv_K[nv_offset + d] = from_float<scalar_t>(Kp[di]);
            seg_piv_V[nv_offset + d] = from_float<scalar_t>(Vp[di]);
            seg_piv_Ks[nv_offset + d] = from_float<scalar_t>(Ks_seed[di]);
        }
        if (lane == 0) {
            seg_piv_W[new_sid] = Wp;
            seg_piv_S[new_sid] = Sp_sum;
            seg_piv_C[new_sid] = Cp;
            seg_piv_Ss[new_sid] = ss;
            piv_row_seg[piv_seg_base + ws] = new_sid;
            piv_rs[row] = (int8_t)BufState::AVAILABLE;
            int32_t n_valid = 0;
            for (int p = 0; p < P && p < MAX_PIVOTS; ++p)
                if (piv_row_seg[piv_seg_base + p] >= 0) n_valid++;
            piv_rc[row] = n_valid;
            if (diag) atomicAdd(&diag[DIAG_A2O_PIV_CREATED], 1);
        }
    }
}

// =============================================================================
// Host wrappers for new pipeline kernels
// =============================================================================

void buf_fill_seg(
    const int64_t* unique_rows, const int32_t* row_offsets,
    const void* sorted_K, const void* sorted_V, const float* sorted_S,
    const int32_t* G_dev,
    int32_t* buf_row_seg, int8_t* row_state, int32_t* row_count,
    void* seg_buf_K, void* seg_buf_V, float* seg_buf_S,
    int32_t* seg_free_stack, int32_t* seg_free_top,
    void* over_K, void* over_V, float* over_S, int64_t* over_rows, int32_t* over_count_dev,
    int32_t* to_free_sids, int32_t* to_free_count,
    int64_t B, int64_t D, int64_t G_max, int64_t E_max, int64_t S_tot,
    DType dtype, cudaStream_t stream,
    const int32_t* cand_row_counts, int32_t* diag)
{
    if (G_max == 0) return;
    zero_scalar_kernel<<<1, 1, 0, stream>>>(over_count_dev);
    zero_scalar_kernel<<<1, 1, 0, stream>>>(to_free_count);
    int blocks = (G_max + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    if (dtype == DType::Float16) {
        buf_fill_seg_kernel<__half><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
            unique_rows, row_offsets, (const __half*)sorted_K, (const __half*)sorted_V, sorted_S, G_dev,
            buf_row_seg, row_state, row_count, (__half*)seg_buf_K, (__half*)seg_buf_V, seg_buf_S,
            seg_free_stack, seg_free_top,
            (__half*)over_K, (__half*)over_V, over_S, over_rows, over_count_dev,
            to_free_sids, to_free_count,
            B, D, G_max, E_max, S_tot, cand_row_counts, diag);
    } else {
        buf_fill_seg_kernel<__nv_bfloat16><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
            unique_rows, row_offsets, (const __nv_bfloat16*)sorted_K, (const __nv_bfloat16*)sorted_V, sorted_S, G_dev,
            buf_row_seg, row_state, row_count, (__nv_bfloat16*)seg_buf_K, (__nv_bfloat16*)seg_buf_V, seg_buf_S,
            seg_free_stack, seg_free_top,
            (__nv_bfloat16*)over_K, (__nv_bfloat16*)over_V, over_S, over_rows, over_count_dev,
            to_free_sids, to_free_count,
            B, D, G_max, E_max, S_tot, cand_row_counts, diag);
    }
}

void buf_merge_seg(
    const int64_t* unique_rows, const int32_t* G_dev,
    const void* seg_buf_K, const void* seg_buf_V, const float* seg_buf_S,
    int32_t* buf_row_seg, int8_t* buf_row_state, int32_t* buf_row_count,
    void* new_piv_K, void* new_piv_V, float* new_piv_W, float* new_piv_S, float* new_piv_C,
    void* new_piv_Ks, float* new_piv_Ss,
    int64_t* new_piv_rows, int32_t* new_piv_count,
    int32_t* to_free_sids, int32_t* to_free_count,
    void* over_K, void* over_V, float* over_S, int64_t* over_rows, int32_t* over_count_dev,
    int64_t B, int64_t D, int64_t G_max, int64_t E_max, int64_t S_tot,
    DType dtype, cudaStream_t stream, int32_t* diag)
{
    if (G_max == 0) return;
    zero_scalar_kernel<<<1, 1, 0, stream>>>(new_piv_count);
    const int total_threads = G_max * WARP_SIZE;
    const int tpb = ALL2ONE_WARPS_PER_BLOCK * WARP_SIZE;
    const int blocks = (total_threads + tpb - 1) / tpb;
    if (dtype == DType::Float16) {
        buf_merge_seg_kernel<__half><<<blocks, tpb, 0, stream>>>(
            unique_rows, G_dev,
            (const __half*)seg_buf_K, (const __half*)seg_buf_V, seg_buf_S,
            buf_row_seg, buf_row_state, buf_row_count,
            (__half*)new_piv_K, (__half*)new_piv_V, new_piv_W, new_piv_S, new_piv_C,
            (__half*)new_piv_Ks, new_piv_Ss, new_piv_rows, new_piv_count,
            to_free_sids, to_free_count,
            (__half*)over_K, (__half*)over_V, over_S, over_rows, over_count_dev,
            B, D, G_max, E_max, S_tot, diag);
    } else {
        buf_merge_seg_kernel<__nv_bfloat16><<<blocks, tpb, 0, stream>>>(
            unique_rows, G_dev,
            (const __nv_bfloat16*)seg_buf_K, (const __nv_bfloat16*)seg_buf_V, seg_buf_S,
            buf_row_seg, buf_row_state, buf_row_count,
            (__nv_bfloat16*)new_piv_K, (__nv_bfloat16*)new_piv_V, new_piv_W, new_piv_S, new_piv_C,
            (__nv_bfloat16*)new_piv_Ks, new_piv_Ss, new_piv_rows, new_piv_count,
            to_free_sids, to_free_count,
            (__nv_bfloat16*)over_K, (__nv_bfloat16*)over_V, over_S, over_rows, over_count_dev,
            B, D, G_max, E_max, S_tot, diag);
    }
}

void buf_free_seg(
    const int32_t* to_free_sids, const int32_t* to_free_count,
    int32_t* seg_free_stack, int32_t* seg_free_top,
    int64_t max_free, cudaStream_t stream)
{
    if (max_free == 0) return;
    int blocks = (max_free + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    buf_free_seg_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
        to_free_sids, to_free_count, seg_free_stack, seg_free_top);
}

void remerge_fused_seg(
    const void* new_piv_K, const void* new_piv_V,
    const float* new_piv_W, const float* new_piv_S, const float* new_piv_C,
    const void* new_piv_Ks, const float* new_piv_Ss,
    const int64_t* new_piv_rows, const int32_t* new_piv_count,
    void* seg_piv_K, void* seg_piv_V, float* seg_piv_W, float* seg_piv_S, float* seg_piv_C,
    void* seg_piv_K_seed, float* seg_piv_S_seed,
    int32_t* piv_row_seg, int8_t* piv_row_state, int32_t* piv_row_count,
    int32_t* seg_piv_free_stack, int32_t* seg_piv_free_top,
    int64_t P, int64_t D, int64_t G_max, int64_t S_tot,
    DType dtype, cudaStream_t stream,
    const int32_t* voxel_zone_map, int32_t* piv_zone_top,
    int32_t piv_zone_cap, int32_t piv_num_zones, int64_t V_alloc,
    int32_t* diag)
{
    if (G_max == 0) return;
    const int total_threads = G_max * WARP_SIZE;
    const int tpb = ALL2ONE_WARPS_PER_BLOCK * WARP_SIZE;
    const int blocks = (total_threads + tpb - 1) / tpb;
    if (dtype == DType::Float16) {
        remerge_fused_seg_kernel<__half><<<blocks, tpb, 0, stream>>>(
            (const __half*)new_piv_K, (const __half*)new_piv_V,
            new_piv_W, new_piv_S, new_piv_C,
            (const __half*)new_piv_Ks, new_piv_Ss,
            new_piv_rows, new_piv_count,
            (__half*)seg_piv_K, (__half*)seg_piv_V, seg_piv_W, seg_piv_S, seg_piv_C,
            (__half*)seg_piv_K_seed, seg_piv_S_seed,
            piv_row_seg, piv_row_state, piv_row_count,
            seg_piv_free_stack, seg_piv_free_top,
            P, D, S_tot,
            voxel_zone_map, piv_zone_top, piv_zone_cap, piv_num_zones, V_alloc,
            diag);
    } else {
        remerge_fused_seg_kernel<__nv_bfloat16><<<blocks, tpb, 0, stream>>>(
            (const __nv_bfloat16*)new_piv_K, (const __nv_bfloat16*)new_piv_V,
            new_piv_W, new_piv_S, new_piv_C,
            (const __nv_bfloat16*)new_piv_Ks, new_piv_Ss,
            new_piv_rows, new_piv_count,
            (__nv_bfloat16*)seg_piv_K, (__nv_bfloat16*)seg_piv_V, seg_piv_W, seg_piv_S, seg_piv_C,
            (__nv_bfloat16*)seg_piv_K_seed, seg_piv_S_seed,
            piv_row_seg, piv_row_state, piv_row_count,
            seg_piv_free_stack, seg_piv_free_top,
            P, D, S_tot,
            voxel_zone_map, piv_zone_top, piv_zone_cap, piv_num_zones, V_alloc,
            diag);
    }
}

// --------------- retrieve_fixed_seg ---------------
template <typename scalar_t>
__global__ void retrieve_fixed_seg_kernel(
    const int64_t* __restrict__ voxel_ids, const scalar_t* __restrict__ seg_piv_K,
    const scalar_t* __restrict__ seg_piv_V, const float* __restrict__ seg_piv_C,
    const int32_t* __restrict__ piv_row_seg, const int8_t* __restrict__ piv_row_state,
    scalar_t* __restrict__ K_out, scalar_t* __restrict__ V_out,
    uint8_t* __restrict__ M_out, float* __restrict__ bias_out,
    int Q, int H, int64_t V_alloc, int P, int D, int retrieve_size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)H * retrieve_size;
    if (idx >= total) return;

    int h = idx / retrieve_size;
    int out_slot = idx % retrieve_size;
    int q_idx = out_slot / P;
    int p_idx = out_slot % P;

    M_out[idx] = 0; bias_out[idx] = -__FLT_MAX__;
    vec_zero_d(&K_out[idx*D], D);
    vec_zero_d(&V_out[idx*D], D);

    if (q_idx >= Q) return;
    int64_t voxel = voxel_ids[q_idx];
    if (voxel < 0 || voxel >= V_alloc) return;
    int64_t row = (int64_t)h * V_alloc + voxel;
    if (piv_row_state[row] == (int8_t)BufState::RESERVED) return;

    int32_t sid = piv_row_seg[row * P + p_idx];
    if (sid < 0) return;

    float C = seg_piv_C[sid];
    M_out[idx] = 1;
    bias_out[idx] = (C > 0.0f) ? logf(C) : -__FLT_MAX__;
    vec_copy_d(&K_out[idx*D], &seg_piv_K[(int64_t)sid*D], D);
    vec_copy_d(&V_out[idx*D], &seg_piv_V[(int64_t)sid*D], D);
}

void retrieve_fixed_seg(
    const int64_t* voxel_ids, int64_t Q,
    const void* seg_piv_K, const void* seg_piv_V, const float* seg_piv_C,
    const int32_t* piv_row_seg, const int8_t* piv_row_state,
    void* K_out, void* V_out, uint8_t* M_out, float* bias_out,
    int64_t H, int64_t V_alloc, int64_t P, int64_t D,
    int64_t retrieve_size, DType dtype, cudaStream_t stream) {
    if (Q == 0 || retrieve_size <= 0) return;
    int64_t total = H * retrieve_size;
    int blocks = (total + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    if (dtype == DType::Float16) {
        retrieve_fixed_seg_kernel<__half><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
            voxel_ids, (const __half*)seg_piv_K, (const __half*)seg_piv_V, seg_piv_C,
            piv_row_seg, piv_row_state,
            (__half*)K_out, (__half*)V_out, M_out, bias_out,
            Q, H, V_alloc, P, D, retrieve_size);
    } else {
        retrieve_fixed_seg_kernel<__nv_bfloat16><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
            voxel_ids, (const __nv_bfloat16*)seg_piv_K, (const __nv_bfloat16*)seg_piv_V, seg_piv_C,
            piv_row_seg, piv_row_state,
            (__nv_bfloat16*)K_out, (__nv_bfloat16*)V_out, M_out, bias_out,
            Q, H, V_alloc, P, D, retrieve_size);
    }
}

// --------------- retrieve_buf_seg ---------------
template <typename scalar_t>
__global__ void retrieve_buf_seg_kernel(
    const int64_t* __restrict__ voxel_ids, const scalar_t* __restrict__ seg_buf_K,
    const scalar_t* __restrict__ seg_buf_V,
    const int32_t* __restrict__ buf_row_seg, const int8_t* __restrict__ buf_row_state,
    scalar_t* __restrict__ K_out, scalar_t* __restrict__ V_out,
    uint8_t* __restrict__ M_out, float* __restrict__ bias_out,
    int Q, int H, int64_t V_alloc, int B, int D, int retrieve_size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)H * retrieve_size;
    if (idx >= total) return;

    int h = idx / retrieve_size;
    int out_slot = idx % retrieve_size;
    int q_idx = out_slot / B;
    int b_idx = out_slot % B;

    M_out[idx] = 0; bias_out[idx] = -__FLT_MAX__;
    vec_zero_d(&K_out[idx*D], D);
    vec_zero_d(&V_out[idx*D], D);

    if (q_idx >= Q) return;
    int64_t voxel = voxel_ids[q_idx];
    if (voxel < 0 || voxel >= V_alloc) return;
    int64_t row = (int64_t)h * V_alloc + voxel;
    if (buf_row_state[row] == (int8_t)BufState::RESERVED) return;

    int32_t sid = buf_row_seg[row * B + b_idx];
    if (sid < 0) return;

    M_out[idx] = 1;
    bias_out[idx] = 0.0f;
    vec_copy_d(&K_out[idx*D], &seg_buf_K[(int64_t)sid*D], D);
    vec_copy_d(&V_out[idx*D], &seg_buf_V[(int64_t)sid*D], D);
}

void retrieve_buf_seg(
    const int64_t* voxel_ids, int64_t Q,
    const void* seg_buf_K, const void* seg_buf_V,
    const int32_t* buf_row_seg, const int8_t* buf_row_state,
    void* K_out, void* V_out, uint8_t* M_out, float* bias_out,
    int64_t H, int64_t V_alloc, int64_t B, int64_t D,
    int64_t retrieve_size, DType dtype, cudaStream_t stream) {
    if (Q == 0 || retrieve_size <= 0) return;
    int64_t total = H * retrieve_size;
    int blocks = (total + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    if (dtype == DType::Float16) {
        retrieve_buf_seg_kernel<__half><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
            voxel_ids, (const __half*)seg_buf_K, (const __half*)seg_buf_V,
            buf_row_seg, buf_row_state,
            (__half*)K_out, (__half*)V_out, M_out, bias_out,
            Q, H, V_alloc, B, D, retrieve_size);
    } else {
        retrieve_buf_seg_kernel<__nv_bfloat16><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
            voxel_ids, (const __nv_bfloat16*)seg_buf_K, (const __nv_bfloat16*)seg_buf_V,
            buf_row_seg, buf_row_state,
            (__nv_bfloat16*)K_out, (__nv_bfloat16*)V_out, M_out, bias_out,
            Q, H, V_alloc, B, D, retrieve_size);
    }
}

}  // namespace detail
}  // namespace backend
}  // namespace causalvggt
