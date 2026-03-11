// Copyright (c) 2024-2026 CausalVGGT Authors
// SPDX-License-Identifier: Apache-2.0
//
// merger_pipeline.cu: KV Merger Pipeline Orchestration
//
// This file implements the main entry points for the merger:
// - merger_insert_and_merge: Full insert-and-merge pipeline
// - merger_retrieve_fixed: Retrieve pivots with fixed-size output
// - merger_reset: Reset all pools and metadata
//
// Completely independent of PyTorch/ATen - uses raw pointers.

#include "include/merger_types.h"
#include "include/merger_kernels.cuh"

#include <cub/cub.cuh>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math_constants.h>
#include <algorithm>
#include <limits>
#include <cstdio>
#include <type_traits>

// Helper to convert float to scalar_t (handles __nv_bfloat16 specialization)
template<typename scalar_t>
__device__ __forceinline__ scalar_t float_to_scalar(float val) {
    if constexpr (std::is_same_v<scalar_t, __nv_bfloat16>) {
        return __float2bfloat16(val);
    } else {
        return static_cast<scalar_t>(val);
    }
}

namespace causalvggt {
namespace backend {

// =============================================================================
// Constants
// =============================================================================

constexpr int THREADS_PER_BLOCK = 256;
constexpr int MAX_DIM = 128;
constexpr int MAX_PIVOTS = 8;
constexpr int MAX_BUDGET = 64;

// =============================================================================
// Helper Kernels
// =============================================================================

/**
 * Fill array with value
 */
template <typename T>
__global__ void fill_kernel(T* data, T value, int64_t count) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        data[idx] = value;
    }
}

/**
 * Add two device scalars: out[0] = a[0] + b[0]
 */
__global__ void add_device_scalars_kernel(
    const int32_t* __restrict__ a,
    const int32_t* __restrict__ b,
    int32_t* __restrict__ out)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        out[0] = a[0] + b[0];
    }
}

/**
 * Zero a device scalar
 */
__global__ void zero_scalar_kernel(int32_t* scalar) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        scalar[0] = 0;
    }
}

/**
 * Copy and clamp device scalar
 */
__global__ void copy_and_clamp_scalar_kernel(
    const int32_t* __restrict__ in,
    int32_t* __restrict__ out,
    int max_val)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int val = *in;
        out[0] = (val > max_val) ? max_val : val;
    }
}

/**
 * Subtract two device scalars: out[0] = a[0] - b[0]
 */
__global__ void subtract_device_scalars_kernel(
    const int32_t* __restrict__ a,
    const int32_t* __restrict__ b,
    int32_t* __restrict__ out)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        out[0] = a[0] - b[0];
    }
}

__global__ void build_combined_orig_idx_kernel(
    const int32_t* __restrict__ packed_orig_idx,
    const int32_t* __restrict__ E_valid_dev,
    const int32_t* __restrict__ overflow_count_dev,
    int32_t* __restrict__ combined_orig_idx,
    int32_t E_input,
    int max_total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int E_valid = *E_valid_dev;
    int overflow_count = *overflow_count_dev;
    int total = E_valid + overflow_count;
    if (idx >= total) return;
    
    if (idx < E_valid) {
        combined_orig_idx[idx] = packed_orig_idx[idx];
    } else {
        combined_orig_idx[idx] = E_input + (idx - E_valid);
    }
}

/**
 * Concatenate two scalar buffers using device-side counts.
 */
template <typename T>
__global__ void concat_scalars_kernel(
    const T* __restrict__ a,
    const int32_t* __restrict__ count_a_dev,
    const T* __restrict__ b,
    const int32_t* __restrict__ count_b_dev,
    T* __restrict__ out,
    int max_a,
    int max_b)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int count_a = *count_a_dev;
    int count_b = *count_b_dev;
    int total = count_a + count_b;
    
    if (idx >= total) return;
    
    if (idx < count_a) {
        out[idx] = a[idx];
    } else {
        out[idx] = b[idx - count_a];
    }
}

/**
 * Concatenate two vector buffers using device-side counts.
 */
template <typename T>
__global__ void concat_vectors_kernel(
    const T* __restrict__ a,
    const int32_t* __restrict__ count_a_dev,
    const T* __restrict__ b,
    const int32_t* __restrict__ count_b_dev,
    T* __restrict__ out,
    int D,
    int max_a,
    int max_b)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int count_a = *count_a_dev;
    int count_b = *count_b_dev;
    int total_elements = (count_a + count_b) * D;
    
    if (idx >= total_elements) return;
    
    int row = idx / D;
    int col = idx % D;
    
    if (row < count_a) {
        out[idx] = a[static_cast<int64_t>(row) * D + col];
    } else {
        int src_row = row - count_a;
        out[idx] = b[static_cast<int64_t>(src_row) * D + col];
    }
}

/**
 * Copy vectors with device-side count guard.
 */
template <typename T>
__global__ void copy_vectors_guarded_kernel(
    const T* __restrict__ in,
    T* __restrict__ out,
    const int32_t* __restrict__ count_dev,
    int D,
    int max_count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int count = *count_dev;
    int total = count * D;
    
    if (idx >= total) return;
    out[idx] = in[idx];
}

/**
 * Copy scalars with device-side count guard.
 */
template <typename T>
__global__ void copy_scalars_guarded_kernel(
    const T* __restrict__ in,
    T* __restrict__ out,
    const int32_t* __restrict__ count_dev,
    int max_count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int count = *count_dev;
    
    if (idx >= count) return;
    out[idx] = in[idx];
}

/**
 * Pack valid tokens kernel.
 * Filters tokens where voxel_id >= 0 && voxel_id < num_voxels.
 * Computes row = head * V_alloc + voxel_id.
 */
template <typename scalar_t>
__global__ void pack_valid_tokens_kernel(
    const scalar_t* __restrict__ K_in,
    const scalar_t* __restrict__ V_in,
    const float* __restrict__ S_in,
    const int64_t* __restrict__ VX_in,
    scalar_t* __restrict__ K_out,
    scalar_t* __restrict__ V_out,
    float* __restrict__ S_out,
    int64_t* __restrict__ rows_out,
    int32_t* __restrict__ E_valid_dev,
    int64_t E,
    int64_t D,
    int64_t H,
    int64_t V_alloc,
    int64_t num_voxels,
    bool vx_per_head)
{
    // Use cooperative groups or atomics for compaction
    // This is a simplified version - in production, use CUB/trust select_if
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= E) return;
    
    int64_t vx;
    int64_t h;
    
    if (vx_per_head) {
        // VX_in is [H, E/H]
        int64_t tokens_per_head = E / H;
        h = idx / tokens_per_head;
        int64_t offset_in_head = idx % tokens_per_head;
        vx = VX_in[h * tokens_per_head + offset_in_head];
    } else {
        // VX_in is [E], token idx determines head
        h = idx / (E / H);
        vx = VX_in[idx];
    }
    
    // Check validity
    if (vx < 0 || vx >= num_voxels) return;
    
    // Atomically allocate output slot
    int out_idx = atomicAdd(E_valid_dev, 1);
    
    // Compute row index
    rows_out[out_idx] = h * V_alloc + vx;
    S_out[out_idx] = S_in[idx];
    
    // Copy K and V vectors
    for (int d = 0; d < D; ++d) {
        K_out[out_idx * D + d] = K_in[idx * D + d];
        V_out[out_idx * D + d] = V_in[idx * D + d];
    }
}

/**
 * Generate sequential indices
 */
__global__ void iota_kernel(int32_t* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = idx;
    }
}

/**
 * Gather vectors kernel
 */
template <typename T>
__global__ void gather_vectors_kernel(
    const T* __restrict__ in,
    const int32_t* __restrict__ perm,
    T* __restrict__ out,
    int n,
    int D)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * D;
    if (idx < total) {
        int i = idx / D;
        int d = idx % D;
        int src_idx = perm[i];
        out[idx] = in[src_idx * D + d];
    }
}

/**
 * Gather scalars kernel
 */
template <typename T>
__global__ void gather_scalars_kernel(
    const T* __restrict__ in,
    const int32_t* __restrict__ perm,
    T* __restrict__ out,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[perm[idx]];
    }
}

/**
 * Materialize rows kernel.
 * For each row that is RESERVED, allocate a slot from the free-list.
 */
__global__ void materialize_rows_kernel(
    const int64_t* __restrict__ unique_rows,
    const int32_t* __restrict__ G_dev,
    int32_t* __restrict__ row_ptr,
    int8_t* __restrict__ row_state,
    int32_t* __restrict__ row_count,
    uint8_t* __restrict__ pool_M,
    int32_t* __restrict__ free_stack,
    int32_t* __restrict__ free_top,
    int64_t B)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int G = *G_dev;
    if (idx >= G) return;
    
    int64_t row = unique_rows[idx];
    
    // Check if row is RESERVED
    if (row_state[row] != static_cast<int8_t>(BufState::RESERVED)) return;
    
    // Atomically pop from free-list
    int32_t top = atomicSub(free_top, 1) - 1;
    if (top < 0) {
        // Out of slots - restore counter
        atomicAdd(free_top, 1);
        return;
    }
    
    int32_t slot = free_stack[top];
    
    // Update row metadata
    row_ptr[row] = slot;
    row_state[row] = static_cast<int8_t>(BufState::AVAILABLE);
    row_count[row] = 0;
    
    // Zero the mask for this slot
    for (int b = 0; b < B; ++b) {
        pool_M[slot * B + b] = 0;
    }
}

/**
 * Get FULL rows kernel.
 * Filters unique_rows to find rows with state == FULL.
 */
__global__ void get_full_rows_kernel(
    const int64_t* __restrict__ unique_rows,
    const int32_t* __restrict__ G_dev,
    const int8_t* __restrict__ row_state,
    int64_t* __restrict__ full_rows,
    int32_t* __restrict__ F_dev)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int G = *G_dev;
    if (idx >= G) return;
    
    int64_t row = unique_rows[idx];
    
    if (row_state[row] == static_cast<int8_t>(BufState::FULL)) {
        int f_idx = atomicAdd(F_dev, 1);
        full_rows[f_idx] = row;
    }
}

/**
 * Clean rows kernel.
 * Return slots to free-list and reset row state to RESERVED.
 */
__global__ void clean_rows_kernel(
    const int64_t* __restrict__ rows,
    const int32_t* __restrict__ F_dev,
    int32_t* __restrict__ row_ptr,
    int8_t* __restrict__ row_state,
    int32_t* __restrict__ row_count,
    int32_t* __restrict__ free_stack,
    int32_t* __restrict__ free_top)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int F = *F_dev;
    if (idx >= F) return;
    
    int64_t row = rows[idx];
    int32_t slot = row_ptr[row];
    
    if (slot >= 0) {
        // Return slot to free-list
        int32_t top = atomicAdd(free_top, 1);
        free_stack[top] = slot;
    }
    
    // Reset row metadata
    row_ptr[row] = -1;
    row_state[row] = static_cast<int8_t>(BufState::RESERVED);
    row_count[row] = 0;
}

/**
 * Retrieve pivots kernel with fixed-size output.
 * For each head and query voxel, gather pivots and compute logit_bias.
 */
template <typename scalar_t>
__global__ void retrieve_fixed_kernel(
    const int64_t* __restrict__ voxel_ids,    // [Q]
    const scalar_t* __restrict__ piv_K,       // [pool_cap, P, D]
    const scalar_t* __restrict__ piv_V,       // [pool_cap, P, D]
    const float* __restrict__ piv_W,          // [pool_cap, P]
    const float* __restrict__ piv_C,          // [pool_cap, P]
    const uint8_t* __restrict__ piv_M,        // [pool_cap, P]
    const int32_t* __restrict__ row_ptr,      // [S_tot]
    const int8_t* __restrict__ row_state,     // [S_tot]
    scalar_t* __restrict__ K_out,             // [H, retrieve_size, D]
    scalar_t* __restrict__ V_out,             // [H, retrieve_size, D]
    uint8_t* __restrict__ M_out,              // [H, retrieve_size]
    float* __restrict__ bias_out,             // [H, retrieve_size]
    int64_t H,
    int64_t V_alloc,
    int64_t P,
    int64_t D,
    int64_t Q,
    int64_t retrieve_size)
{
    // Thread index covers all (h, output_slot) pairs
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_outputs = H * retrieve_size;
    
    if (idx >= total_outputs) return;
    
    int64_t h = idx / retrieve_size;
    int64_t out_slot = idx % retrieve_size;
    
    // Compute which (voxel, pivot) this output slot corresponds to
    // Output layout: for each head, we have retrieve_size slots
    // If retrieve_size <= Q*P, we pack top pivots by weight
    // For simplicity, this version just packs all Q voxels' P pivots
    int64_t q_idx = out_slot / P;
    int64_t p_idx = out_slot % P;
    
    // Initialize output as invalid
    M_out[idx] = 0;
    bias_out[idx] = -__FLT_MAX__;  // -inf
    for (int d = 0; d < D; ++d) {
        K_out[idx * D + d] = float_to_scalar<scalar_t>(0.0f);
        V_out[idx * D + d] = float_to_scalar<scalar_t>(0.0f);
    }
    
    // Check if within valid range
    if (q_idx >= Q) return;
    
    int64_t voxel = voxel_ids[q_idx];
    int64_t row = h * V_alloc + voxel;
    
    int32_t slot = row_ptr[row];
    int8_t state = row_state[row];
    
    // Check if row is available
    if (slot < 0 || state == static_cast<int8_t>(BufState::RESERVED)) return;
    
    // Read pivot data
    int64_t piv_scalar_idx = slot * P + p_idx;
    int64_t piv_vec_idx = slot * P * D + p_idx * D;
    
    uint8_t mask = piv_M[piv_scalar_idx];
    if (mask == 0) return;
    
    float C = piv_C[piv_scalar_idx];
    float log_c = (C > 0.0f) ? logf(C) : -__FLT_MAX__;
    
    // Write output
    M_out[idx] = mask;
    bias_out[idx] = log_c;
    for (int d = 0; d < D; ++d) {
        K_out[idx * D + d] = piv_K[piv_vec_idx + d];
        V_out[idx * D + d] = piv_V[piv_vec_idx + d];
    }
}

// =============================================================================
// MergerWorkspace Implementation
// =============================================================================

static size_t compute_cub_workspace_for_group_by_row(int64_t N) {
    // group_by_row carves these intermediate buffers from the CUB temp space:
    //   indices:              N * sizeof(int32_t)  + alignment
    //   compound_keys:        N * sizeof(int64_t)
    //   sorted_compound_keys: N * sizeof(int64_t)
    //   sorted_rows:          N * sizeof(int64_t)
    //   counts:               N * sizeof(int32_t)
    size_t gbr_internal = ((N * sizeof(int32_t) + 7) & ~size_t(7))
                         + N * sizeof(int64_t) * 3
                         + N * sizeof(int32_t);
    
    // Query CUB for actual temp requirements.
    // CRITICAL: use (int64_t)N to match the NumItemsT type used in actual
    // group_by_row calls.  CUB deduces OffsetT from NumItemsT; querying with
    // (int) yields a smaller temp estimate than int64_t, causing OOB writes.
    size_t sort_temp = 0;
    cub::DeviceRadixSort::SortPairs(nullptr, sort_temp,
        (int64_t*)nullptr, (int64_t*)nullptr,
        (int32_t*)nullptr, (int32_t*)nullptr,
        (int64_t)N, 0, (int)(sizeof(int64_t) * 8));
    
    size_t rle_temp = 0;
    cub::DeviceRunLengthEncode::Encode(nullptr, rle_temp,
        (int64_t*)nullptr, (int64_t*)nullptr,
        (int32_t*)nullptr, (int32_t*)nullptr,
        (int64_t)N);
    
    size_t scan_temp = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, scan_temp,
        (int32_t*)nullptr, (int32_t*)nullptr,
        (int64_t)N);
    
    size_t cub_needed = std::max({sort_temp, rle_temp, scan_temp});
    
    return std::max(gbr_internal + cub_needed, (size_t)DEFAULT_WORKSPACE_SIZE);
}

size_t MergerWorkspace::required(
    int64_t E_max,
    int64_t G_max,
    int64_t D,
    int64_t overflow_max,
    DType dtype,
    int64_t B_max,
    bool skip_packed,
    bool skip_combined_kv,
    int64_t new_piv_cap)
{
    size_t elem_size = dtype_size(dtype);
    int64_t combined_max = overflow_max + E_max;
    int64_t piv_max = (new_piv_cap >= 0) ? std::min(G_max, new_piv_cap) : G_max;
    
    size_t total = 0;
    
    // Input packing (only for insert_and_merge with internal overflow)
    if (!skip_packed) {
        total += (E_max * sizeof(int64_t) + 15) & ~15;              // packed_rows
        total += (E_max * D * elem_size + 15) & ~15;                // packed_K
        total += (E_max * D * elem_size + 15) & ~15;                // packed_V
        total += (E_max * sizeof(float) + 15) & ~15;                // packed_S
        total += (sizeof(int32_t) + 15) & ~15;                      // E_valid_dev
        total += (E_max * sizeof(int32_t) + 15) & ~15;              // packed_orig_idx
    }
    
    // Combined input: always keep combined_orig_idx; K/V/S/rows only when not skipped
    total += (combined_max * sizeof(int32_t) + 15) & ~15;           // combined_orig_idx (always)
    if (!skip_combined_kv) {
        total += (combined_max * sizeof(int64_t) + 15) & ~15;       // combined_rows
        total += (combined_max * D * elem_size + 15) & ~15;         // combined_K
        total += (combined_max * D * elem_size + 15) & ~15;         // combined_V
        total += (combined_max * sizeof(float) + 15) & ~15;         // combined_S
        total += (sizeof(int32_t) + 15) & ~15;                      // E_total_dev
    }
    
    // Group by row output
    total += (G_max * sizeof(int64_t) + 15) & ~15;              // unique_rows
    total += ((G_max + 1) * sizeof(int32_t) + 15) & ~15;        // row_offsets
    total += (combined_max * sizeof(int32_t) + 15) & ~15;       // sorted_indices
    total += (combined_max * D * elem_size + 15) & ~15;         // sorted_K
    total += (combined_max * D * elem_size + 15) & ~15;         // sorted_V
    total += (combined_max * sizeof(float) + 15) & ~15;         // sorted_S
    total += (sizeof(int32_t) + 15) & ~15;                      // G_dev
    
    // Buffer update overflow
    total += (combined_max * D * elem_size + 15) & ~15;         // over_K
    total += (combined_max * D * elem_size + 15) & ~15;         // over_V
    total += (combined_max * sizeof(float) + 15) & ~15;         // over_S
    total += (combined_max * sizeof(int64_t) + 15) & ~15;       // over_rows
    total += (sizeof(int32_t) + 15) & ~15;                      // over_count_dev
    
    // Full rows
    total += (G_max * sizeof(int64_t) + 15) & ~15;              // full_rows
    total += (sizeof(int32_t) + 15) & ~15;                      // F_dev
    
    // One2one candidates
    total += (combined_max * D * elem_size + 15) & ~15;         // cand_K
    total += (combined_max * D * elem_size + 15) & ~15;         // cand_V
    total += (combined_max * sizeof(float) + 15) & ~15;         // cand_S
    total += (combined_max * sizeof(int64_t) + 15) & ~15;       // cand_rows
    total += (sizeof(int32_t) + 15) & ~15;                      // cand_count_dev
    
    // Candidate regrouping buffers
    total += (combined_max * sizeof(int64_t) + 15) & ~15;       // cand_unique_rows
    total += ((combined_max + 1) * sizeof(int32_t) + 15) & ~15; // cand_row_offsets
    total += (combined_max * sizeof(int32_t) + 15) & ~15;       // cand_sorted_indices
    total += (combined_max * D * elem_size + 15) & ~15;         // cand_sorted_K
    total += (combined_max * D * elem_size + 15) & ~15;         // cand_sorted_V
    total += (combined_max * sizeof(float) + 15) & ~15;         // cand_sorted_S
    total += (sizeof(int32_t) + 15) & ~15;                      // cand_G_dev

    // New pivot tokens (K2 -> remerge), capped by new_piv_cap
    total += (piv_max * D * elem_size + 15) & ~15;              // new_piv_K
    total += (piv_max * D * elem_size + 15) & ~15;              // new_piv_V
    total += (piv_max * sizeof(float) + 15) & ~15;              // new_piv_W
    total += (piv_max * sizeof(float) + 15) & ~15;              // new_piv_S
    total += (piv_max * sizeof(float) + 15) & ~15;              // new_piv_C
    total += (piv_max * D * elem_size + 15) & ~15;              // new_piv_Ks
    total += (piv_max * sizeof(float) + 15) & ~15;              // new_piv_Ss
    total += (piv_max * sizeof(int64_t) + 15) & ~15;            // new_piv_rows
    total += (sizeof(int32_t) + 15) & ~15;                      // new_piv_count_dev

    // Buffer segment IDs to free (K1 evicts + K2 merges -> K3)
    int64_t free_max = 2 * piv_max * B_max;
    total += (free_max * sizeof(int32_t) + 15) & ~15;           // to_free_sids
    total += (sizeof(int32_t) + 15) & ~15;                      // to_free_count_dev

    // Diagnostic counters
    total += (DIAG_COUNT * sizeof(int32_t) + 15) & ~15;
    
    // CUB temporary storage
    total += compute_cub_workspace_for_group_by_row(combined_max);
    
    return total;
}

MergerWorkspace MergerWorkspace::fromChunk(
    char*& chunk,
    int64_t E_max,
    int64_t G_max,
    int64_t D,
    int64_t overflow_max,
    DType dtype,
    int64_t B_max,
    bool skip_packed,
    bool skip_combined_kv,
    int64_t new_piv_cap)
{
    MergerWorkspace ws;
    memset(&ws, 0, sizeof(ws));
    size_t elem_size = dtype_size(dtype);
    int64_t combined_max = overflow_max + E_max;
    int64_t piv_max = (new_piv_cap >= 0) ? std::min(G_max, new_piv_cap) : G_max;
    
    // Input packing (only for insert_and_merge with internal overflow)
    if (!skip_packed) {
        obtain(chunk, ws.packed_rows, E_max);
        obtain_void(chunk, ws.packed_K, E_max * D * elem_size);
        obtain_void(chunk, ws.packed_V, E_max * D * elem_size);
        obtain(chunk, ws.packed_S, E_max);
        obtain(chunk, ws.E_valid_dev, 1);
        obtain(chunk, ws.packed_orig_idx, E_max);
    }
    
    // Combined input: always keep combined_orig_idx; K/V/S/rows only when not skipped
    obtain(chunk, ws.combined_orig_idx, combined_max);
    if (!skip_combined_kv) {
        obtain(chunk, ws.combined_rows, combined_max);
        obtain_void(chunk, ws.combined_K, combined_max * D * elem_size);
        obtain_void(chunk, ws.combined_V, combined_max * D * elem_size);
        obtain(chunk, ws.combined_S, combined_max);
        obtain(chunk, ws.E_total_dev, 1);
    }
    
    // Group by row output
    obtain(chunk, ws.unique_rows, G_max);
    obtain(chunk, ws.row_offsets, G_max + 1);
    obtain(chunk, ws.sorted_indices, combined_max);
    obtain_void(chunk, ws.sorted_K, combined_max * D * elem_size);
    obtain_void(chunk, ws.sorted_V, combined_max * D * elem_size);
    obtain(chunk, ws.sorted_S, combined_max);
    obtain(chunk, ws.G_dev, 1);
    
    // Buffer update overflow
    obtain_void(chunk, ws.over_K, combined_max * D * elem_size);
    obtain_void(chunk, ws.over_V, combined_max * D * elem_size);
    obtain(chunk, ws.over_S, combined_max);
    obtain(chunk, ws.over_rows, combined_max);
    obtain(chunk, ws.over_count_dev, 1);
    
    // Full rows
    obtain(chunk, ws.full_rows, G_max);
    obtain(chunk, ws.F_dev, 1);
    
    // One2one candidates
    obtain_void(chunk, ws.cand_K, combined_max * D * elem_size);
    obtain_void(chunk, ws.cand_V, combined_max * D * elem_size);
    obtain(chunk, ws.cand_S, combined_max);
    obtain(chunk, ws.cand_rows, combined_max);
    obtain(chunk, ws.cand_count_dev, 1);
    
    // Candidate regrouping buffers
    obtain(chunk, ws.cand_unique_rows, combined_max);
    obtain(chunk, ws.cand_row_offsets, combined_max + 1);
    obtain(chunk, ws.cand_sorted_indices, combined_max);
    obtain_void(chunk, ws.cand_sorted_K, combined_max * D * elem_size);
    obtain_void(chunk, ws.cand_sorted_V, combined_max * D * elem_size);
    obtain(chunk, ws.cand_sorted_S, combined_max);
    obtain(chunk, ws.cand_G_dev, 1);

    // New pivot tokens (K2 -> remerge), capped
    obtain_void(chunk, ws.new_piv_K, piv_max * D * elem_size);
    obtain_void(chunk, ws.new_piv_V, piv_max * D * elem_size);
    obtain(chunk, ws.new_piv_W, piv_max);
    obtain(chunk, ws.new_piv_S, piv_max);
    obtain(chunk, ws.new_piv_C, piv_max);
    obtain_void(chunk, ws.new_piv_Ks, piv_max * D * elem_size);
    obtain(chunk, ws.new_piv_Ss, piv_max);
    obtain(chunk, ws.new_piv_rows, piv_max);
    obtain(chunk, ws.new_piv_count_dev, 1);

    // Buffer segment IDs to free (K1 evicts + K2 merges -> K3)
    int64_t free_max = 2 * piv_max * B_max;
    obtain(chunk, ws.to_free_sids, free_max);
    obtain(chunk, ws.to_free_count_dev, 1);

    // Diagnostic counters
    obtain(chunk, ws.diag, (int64_t)DIAG_COUNT);
    
    // CUB temporary storage
    size_t cub_ws = compute_cub_workspace_for_group_by_row(combined_max);
    obtain_void(chunk, ws.cub_temp, cub_ws);
    ws.cub_temp_size = cub_ws;
    
    return ws;
}

// =============================================================================
// RetrieveWorkspace Implementation
// =============================================================================

size_t RetrieveWorkspace::required(int64_t H, int64_t Q, int64_t P, int64_t D, DType dtype) {
    size_t elem_size = dtype_size(dtype);
    int64_t total_pivots = H * Q * P;
    
    size_t total = 0;
    
    // Gather buffers
    total += (total_pivots * D * elem_size + 15) & ~15;  // gather_K
    total += (total_pivots * D * elem_size + 15) & ~15;  // gather_V
    total += (total_pivots * sizeof(float) + 15) & ~15;  // gather_W
    total += (total_pivots * sizeof(float) + 15) & ~15;  // gather_C
    total += (total_pivots * sizeof(uint8_t) + 15) & ~15; // gather_M
    
    // Sort buffers
    total += (total_pivots * sizeof(float) + 15) & ~15;   // sort_keys
    total += (total_pivots * sizeof(float) + 15) & ~15;   // sort_keys_out
    total += (total_pivots * sizeof(int32_t) + 15) & ~15; // sort_values
    total += (total_pivots * sizeof(int32_t) + 15) & ~15; // sort_values_out
    total += ((H + 1) * sizeof(int32_t) + 15) & ~15;      // segment_offsets
    
    // CUB temporary storage (estimate based on segmented sort requirements)
    // CUB DeviceSegmentedRadixSort requires roughly O(H + total_pivots) additional space
    size_t cub_estimate = std::max(size_t(1024 * 1024), static_cast<size_t>(total_pivots * 4));
    total += (cub_estimate + 15) & ~15;
    
    return total;
}

RetrieveWorkspace RetrieveWorkspace::fromChunk(
    char*& chunk,
    int64_t H,
    int64_t Q,
    int64_t P,
    int64_t D,
    DType dtype)
{
    RetrieveWorkspace ws;
    size_t elem_size = dtype_size(dtype);
    int64_t total_pivots = H * Q * P;
    
    // Gather buffers
    obtain_void(chunk, ws.gather_K, static_cast<size_t>(total_pivots * D) * elem_size);
    obtain_void(chunk, ws.gather_V, static_cast<size_t>(total_pivots * D) * elem_size);
    obtain(chunk, ws.gather_W, static_cast<size_t>(total_pivots));
    obtain(chunk, ws.gather_C, static_cast<size_t>(total_pivots));
    obtain(chunk, ws.gather_M, static_cast<size_t>(total_pivots));
    
    // Sort buffers
    obtain(chunk, ws.sort_keys, static_cast<size_t>(total_pivots));
    obtain(chunk, ws.sort_keys_out, static_cast<size_t>(total_pivots));
    obtain(chunk, ws.sort_values, static_cast<size_t>(total_pivots));
    obtain(chunk, ws.sort_values_out, static_cast<size_t>(total_pivots));
    obtain(chunk, ws.segment_offsets, static_cast<size_t>(H + 1));
    
    // CUB temporary storage
    size_t cub_size = std::max(size_t(1024 * 1024), static_cast<size_t>(total_pivots * 4));
    obtain_void(chunk, ws.cub_temp, cub_size);
    ws.cub_temp_size = cub_size;
    
    return ws;
}

// =============================================================================
// Sorted Retrieve Kernels
// =============================================================================

/**
 * Kernel to gather all pivots from queried voxels for sorted retrieval.
 * Output layout: [H, Q, P] flattened - each head gets Q*P pivot slots.
 */
template <typename scalar_t>
__global__ void retrieve_gather_kernel(
    const int64_t* __restrict__ voxel_ids,    // [Q]
    const scalar_t* __restrict__ piv_K,       // [pool_cap, P, D]
    const scalar_t* __restrict__ piv_V,       // [pool_cap, P, D]
    const float* __restrict__ piv_W,          // [pool_cap, P]
    const float* __restrict__ piv_C,          // [pool_cap, P]
    const uint8_t* __restrict__ piv_M,        // [pool_cap, P]
    const int32_t* __restrict__ row_ptr,      // [S_tot]
    const int8_t* __restrict__ row_state,     // [S_tot]
    scalar_t* __restrict__ gather_K,          // [H * Q * P, D]
    scalar_t* __restrict__ gather_V,          // [H * Q * P, D]
    float* __restrict__ gather_W,             // [H * Q * P]
    float* __restrict__ gather_C,             // [H * Q * P]
    uint8_t* __restrict__ gather_M,           // [H * Q * P]
    int64_t H,
    int64_t V_alloc,
    int64_t P,
    int64_t D,
    int64_t Q)
{
    // Thread index covers all (h, q, p) tuples
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = H * Q * P;
    
    if (idx >= total) return;
    
    int64_t QP = Q * P;
    int64_t h = idx / QP;
    int64_t qp = idx % QP;
    int64_t q = qp / P;
    int64_t p = qp % P;
    
    // Initialize as invalid
    gather_W[idx] = -CUDART_INF_F;  // Will sort to end
    gather_C[idx] = 0.0f;
    gather_M[idx] = 0;
    for (int d = 0; d < D; ++d) {
        gather_K[idx * D + d] = float_to_scalar<scalar_t>(0.0f);
        gather_V[idx * D + d] = float_to_scalar<scalar_t>(0.0f);
    }
    
    // Get voxel and compute row
    int64_t voxel = voxel_ids[q];
    int64_t row = h * V_alloc + voxel;
    
    int32_t slot = row_ptr[row];
    int8_t state = row_state[row];
    
    // Check if row has data
    if (slot < 0 || state == static_cast<int8_t>(BufState::RESERVED)) return;
    
    // Read pivot data
    int64_t piv_scalar_idx = slot * P + p;
    int64_t piv_vec_idx = slot * P * D + p * D;
    
    uint8_t mask = piv_M[piv_scalar_idx];
    if (mask == 0) return;
    
    // Store gathered data
    gather_W[idx] = piv_W[piv_scalar_idx];
    gather_C[idx] = piv_C[piv_scalar_idx];
    gather_M[idx] = mask;
    for (int d = 0; d < D; ++d) {
        gather_K[idx * D + d] = piv_K[piv_vec_idx + d];
        gather_V[idx * D + d] = piv_V[piv_vec_idx + d];
    }
}

/**
 * Initialize sort keys (negative W for descending-by-W) and values (indices).
 *
 * Contract alignment note:
 * Python baseline (`layers/merger.py::retrieve`) ranks pivots by W, while
 * logit_bias is still computed from C. CUDA retrieve follows the same contract:
 * sort by W first, then emit bias=log(C) for valid outputs.
 */
__global__ void retrieve_init_sort_kernel(
    const float* __restrict__ gather_W,    // [H * Q * P] - used for sorting
    const float* __restrict__ gather_C,    // [H * Q * P] - kept for interface symmetry
    float* __restrict__ sort_keys,         // [H * Q * P]
    int32_t* __restrict__ sort_values,     // [H * Q * P]
    int64_t total)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    
    // Negate W for descending sort (CUB segmented sort is ascending).
    sort_keys[idx] = -gather_W[idx];
    sort_values[idx] = static_cast<int32_t>(idx);
}

/**
 * Initialize segment offsets for CUB segmented sort.
 * Each head is a segment of Q*P elements.
 */
__global__ void retrieve_init_segments_kernel(
    int32_t* __restrict__ segment_offsets,  // [H + 1]
    int64_t QP,  // Q * P
    int64_t H)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > H) return;
    
    segment_offsets[idx] = static_cast<int32_t>(idx * QP);
}

/**
 * Scatter sorted pivots to output using sorted indices.
 * Handles truncation (retrieve_size < Q*P) and padding (retrieve_size > Q*P).
 */
template <typename scalar_t>
__global__ void retrieve_scatter_kernel(
    const int32_t* __restrict__ sorted_indices,  // [H * Q * P] - indices into gather arrays
    const scalar_t* __restrict__ gather_K,       // [H * Q * P, D]
    const scalar_t* __restrict__ gather_V,       // [H * Q * P, D]
    const float* __restrict__ gather_C,          // [H * Q * P]
    const uint8_t* __restrict__ gather_M,        // [H * Q * P]
    scalar_t* __restrict__ K_out,                // [H, retrieve_size, D]
    scalar_t* __restrict__ V_out,                // [H, retrieve_size, D]
    uint8_t* __restrict__ M_out,                 // [H, retrieve_size]
    float* __restrict__ bias_out,                // [H, retrieve_size]
    int64_t H,
    int64_t QP,          // Q * P
    int64_t D,
    int64_t retrieve_size)
{
    // Thread index covers all output positions (h, out_pos)
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_outputs = H * retrieve_size;
    
    if (idx >= total_outputs) return;
    
    int64_t h = idx / retrieve_size;
    int64_t out_pos = idx % retrieve_size;
    
    // Output index
    int64_t out_scalar_idx = idx;
    int64_t out_vec_idx = idx * D;
    
    // Initialize as invalid
    M_out[out_scalar_idx] = 0;
    bias_out[out_scalar_idx] = -CUDART_INF_F;
    for (int d = 0; d < D; ++d) {
        K_out[out_vec_idx + d] = float_to_scalar<scalar_t>(0.0f);
        V_out[out_vec_idx + d] = float_to_scalar<scalar_t>(0.0f);
    }
    
    // Check if within valid range
    if (out_pos >= QP) return;  // Padding region
    
    // Get sorted index (already sorted by weight descending within each head)
    int64_t sorted_idx_in_segment = h * QP + out_pos;
    int32_t src_idx = sorted_indices[sorted_idx_in_segment];
    
    // The src_idx is the original gather index
    uint8_t mask = gather_M[src_idx];
    if (mask == 0) return;
    
    float C = gather_C[src_idx];
    float log_c = (C > 0.0f) ? logf(C) : -CUDART_INF_F;
    
    // Write output
    M_out[out_scalar_idx] = mask;
    bias_out[out_scalar_idx] = log_c;
    int64_t src_vec_idx = src_idx * D;
    for (int d = 0; d < D; ++d) {
        K_out[out_vec_idx + d] = gather_K[src_vec_idx + d];
        V_out[out_vec_idx + d] = gather_V[src_vec_idx + d];
    }
}

// =============================================================================
// Backend Function Implementations
// =============================================================================

void merger_insert_and_merge(
    const MergerConfig& config,
    const MergerViews& views,
    MergerWorkspace& workspace,
    const InsertAndMergeInputs& inputs,
    cudaStream_t stream)
{
    // Early exit if no input
    if (inputs.E == 0) return;
    
    int64_t H = config.H;
    int64_t D = config.D;
    int64_t P = config.P;
    int64_t B = config.B;
    int64_t V_alloc = config.V_alloc;
    int64_t overflow_max = config.overflow_max;
    int64_t S_tot = config.S_tot;
    
    // =========================================================================
    // Step 1: Pack valid tokens
    // =========================================================================
    // Filters tokens where voxel_id >= 0 && voxel_id < num_voxels
    // Computes row = head * V_alloc + voxel_id
    detail::pack_valid_tokens(
        inputs.K_new,
        inputs.V_new,
        inputs.S_new,
        inputs.VX_new,
        inputs.E,
        D,
        H,
        V_alloc,
        inputs.num_voxels,
        inputs.vx_per_head,
        workspace.packed_rows,
        workspace.packed_K,
        workspace.packed_V,
        workspace.packed_S,
        workspace.E_valid_dev,
        workspace.packed_orig_idx,
        config.dtype,
        stream);
    
    // =========================================================================
    // Step 2: Concat overflow + new tokens
    // =========================================================================
    // E_total = overflow_count + E_valid
    add_device_scalars_kernel<<<1, 1, 0, stream>>>(
        views.overflow_count,
        workspace.E_valid_dev,
        workspace.E_total_dev);
    
    int64_t combined_max = inputs.E + overflow_max;
    int blocks_combined = (combined_max + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    int blocks_combined_vec = (combined_max * D + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    // FIX: Order changed to match Hybrid (Python): new tokens FIRST, overflow SECOND
    // This ensures consistent token ordering when buffer_topb_update selects by score.
    // When scores are tied, different ordering can lead to different token retention,
    // which accumulates over frames causing accuracy drift.
    if (config.dtype == DType::Float16) {
        concat_vectors_kernel<__half><<<blocks_combined_vec, THREADS_PER_BLOCK, 0, stream>>>(
            reinterpret_cast<const __half*>(workspace.packed_K),
            workspace.E_valid_dev,
            reinterpret_cast<const __half*>(views.overflow_K),
            views.overflow_count,
            reinterpret_cast<__half*>(workspace.combined_K),
            static_cast<int>(D), inputs.E, overflow_max);
        
        concat_vectors_kernel<__half><<<blocks_combined_vec, THREADS_PER_BLOCK, 0, stream>>>(
            reinterpret_cast<const __half*>(workspace.packed_V),
            workspace.E_valid_dev,
            reinterpret_cast<const __half*>(views.overflow_V),
            views.overflow_count,
            reinterpret_cast<__half*>(workspace.combined_V),
            static_cast<int>(D), inputs.E, overflow_max);
    } else {
        concat_vectors_kernel<__nv_bfloat16><<<blocks_combined_vec, THREADS_PER_BLOCK, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(workspace.packed_K),
            workspace.E_valid_dev,
            reinterpret_cast<const __nv_bfloat16*>(views.overflow_K),
            views.overflow_count,
            reinterpret_cast<__nv_bfloat16*>(workspace.combined_K),
            static_cast<int>(D), inputs.E, overflow_max);
        
        concat_vectors_kernel<__nv_bfloat16><<<blocks_combined_vec, THREADS_PER_BLOCK, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(workspace.packed_V),
            workspace.E_valid_dev,
            reinterpret_cast<const __nv_bfloat16*>(views.overflow_V),
            views.overflow_count,
            reinterpret_cast<__nv_bfloat16*>(workspace.combined_V),
            static_cast<int>(D), inputs.E, overflow_max);
    }
    
    concat_scalars_kernel<float><<<blocks_combined, THREADS_PER_BLOCK, 0, stream>>>(
        workspace.packed_S,
        workspace.E_valid_dev,
        views.overflow_S,
        views.overflow_count,
        workspace.combined_S,
        inputs.E, overflow_max);
    
    concat_scalars_kernel<int64_t><<<blocks_combined, THREADS_PER_BLOCK, 0, stream>>>(
        workspace.packed_rows,
        workspace.E_valid_dev,
        views.overflow_rows,
        views.overflow_count,
        workspace.combined_rows,
        inputs.E, overflow_max);
    
    build_combined_orig_idx_kernel<<<blocks_combined, THREADS_PER_BLOCK, 0, stream>>>(
        workspace.packed_orig_idx,
        workspace.E_valid_dev,
        views.overflow_count,
        workspace.combined_orig_idx,
        static_cast<int32_t>(inputs.E),
        combined_max);
    
    // Reset overflow count for this iteration
    zero_scalar_kernel<<<1, 1, 0, stream>>>(const_cast<int32_t*>(views.overflow_count));
    
    // =========================================================================
    // Step 3: Group by row using CUB sort + run-length encode
    // =========================================================================
    // NOTE: group_by_row uses fixed-size buffers and guards with device-side G_dev,
    // so downstream kernels can operate async. However, we pass combined_max as E
    // since E_total_dev is a device scalar.
    detail::group_by_row(
        workspace.combined_rows,
        workspace.combined_K,
        workspace.combined_V,
        workspace.combined_S,
        combined_max,  // Use max size, kernel guards with device-side count
        D,
        workspace.unique_rows,
        workspace.row_offsets,
        workspace.sorted_indices,
        workspace.sorted_K,
        workspace.sorted_V,
        workspace.sorted_S,
        workspace.G_dev,
        workspace.cub_temp,
        workspace.cub_temp_size,
        config.dtype,
        stream,
        workspace.combined_orig_idx);
    
    // G_max for downstream operations (allocated size)
    int64_t G_max = combined_max;
    
    // =========================================================================
    // Step 4: Materialize buffer rows
    // =========================================================================
    // Allocate pool slots for RESERVED rows in buffer pool
    int32_t* buf_n_mat_dev = workspace.cand_count_dev;  // Reuse workspace scalar
    detail::materialize_rows(
        workspace.unique_rows,
        workspace.G_dev,
        views.buf_row_ptr,
        views.buf_row_state,
        views.buf_row_count,
        views.buf_pool_M,
        views.buf_free_stack,
        views.buf_free_top,
        B,
        G_max,
        buf_n_mat_dev,
        config.S_tot,
        stream);
    
    // =========================================================================
    // Step 5: REMOVED - pivot rows are NOT materialized before one2one_merge.
    // Python keeps new-voxel pivot rows RESERVED until buffer is FULL and
    // all2one creates pivots.  Materializing here would make them AVAILABLE,
    // causing one2one to create pivots directly from tokens (divergent path).
    // Pivot materialization is now done for FULL rows only, before all2one.
    // =========================================================================
    
    // =========================================================================
    // Step 6: One2one merge (optional - similarity-based merge into pivots)
    // =========================================================================
    // Tokens similar to existing pivots get merged, others become candidates
    detail::one2one_merge(
        workspace.unique_rows,
        workspace.row_offsets,
        workspace.sorted_K,
        workspace.sorted_V,
        workspace.sorted_S,
        workspace.G_dev,
        views.piv_pool_K,
        views.piv_pool_V,
        views.piv_pool_W,
        views.piv_pool_S,
        views.piv_pool_C,
        views.piv_pool_K_seed,
        views.piv_pool_M,
        views.piv_row_ptr,
        views.piv_row_state,
        workspace.cand_K,
        workspace.cand_V,
        workspace.cand_S,
        workspace.cand_rows,
        workspace.cand_count_dev,
        config.sim_thresh,
        config.replace_thresh,
        config.score_thresh,
        P,
        D,
        G_max,
        combined_max,
        config.dtype,
        stream,
        nullptr,
        workspace.diag);
    
    // =========================================================================
    // Step 7: Buffer top-B update
    // =========================================================================
    // Update buffer slots with top-B tokens by score, overflow goes to over_*
    detail::buffer_topb_update(
        workspace.unique_rows,
        workspace.row_offsets,
        workspace.sorted_K,
        workspace.sorted_V,
        workspace.sorted_S,
        workspace.G_dev,
        views.buf_row_ptr,
        views.buf_row_state,
        views.buf_row_count,
        views.buf_pool_K,
        views.buf_pool_V,
        views.buf_pool_S,
        views.buf_pool_M,
        workspace.over_K,
        workspace.over_V,
        workspace.over_S,
        workspace.over_rows,
        workspace.over_count_dev,
        B,
        D,
        G_max,
        combined_max,
        config.S_tot,
        config.dtype,
        stream,
        nullptr,
        workspace.diag);
    
    // =========================================================================
    // Steps 8+8b+9+10 (fused): filter FULL → materialize pivot → all2one
    //                           merge → clean buffer, all in one kernel.
    // =========================================================================
    detail::all2one_merge_fused(
        workspace.unique_rows,
        workspace.G_dev,
        views.buf_pool_K,
        views.buf_pool_V,
        views.buf_pool_S,
        views.buf_pool_M,
        views.piv_pool_K,
        views.piv_pool_V,
        views.piv_pool_W,
        views.piv_pool_S,
        views.piv_pool_C,
        views.piv_pool_K_seed,
        views.piv_pool_S_seed,
        views.piv_pool_M,
        views.buf_row_ptr,
        views.buf_row_state,
        views.buf_row_count,
        views.piv_row_ptr,
        views.piv_row_state,
        views.piv_row_count,
        views.piv_free_stack,
        views.piv_free_top,
        views.buf_free_stack,
        views.buf_free_top,
        B,
        P,
        D,
        G_max,
        config.S_tot,
        config.dtype,
        stream,
        workspace.diag);
    
    // =========================================================================
    // Step 10.5: Overflow pivot absorption
    // =========================================================================
    // After all2one_merge creates NEW pivots, overflow tokens that previously
    // had no matching pivot may now be absorbable. Run a second one2one_merge
    // pass on the overflow against the updated pivot pool.
    //
    // Workspace reuse: unique_rows/row_offsets/sorted_K/V/S/G_dev (Step 3)
    // and cand_K/V/S/rows/cand_count_dev (Step 6) are dead after Step 10.
    
    bool ran_overflow_absorb = false;
    {
        int32_t over_count_host = 0;
        cudaMemcpyAsync(&over_count_host, workspace.over_count_dev,
                         sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        
        if (over_count_host > 0) {
            // 10.5a: Group overflow tokens by row (reuse Step 3 output buffers)
            detail::group_by_row(
                workspace.over_rows,
                workspace.over_K,
                workspace.over_V,
                workspace.over_S,
                over_count_host,
                D,
                workspace.unique_rows,
                workspace.row_offsets,
                workspace.sorted_indices,
                workspace.sorted_K,
                workspace.sorted_V,
                workspace.sorted_S,
                workspace.G_dev,
                workspace.cub_temp,
                workspace.cub_temp_size,
                config.dtype,
                stream,
                nullptr);
            
            // 10.5b: One2one merge overflow vs updated pivot pool.
            // replace_thresh=10.0 ensures ALL non-absorbed tokens become
            // candidates (sim is in [-1,1], so sim < 10 is always true).
            // score_thresh=-1.0 makes gate negative, so sc > gate always holds.
            detail::one2one_merge(
                workspace.unique_rows,
                workspace.row_offsets,
                workspace.sorted_K,
                workspace.sorted_V,
                workspace.sorted_S,
                workspace.G_dev,
                views.piv_pool_K,
                views.piv_pool_V,
                views.piv_pool_W,
                views.piv_pool_S,
                views.piv_pool_C,
                views.piv_pool_K_seed,
                views.piv_pool_M,
                views.piv_row_ptr,
                views.piv_row_state,
                workspace.cand_K,
                workspace.cand_V,
                workspace.cand_S,
                workspace.cand_rows,
                workspace.cand_count_dev,
                config.sim_thresh,
                10.0f,
                -1.0f,
                P,
                D,
                over_count_host,
                over_count_host,
                config.dtype,
                stream,
                nullptr,
                nullptr);
            
            // 10.5c: Track absorption count via device-side subtraction
            if (workspace.diag) {
                subtract_device_scalars_kernel<<<1, 1, 0, stream>>>(
                    workspace.over_count_dev,
                    workspace.cand_count_dev,
                    &workspace.diag[DIAG_OVER_ABSORBED]);
            }
            
            ran_overflow_absorb = true;
        }
    }
    
    // =========================================================================
    // Step 11: Copy overflow to carry-out buffer
    // =========================================================================
    // Source is cand_* (reduced overflow) if Step 10.5 ran, else over_* (raw).
    const int32_t* src_count = ran_overflow_absorb
        ? workspace.cand_count_dev : workspace.over_count_dev;
    const void*    src_K     = ran_overflow_absorb
        ? workspace.cand_K : workspace.over_K;
    const void*    src_V     = ran_overflow_absorb
        ? workspace.cand_V : workspace.over_V;
    const float*   src_S     = ran_overflow_absorb
        ? workspace.cand_S : workspace.over_S;
    const int64_t* src_rows  = ran_overflow_absorb
        ? workspace.cand_rows : workspace.over_rows;
    
    copy_and_clamp_scalar_kernel<<<1, 1, 0, stream>>>(
        src_count,
        const_cast<int32_t*>(views.overflow_count),
        static_cast<int>(overflow_max));
    
    int64_t over_max = combined_max;
    int blocks_over = (over_max + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    int blocks_over_vec = (over_max * D + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    if (config.dtype == DType::Float16) {
        copy_vectors_guarded_kernel<__half><<<blocks_over_vec, THREADS_PER_BLOCK, 0, stream>>>(
            reinterpret_cast<const __half*>(src_K),
            reinterpret_cast<__half*>(const_cast<void*>(views.overflow_K)),
            src_count,
            static_cast<int>(D), over_max);
        
        copy_vectors_guarded_kernel<__half><<<blocks_over_vec, THREADS_PER_BLOCK, 0, stream>>>(
            reinterpret_cast<const __half*>(src_V),
            reinterpret_cast<__half*>(const_cast<void*>(views.overflow_V)),
            src_count,
            static_cast<int>(D), over_max);
    } else {
        copy_vectors_guarded_kernel<__nv_bfloat16><<<blocks_over_vec, THREADS_PER_BLOCK, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(src_K),
            reinterpret_cast<__nv_bfloat16*>(const_cast<void*>(views.overflow_K)),
            src_count,
            static_cast<int>(D), over_max);
        
        copy_vectors_guarded_kernel<__nv_bfloat16><<<blocks_over_vec, THREADS_PER_BLOCK, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(src_V),
            reinterpret_cast<__nv_bfloat16*>(const_cast<void*>(views.overflow_V)),
            src_count,
            static_cast<int>(D), over_max);
    }
    
    copy_scalars_guarded_kernel<float><<<blocks_over, THREADS_PER_BLOCK, 0, stream>>>(
        src_S,
        const_cast<float*>(views.overflow_S),
        src_count,
        over_max);
    
    copy_scalars_guarded_kernel<int64_t><<<blocks_over, THREADS_PER_BLOCK, 0, stream>>>(
        src_rows,
        const_cast<int64_t*>(views.overflow_rows),
        src_count,
        over_max);
}

void merger_retrieve_fixed(
    const MergerConfig& config,
    const MergerViews& views,
    const int64_t* voxel_ids,
    int64_t Q,
    int64_t retrieve_size,
    RetrieveOutputs& outputs,
    cudaStream_t stream,
    void* /*ws_buffer*/,
    size_t /*ws_buffer_size*/)
{
    if (Q == 0 || retrieve_size <= 0) return;

    // Single-kernel retrieve: each thread handles one (head, out_slot) pair,
    // reading directly from the pivot pool. W-based sorting is performed in
    // MergerWrapper::retrieve() after this kernel returns.
    detail::retrieve_fixed(
        voxel_ids, Q,
        views.piv_pool_K, views.piv_pool_V,
        views.piv_pool_W, views.piv_pool_C, views.piv_pool_M,
        views.piv_row_ptr, views.piv_row_state,
        outputs.K_out, outputs.V_out, outputs.M_out, outputs.bias_out,
        config.H, config.V_alloc, config.P, config.D,
        retrieve_size, config.dtype, stream);
}

void merger_retrieve_buf(
    const MergerConfig& config,
    const MergerViews& views,
    const int64_t* voxel_ids,
    int64_t Q,
    int64_t retrieve_size,
    RetrieveOutputs& outputs,
    cudaStream_t stream)
{
    if (Q == 0 || retrieve_size <= 0) return;

    // Single-kernel retrieve for buffer pool: each thread handles one (head, out_slot) pair,
    // reading directly from the buffer pool. Buffer tokens have bias=0.0 (neutral).
    detail::retrieve_buf(
        voxel_ids, Q,
        views.buf_pool_K, views.buf_pool_V, views.buf_pool_M,
        views.buf_row_ptr, views.buf_row_state,
        outputs.K_out, outputs.V_out, outputs.M_out, outputs.bias_out,
        config.H, config.V_alloc, config.B, config.D,
        retrieve_size, config.dtype, stream);
}

void merger_reset(
    const MergerConfig& config,
    MergerViews& views,
    cudaStream_t stream)
{
    int64_t S_tot = config.S_tot;
    int64_t buf_pool_cap = config.buf_pool_cap;
    int64_t piv_pool_cap = config.piv_pool_cap;
    int64_t B = config.B;
    int64_t P = config.P;
    
    // Reset buffer pool metadata
    int blocks_s = (S_tot + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    fill_kernel<<<blocks_s, THREADS_PER_BLOCK, 0, stream>>>(
        views.buf_row_ptr, static_cast<int32_t>(-1), S_tot);
    fill_kernel<<<blocks_s, THREADS_PER_BLOCK, 0, stream>>>(
        views.buf_row_state, static_cast<int8_t>(0), S_tot);
    fill_kernel<<<blocks_s, THREADS_PER_BLOCK, 0, stream>>>(
        views.buf_row_count, static_cast<int32_t>(0), S_tot);
    
    int blocks_buf_m = (buf_pool_cap * B + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    fill_kernel<<<blocks_buf_m, THREADS_PER_BLOCK, 0, stream>>>(
        views.buf_pool_M, static_cast<uint8_t>(0), buf_pool_cap * B);
    
    // Reset buffer free-list
    int blocks_buf_pool = (buf_pool_cap + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    iota_kernel<<<blocks_buf_pool, THREADS_PER_BLOCK, 0, stream>>>(
        views.buf_free_stack, buf_pool_cap);
    fill_kernel<<<1, 1, 0, stream>>>(
        views.buf_free_top, static_cast<int32_t>(buf_pool_cap), 1);
    
    // Reset pivot pool metadata
    fill_kernel<<<blocks_s, THREADS_PER_BLOCK, 0, stream>>>(
        views.piv_row_ptr, static_cast<int32_t>(-1), S_tot);
    fill_kernel<<<blocks_s, THREADS_PER_BLOCK, 0, stream>>>(
        views.piv_row_state, static_cast<int8_t>(0), S_tot);
    fill_kernel<<<blocks_s, THREADS_PER_BLOCK, 0, stream>>>(
        views.piv_row_count, static_cast<int32_t>(0), S_tot);
    
    int blocks_piv_m = (piv_pool_cap * P + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    fill_kernel<<<blocks_piv_m, THREADS_PER_BLOCK, 0, stream>>>(
        views.piv_pool_M, static_cast<uint8_t>(0), piv_pool_cap * P);
    
    // Reset pivot free-list
    int blocks_piv_pool = (piv_pool_cap + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    iota_kernel<<<blocks_piv_pool, THREADS_PER_BLOCK, 0, stream>>>(
        views.piv_free_stack, piv_pool_cap);
    fill_kernel<<<1, 1, 0, stream>>>(
        views.piv_free_top, static_cast<int32_t>(piv_pool_cap), 1);
    
    // Reset overflow count
    zero_scalar_kernel<<<1, 1, 0, stream>>>(views.overflow_count);
}

}  // namespace backend
}  // namespace causalvggt
