// Copyright (c) 2024-2026 CausalVGGT Authors
// SPDX-License-Identifier: Apache-2.0
//
// merger_kernels.cuh: CUDA kernel declarations for KV Merger
//
// This file declares raw pointer versions of all merger kernels.
// These functions are called by merger_pipeline.cu and take
// raw pointers instead of torch::Tensor.
//
// Pipeline steps:
// - pack_valid_tokens: Filter and compact valid tokens
// - group_by_row: Sort and group tokens by row index
// - materialize_rows: Allocate pool slots for new rows
// - one2one_merge: Similarity-based merge into existing pivots
// - buffer_topb_update: Update buffer with top-B tokens
// - all2one_merge: Cluster merge FULL buffers into pivots
// - retrieve_fixed: Retrieve pivots with fixed-size output

#pragma once

#include "merger_types.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace causalvggt {
namespace backend {
namespace detail {

// =============================================================================
// Step 1: Pack valid tokens
// =============================================================================

/**
 * Pack valid tokens: filter by validity mask and compute row indices.
 * 
 * Compacts tokens where voxel indices are valid (0 <= vx < num_voxels)
 * and computes row indices as row = head * V_alloc + voxel.
 * 
 * @param K_in [E, D] input keys (void* for half/bfloat16)
 * @param V_in [E, D] input values
 * @param S_in [E] input scores (float)
 * @param VX_in [E] or [H, E/H] input voxel indices (int64)
 * @param E number of input tokens
 * @param D head dimension
 * @param H number of heads
 * @param V_alloc voxel allocation stride
 * @param num_voxels current voxel count (for validity check)
 * @param vx_per_head if true, VX_in is [H, E/H] not [E]
 * @param rows_out [E] output row indices (int64)
 * @param K_out [E, D] output keys
 * @param V_out [E, D] output values
 * @param S_out [E] output scores
 * @param E_valid_dev [1] device scalar output: valid count
 * @param dtype data type (Float16 or BFloat16)
 * @param stream CUDA stream
 */
void pack_valid_tokens(
    const void* K_in,
    const void* V_in,
    const float* S_in,
    const int64_t* VX_in,
    int64_t E,
    int64_t D,
    int64_t H,
    int64_t V_alloc,
    int64_t num_voxels,
    bool vx_per_head,
    int64_t* rows_out,
    void* K_out,
    void* V_out,
    float* S_out,
    int32_t* E_valid_dev,
    int32_t* orig_idx_out,
    DType dtype,
    cudaStream_t stream);

// =============================================================================
// Step 3: Group by row
// =============================================================================

/**
 * Group tokens by row index using CUB sort and run-length encode.
 * 
 * @param rows [E] row indices (int64)
 * @param K [E, D] keys
 * @param V [E, D] values
 * @param S [E] scores
 * @param E total tokens
 * @param D head dimension
 * @param unique_rows [E] output unique rows (int64)
 * @param row_offsets [E+1] output CSR offsets (int32)
 * @param sorted_indices [E] output permutation (int32)
 * @param sorted_K [E, D] output reordered keys
 * @param sorted_V [E, D] output reordered values
 * @param sorted_S [E] output reordered scores
 * @param G_dev [1] device scalar output: number of unique rows
 * @param temp_storage workspace for CUB operations
 * @param temp_storage_size size of workspace
 * @param dtype data type
 * @param stream CUDA stream
 */
void group_by_row(
    const int64_t* rows,
    const void* K,
    const void* V,
    const float* S,
    int64_t E,
    int64_t D,
    int64_t* unique_rows,
    int32_t* row_offsets,
    int32_t* sorted_indices,
    void* sorted_K,
    void* sorted_V,
    float* sorted_S,
    int32_t* G_dev,
    void* temp_storage,
    size_t temp_storage_size,
    DType dtype,
    cudaStream_t stream,
    const int32_t* orig_idx = nullptr);

// =============================================================================
// Step 4: Materialize rows
// =============================================================================

/**
 * Materialize RESERVED rows by allocating pool slots.
 * 
 * @param unique_rows [G] row indices (int64)
 * @param G_dev [1] device scalar with actual G (int32)
 * @param row_ptr [S_tot] row to slot mapping (int32)
 * @param row_state [S_tot] row states (int8)
 * @param row_count [S_tot] token counts (int32)
 * @param pool_M [pool_cap, B] slot masks (uint8)
 * @param free_stack [pool_cap] free-list (int32)
 * @param free_top [1] free-list top (int32)
 * @param B buffer capacity per slot
 * @param G_max allocated size of unique_rows
 * @param n_materialized_dev [1] output device scalar (int32)
 * @param stream CUDA stream
 * @param cand_row_counts optional [G] per-row candidate counts; if non-null,
 *        rows with count==0 are skipped to avoid empty-shell slots
 */
void materialize_rows(
    const int64_t* unique_rows,
    const int32_t* G_dev,
    int32_t* row_ptr,
    int8_t* row_state,
    int32_t* row_count,
    uint8_t* pool_M,
    int32_t* free_stack,
    int32_t* free_top,
    int64_t B,
    int64_t G_max,
    int32_t* n_materialized_dev,
    int64_t S_tot,
    cudaStream_t stream,
    const int32_t* cand_row_counts = nullptr);

// =============================================================================
// Step 6: One2one merge
// =============================================================================

/**
 * One2one merge: similarity-based merge into existing pivots.
 * 
 * @param unique_rows [G] row indices
 * @param row_offsets [G+1] CSR offsets
 * @param sorted_K [E, D] sorted keys
 * @param sorted_V [E, D] sorted values
 * @param sorted_S [E] sorted scores
 * @param G_dev [1] device scalar with actual G
 * @param piv_K [pool_cap, P, D] pivot keys
 * @param piv_V [pool_cap, P, D] pivot values
 * @param piv_W [pool_cap, P] pivot weights
 * @param piv_S [pool_cap, P] pivot scores
 * @param piv_C [pool_cap, P] pivot counts
 * @param piv_K_seed [pool_cap, P, D] pivot seed keys
 * @param piv_M [pool_cap, P] pivot masks
 * @param row_ptr [S_tot] row to slot mapping
 * @param row_state [S_tot] row states
 * @param cand_K [E, D] output candidate keys
 * @param cand_V [E, D] output candidate values
 * @param cand_S [E] output candidate scores
 * @param cand_rows [E] output candidate rows
 * @param cand_count_dev [1] output candidate count
 * @param sim_thresh similarity threshold
 * @param replace_thresh replace threshold
 * @param score_thresh score threshold
 * @param P pivot capacity
 * @param D head dimension
 * @param G_max allocated size
 * @param E_max max tokens
 * @param dtype data type
 * @param stream CUDA stream
 */
/**
 * @param cand_row_counts [G_max] optional output per-row candidate counts.
 *   When non-null (CSR mode): candidates are written at CSR positions
 *   [row_offsets[ri] .. row_offsets[ri]+count) into cand_K/V/S.
 *   cand_rows and cand_count_dev are unused.
 *   When null (flat mode): original atomicAdd-based flat output.
 */
void one2one_merge(
    const int64_t* unique_rows,
    const int32_t* row_offsets,
    const void* sorted_K,
    const void* sorted_V,
    const float* sorted_S,
    const int32_t* G_dev,
    void* piv_K,
    void* piv_V,
    float* piv_W,
    float* piv_S,
    float* piv_C,
    const void* piv_K_seed,
    uint8_t* piv_M,
    const int32_t* row_ptr,
    const int8_t* row_state,
    void* cand_K,
    void* cand_V,
    float* cand_S,
    int64_t* cand_rows,
    int32_t* cand_count_dev,
    float sim_thresh,
    float replace_thresh,
    float score_thresh,
    int64_t P,
    int64_t D,
    int64_t G_max,
    int64_t E_max,
    DType dtype,
    cudaStream_t stream,
    int32_t* cand_row_counts = nullptr,
    int32_t* diag = nullptr);

// =============================================================================
// Step 7: Buffer top-B update
// =============================================================================

/**
 * Buffer update with top-B selection.
 * 
 * @param unique_rows [G] row indices
 * @param row_offsets [G+1] CSR offsets
 * @param sorted_K [E, D] sorted keys
 * @param sorted_V [E, D] sorted values
 * @param sorted_S [E] sorted scores
 * @param G_dev [1] device scalar with actual G
 * @param row_ptr [S_tot] row to slot mapping
 * @param row_state [S_tot] row states
 * @param row_count [S_tot] token counts
 * @param pool_K [pool_cap, B, D] buffer keys
 * @param pool_V [pool_cap, B, D] buffer values
 * @param pool_S [pool_cap, B] buffer scores
 * @param pool_M [pool_cap, B] buffer masks
 * @param over_K [E, D] output overflow keys
 * @param over_V [E, D] output overflow values
 * @param over_S [E] output overflow scores
 * @param over_rows [E] output overflow rows
 * @param over_count_dev [1] output overflow count
 * @param B buffer capacity
 * @param D head dimension
 * @param G_max allocated size
 * @param E_max max tokens
 * @param dtype data type
 * @param stream CUDA stream
 */
/**
 * @param cand_row_counts [G_max] optional per-row candidate counts.
 *   When non-null: new-token range for row ri is
 *   [row_offsets[ri], row_offsets[ri] + cand_row_counts[ri]).
 *   When null: range is [row_offsets[ri], row_offsets[ri+1]).
 */
void buffer_topb_update(
    const int64_t* unique_rows,
    const int32_t* row_offsets,
    const void* sorted_K,
    const void* sorted_V,
    const float* sorted_S,
    const int32_t* G_dev,
    int32_t* row_ptr,
    int8_t* row_state,
    int32_t* row_count,
    void* pool_K,
    void* pool_V,
    float* pool_S,
    uint8_t* pool_M,
    void* over_K,
    void* over_V,
    float* over_S,
    int64_t* over_rows,
    int32_t* over_count_dev,
    int64_t B,
    int64_t D,
    int64_t G_max,
    int64_t E_max,
    int64_t S_tot,
    DType dtype,
    cudaStream_t stream,
    const int32_t* cand_row_counts = nullptr,
    int32_t* diag = nullptr);

// =============================================================================
// Step 8: Get full rows
// =============================================================================

/**
 * Get rows with state == FULL.
 * 
 * @param unique_rows [G] row indices
 * @param G_dev [1] device scalar with actual G
 * @param row_state [S_tot] row states
 * @param full_rows [G] output FULL rows
 * @param F_dev [1] output FULL row count
 * @param G_max allocated size
 * @param temp_storage workspace
 * @param temp_storage_size workspace size
 * @param stream CUDA stream
 */
void get_full_rows(
    const int64_t* unique_rows,
    const int32_t* G_dev,
    const int8_t* row_state,
    int64_t* full_rows,
    int32_t* F_dev,
    int64_t G_max,
    int64_t S_tot,
    void* temp_storage,
    size_t temp_storage_size,
    cudaStream_t stream);

// =============================================================================
// Steps 8+8b+9+10 fused: All2one merge (filter FULL + materialize pivot
//                         + cluster merge + clean buffer) in one kernel
// =============================================================================

/**
 * Fused all2one merge for unique rows.
 * Each thread checks one row. Only FULL buffer rows are processed; all others
 * are skipped. This eliminates the separate get_full_rows, materialize_rows
 * (for pivots), all2one_merge, and clean_rows steps, as well as the host-
 * device synchronization that was needed to decide whether to run those steps.
 *
 * @param unique_rows  [G]  row indices (from group_by_row or cand_unique_rows)
 * @param G_dev        [1]  device scalar with actual G
 * @param buf_K        [pool_cap, B, D]  buffer keys
 * @param buf_V        [pool_cap, B, D]  buffer values
 * @param buf_S        [pool_cap, B]     buffer scores
 * @param buf_M        [pool_cap, B]     buffer masks
 * @param piv_K        [pool_cap, P, D]  pivot keys
 * @param piv_V        [pool_cap, P, D]  pivot values
 * @param piv_W        [pool_cap, P]     pivot weights
 * @param piv_S        [pool_cap, P]     pivot scores
 * @param piv_C        [pool_cap, P]     pivot counts
 * @param piv_K_seed   [pool_cap, P, D]  pivot seed keys
 * @param piv_S_seed   [pool_cap, P]     pivot seed scores
 * @param piv_M        [pool_cap, P]     pivot masks
 * @param buf_row_ptr  [S_tot]  buffer row→slot
 * @param buf_row_state [S_tot] buffer row states
 * @param buf_row_count [S_tot] buffer row counts
 * @param piv_row_ptr   [S_tot] pivot row→slot (read/write)
 * @param piv_row_state [S_tot] pivot row states (read/write)
 * @param piv_row_count [S_tot] pivot row counts (read/write)
 * @param piv_free_stack [pool_cap] pivot free-list
 * @param piv_free_top   [1]        pivot free-list top
 * @param buf_free_stack [pool_cap] buffer free-list
 * @param buf_free_top   [1]        buffer free-list top
 * @param B buffer capacity
 * @param P pivot capacity
 * @param D head dimension
 * @param G_max allocated size
 * @param S_tot total rows
 * @param dtype data type
 * @param stream CUDA stream
 */
void all2one_merge_fused(
    const int64_t* unique_rows,
    const int32_t* G_dev,
    const void* buf_K,
    const void* buf_V,
    const float* buf_S,
    uint8_t* buf_M,
    void* piv_K,
    void* piv_V,
    float* piv_W,
    float* piv_S,
    float* piv_C,
    void* piv_K_seed,
    float* piv_S_seed,
    uint8_t* piv_M,
    int32_t* buf_row_ptr,
    int8_t* buf_row_state,
    int32_t* buf_row_count,
    int32_t* piv_row_ptr,
    int8_t* piv_row_state,
    int32_t* piv_row_count,
    int32_t* piv_free_stack,
    int32_t* piv_free_top,
    int32_t* buf_free_stack,
    int32_t* buf_free_top,
    int64_t B,
    int64_t P,
    int64_t D,
    int64_t G_max,
    int64_t S_tot,
    DType dtype,
    cudaStream_t stream,
    int32_t* diag = nullptr);

// =============================================================================
// Step 10: Clean rows (kept for non-fused path)
// =============================================================================

/**
 * Clean rows: reset to RESERVED state and return slots to free-list.
 * 
 * @param rows [F] rows to clean
 * @param F_dev [1] device scalar with actual F
 * @param row_ptr [S_tot] row to slot mapping
 * @param row_state [S_tot] row states
 * @param row_count [S_tot] row counts
 * @param free_stack [pool_cap] free-list
 * @param free_top [1] free-list top
 * @param F_max allocated size
 * @param stream CUDA stream
 */
void clean_rows(
    const int64_t* rows,
    const int32_t* F_dev,
    int32_t* row_ptr,
    int8_t* row_state,
    int32_t* row_count,
    int32_t* free_stack,
    int32_t* free_top,
    int64_t F_max,
    int64_t S_tot,
    cudaStream_t stream);

// =============================================================================
// Retrieve
// =============================================================================

/**
 * Retrieve pivot data with fixed-size output.
 * 
 * @param voxel_ids [Q] query voxel indices
 * @param Q number of queries
 * @param piv_K [pool_cap, P, D] pivot keys
 * @param piv_V [pool_cap, P, D] pivot values
 * @param piv_W [pool_cap, P] pivot weights
 * @param piv_C [pool_cap, P] pivot counts
 * @param piv_M [pool_cap, P] pivot masks
 * @param row_ptr [S_tot] row to slot mapping
 * @param row_state [S_tot] row states
 * @param K_out [H, retrieve_size, D] output keys
 * @param V_out [H, retrieve_size, D] output values
 * @param M_out [H, retrieve_size] output masks
 * @param bias_out [H, retrieve_size] output logit bias
 * @param H number of heads
 * @param V_alloc voxel allocation stride
 * @param P pivot capacity
 * @param D head dimension
 * @param retrieve_size output size per head
 * @param dtype data type
 * @param stream CUDA stream
 */
void retrieve_fixed(
    const int64_t* voxel_ids,
    int64_t Q,
    const void* piv_K,
    const void* piv_V,
    const float* piv_W,
    const float* piv_C,
    const uint8_t* piv_M,
    const int32_t* row_ptr,
    const int8_t* row_state,
    void* K_out,
    void* V_out,
    uint8_t* M_out,
    float* bias_out,
    int64_t H,
    int64_t V_alloc,
    int64_t P,
    int64_t D,
    int64_t retrieve_size,
    DType dtype,
    cudaStream_t stream);

/**
 * Retrieve buffer (unmerged) tokens from the buffer pool.
 * 
 * Layout: output[h, out_slot] where out_slot = q_idx * B + b_idx
 *   - Reads from buf_pool_K/V/M at slot = buf_row_ptr[row]
 *   - Buffer tokens have bias = 0.0 (neutral weighting)
 *   - Invalid positions have M=0, bias=-inf
 * 
 * @param voxel_ids [Q] voxel indices to retrieve
 * @param Q number of queried voxels
 * @param buf_K buffer pool keys [pool_cap, B, D]
 * @param buf_V buffer pool values [pool_cap, B, D]
 * @param buf_M buffer pool mask [pool_cap, B]
 * @param row_ptr [S_tot] row -> slot mapping
 * @param row_state [S_tot] row state
 * @param K_out [H, retrieve_size, D] output keys
 * @param V_out [H, retrieve_size, D] output values
 * @param M_out [H, retrieve_size] output mask
 * @param bias_out [H, retrieve_size] output bias (0 for valid, -inf for invalid)
 * @param H number of heads
 * @param V_alloc voxel allocation
 * @param B buffer capacity per voxel
 * @param D head dimension
 * @param retrieve_size output size per head
 * @param dtype data type
 * @param stream CUDA stream
 */
void retrieve_buf(
    const int64_t* voxel_ids,
    int64_t Q,
    const void* buf_K,
    const void* buf_V,
    const uint8_t* buf_M,
    const int32_t* row_ptr,
    const int8_t* row_state,
    void* K_out,
    void* V_out,
    uint8_t* M_out,
    float* bias_out,
    int64_t H,
    int64_t V_alloc,
    int64_t B,
    int64_t D,
    int64_t retrieve_size,
    DType dtype,
    cudaStream_t stream);

// =============================================================================
// Pool State Validation
// =============================================================================

/**
 * Validation result structure.
 * Contains error counts for various consistency checks.
 */
struct ValidationResult {
    int32_t orphaned_slots;          // Slots with data but no row pointer
    int32_t invalid_row_ptrs;        // Rows with pointers to invalid slots
    int32_t state_count_mismatch;    // Rows where state doesn't match count
    int32_t state_mask_mismatch;     // Rows where state doesn't match pool mask
    int32_t free_stack_errors;       // Errors in free stack (duplicate/invalid entries)
    int32_t total_errors;            // Sum of all errors
    
    ValidationResult() : orphaned_slots(0), invalid_row_ptrs(0), 
                         state_count_mismatch(0), state_mask_mismatch(0),
                         free_stack_errors(0), total_errors(0) {}
    
    bool is_valid() const { return total_errors == 0; }
};

/**
 * Validate buffer pool state consistency.
 * 
 * Checks:
 * - All active rows have valid slot pointers within pool_cap
 * - Row states match slot occupancy (AVAILABLE has M[slot] partially set, FULL has all set)
 * - Row counts match number of valid entries in slot
 * - Free stack contains valid slot indices without duplicates
 * - No orphaned slots (data in pool but no row pointing to it)
 * 
 * @param row_ptr [S_tot] row to slot mapping
 * @param row_state [S_tot] row states
 * @param row_count [S_tot] row token counts
 * @param pool_M [pool_cap, B] slot masks
 * @param free_stack [pool_cap] free-list
 * @param free_top [1] free-list top
 * @param result_dev [1] output validation result (device memory)
 * @param S_tot total rows
 * @param pool_cap pool capacity
 * @param B buffer capacity per slot
 * @param stream CUDA stream
 */
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
    cudaStream_t stream);

/**
 * Validate pivot pool state consistency.
 * 
 * Similar checks as buffer pool, but for pivot pool.
 * 
 * @param row_ptr [S_tot] row to slot mapping
 * @param row_state [S_tot] row states
 * @param row_count [S_tot] row pivot counts
 * @param pool_M [pool_cap, P] slot masks
 * @param pool_W [pool_cap, P] slot weights (W > 0 if valid)
 * @param free_stack [pool_cap] free-list
 * @param free_top [1] free-list top
 * @param result_dev [1] output validation result (device memory)
 * @param S_tot total rows
 * @param pool_cap pool capacity
 * @param P pivot capacity per slot
 * @param stream CUDA stream
 */
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
    cudaStream_t stream);

// =============================================================================
// Segmented Pool Kernel Declarations (1 token = 1 segment)
// These _seg variants use row_seg indirection instead of slot-based access.
// =============================================================================

void materialize_rows_seg(
    const int64_t* unique_rows,
    const int32_t* G_dev,
    int32_t* row_seg,       // [S_tot * B] indirection table
    int8_t* row_state,
    int32_t* row_count,
    int32_t* seg_free_stack,
    int32_t* seg_free_top,
    int64_t B,
    int64_t G_max,
    int64_t S_tot,
    cudaStream_t stream,
    const int32_t* cand_row_counts = nullptr);

void one2one_merge_seg(
    const int64_t* unique_rows,
    const int32_t* row_offsets,
    const void* sorted_K,
    const void* sorted_V,
    const float* sorted_S,
    const int32_t* G_dev,
    // Pivot seg pool
    void* seg_piv_K,
    void* seg_piv_V,
    float* seg_piv_W,
    float* seg_piv_S,
    float* seg_piv_C,
    const void* seg_piv_K_seed,
    const int32_t* piv_row_seg,  // [S_tot * P]
    const int8_t* piv_row_state,
    const int32_t* piv_row_count,
    // Candidate output (same workspace layout)
    void* cand_K,
    void* cand_V,
    float* cand_S,
    int64_t* cand_rows,
    int32_t* cand_count_dev,
    float sim_thresh,
    float replace_thresh,
    float score_thresh,
    int64_t P,
    int64_t D,
    int64_t G_max,
    int64_t E_max,
    DType dtype,
    cudaStream_t stream,
    int32_t* cand_row_counts = nullptr,
    int32_t* diag = nullptr);

// Deprecated: replaced by buf_fill_seg + buf_merge_seg + buf_free_seg + remerge_fused_seg
#if 0
void buffer_topb_update_seg(...);
void all2one_merge_fused_seg(...);
#endif

// New pipeline: temporally separated pop/push kernels
void buf_fill_seg(
    const int64_t* unique_rows,
    const int32_t* row_offsets,
    const void* sorted_K,
    const void* sorted_V,
    const float* sorted_S,
    const int32_t* G_dev,
    int32_t* buf_row_seg,
    int8_t* row_state,
    int32_t* row_count,
    void* seg_buf_K,
    void* seg_buf_V,
    float* seg_buf_S,
    int32_t* seg_free_stack,
    int32_t* seg_free_top,
    void* over_K,
    void* over_V,
    float* over_S,
    int64_t* over_rows,
    int32_t* over_count_dev,
    int32_t* to_free_sids,
    int32_t* to_free_count,
    int64_t B,
    int64_t D,
    int64_t G_max,
    int64_t E_max,
    int64_t S_tot,
    DType dtype,
    cudaStream_t stream,
    const int32_t* cand_row_counts = nullptr,
    int32_t* diag = nullptr);

void buf_merge_seg(
    const int64_t* unique_rows,
    const int32_t* G_dev,
    const void* seg_buf_K,
    const void* seg_buf_V,
    const float* seg_buf_S,
    int32_t* buf_row_seg,
    int8_t* buf_row_state,
    int32_t* buf_row_count,
    void* new_piv_K,
    void* new_piv_V,
    float* new_piv_W,
    float* new_piv_S,
    float* new_piv_C,
    void* new_piv_Ks,
    float* new_piv_Ss,
    int64_t* new_piv_rows,
    int32_t* new_piv_count,
    int32_t* to_free_sids,
    int32_t* to_free_count,
    void* over_K,
    void* over_V,
    float* over_S,
    int64_t* over_rows,
    int32_t* over_count_dev,
    int64_t B,
    int64_t D,
    int64_t G_max,
    int64_t E_max,
    int64_t S_tot,
    DType dtype,
    cudaStream_t stream,
    int32_t* diag = nullptr);

void buf_free_seg(
    const int32_t* to_free_sids,
    const int32_t* to_free_count,
    int32_t* seg_free_stack,
    int32_t* seg_free_top,
    int64_t max_free,
    cudaStream_t stream);

void remerge_fused_seg(
    const void* new_piv_K,
    const void* new_piv_V,
    const float* new_piv_W,
    const float* new_piv_S,
    const float* new_piv_C,
    const void* new_piv_Ks,
    const float* new_piv_Ss,
    const int64_t* new_piv_rows,
    const int32_t* new_piv_count,
    void* seg_piv_K,
    void* seg_piv_V,
    float* seg_piv_W,
    float* seg_piv_S,
    float* seg_piv_C,
    void* seg_piv_K_seed,
    float* seg_piv_S_seed,
    int32_t* piv_row_seg,
    int8_t* piv_row_state,
    int32_t* piv_row_count,
    int32_t* seg_piv_free_stack,
    int32_t* seg_piv_free_top,
    int64_t P,
    int64_t D,
    int64_t G_max,
    int64_t S_tot,
    DType dtype,
    cudaStream_t stream,
    const int32_t* voxel_zone_map = nullptr,
    int32_t* piv_zone_top = nullptr,
    int32_t piv_zone_cap = 0,
    int32_t piv_num_zones = 0,
    int64_t V_alloc = 0,
    int32_t* diag = nullptr);

void retrieve_fixed_seg(
    const int64_t* voxel_ids,
    int64_t Q,
    const void* seg_piv_K,
    const void* seg_piv_V,
    const float* seg_piv_C,
    const int32_t* piv_row_seg,
    const int8_t* piv_row_state,
    void* K_out,
    void* V_out,
    uint8_t* M_out,
    float* bias_out,
    int64_t H,
    int64_t V_alloc,
    int64_t P,
    int64_t D,
    int64_t retrieve_size,
    DType dtype,
    cudaStream_t stream);

void retrieve_buf_seg(
    const int64_t* voxel_ids,
    int64_t Q,
    const void* seg_buf_K,
    const void* seg_buf_V,
    const int32_t* buf_row_seg,
    const int8_t* buf_row_state,
    void* K_out,
    void* V_out,
    uint8_t* M_out,
    float* bias_out,
    int64_t H,
    int64_t V_alloc,
    int64_t B,
    int64_t D,
    int64_t retrieve_size,
    DType dtype,
    cudaStream_t stream);

}  // namespace detail
}  // namespace backend
}  // namespace causalvggt
