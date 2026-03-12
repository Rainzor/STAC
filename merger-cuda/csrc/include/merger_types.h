// Copyright (c) 2024-2026 CausalVGGT Authors
// SPDX-License-Identifier: Apache-2.0
//
// merger_types.h: Type definitions for KV Merger
// 
// This header is completely independent of PyTorch/ATen.
// It uses raw pointers and explicit sizes for all data.
// The binding layer (merger_wrapper.h) owns torch::Tensor storage
// and passes raw pointers to these functions.
//
// Contents:
// - MergerConfig: Configuration struct for merger hyperparameters
// - MergerViews: Raw pointer views to all pools and metadata
// - MergerWorkspace: Temporary workspace for kernel operations
// - InsertAndMergeInputs/RetrieveOutputs: I/O structures
// - Backend function declarations

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstddef>

namespace causalvggt {
namespace backend {

// =============================================================================
// Constants
// =============================================================================

constexpr int64_t OVERFLOW_MAX_CAP = 65536;  // Increased from 16384 to handle larger batches
constexpr size_t DEFAULT_WORKSPACE_SIZE = 32 * 1024 * 1024;  // 32MB
// Hard kernel limits. Keep aligned with kernel implementations.
constexpr int64_t KERNEL_MAX_DIM = 128;
constexpr int64_t KERNEL_MAX_PIVOTS = 8;
constexpr int64_t KERNEL_MAX_BUFFER = 64;

// Data type enum (since we can't use torch::Dtype)
enum class DType : int {
    Float16 = 0,
    BFloat16 = 1
};

// Diagnostic counters for overflow analysis (gated by MERGER_MEM_PROFILE=1)
enum DiagIdx : int {
    // one2one_merge token fates
    DIAG_O2O_ABSORBED = 0,      // merged into pivot (sim >= sim_thresh)
    DIAG_O2O_CAND_RESERVED,     // candidate: row RESERVED / no slot
    DIAG_O2O_CAND_NO_PIVOT,     // candidate: row has slot but 0 valid pivots
    DIAG_O2O_CAND_LOW_SIM,      // candidate: sim < replace_thresh & score > gate
    DIAG_O2O_DROPPED,           // dropped: sim in middle, or score too low
    // buffer_topb token fates
    DIAG_BUF_KEPT,              // kept in buffer top-B
    DIAG_BUF_OVER_NO_SLOT,      // overflow: row not AVAILABLE (no buffer slot)
    DIAG_BUF_OVER_EXCESS,       // overflow: buffer full, excess beyond B
    DIAG_BUF_ROWS_FULL,         // rows that transitioned to FULL
    // all2one_merge_fused
    DIAG_A2O_ROWS_MERGED,       // FULL rows processed (buffer→pivot)
    DIAG_A2O_PIV_CREATED,       // new pivot slots materialized
    // overflow pivot absorption (Step 10.5)
    DIAG_OVER_ABSORBED,          // overflow tokens absorbed by new pivots
    // Step 10.5 per-token breakdown (reuses one2one_merge diag path)
    DIAG_OVER_O2O_ABSORBED,      // overflow tokens merged into pivot
    DIAG_OVER_O2O_RESERVED,      // overflow tokens hitting RESERVED rows
    DIAG_OVER_O2O_NO_PIVOT,      // overflow tokens hitting rows with 0 valid pivots
    DIAG_OVER_O2O_LOW_SIM,       // overflow tokens: sim < thresh, became candidate
    DIAG_OVER_O2O_DROPPED,       // overflow tokens: dropped (score too low)
    // Pivot zone allocator profiling
    DIAG_ZONE_ALLOC_TOTAL,       // total zone_alloc_piv calls
    DIAG_ZONE_ALLOC_HIT,         // allocated from target zone (distance 0)
    DIAG_ZONE_ALLOC_SPILL,       // allocated from a neighbor zone (distance >= 1)
    DIAG_ZONE_ALLOC_SPILL_DIST,  // sum of spill distances (for computing average)
    DIAG_ZONE_ALLOC_EXHAUSTED,   // all zones exhausted, returned -1
    DIAG_ZONE_ALLOC_VICTIM_REUSE,// pivot allocated via victim SID reuse (no zone_alloc)
    DIAG_COUNT                   // sentinel
};

// Buffer state enum (matches Python BufState)
enum class BufState : int8_t {
    RESERVED = 0,
    AVAILABLE = 1,
    FULL = 2,
    FREE = 3,
    HELD = 4
};

// =============================================================================
// MergerConfig: Configuration struct for merger hyperparameters
// =============================================================================

struct MergerConfig {
    int64_t H;              // Number of attention heads
    int64_t D;              // Head dimension
    int64_t P;              // Pivot capacity per voxel
    int64_t B;              // Buffer capacity per voxel
    int64_t V_alloc;        // Allocated voxel capacity
    int64_t S_tot;          // Total rows = H * V_alloc
    int64_t pool_cap;       // Legacy: max(buf_pool_cap, piv_pool_cap)
    int64_t buf_pool_cap;   // Buffer pool capacity (slots)
    int64_t piv_pool_cap;   // Pivot pool capacity (slots)
    
    float sim_thresh;       // Similarity threshold for merging (default: 0.75)
    float replace_thresh;   // Replace threshold for new pivots (default: 0.5)
    float score_thresh;     // Score threshold ratio (default: 0.2)
    
    int64_t overflow_max;   // Maximum overflow tokens to carry
    
    DType dtype;            // Data type (Float16 or BFloat16)

    // Pivot zone allocator (Morton-based spatial locality)
    int32_t piv_num_zones;  // Z: number of Morton zones (default 512 for zone_bits=3)
    int32_t piv_zone_cap;   // seg_piv_cap / Z: segments per zone (derived)
    
    // Default constructor with sensible defaults
    MergerConfig()
        : H(16), D(64), P(4), B(8), V_alloc(1024)
        , S_tot(H * V_alloc), pool_cap(S_tot)
        , buf_pool_cap(S_tot), piv_pool_cap(S_tot)
        , sim_thresh(0.75f), replace_thresh(0.5f), score_thresh(0.2f)
        , overflow_max(OVERFLOW_MAX_CAP)
        , dtype(DType::Float16)
        , piv_num_zones(512), piv_zone_cap(0) {}
    
    // Compute S_tot from H and V_alloc
    void update_derived() {
        S_tot = H * V_alloc;
    }
};

// =============================================================================
// MergerViews: Raw pointer views to all pools and metadata
// =============================================================================

/**
 * MergerViews contains raw pointers and strides for all merger storage.
 * The binding layer is responsible for:
 * - Allocating tensors with appropriate sizes
 * - Updating these pointers when tensors are reallocated
 * - Ensuring memory is valid during kernel execution
 */
struct MergerViews {
    // -------------------------------------------------------------------------
    // Buffer Pool (unmerged tokens)
    // -------------------------------------------------------------------------
    void* buf_pool_K;           // [pool_cap, B, D] half/bfloat16
    void* buf_pool_V;           // [pool_cap, B, D] half/bfloat16
    float* buf_pool_S;          // [pool_cap, B] float32
    uint8_t* buf_pool_M;        // [pool_cap, B] uint8
    
    int32_t* buf_row_ptr;       // [S_tot] int32, row -> slot, -1 = unmaterialized
    int8_t* buf_row_state;      // [S_tot] int8, BufState enum
    int32_t* buf_row_count;     // [S_tot] int32, token count
    
    int32_t* buf_free_stack;    // [pool_cap] int32, free slot indices
    int32_t* buf_free_top;      // [1] int32, atomic counter
    
    // -------------------------------------------------------------------------
    // Pivot Pool (merged tokens)
    // -------------------------------------------------------------------------
    void* piv_pool_K;           // [pool_cap, P, D] half/bfloat16
    void* piv_pool_V;           // [pool_cap, P, D] half/bfloat16
    float* piv_pool_W;          // [pool_cap, P] float32, weights
    float* piv_pool_S;          // [pool_cap, P] float32, score sums
    float* piv_pool_C;          // [pool_cap, P] float32, token counts
    uint8_t* piv_pool_M;        // [pool_cap, P] uint8, validity mask
    void* piv_pool_K_seed;      // [pool_cap, P, D] half/bfloat16, normalized seed keys
    float* piv_pool_S_seed;     // [pool_cap, P] float32, seed scores
    
    int32_t* piv_row_ptr;       // [S_tot] int32
    int8_t* piv_row_state;      // [S_tot] int8
    int32_t* piv_row_count;     // [S_tot] int32
    
    int32_t* piv_free_stack;    // [pool_cap] int32
    int32_t* piv_free_top;      // [1] int32
    
    // -------------------------------------------------------------------------
    // Overflow Carry-in/Carry-out Storage
    // -------------------------------------------------------------------------
    void* overflow_K;           // [overflow_max, D] half/bfloat16
    void* overflow_V;           // [overflow_max, D] half/bfloat16
    float* overflow_S;          // [overflow_max] float32
    int64_t* overflow_rows;     // [overflow_max] int64
    int32_t* overflow_count;    // [1] int32, device scalar with actual count
    
    // Default constructor initializes all pointers to nullptr
    MergerViews() {
        buf_pool_K = nullptr;
        buf_pool_V = nullptr;
        buf_pool_S = nullptr;
        buf_pool_M = nullptr;
        buf_row_ptr = nullptr;
        buf_row_state = nullptr;
        buf_row_count = nullptr;
        buf_free_stack = nullptr;
        buf_free_top = nullptr;
        
        piv_pool_K = nullptr;
        piv_pool_V = nullptr;
        piv_pool_W = nullptr;
        piv_pool_S = nullptr;
        piv_pool_C = nullptr;
        piv_pool_M = nullptr;
        piv_pool_K_seed = nullptr;
        piv_pool_S_seed = nullptr;
        piv_row_ptr = nullptr;
        piv_row_state = nullptr;
        piv_row_count = nullptr;
        piv_free_stack = nullptr;
        piv_free_top = nullptr;
        
        overflow_K = nullptr;
        overflow_V = nullptr;
        overflow_S = nullptr;
        overflow_rows = nullptr;
        overflow_count = nullptr;
    }
};

// =============================================================================
// MergerWorkspace: Temporary workspace for kernel operations
// =============================================================================

/**
 * MergerWorkspace maps a contiguous byte buffer into typed pointers.
 * Uses the diff-gaussian-rasterization pattern:
 * - required() returns the minimum bytes needed
 * - fromChunk() maps a byte buffer to typed pointers
 * 
 * The workspace is used for:
 * - CUB temporary storage
 * - Intermediate buffers for sorting/grouping
 * - Device scalars (E_total, G, F counts)
 */
struct MergerWorkspace {
    // -------------------------------------------------------------------------
    // Intermediate arrays for insert_and_merge pipeline
    // -------------------------------------------------------------------------
    
    // Input packing (Step 1)
    int64_t* packed_rows;       // [E_max] row indices after filtering
    void* packed_K;             // [E_max, D] keys after filtering
    void* packed_V;             // [E_max, D] values after filtering
    float* packed_S;            // [E_max] scores after filtering
    int32_t* E_valid_dev;       // [1] device scalar: valid input count
    int32_t* packed_orig_idx;   // [E_max] original input indices for deterministic ordering
    
    // Combined input (Step 2 - overflow + new)
    int32_t* combined_orig_idx; // [overflow_max + E_max] original indices for combined tokens
    int64_t* combined_rows;     // [overflow_max + E_max]
    void* combined_K;           // [overflow_max + E_max, D]
    void* combined_V;           // [overflow_max + E_max, D]
    float* combined_S;          // [overflow_max + E_max]
    int32_t* E_total_dev;       // [1] device scalar: total input count
    
    // Group by row output (Step 3)
    int64_t* unique_rows;       // [G_max] unique row indices
    int32_t* row_offsets;       // [G_max + 1] CSR-style offsets
    int32_t* sorted_indices;    // [E_total_max] permutation
    void* sorted_K;             // [E_total_max, D]
    void* sorted_V;             // [E_total_max, D]
    float* sorted_S;            // [E_total_max]
    int32_t* G_dev;             // [1] device scalar: number of unique rows
    
    // Buffer update overflow (Step 7)
    void* over_K;               // [over_max, D]
    void* over_V;               // [over_max, D]
    float* over_S;              // [over_max]
    int64_t* over_rows;         // [over_max]
    int32_t* over_count_dev;    // [1] device scalar
    
    // Full rows for all2one merge (Step 9)
    int64_t* full_rows;         // [F_max]
    int32_t* F_dev;             // [1] device scalar
    
    // One2one merge candidates (Step 6, optional)
    void* cand_K;               // [cand_max, D]
    void* cand_V;               // [cand_max, D]
    float* cand_S;              // [cand_max]
    int64_t* cand_rows;         // [cand_max]
    int32_t* cand_count_dev;    // [1] device scalar
    
    // Candidate regrouping buffers (Step 6b - regroup candidates after one2one merge)
    int64_t* cand_unique_rows;      // [cand_max] unique rows from candidates
    int32_t* cand_row_offsets;      // [cand_max + 1] CSR offsets
    int32_t* cand_sorted_indices;   // [cand_max] permutation
    void* cand_sorted_K;            // [cand_max, D] sorted candidate keys
    void* cand_sorted_V;            // [cand_max, D] sorted candidate values
    float* cand_sorted_S;           // [cand_max] sorted candidate scores
    int32_t* cand_G_dev;            // [1] device scalar: number of unique candidate rows
    
    // New pivot tokens output by buf_merge_seg (K2) for remerge_fused_seg
    void* new_piv_K;            // [G_max, D] merged pivot keys
    void* new_piv_V;            // [G_max, D] merged pivot values
    float* new_piv_W;           // [G_max] merged pivot weights
    float* new_piv_S;           // [G_max] merged pivot score sums
    float* new_piv_C;           // [G_max] merged pivot counts
    void* new_piv_Ks;           // [G_max, D] seed keys
    float* new_piv_Ss;          // [G_max] seed scores
    int64_t* new_piv_rows;      // [G_max] row IDs
    int32_t* new_piv_count_dev; // [1] device scalar

    // Buffer segment IDs to free (written by K2, consumed by K3)
    int32_t* to_free_sids;      // [G_max * B_max] segment IDs
    int32_t* to_free_count_dev; // [1] device scalar

    // CUB temporary storage
    void* cub_temp;             // CUB scratch space
    size_t cub_temp_size;       // Size of CUB scratch
    
    // Diagnostic counters (optional, nullptr when profiling is off)
    int32_t* diag;              // [DIAG_COUNT] atomic counters
    
    // -------------------------------------------------------------------------
    // Static Methods: required() and fromChunk()
    // -------------------------------------------------------------------------
    
    /**
     * Calculate required workspace size for given parameters.
     * 
     * @param E_max Maximum input tokens per call
     * @param G_max Maximum unique rows (typically E_max)
     * @param D Head dimension
     * @param overflow_max Maximum overflow capacity
     * @param dtype Data type (Float16 or BFloat16)
     * @return Required workspace size in bytes
     */
    static size_t required(
        int64_t E_max,
        int64_t G_max,
        int64_t D,
        int64_t overflow_max,
        DType dtype,
        int64_t B_max = 8,
        bool skip_packed = false,
        bool skip_combined_kv = false,
        int64_t new_piv_cap = -1);

    static MergerWorkspace fromChunk(
        char*& chunk,
        int64_t E_max,
        int64_t G_max,
        int64_t D,
        int64_t overflow_max,
        DType dtype,
        int64_t B_max = 8,
        bool skip_packed = false,
        bool skip_combined_kv = false,
        int64_t new_piv_cap = -1);
};

// =============================================================================
// RetrieveWorkspace: Temporary workspace for sorted retrieve
// =============================================================================

/**
 * RetrieveWorkspace maps a contiguous byte buffer into typed pointers
 * for the sorted retrieve operation.
 * 
 * The workspace is used for:
 * - Gathering all pivots with weights
 * - CUB segmented sort for sorting by weight descending
 * - Temporary buffers for sorted indices
 */
struct RetrieveWorkspace {
    // Gather buffers for pivots [H * Q * P elements]
    void* gather_K;             // [H * Q * P, D] half/bfloat16
    void* gather_V;             // [H * Q * P, D] half/bfloat16
    float* gather_W;            // [H * Q * P] float32, weights for sorting
    float* gather_C;            // [H * Q * P] float32, token counts
    uint8_t* gather_M;          // [H * Q * P] uint8, validity mask
    
    // Sort buffers
    float* sort_keys;           // [H * Q * P] negative weights for descending sort
    float* sort_keys_out;       // [H * Q * P] sorted keys output
    int32_t* sort_values;       // [H * Q * P] original indices 0..Q*P-1
    int32_t* sort_values_out;   // [H * Q * P] sorted indices output
    int32_t* segment_offsets;   // [H + 1] segment boundaries for CUB
    
    // CUB temporary storage
    void* cub_temp;             // CUB scratch space
    size_t cub_temp_size;       // Size of CUB scratch
    
    /**
     * Calculate required workspace size for retrieve.
     * 
     * @param H Number of heads
     * @param Q Number of query voxels
     * @param P Pivot capacity per voxel
     * @param D Head dimension
     * @param dtype Data type (Float16 or BFloat16)
     * @return Required workspace size in bytes
     */
    static size_t required(int64_t H, int64_t Q, int64_t P, int64_t D, DType dtype);
    
    /**
     * Map a byte buffer to typed pointers.
     */
    static RetrieveWorkspace fromChunk(
        char*& chunk,
        int64_t H,
        int64_t Q,
        int64_t P,
        int64_t D,
        DType dtype);
};

// =============================================================================
// Input/Output Structures for Backend Functions
// =============================================================================

/**
 * Input structure for insert_and_merge.
 * Contains raw pointers to input tensors with explicit sizes.
 */
struct InsertAndMergeInputs {
    const void* K_new;          // [E, D] New key vectors
    const void* V_new;          // [E, D] New value vectors
    const float* S_new;         // [E] New scores
    const int64_t* VX_new;      // [E] or [H, E/H] Voxel indices (-1 = invalid)
    int64_t E;                  // Number of input tokens
    int64_t num_voxels;         // Current number of voxels (for validity check)
    
    // Optional per-head voxel layout (if VX_new is [H, E/H])
    bool vx_per_head;           // If true, VX_new is [H, E/H] not [E]
    
    InsertAndMergeInputs()
        : K_new(nullptr), V_new(nullptr), S_new(nullptr), VX_new(nullptr)
        , E(0), num_voxels(0), vx_per_head(false) {}
};

/**
 * Output structure for retrieve.
 * Contains raw pointers to output buffers.
 */
struct RetrieveOutputs {
    void* K_out;                // [H, retrieve_size, D] Output keys
    void* V_out;                // [H, retrieve_size, D] Output values
    uint8_t* M_out;             // [H, retrieve_size] Output mask
    float* bias_out;            // [H, retrieve_size] Output logit bias (log(C) or -inf)
    
    RetrieveOutputs()
        : K_out(nullptr), V_out(nullptr), M_out(nullptr), bias_out(nullptr) {}
};

// =============================================================================
// Backend Function Declarations
// =============================================================================

/**
 * Insert new tokens and perform merge operations.
 * 
 * This implements the full merge pipeline:
 * 1. Filter valid tokens (VX >= 0 && VX < num_voxels)
 * 2. Carry-in overflow tokens from previous call
 * 3. Compute row indices: row = head * V_alloc + voxel_id
 * 4. Group by row (sort + run-length encode)
 * 5. Materialize buffer and pivot rows (allocate slots)
 * 6. Optional: one2one_merge for similarity-based pivot update
 * 7. Buffer update with top-B selection
 * 8. Find FULL buffer rows
 * 9. All2one merge: cluster merge FULL buffers into pivots
 * 10. Carry-out overflow tokens for next call
 * 
 * @param config Merger configuration
 * @param views Raw pointer views to pools/metadata
 * @param workspace Temporary workspace
 * @param inputs Input tokens
 * @param stream CUDA stream
 */
void merger_insert_and_merge(
    const MergerConfig& config,
    const MergerViews& views,
    MergerWorkspace& workspace,
    const InsertAndMergeInputs& inputs,
    cudaStream_t stream);

/**
 * Retrieve pivot data with fixed-size output.
 * 
 * For each head, retrieves up to retrieve_size pivots from the queried voxels.
 * Output is fixed-size: K/V [H, retrieve_size, D], M/bias [H, retrieve_size].
 * Invalid slots have M=0 and bias=-inf.
 * 
 * This implements CUDA-side top-k selection so Python does no sorting.
 * 
 * @param config Merger configuration
 * @param views Raw pointer views to pools/metadata
 * @param voxel_ids [Q] Voxel indices to retrieve
 * @param Q Number of query voxels
 * @param retrieve_size Number of tokens to retrieve per head
 * @param outputs Output buffers (caller-allocated)
 * @param stream CUDA stream
 */
void merger_retrieve_fixed(
    const MergerConfig& config,
    const MergerViews& views,
    const int64_t* voxel_ids,
    int64_t Q,
    int64_t retrieve_size,
    RetrieveOutputs& outputs,
    cudaStream_t stream,
    void* ws_buffer = nullptr,
    size_t ws_buffer_size = 0);

/**
 * Retrieve buffer (unmerged) tokens from the buffer pool.
 * 
 * Unlike pivot tokens which have log(C) bias, buffer tokens have bias=0.0.
 * Output layout: [H, retrieve_size, D] where retrieve_size = Q * B by default.
 * 
 * @param config Merger configuration
 * @param views Raw pointer views to pools/metadata  
 * @param voxel_ids Array of voxel indices to retrieve [Q]
 * @param Q Number of queried voxels
 * @param retrieve_size Number of tokens to retrieve per head (default Q*B)
 * @param outputs Output buffers (caller-allocated)
 * @param stream CUDA stream
 */
void merger_retrieve_buf(
    const MergerConfig& config,
    const MergerViews& views,
    const int64_t* voxel_ids,
    int64_t Q,
    int64_t retrieve_size,
    RetrieveOutputs& outputs,
    cudaStream_t stream);

/**
 * Reset all pools and metadata.
 * Clears all data but keeps allocated memory.
 * 
 * @param config Merger configuration
 * @param views Raw pointer views to pools/metadata
 * @param stream CUDA stream
 */
void merger_reset(
    const MergerConfig& config,
    MergerViews& views,
    cudaStream_t stream);

// =============================================================================
// Helper: Obtain aligned pointer from chunk (like rasterizer_impl.h)
// =============================================================================

template <typename T>
inline void obtain(char*& chunk, T*& ptr, size_t count, size_t alignment = alignof(T)) {
    size_t offset = (reinterpret_cast<uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1);
    ptr = reinterpret_cast<T*>(offset);
    chunk = reinterpret_cast<char*>(ptr + count);
}

// Specialization for void* (used for half/bfloat16 which are 2 bytes)
inline void obtain_void(char*& chunk, void*& ptr, size_t bytes, size_t alignment = 16) {
    size_t offset = (reinterpret_cast<uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1);
    ptr = reinterpret_cast<void*>(offset);
    chunk = reinterpret_cast<char*>(offset + bytes);
}

// Get element size for dtype
inline size_t dtype_size(DType dtype) {
    return (dtype == DType::Float16 || dtype == DType::BFloat16) ? 2 : 4;
}

// =============================================================================
// Segmented Pool Storage Types (1 token = 1 segment, no internal fragmentation)
// =============================================================================

struct SegPoolViews {
    // Buffer segments: each segment holds exactly 1 token
    void* seg_buf_K;             // [seg_buf_cap, D] half/bfloat16
    void* seg_buf_V;             // [seg_buf_cap, D] half/bfloat16
    float* seg_buf_S;            // [seg_buf_cap] float32
    int32_t* buf_row_seg;        // [S_tot * B] int32, (row,b)->seg_id, -1=empty
    int32_t* buf_row_count;      // [S_tot] int32
    int8_t* buf_row_state;       // [S_tot] int8
    int32_t* seg_buf_free_stack; // [seg_buf_cap] int32
    int32_t* seg_buf_free_top;   // [1] int32 atomic

    // Pivot segments: each segment holds exactly 1 pivot
    void* seg_piv_K;             // [seg_piv_cap, D]
    void* seg_piv_V;             // [seg_piv_cap, D]
    float* seg_piv_W;            // [seg_piv_cap]
    float* seg_piv_S;            // [seg_piv_cap]
    float* seg_piv_C;            // [seg_piv_cap]
    void* seg_piv_K_seed;        // [seg_piv_cap, D]
    float* seg_piv_S_seed;       // [seg_piv_cap]
    int32_t* piv_row_seg;        // [S_tot * P] int32, (row,p)->seg_id, -1=empty
    int32_t* piv_row_count;      // [S_tot] int32
    int8_t* piv_row_state;       // [S_tot] int8
    int32_t* seg_piv_free_stack; // [seg_piv_cap] int32
    int32_t* seg_piv_free_top;   // [1] int32 atomic

    int64_t seg_buf_cap;
    int64_t seg_piv_cap;

    // Pivot zone allocator (Morton-based spatial locality)
    int32_t* voxel_zone_map;    // [V_alloc] voxel_id -> zone_id, nullptr if disabled
    int32_t* piv_zone_top;      // [Z] per-zone atomic stack tops, nullptr if disabled
    int32_t  piv_num_zones;     // Z
    int32_t  piv_zone_cap;      // segments per zone

    SegPoolViews() {
        seg_buf_K = seg_buf_V = nullptr;
        seg_buf_S = nullptr;
        buf_row_seg = buf_row_count = nullptr;
        buf_row_state = nullptr;
        seg_buf_free_stack = seg_buf_free_top = nullptr;
        seg_piv_K = seg_piv_V = nullptr;
        seg_piv_W = seg_piv_S = seg_piv_C = nullptr;
        seg_piv_K_seed = nullptr;
        seg_piv_S_seed = nullptr;
        piv_row_seg = piv_row_count = nullptr;
        piv_row_state = nullptr;
        seg_piv_free_stack = seg_piv_free_top = nullptr;
        seg_buf_cap = seg_piv_cap = 0;
        voxel_zone_map = nullptr;
        piv_zone_top = nullptr;
        piv_num_zones = 0;
        piv_zone_cap = 0;
    }
};

}  // namespace backend
}  // namespace causalvggt
