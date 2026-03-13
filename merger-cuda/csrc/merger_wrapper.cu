// Copyright (c) 2024-2026 CausalVGGT Authors
// SPDX-License-Identifier: Apache-2.0
//
// MergerWrapper Implementation

#include "merger_wrapper.h"
#include "include/merger_kernels.cuh"
#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>

namespace causalvggt {

static bool merger_mem_profile_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char* v = std::getenv("MERGER_MEM_PROFILE");
        cached = (v && std::string(v) == "1") ? 1 : 0;
    }
    return cached == 1;
}

static void print_cuda_mem(const char* label) {
    if (!merger_mem_profile_enabled()) return;
    size_t free_bytes = 0, total_bytes = 0;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    double used_gb = (double)(total_bytes - free_bytes) / (1024.0 * 1024.0 * 1024.0);
    double free_gb = (double)free_bytes / (1024.0 * 1024.0 * 1024.0);
    double total_gb = (double)total_bytes / (1024.0 * 1024.0 * 1024.0);
    fprintf(stderr, "  [MEM-CUDA] %-42s | used=%.2f GB | free=%.2f GB | total=%.2f GB\n",
            label, used_gb, free_gb, total_gb);
    fflush(stderr);
}

template <typename T>
__global__ void remap_row_metadata_kernel_wrapper(
    const T* __restrict__ old_data, T* __restrict__ new_data,
    int64_t H, int64_t old_alloc, int64_t new_alloc, T init_value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = H * old_alloc;
    if (idx >= total) return;
    int64_t h = idx / old_alloc;
    int64_t v = idx % old_alloc;
    int64_t old_row = h * old_alloc + v;
    int64_t new_row = h * new_alloc + v;
    new_data[new_row] = old_data[old_row];
}

template <typename T>
__global__ void remap_seg_row_metadata_kernel(
    const T* __restrict__ old_data, T* __restrict__ new_data,
    int64_t H, int64_t old_alloc, int64_t new_alloc, int64_t W) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = H * old_alloc * W;
    if (idx >= total) return;
    int64_t hw = idx / W;
    int64_t w = idx % W;
    int64_t h = hw / old_alloc;
    int64_t v = hw % old_alloc;
    new_data[(h * new_alloc + v) * W + w] = old_data[(h * old_alloc + v) * W + w];
}

// =============================================================================
// Helper Kernels for insert_and_merge pipeline
// =============================================================================

constexpr int WRAPPER_THREADS_PER_BLOCK = 256;

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
 * Copy and clamp device scalar: out = min(in, max_val)
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

/**
 * Clamp overflow count and track dropped tokens.
 */
__global__ void clamp_overflow_with_stats_kernel(
    const int32_t* __restrict__ in_count,
    int32_t* __restrict__ out_count,
    int64_t* __restrict__ dropped_total,
    int64_t* __restrict__ dropped_last,
    int max_val)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int raw = *in_count;
        int clamped = (raw > max_val) ? max_val : raw;
        int dropped = raw - clamped;
        out_count[0] = clamped;
        dropped_last[0] = static_cast<int64_t>(dropped);
        if (dropped > 0) {
            atomicAdd(reinterpret_cast<unsigned long long*>(dropped_total),
                      static_cast<unsigned long long>(dropped));
        }
    }
}

/**
 * Build combined orig_idx: [packed_orig_idx[0:E_valid], E_input+0, E_input+1, ...]
 * New tokens keep their original input indices; overflow tokens get indices
 * starting from E_input to ensure they sort AFTER new tokens within same row.
 */
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
 * out = [a[0:count_a], b[0:count_b]]
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
 * out = [a[0:count_a, :], b[0:count_b, :]]
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
 * out[0:count, :] = in[0:count, :]
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
 * out[0:count] = in[0:count]
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
 * Fill array with constant value
 */
template <typename T>
__global__ void fill_array_kernel(T* data, T value, int64_t count) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < count) {
        data[idx] = value;
    }
}

/**
 * Generate sequential indices [0, 1, 2, ..., n-1]
 */
__global__ void iota_kernel(int32_t* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = idx;
    }
}

// =============================================================================
// Pool statistics collection kernel (diagnostics only, gated by MERGER_MEM_PROFILE)
// Each thread processes one logical row from [0, S_tot).
// Layout of stats[] output buffer (POOL_STATS_SIZE int32s):
//   [0..4]   buf row state histogram {RESERVED, AVAILABLE, FULL, FREE, HELD}
//   [5..9]   piv row state histogram {RESERVED, AVAILABLE, FULL, FREE, HELD}
//   [10..18] buf internal fill histogram: slots with 0..B valid tokens (AVAILABLE+FULL)
//   [20..24] piv internal fill histogram: slots with 0..P valid pivots (AVAILABLE rows)
// =============================================================================
constexpr int POOL_STATS_SIZE = 32;

__global__ void collect_pool_stats_kernel(
    const int8_t* __restrict__ buf_row_state,
    const int32_t* __restrict__ buf_row_ptr,
    const uint8_t* __restrict__ buf_pool_M,
    const int8_t* __restrict__ piv_row_state,
    const int32_t* __restrict__ piv_row_ptr,
    const uint8_t* __restrict__ piv_pool_M,
    int32_t* __restrict__ stats,
    int64_t S_tot, int B, int P)
{
    int64_t row = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= S_tot) return;

    int8_t bs = buf_row_state[row];
    if (bs >= 0 && bs <= 4) atomicAdd(&stats[(int)bs], 1);

    int8_t ps = piv_row_state[row];
    if (ps >= 0 && ps <= 4) atomicAdd(&stats[5 + (int)ps], 1);

    // Buffer internal fill: count valid tokens per active slot
    if ((bs == 1 || bs == 2) && buf_row_ptr[row] >= 0) {
        int32_t slot = buf_row_ptr[row];
        int count = 0;
        for (int b = 0; b < B; ++b)
            if (buf_pool_M[(int64_t)slot * B + b]) count++;
        if (count >= 0 && count <= 8) atomicAdd(&stats[10 + count], 1);
    }

    // Pivot internal fill: count valid pivots per active slot
    if (ps == 1 && piv_row_ptr[row] >= 0) {
        int32_t slot = piv_row_ptr[row];
        int count = 0;
        for (int p = 0; p < P; ++p)
            if (piv_pool_M[(int64_t)slot * P + p]) count++;
        if (count >= 0 && count <= 4) atomicAdd(&stats[20 + count], 1);
    }
}

// =============================================================================
// Segmented Pool Statistics Kernel
// Layout of stats[] (SEG_POOL_STATS_SIZE int32s):
//   [0..4]   buf row state histogram {RESERVED, AVAILABLE, FULL, FREE, HELD}
//   [5..9]   piv row state histogram {RESERVED, AVAILABLE, FULL, FREE, HELD}
//   [10..18] buf internal fill histogram: rows with 0..B valid segments
//   [20..24] piv internal fill histogram: rows with 0..P valid segments
//   [25]     total buf segments in use (sum of per-row valid segment counts)
//   [26]     total piv segments in use (sum of per-row valid segment counts)
// =============================================================================
constexpr int SEG_POOL_STATS_SIZE = 32;

// Pivot C distribution statistics:
// [0]: total active pivot rows (with >=1 pivot)
// [1]: total valid pivots across all rows
// [2..11]: total_C histogram (bucket i = count of rows with total_C in bucket i)
//   buckets: [0,1), [1,5), [5,10), [10,20), [20,50), [50,100), [100,200), [200,500), [500,1000), [1000,+inf)
// [12]: rows with 1 pivot
// [13]: rows with 2 pivots
// [14]: rows with 3 pivots
// [15]: rows with 4 pivots
// [16]: skewed rows (max_C / total_C > 0.8, among rows with >=2 pivots)
// [17]: balanced rows (max_C / total_C <= 0.5, among rows with >=2 pivots)
// [18]: sum of total_C (low 32 bits, for average)
// [19]: sum of max_C (for average skew)
// [20]: rows with >=2 pivots
constexpr int PIV_C_STATS_SIZE = 24;

__device__ inline int total_c_bucket(float c) {
    if (c < 1.0f)    return 0;
    if (c < 5.0f)    return 1;
    if (c < 10.0f)   return 2;
    if (c < 20.0f)   return 3;
    if (c < 50.0f)   return 4;
    if (c < 100.0f)  return 5;
    if (c < 200.0f)  return 6;
    if (c < 500.0f)  return 7;
    if (c < 1000.0f) return 8;
    return 9;
}

__global__ void collect_piv_c_stats_kernel(
    const int8_t* __restrict__ piv_row_state,
    const int32_t* __restrict__ piv_row_seg,
    const float* __restrict__ seg_piv_C,
    int32_t* __restrict__ stats,
    int64_t S_tot, int P)
{
    int64_t row = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= S_tot) return;

    int8_t ps = piv_row_state[row];
    if (ps != 1) return;  // only AVAILABLE pivot rows

    int n_valid = 0;
    float total_C = 0.0f, max_C = 0.0f;
    for (int p = 0; p < P; ++p) {
        int32_t seg = piv_row_seg[row * P + p];
        if (seg >= 0) {
            float c = seg_piv_C[seg];
            total_C += c;
            if (c > max_C) max_C = c;
            n_valid++;
        }
    }
    if (n_valid == 0) return;

    atomicAdd(&stats[0], 1);               // active rows
    atomicAdd(&stats[1], n_valid);          // total valid pivots
    int bkt = total_c_bucket(total_C);
    atomicAdd(&stats[2 + bkt], 1);          // total_C histogram

    if (n_valid >= 1 && n_valid <= 4)
        atomicAdd(&stats[11 + n_valid], 1); // pivot count histogram

    atomicAdd(&stats[18], (int32_t)total_C); // sum total_C
    atomicAdd(&stats[19], (int32_t)max_C);   // sum max_C

    if (n_valid >= 2) {
        atomicAdd(&stats[20], 1);           // multi-pivot rows
        float ratio = max_C / fmaxf(total_C, 1e-6f);
        if (ratio > 0.8f)
            atomicAdd(&stats[16], 1);       // skewed
        if (ratio <= 0.5f)
            atomicAdd(&stats[17], 1);       // balanced
    }
}

__global__ void collect_seg_pool_stats_kernel(
    const int8_t* __restrict__ buf_row_state,
    const int32_t* __restrict__ buf_row_seg,
    const int8_t* __restrict__ piv_row_state,
    const int32_t* __restrict__ piv_row_seg,
    int32_t* __restrict__ stats,
    int64_t S_tot, int B, int P)
{
    int64_t row = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= S_tot) return;

    int8_t bs = buf_row_state[row];
    if (bs >= 0 && bs <= 4) atomicAdd(&stats[(int)bs], 1);

    int8_t ps = piv_row_state[row];
    if (ps >= 0 && ps <= 4) atomicAdd(&stats[5 + (int)ps], 1);

    if (bs == 1 || bs == 2) {
        int count = 0;
        for (int b = 0; b < B; ++b)
            if (buf_row_seg[row * B + b] >= 0) count++;
        if (count >= 0 && count <= 8) atomicAdd(&stats[10 + count], 1);
        atomicAdd(&stats[25], count);
    }

    if (ps == 1) {
        int count = 0;
        for (int p = 0; p < P; ++p)
            if (piv_row_seg[row * P + p] >= 0) count++;
        if (count >= 0 && count <= 4) atomicAdd(&stats[20 + count], 1);
        atomicAdd(&stats[26], count);
    }
}

MergerWrapper::~MergerWrapper() {
    if (retrieve_ws_buffer_) {
        cudaFree(retrieve_ws_buffer_);
        retrieve_ws_buffer_ = nullptr;
        retrieve_ws_size_ = 0;
        retrieve_ws_Q_ = 0;
    }
}

MergerWrapper::MergerWrapper(
    int64_t num_heads, int64_t head_dim, int64_t pivot_cap, int64_t budget_cap,
    int64_t init_voxels, torch::Dtype dtype, torch::Device device, bool seg_mode)
    : dtype_(dtype), device_(device), overflow_count_host_(0), workspace_E_max_(0),
      workspace_skip_packed_(false), workspace_skip_combined_kv_(false),
      workspace_new_piv_cap_(-1),
      retrieve_ws_buffer_(nullptr), retrieve_ws_size_(0), retrieve_ws_Q_(0) {
    TORCH_CHECK(num_heads > 0 && head_dim > 0 && pivot_cap > 0 && budget_cap > 0);
    TORCH_CHECK(
        head_dim <= backend::KERNEL_MAX_DIM,
        "MergerWrapper head_dim=", head_dim,
        " exceeds kernel hard limit MAX_DIM=", backend::KERNEL_MAX_DIM);
    TORCH_CHECK(
        pivot_cap <= backend::KERNEL_MAX_PIVOTS,
        "MergerWrapper pivot_cap=", pivot_cap,
        " exceeds kernel hard limit MAX_PIVOTS=", backend::KERNEL_MAX_PIVOTS);
    TORCH_CHECK(
        budget_cap <= backend::KERNEL_MAX_BUFFER,
        "MergerWrapper budget_cap=", budget_cap,
        " exceeds kernel hard limit MAX_BUFFER=", backend::KERNEL_MAX_BUFFER);
    TORCH_CHECK(dtype == torch::kFloat16 || dtype == torch::kBFloat16);
    TORCH_CHECK(device.is_cuda());
    
    config_.H = num_heads;
    config_.D = head_dim;
    config_.P = pivot_cap;
    config_.B = budget_cap;
    config_.V_alloc = std::max(init_voxels, int64_t(1));
    config_.S_tot = config_.H * config_.V_alloc;
    overflow_cap_ = backend::OVERFLOW_MAX_CAP;
    config_.overflow_max = overflow_cap_;
    config_.dtype = (dtype == torch::kFloat16) ? backend::DType::Float16 : backend::DType::BFloat16;

    if (seg_mode) {
        use_seg_mode_ = true;
        buf_pool_cap_ = 0;
        piv_pool_cap_ = 0;
        config_.buf_pool_cap = 0;
        config_.piv_pool_cap = 0;
        config_.pool_cap = 0;
        buf_pool_growth_ = 0;
        piv_pool_growth_ = 0;
        init_row_metadata_only();
        init_seg_pools();
        update_seg_views();
    } else {
        buf_pool_cap_ = std::min(config_.S_tot, std::max(config_.S_tot / 2, int64_t(8192)));
        piv_pool_cap_ = std::min(config_.S_tot, std::max(config_.S_tot / 4, int64_t(8192)));
        config_.buf_pool_cap = buf_pool_cap_;
        config_.piv_pool_cap = piv_pool_cap_;
        config_.pool_cap = std::max(buf_pool_cap_, piv_pool_cap_);
        buf_pool_growth_ = std::max(int64_t(8192), buf_pool_cap_ / 4);
        piv_pool_growth_ = std::max(int64_t(8192), piv_pool_cap_ / 4);
        init_pools();
        update_views();
    }
}

void MergerWrapper::init_pools() {
    c10::cuda::CUDAGuard guard(device_);
    int64_t D = config_.D, P = config_.P, B = config_.B, S_tot = config_.S_tot;
    
    print_cuda_mem("init_pools:before");
    if (merger_mem_profile_enabled()) {
        int elem_kv = (dtype_ == torch::kFloat16 || dtype_ == torch::kBFloat16) ? 2 : 4;
        double buf_mb = (double)buf_pool_cap_ * B * D * elem_kv * 2 / (1024.0 * 1024.0);
        double piv_mb = (double)piv_pool_cap_ * P * D * elem_kv * 4 / (1024.0 * 1024.0);
        double meta_mb = (double)S_tot * (4+1+4) * 2 / (1024.0 * 1024.0);
        double over_mb = (double)overflow_cap_ * (D * elem_kv * 2 + 4 + 8) / (1024.0 * 1024.0);
        fprintf(stderr, "  [MEM-CUDA] init_pools: buf_pool_cap=%ld, piv_pool_cap=%ld, S_tot=%ld, "
                "buf_pool=%.1fMB, piv_pool=%.1fMB, metadata=%.1fMB, overflow=%.1fMB\n",
                buf_pool_cap_, piv_pool_cap_, S_tot, buf_mb, piv_mb, meta_mb, over_mb);
    }

    auto opts_kv = torch::TensorOptions().dtype(dtype_).device(device_);
    auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(device_);
    auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32).device(device_);
    auto opts_i8 = torch::TensorOptions().dtype(torch::kInt8).device(device_);
    auto opts_u8 = torch::TensorOptions().dtype(torch::kUInt8).device(device_);
    auto opts_i64 = torch::TensorOptions().dtype(torch::kInt64).device(device_);
    
    buf_pool_K_ = torch::zeros({buf_pool_cap_, B, D}, opts_kv);
    buf_pool_V_ = torch::zeros({buf_pool_cap_, B, D}, opts_kv);
    buf_pool_S_ = torch::zeros({buf_pool_cap_, B}, opts_f32);
    buf_pool_M_ = torch::zeros({buf_pool_cap_, B}, opts_u8);
    buf_row_ptr_ = torch::full({S_tot}, -1, opts_i32);
    buf_row_state_ = torch::zeros({S_tot}, opts_i8);
    buf_row_count_ = torch::zeros({S_tot}, opts_i32);
    buf_free_stack_ = torch::arange(buf_pool_cap_, opts_i32);
    buf_free_top_ = torch::tensor({(int64_t)buf_pool_cap_}, opts_i32);
    
    piv_pool_K_ = torch::zeros({piv_pool_cap_, P, D}, opts_kv);
    piv_pool_V_ = torch::zeros({piv_pool_cap_, P, D}, opts_kv);
    piv_pool_W_ = torch::zeros({piv_pool_cap_, P}, opts_f32);
    piv_pool_S_ = torch::zeros({piv_pool_cap_, P}, opts_f32);
    piv_pool_C_ = torch::zeros({piv_pool_cap_, P}, opts_f32);
    piv_pool_M_ = torch::zeros({piv_pool_cap_, P}, opts_u8);
    piv_pool_K_seed_ = torch::zeros({piv_pool_cap_, P, D}, opts_kv);
    piv_pool_S_seed_ = torch::zeros({piv_pool_cap_, P}, opts_f32);
    piv_row_ptr_ = torch::full({S_tot}, -1, opts_i32);
    piv_row_state_ = torch::zeros({S_tot}, opts_i8);
    piv_row_count_ = torch::zeros({S_tot}, opts_i32);
    piv_free_stack_ = torch::arange(piv_pool_cap_, opts_i32);
    piv_free_top_ = torch::tensor({(int64_t)piv_pool_cap_}, opts_i32);
    
    overflow_K_ = torch::empty({overflow_cap_, D}, opts_kv);
    overflow_V_ = torch::empty({overflow_cap_, D}, opts_kv);
    overflow_S_ = torch::empty({overflow_cap_}, opts_f32);
    overflow_rows_ = torch::empty({overflow_cap_}, opts_i64);
    overflow_count_dev_ = torch::zeros({1}, opts_i32);
    overflow_dropped_count_dev_ = torch::zeros({1}, opts_i64);
    overflow_dropped_last_dev_ = torch::zeros({1}, opts_i64);
    
    workspace_ = torch::empty({static_cast<int64_t>(backend::DEFAULT_WORKSPACE_SIZE)},
        torch::TensorOptions().dtype(torch::kUInt8).device(device_));
    print_cuda_mem("init_pools:after");
}

void MergerWrapper::init_row_metadata_only() {
    c10::cuda::CUDAGuard guard(device_);
    int64_t S_tot = config_.S_tot;

    auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32).device(device_);
    auto opts_i8 = torch::TensorOptions().dtype(torch::kInt8).device(device_);
    auto opts_i64 = torch::TensorOptions().dtype(torch::kInt64).device(device_);

    buf_row_state_ = torch::zeros({S_tot}, opts_i8);
    buf_row_count_ = torch::zeros({S_tot}, opts_i32);
    piv_row_state_ = torch::zeros({S_tot}, opts_i8);
    piv_row_count_ = torch::zeros({S_tot}, opts_i32);

    if (merger_mem_profile_enabled()) {
        double meta_mb = (double)S_tot * (1 + 4) * 2 / (1024.0 * 1024.0);
        fprintf(stderr, "  [MEM-CUDA] init_row_metadata_only: S_tot=%ld, metadata=%.1fMB\n",
                S_tot, meta_mb);
        fflush(stderr);
    }
}

void MergerWrapper::update_views() {
    views_.buf_pool_K = buf_pool_K_.data_ptr();
    views_.buf_pool_V = buf_pool_V_.data_ptr();
    views_.buf_pool_S = buf_pool_S_.data_ptr<float>();
    views_.buf_pool_M = buf_pool_M_.data_ptr<uint8_t>();
    views_.buf_row_ptr = buf_row_ptr_.data_ptr<int32_t>();
    views_.buf_row_state = buf_row_state_.data_ptr<int8_t>();
    views_.buf_row_count = buf_row_count_.data_ptr<int32_t>();
    views_.buf_free_stack = buf_free_stack_.data_ptr<int32_t>();
    views_.buf_free_top = buf_free_top_.data_ptr<int32_t>();
    
    views_.piv_pool_K = piv_pool_K_.data_ptr();
    views_.piv_pool_V = piv_pool_V_.data_ptr();
    views_.piv_pool_W = piv_pool_W_.data_ptr<float>();
    views_.piv_pool_S = piv_pool_S_.data_ptr<float>();
    views_.piv_pool_C = piv_pool_C_.data_ptr<float>();
    views_.piv_pool_M = piv_pool_M_.data_ptr<uint8_t>();
    views_.piv_pool_K_seed = piv_pool_K_seed_.data_ptr();
    views_.piv_pool_S_seed = piv_pool_S_seed_.data_ptr<float>();
    views_.piv_row_ptr = piv_row_ptr_.data_ptr<int32_t>();
    views_.piv_row_state = piv_row_state_.data_ptr<int8_t>();
    views_.piv_row_count = piv_row_count_.data_ptr<int32_t>();
    views_.piv_free_stack = piv_free_stack_.data_ptr<int32_t>();
    views_.piv_free_top = piv_free_top_.data_ptr<int32_t>();
    
    views_.overflow_K = overflow_K_.data_ptr();
    views_.overflow_V = overflow_V_.data_ptr();
    views_.overflow_S = overflow_S_.data_ptr<float>();
    views_.overflow_rows = overflow_rows_.data_ptr<int64_t>();
    views_.overflow_count = overflow_count_dev_.data_ptr<int32_t>();
}

void MergerWrapper::ensure_capacity(int64_t num_voxels) {
    if (num_voxels <= config_.V_alloc) return;
    c10::cuda::CUDAGuard guard(device_);
    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    cudaStreamSynchronize(stream);
    int64_t old_alloc = config_.V_alloc;
    int64_t new_alloc = std::max(num_voxels, old_alloc * 3 / 2);

    if (merger_mem_profile_enabled()) {
        fprintf(stderr, "  [MEM-CUDA] ensure_capacity: V_alloc %ld->%ld, S_tot %ld->%ld\n",
                old_alloc, new_alloc, config_.S_tot, config_.H * new_alloc);
    }
    print_cuda_mem("ensure_capacity:before-expand");

    int64_t old_S_tot = config_.S_tot;
    expand_row_metadata(old_alloc, new_alloc);
    config_.V_alloc = new_alloc;
    config_.S_tot = config_.H * config_.V_alloc;

    if (use_seg_mode_ && seg_buf_cap_ > 0) {
        expand_seg_row_metadata(old_S_tot, config_.S_tot);

        // Expand voxel_zone_map to match new V_alloc (zero-fill new entries)
        if (voxel_zone_map_.defined() && voxel_zone_map_.size(0) < new_alloc) {
            auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32).device(device_);
            auto old_map = voxel_zone_map_;
            voxel_zone_map_ = torch::zeros({new_alloc}, opts_i32);
            voxel_zone_map_.narrow(0, 0, old_alloc).copy_(old_map);
        }
    }

    if (!use_seg_mode_) {
        if (config_.S_tot > buf_pool_cap_) {
            int64_t new_cap = std::min(config_.S_tot,
                                       std::max(buf_pool_cap_ * 3 / 2, buf_pool_cap_ + int64_t(8192)));
            if (merger_mem_profile_enabled()) {
                fprintf(stderr, "  [MEM-CUDA] expand_buf_pool: buf_pool_cap %ld->%ld\n",
                        buf_pool_cap_, new_cap);
                fflush(stderr);
            }
            expand_buf_pool_capacity(buf_pool_cap_, new_cap);
            buf_pool_cap_ = new_cap;
            config_.buf_pool_cap = buf_pool_cap_;
            buf_pool_growth_ = std::max(int64_t(8192), buf_pool_cap_ / 4);
        }
        if (config_.S_tot > piv_pool_cap_) {
            int64_t new_cap = std::min(config_.S_tot,
                                       std::max(piv_pool_cap_ * 3 / 2, piv_pool_cap_ + int64_t(8192)));
            if (merger_mem_profile_enabled()) {
                fprintf(stderr, "  [MEM-CUDA] expand_piv_pool: piv_pool_cap %ld->%ld\n",
                        piv_pool_cap_, new_cap);
                fflush(stderr);
            }
            expand_piv_pool_capacity(piv_pool_cap_, new_cap);
            piv_pool_cap_ = new_cap;
            config_.piv_pool_cap = piv_pool_cap_;
            piv_pool_growth_ = std::max(int64_t(8192), piv_pool_cap_ / 4);
        }
        config_.pool_cap = std::max(buf_pool_cap_, piv_pool_cap_);
    }

    cudaStreamSynchronize(stream);
    if (!use_seg_mode_) {
        update_views();
    }
    if (use_seg_mode_ && seg_buf_cap_ > 0) {
        update_seg_views();
    }
    print_cuda_mem("ensure_capacity:after-expand");
}

void MergerWrapper::ensure_pool_slots(cudaStream_t stream) {
    cudaStreamSynchronize(stream);
    int32_t buf_free = buf_free_top_.item<int32_t>();
    int32_t piv_free = piv_free_top_.item<int32_t>();
    bool did_grow = false;

    // --- Buffer pool growth ---
    if (buf_pool_cap_ < config_.S_tot && buf_free < buf_pool_growth_) {
        int64_t grow_by = std::max(buf_pool_growth_, buf_pool_cap_ / 4);
        int64_t new_cap = std::min(buf_pool_cap_ + grow_by, config_.S_tot);
        if (new_cap > buf_pool_cap_) {
            if (merger_mem_profile_enabled()) {
                fprintf(stderr, "  [MEM-CUDA] ensure_pool_slots: buf_free=%d < threshold=%ld, "
                        "buf_pool_cap %ld->%ld (S_tot=%ld)\n",
                        buf_free, buf_pool_growth_, buf_pool_cap_, new_cap, config_.S_tot);
                fflush(stderr);
            }
            expand_buf_pool_capacity(buf_pool_cap_, new_cap);
            buf_pool_cap_ = new_cap;
            config_.buf_pool_cap = buf_pool_cap_;
            buf_pool_growth_ = std::max(int64_t(8192), buf_pool_cap_ / 4);
            did_grow = true;
        }
    } else if (buf_pool_cap_ >= config_.S_tot && buf_free < buf_pool_growth_) {
        fprintf(stderr, "  [WARN] ensure_pool_slots: buf_pool_cap=%ld reached S_tot=%ld, "
                "but buf_free=%d < threshold=%ld. Cannot grow further.\n",
                buf_pool_cap_, config_.S_tot, buf_free, buf_pool_growth_);
        fflush(stderr);
    }

    // --- Pivot pool growth ---
    if (piv_pool_cap_ < config_.S_tot && piv_free < piv_pool_growth_) {
        int64_t grow_by = std::max(piv_pool_growth_, piv_pool_cap_ / 4);
        int64_t new_cap = std::min(piv_pool_cap_ + grow_by, config_.S_tot);
        if (new_cap > piv_pool_cap_) {
            if (merger_mem_profile_enabled()) {
                fprintf(stderr, "  [MEM-CUDA] ensure_pool_slots: piv_free=%d < threshold=%ld, "
                        "piv_pool_cap %ld->%ld (S_tot=%ld)\n",
                        piv_free, piv_pool_growth_, piv_pool_cap_, new_cap, config_.S_tot);
                fflush(stderr);
            }
            expand_piv_pool_capacity(piv_pool_cap_, new_cap);
            piv_pool_cap_ = new_cap;
            config_.piv_pool_cap = piv_pool_cap_;
            piv_pool_growth_ = std::max(int64_t(8192), piv_pool_cap_ / 4);
            did_grow = true;
        }
    } else if (piv_pool_cap_ >= config_.S_tot && piv_free < piv_pool_growth_) {
        fprintf(stderr, "  [WARN] ensure_pool_slots: piv_pool_cap=%ld reached S_tot=%ld, "
                "but piv_free=%d < threshold=%ld. Cannot grow further.\n",
                piv_pool_cap_, config_.S_tot, piv_free, piv_pool_growth_);
        fflush(stderr);
    }

    if (did_grow) {
        config_.pool_cap = std::max(buf_pool_cap_, piv_pool_cap_);
        update_views();
    }
}

void MergerWrapper::print_pool_stats(cudaStream_t stream) {
    if (!merger_mem_profile_enabled()) return;

    auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32).device(device_);
    auto stats_dev = torch::zeros({POOL_STATS_SIZE}, opts_i32);

    int64_t S_tot = config_.S_tot;
    int blocks = (S_tot + 255) / 256;
    collect_pool_stats_kernel<<<blocks, 256, 0, stream>>>(
        views_.buf_row_state, views_.buf_row_ptr, views_.buf_pool_M,
        views_.piv_row_state, views_.piv_row_ptr, views_.piv_pool_M,
        stats_dev.data_ptr<int32_t>(), S_tot, config_.B, config_.P);

    cudaStreamSynchronize(stream);
    auto stats_cpu = stats_dev.cpu();
    int32_t* s = stats_cpu.data_ptr<int32_t>();

    int32_t buf_free = buf_free_top_.item<int32_t>();
    int32_t piv_free = piv_free_top_.item<int32_t>();

    int P = (int)config_.P;
    int B = (int)config_.B;

    // buf active = AVAILABLE + FULL
    int32_t buf_active = s[1] + s[2];
    // piv active = AVAILABLE (pivots don't go to FULL in our state machine)
    int32_t piv_active = s[6];

    float buf_ext_util = buf_pool_cap_ > 0 ? 100.0f * buf_active / buf_pool_cap_ : 0.0f;
    float piv_ext_util = piv_pool_cap_ > 0 ? 100.0f * piv_active / piv_pool_cap_ : 0.0f;

    // Compute weighted average fill rates
    int64_t buf_total_tokens = 0, buf_total_slots = 0;
    for (int b = 0; b <= B && b <= 8; ++b) {
        buf_total_tokens += (int64_t)b * s[10 + b];
        buf_total_slots += s[10 + b];
    }
    float buf_avg_fill = buf_total_slots > 0
        ? (float)buf_total_tokens / (buf_total_slots * B) * 100.0f : 0.0f;

    int64_t piv_total_pivots = 0, piv_total_slots = 0;
    for (int p = 0; p <= P && p <= 4; ++p) {
        piv_total_pivots += (int64_t)p * s[20 + p];
        piv_total_slots += s[20 + p];
    }
    float piv_avg_fill = piv_total_slots > 0
        ? (float)piv_total_pivots / (piv_total_slots * P) * 100.0f : 0.0f;

    // Print in key=value format for easy parsing
    fprintf(stderr,
        "[POOL-STAT] buf_pool_cap=%ld piv_pool_cap=%ld S_tot=%ld"
        " buf_free=%d buf_R=%d buf_A=%d buf_F=%d buf_Fr=%d buf_H=%d"
        " buf_ext_util=%.1f%% buf_avg_fill=%.1f%%"
        " buf_fill=",
        buf_pool_cap_, piv_pool_cap_, S_tot,
        buf_free, s[0], s[1], s[2], s[3], s[4],
        buf_ext_util, buf_avg_fill);
    for (int b = 0; b <= B && b <= 8; ++b)
        fprintf(stderr, "%s%d", b ? "," : "", s[10 + b]);

    fprintf(stderr,
        " piv_free=%d piv_R=%d piv_A=%d piv_F=%d piv_Fr=%d piv_H=%d"
        " piv_ext_util=%.1f%% piv_avg_fill=%.1f%%"
        " piv_fill=",
        piv_free, s[5], s[6], s[7], s[8], s[9],
        piv_ext_util, piv_avg_fill);
    for (int p = 0; p <= P && p <= 4; ++p)
        fprintf(stderr, "%s%d", p ? "," : "", s[20 + p]);

    fprintf(stderr, "\n");
    fflush(stderr);
}

void MergerWrapper::print_seg_pool_stats(cudaStream_t stream) {
    if (!merger_mem_profile_enabled()) return;
    if (!use_seg_mode_ || seg_buf_cap_ == 0) return;

    auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32).device(device_);
    auto stats_dev = torch::zeros({SEG_POOL_STATS_SIZE}, opts_i32);

    int64_t S_tot = config_.S_tot;
    int blocks = (S_tot + 255) / 256;
    collect_seg_pool_stats_kernel<<<blocks, 256, 0, stream>>>(
        buf_row_state_.data_ptr<int8_t>(),
        buf_row_seg_.data_ptr<int32_t>(),
        piv_row_state_.data_ptr<int8_t>(),
        piv_row_seg_.data_ptr<int32_t>(),
        stats_dev.data_ptr<int32_t>(), S_tot, config_.B, config_.P);

    cudaStreamSynchronize(stream);
    auto stats_cpu = stats_dev.cpu();
    int32_t* s = stats_cpu.data_ptr<int32_t>();

    int32_t buf_free_top = seg_buf_free_top_.item<int32_t>();
    int32_t piv_free_top = (piv_zone_top_.defined() && config_.piv_num_zones > 0)
        ? piv_zone_top_.sum().item<int32_t>()
        : seg_piv_free_top_.item<int32_t>();

    int P = (int)config_.P;
    int B = (int)config_.B;

    int32_t buf_segs_used = s[25];
    int32_t piv_segs_used = s[26];

    int32_t buf_active_rows = s[1] + s[2];
    int32_t piv_active_rows = s[6];

    float buf_seg_util = seg_buf_cap_ > 0 ? 100.0f * buf_segs_used / seg_buf_cap_ : 0.0f;
    float piv_seg_util = seg_piv_cap_ > 0 ? 100.0f * piv_segs_used / seg_piv_cap_ : 0.0f;

    float buf_avg_fill = buf_active_rows > 0
        ? (float)buf_segs_used / (buf_active_rows * B) * 100.0f : 0.0f;
    float piv_avg_fill = piv_active_rows > 0
        ? (float)piv_segs_used / (piv_active_rows * P) * 100.0f : 0.0f;

    int elem_kv = (dtype_ == torch::kFloat16 || dtype_ == torch::kBFloat16) ? 2 : 4;
    double buf_pool_mb = (double)seg_buf_cap_ * (2.0 * config_.D * elem_kv + 4 + 4) / (1024.0 * 1024.0);
    double buf_indir_mb = (double)S_tot * B * 4.0 / (1024.0 * 1024.0);
    double piv_pool_mb = (double)seg_piv_cap_ * (3.0 * config_.D * elem_kv + 5 * 4.0) / (1024.0 * 1024.0);
    double piv_indir_mb = (double)S_tot * P * 4.0 / (1024.0 * 1024.0);

    fprintf(stderr,
        "[SEG-POOL-STAT] seg_buf_cap=%ld seg_piv_cap=%ld S_tot=%ld"
        " buf_free_segs=%d buf_used_segs=%d buf_R=%d buf_A=%d buf_F=%d buf_Fr=%d buf_H=%d"
        " buf_seg_util=%.1f%% buf_avg_fill=%.1f%%"
        " buf_fill=",
        seg_buf_cap_, seg_piv_cap_, S_tot,
        buf_free_top, buf_segs_used,
        s[0], s[1], s[2], s[3], s[4],
        buf_seg_util, buf_avg_fill);
    for (int b = 0; b <= B && b <= 8; ++b)
        fprintf(stderr, "%s%d", b ? "," : "", s[10 + b]);

    fprintf(stderr,
        " piv_free_segs=%d piv_used_segs=%d piv_R=%d piv_A=%d piv_F=%d piv_Fr=%d piv_H=%d"
        " piv_seg_util=%.1f%% piv_avg_fill=%.1f%%"
        " piv_fill=",
        piv_free_top, piv_segs_used,
        s[5], s[6], s[7], s[8], s[9],
        piv_seg_util, piv_avg_fill);
    for (int p = 0; p <= P && p <= 4; ++p)
        fprintf(stderr, "%s%d", p ? "," : "", s[20 + p]);

    fprintf(stderr,
        " buf_pool_MB=%.1f+%.1f piv_pool_MB=%.1f+%.1f total_seg_MB=%.1f\n",
        buf_pool_mb, buf_indir_mb, piv_pool_mb, piv_indir_mb,
        buf_pool_mb + buf_indir_mb + piv_pool_mb + piv_indir_mb);
    fflush(stderr);

    // Pivot zone utilization distribution
    if (piv_zone_top_.defined() && config_.piv_num_zones > 0) {
        int32_t Z = config_.piv_num_zones;
        int32_t zone_cap = config_.piv_zone_cap;
        auto zt_cpu = piv_zone_top_.cpu();
        int32_t* zt = zt_cpu.data_ptr<int32_t>();

        int32_t zones_empty = 0, zones_low = 0, zones_mid = 0, zones_high = 0, zones_full = 0;
        int32_t total_free = 0, min_free = zone_cap, max_free = 0;
        int32_t max_used_zone = -1;
        int32_t max_used_val = 0;
        for (int32_t z = 0; z < Z; z++) {
            int32_t free_slots = zt[z];
            int32_t used = zone_cap - free_slots;
            total_free += free_slots;
            if (free_slots < min_free) min_free = free_slots;
            if (free_slots > max_free) max_free = free_slots;
            if (used > max_used_val) { max_used_val = used; max_used_zone = z; }

            float usage_pct = zone_cap > 0 ? 100.0f * used / zone_cap : 0.0f;
            if (used == 0) zones_empty++;
            else if (usage_pct < 25.0f) zones_low++;
            else if (usage_pct < 75.0f) zones_mid++;
            else if (used < zone_cap) zones_high++;
            else zones_full++;
        }
        int32_t total_used = (int32_t)seg_piv_cap_ - total_free;
        int32_t zones_active = Z - zones_empty;
        float avg_used = zones_active > 0 ? (float)total_used / zones_active : 0.0f;

        fprintf(stderr,
            "[ZONE-UTIL] Z=%d zone_cap=%d | total_used=%d/%ld(%.1f%%)"
            " active_zones=%d/%d"
            " | empty=%d <25%%=%d 25-75%%=%d >75%%=%d full=%d"
            " | avg_used_per_active=%.1f max_zone=%d(used=%d)"
            " free_range=[%d,%d]\n",
            Z, zone_cap, total_used, seg_piv_cap_,
            100.0f * total_used / seg_piv_cap_,
            zones_active, Z,
            zones_empty, zones_low, zones_mid, zones_high, zones_full,
            avg_used, max_used_zone, max_used_val,
            min_free, max_free);
        fflush(stderr);
    }

    // Pivot C distribution stats
    auto pc_stats_dev = torch::zeros({PIV_C_STATS_SIZE}, opts_i32);
    collect_piv_c_stats_kernel<<<blocks, 256, 0, stream>>>(
        piv_row_state_.data_ptr<int8_t>(),
        piv_row_seg_.data_ptr<int32_t>(),
        seg_piv_C_.data_ptr<float>(),
        pc_stats_dev.data_ptr<int32_t>(), S_tot, P);
    cudaStreamSynchronize(stream);
    auto pc_cpu = pc_stats_dev.cpu();
    int32_t* pc = pc_cpu.data_ptr<int32_t>();

    int32_t active_rows = pc[0];
    int32_t total_pivots = pc[1];
    int32_t sum_total_c = pc[18];
    int32_t sum_max_c   = pc[19];
    int32_t multi_rows  = pc[20];
    int32_t skewed      = pc[16];
    int32_t balanced    = pc[17];

    float avg_c_per_row = active_rows > 0 ? (float)sum_total_c / active_rows : 0.0f;
    float avg_piv_per_row = active_rows > 0 ? (float)total_pivots / active_rows : 0.0f;
    float avg_c_per_piv = total_pivots > 0 ? (float)sum_total_c / total_pivots : 0.0f;
    float skew_pct = multi_rows > 0 ? 100.0f * skewed / multi_rows : 0.0f;
    float bal_pct  = multi_rows > 0 ? 100.0f * balanced / multi_rows : 0.0f;

    static const char* bkt_labels[] = {
        "<1", "1-5", "5-10", "10-20", "20-50", "50-100", "100-200", "200-500", "500-1k", "1k+"};
    // fprintf(stderr,
    //     "[PIV-C-STAT] active_rows=%d total_pivots=%d avg_piv_per_row=%.2f"
    //     " avg_C_per_row=%.1f avg_C_per_piv=%.1f"
    //     " multi_piv_rows=%d skewed(>80%%)=%d(%.1f%%) balanced(<=50%%)=%d(%.1f%%)"
    //     " piv_count=1:%d,2:%d,3:%d,4:%d"
    //     " total_C_hist=",
    //     active_rows, total_pivots, avg_piv_per_row,
    //     avg_c_per_row, avg_c_per_piv,
    //     multi_rows, skewed, skew_pct, balanced, bal_pct,
    //     pc[12], pc[13], pc[14], pc[15]);
    for (int i = 0; i < 10; ++i)
        fprintf(stderr, "%s%s:%d", i ? "," : "", bkt_labels[i], pc[2 + i]);
    fprintf(stderr, "\n");
    fflush(stderr);
}

void MergerWrapper::expand_row_metadata(int64_t old_alloc, int64_t new_alloc) {
    c10::cuda::CUDAGuard guard(device_);
    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    int64_t H = config_.H;
    int64_t new_S_tot = H * new_alloc;
    
    auto old_buf_state = buf_row_state_, old_buf_count = buf_row_count_;
    auto old_piv_state = piv_row_state_, old_piv_count = piv_row_count_;
    
    auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32).device(device_);
    auto opts_i8 = torch::TensorOptions().dtype(torch::kInt8).device(device_);
    
    buf_row_state_ = torch::zeros({new_S_tot}, opts_i8);
    buf_row_count_ = torch::zeros({new_S_tot}, opts_i32);
    piv_row_state_ = torch::zeros({new_S_tot}, opts_i8);
    piv_row_count_ = torch::zeros({new_S_tot}, opts_i32);
    
    int64_t total = H * old_alloc;
    if (total > 0) {
        int blk = (total + 255) / 256;
        remap_row_metadata_kernel_wrapper<int8_t><<<blk, 256, 0, stream>>>(
            old_buf_state.data_ptr<int8_t>(), buf_row_state_.data_ptr<int8_t>(), H, old_alloc, new_alloc, (int8_t)0);
        remap_row_metadata_kernel_wrapper<int32_t><<<blk, 256, 0, stream>>>(
            old_buf_count.data_ptr<int32_t>(), buf_row_count_.data_ptr<int32_t>(), H, old_alloc, new_alloc, 0);
        remap_row_metadata_kernel_wrapper<int8_t><<<blk, 256, 0, stream>>>(
            old_piv_state.data_ptr<int8_t>(), piv_row_state_.data_ptr<int8_t>(), H, old_alloc, new_alloc, (int8_t)0);
        remap_row_metadata_kernel_wrapper<int32_t><<<blk, 256, 0, stream>>>(
            old_piv_count.data_ptr<int32_t>(), piv_row_count_.data_ptr<int32_t>(), H, old_alloc, new_alloc, 0);
    }

    if (!use_seg_mode_) {
        auto old_buf_ptr = buf_row_ptr_;
        auto old_piv_ptr = piv_row_ptr_;
        buf_row_ptr_ = torch::full({new_S_tot}, -1, opts_i32);
        piv_row_ptr_ = torch::full({new_S_tot}, -1, opts_i32);
        if (total > 0) {
            int blk = (total + 255) / 256;
            remap_row_metadata_kernel_wrapper<int32_t><<<blk, 256, 0, stream>>>(
                old_buf_ptr.data_ptr<int32_t>(), buf_row_ptr_.data_ptr<int32_t>(), H, old_alloc, new_alloc, -1);
            remap_row_metadata_kernel_wrapper<int32_t><<<blk, 256, 0, stream>>>(
                old_piv_ptr.data_ptr<int32_t>(), piv_row_ptr_.data_ptr<int32_t>(), H, old_alloc, new_alloc, -1);
        }
    }
}

void MergerWrapper::expand_buf_pool_capacity(int64_t old_cap, int64_t new_cap) {
    c10::cuda::CUDAGuard guard(device_);
    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    int64_t D = config_.D, B = config_.B;

    print_cuda_mem("expand_buf_pool:before");
    
    auto opts_kv = torch::TensorOptions().dtype(dtype_).device(device_);
    auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(device_);
    auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32).device(device_);
    auto opts_u8 = torch::TensorOptions().dtype(torch::kUInt8).device(device_);
    
    cudaStreamSynchronize(stream);
    c10::cuda::CUDACachingAllocator::emptyCache();
    
    auto expand_tensor = [&](torch::Tensor& member, std::vector<int64_t> new_shape,
                             const torch::TensorOptions& opts) {
        torch::Tensor old_data = member;
        member = torch::zeros(new_shape, opts);
        member.slice(0, 0, old_cap).copy_(old_data, /*non_blocking=*/true);
        cudaStreamSynchronize(stream);
    };
    
    expand_tensor(buf_pool_K_, {new_cap, B, D}, opts_kv);
    expand_tensor(buf_pool_V_, {new_cap, B, D}, opts_kv);
    expand_tensor(buf_pool_S_, {new_cap, B}, opts_f32);
    expand_tensor(buf_pool_M_, {new_cap, B}, opts_u8);
    
    {
        int32_t buf_old_top = buf_free_top_.item<int32_t>();
        int32_t buf_new_top = buf_old_top + (int32_t)(new_cap - old_cap);
        torch::Tensor old_stack = buf_free_stack_;
        buf_free_stack_ = torch::empty({new_cap}, opts_i32);
        buf_free_stack_.slice(0, 0, buf_old_top).copy_(old_stack.slice(0, 0, buf_old_top));
        buf_free_stack_.slice(0, buf_old_top, buf_new_top).copy_(
            torch::arange(old_cap, new_cap, opts_i32));
        buf_free_top_ = torch::tensor({buf_new_top}, opts_i32);
        cudaStreamSynchronize(stream);
    }
    
    print_cuda_mem("expand_buf_pool:after");
}

void MergerWrapper::expand_piv_pool_capacity(int64_t old_cap, int64_t new_cap) {
    c10::cuda::CUDAGuard guard(device_);
    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    int64_t D = config_.D, P = config_.P;

    print_cuda_mem("expand_piv_pool:before");
    
    auto opts_kv = torch::TensorOptions().dtype(dtype_).device(device_);
    auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(device_);
    auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32).device(device_);
    auto opts_u8 = torch::TensorOptions().dtype(torch::kUInt8).device(device_);
    
    cudaStreamSynchronize(stream);
    c10::cuda::CUDACachingAllocator::emptyCache();
    
    auto expand_tensor = [&](torch::Tensor& member, std::vector<int64_t> new_shape,
                             const torch::TensorOptions& opts) {
        torch::Tensor old_data = member;
        member = torch::zeros(new_shape, opts);
        member.slice(0, 0, old_cap).copy_(old_data, /*non_blocking=*/true);
        cudaStreamSynchronize(stream);
    };
    
    expand_tensor(piv_pool_K_, {new_cap, P, D}, opts_kv);
    expand_tensor(piv_pool_V_, {new_cap, P, D}, opts_kv);
    expand_tensor(piv_pool_W_, {new_cap, P}, opts_f32);
    expand_tensor(piv_pool_S_, {new_cap, P}, opts_f32);
    expand_tensor(piv_pool_C_, {new_cap, P}, opts_f32);
    expand_tensor(piv_pool_M_, {new_cap, P}, opts_u8);
    expand_tensor(piv_pool_K_seed_, {new_cap, P, D}, opts_kv);
    expand_tensor(piv_pool_S_seed_, {new_cap, P}, opts_f32);
    
    {
        int32_t piv_old_top = piv_free_top_.item<int32_t>();
        int32_t piv_new_top = piv_old_top + (int32_t)(new_cap - old_cap);
        torch::Tensor old_stack = piv_free_stack_;
        piv_free_stack_ = torch::empty({new_cap}, opts_i32);
        piv_free_stack_.slice(0, 0, piv_old_top).copy_(old_stack.slice(0, 0, piv_old_top));
        piv_free_stack_.slice(0, piv_old_top, piv_new_top).copy_(
            torch::arange(old_cap, new_cap, opts_i32));
        piv_free_top_ = torch::tensor({piv_new_top}, opts_i32);
        cudaStreamSynchronize(stream);
    }
    
    print_cuda_mem("expand_piv_pool:after");
}

void MergerWrapper::grow_overflow_if_needed(int64_t needed, cudaStream_t stream) {
    if (needed <= overflow_cap_) return;
    int64_t new_cap = std::max(needed, overflow_cap_ * 2);
    if (merger_mem_profile_enabled()) {
        fprintf(stderr, "  [MEM-CUDA] grow_overflow: %ld -> %ld\n", overflow_cap_, new_cap);
    }
    print_cuda_mem("grow_overflow:before");
    int64_t D = config_.D;
    
    auto opts_kv = torch::TensorOptions().dtype(dtype_).device(device_);
    auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(device_);
    auto opts_i64 = torch::TensorOptions().dtype(torch::kInt64).device(device_);
    
    auto old_K = overflow_K_, old_V = overflow_V_;
    auto old_S = overflow_S_, old_rows = overflow_rows_;
    
    overflow_K_ = torch::empty({new_cap, D}, opts_kv);
    overflow_V_ = torch::empty({new_cap, D}, opts_kv);
    overflow_S_ = torch::empty({new_cap}, opts_f32);
    overflow_rows_ = torch::empty({new_cap}, opts_i64);
    
    // Copy existing overflow data (up to old_cap elements; actual count may be less)
    overflow_K_.slice(0, 0, overflow_cap_).copy_(old_K, /*non_blocking=*/true);
    overflow_V_.slice(0, 0, overflow_cap_).copy_(old_V, /*non_blocking=*/true);
    overflow_S_.slice(0, 0, overflow_cap_).copy_(old_S, /*non_blocking=*/true);
    overflow_rows_.slice(0, 0, overflow_cap_).copy_(old_rows, /*non_blocking=*/true);
    
    overflow_cap_ = new_cap;
    // NOTE: Do NOT update config_.overflow_max here. It stays at its initial
    // value for workspace sizing purposes. Only overflow_cap_ tracks the
    // actual carry-out buffer capacity.
    
    views_.overflow_K = overflow_K_.data_ptr();
    views_.overflow_V = overflow_V_.data_ptr();
    views_.overflow_S = overflow_S_.data_ptr<float>();
    views_.overflow_rows = overflow_rows_.data_ptr<int64_t>();
    print_cuda_mem("grow_overflow:after");
}

void MergerWrapper::ensure_workspace(int64_t E_max,
                                     bool skip_packed,
                                     bool skip_combined_kv,
                                     int64_t new_piv_cap) {
    bool mode_changed = (skip_packed != workspace_skip_packed_) ||
                        (skip_combined_kv != workspace_skip_combined_kv_) ||
                        (new_piv_cap != workspace_new_piv_cap_);

    // Shrink the workspace when E_max drops well below the high-water mark.
    // This reclaims memory after transient edge-count spikes.
    bool should_shrink = workspace_.defined() &&
                         (E_max < workspace_E_max_ / 2) &&
                         (workspace_.nbytes() > 256ULL * 1024 * 1024);

    if (E_max <= workspace_E_max_ && !mode_changed && !should_shrink) return;

    int64_t overflow_max_for_ws = skip_combined_kv ? 0 : config_.overflow_max;
    size_t req = backend::MergerWorkspace::required(
        E_max, E_max, config_.D, overflow_max_for_ws, config_.dtype, config_.B,
        skip_packed, skip_combined_kv, new_piv_cap);

    // Add 25% headroom to reduce future realloc frequency
    size_t alloc_size = req + req / 4;

    bool needs_alloc = ((size_t)workspace_.numel() < req) || mode_changed || should_shrink;
    if (merger_mem_profile_enabled()) {
        fprintf(stderr, "  [MEM-CUDA] ensure_workspace: E_max %ld->%ld, req=%.1fMB%s%s%s\n",
                workspace_E_max_, E_max, (double)req / (1024.0 * 1024.0),
                needs_alloc ? " (REALLOC)" : " (reuse)",
                should_shrink ? " (SHRINK)" : "",
                skip_packed ? " [slim]" : "");
    }
    if (needs_alloc) {
        print_cuda_mem("ensure_workspace:before-alloc");
        // Release old workspace FIRST so old + new don't coexist in GPU memory.
        // The workspace is purely temporary (no persistent data to preserve).
        workspace_ = torch::Tensor();
        workspace_ = torch::empty({(int64_t)alloc_size}, torch::TensorOptions().dtype(torch::kUInt8).device(device_));
        print_cuda_mem("ensure_workspace:after-alloc");
    }
    workspace_E_max_ = E_max;
    workspace_skip_packed_ = skip_packed;
    workspace_skip_combined_kv_ = skip_combined_kv;
    workspace_new_piv_cap_ = new_piv_cap;
}

backend::MergerWorkspace MergerWrapper::get_workspace(int64_t E_max,
                                                      bool skip_packed,
                                                      bool skip_combined_kv,
                                                      int64_t new_piv_cap) {
    ensure_workspace(E_max, skip_packed, skip_combined_kv, new_piv_cap);
    int64_t overflow_max_for_ws = skip_combined_kv ? 0 : config_.overflow_max;
    char* chunk = reinterpret_cast<char*>(workspace_.data_ptr());
    return backend::MergerWorkspace::fromChunk(
        chunk, E_max, E_max, config_.D, overflow_max_for_ws, config_.dtype, config_.B,
        skip_packed, skip_combined_kv, new_piv_cap);
}

void MergerWrapper::ensure_retrieve_workspace(int64_t /*Q*/, cudaStream_t /*stream*/) {
    // No-op: fused single-kernel retrieve needs no workspace.
}

void MergerWrapper::reset() {
    c10::cuda::CUDAGuard guard(device_);
    overflow_dropped_count_dev_.zero_();
    overflow_dropped_last_dev_.zero_();
    overflow_count_host_ = 0;
    backend::merger_reset(config_, views_, c10::cuda::getCurrentCUDAStream().stream());
}

bool MergerWrapper::has_overflow() const { return overflow_count_dev_.item<int32_t>() > 0; }
int64_t MergerWrapper::overflow_count() const { return overflow_count_dev_.item<int32_t>(); }
int64_t MergerWrapper::dropped_overflow_count() const { return overflow_dropped_count_dev_.item<int64_t>(); }
int64_t MergerWrapper::dropped_overflow_last() const { return overflow_dropped_last_dev_.item<int64_t>(); }
int64_t MergerWrapper::valid_pivot_count() const { return piv_pool_M_.sum().item<int64_t>(); }

int64_t MergerWrapper::workspace_bytes() const {
    return workspace_.defined() ? workspace_.nbytes() : 0;
}

std::map<std::string, int64_t> MergerWrapper::pool_stats() const {
    std::map<std::string, int64_t> result;

    // Actual token counts from per-row counters (GPU reductions).
    result["buf_data_count"] = buf_row_count_.sum().item<int64_t>();
    result["piv_data_count"] = piv_row_count_.sum().item<int64_t>();

    if (use_seg_mode_) {
        // Segmented mode: 1 token per segment slot.
        int32_t buf_free = seg_buf_free_top_.item<int32_t>();
        int32_t piv_free = (piv_zone_top_.defined() && config_.piv_num_zones > 0)
            ? piv_zone_top_.sum().item<int32_t>()
            : seg_piv_free_top_.item<int32_t>();

        result["buf_alloc_count"] = seg_buf_cap_;
        result["buf_used_slots"]  = seg_buf_cap_ - buf_free;
        result["piv_alloc_count"] = seg_piv_cap_;
        result["piv_used_slots"]  = seg_piv_cap_ - piv_free;
    } else {
        // Contiguous mode: B tokens per buffer slot, P tokens per pivot slot.
        int32_t buf_free = buf_free_top_.item<int32_t>();
        int32_t piv_free = piv_free_top_.item<int32_t>();

        result["buf_alloc_count"] = buf_pool_cap_ * config_.B;
        result["buf_used_slots"]  = buf_pool_cap_ - buf_free;
        result["piv_alloc_count"] = piv_pool_cap_ * config_.P;
        result["piv_used_slots"]  = piv_pool_cap_ - piv_free;
    }

    return result;
}

void MergerWrapper::push_diagnostic(const char* fmt, ...) {
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    if (n >= 0 && n < (int)sizeof(buf)) {
        diagnostic_messages_.emplace_back(buf);
    }
}

std::vector<std::string> MergerWrapper::take_diagnostics() {
    std::vector<std::string> out;
    out.swap(diagnostic_messages_);
    return out;
}

void MergerWrapper::insert_and_merge(
    torch::Tensor K_new, 
    torch::Tensor V_new, 
    torch::Tensor S_new, 
    torch::Tensor VX_new,
    int64_t num_voxels, 
    double sim_thresh, 
    double replace_thresh, 
    double score_thresh) 
{
    c10::cuda::CUDAGuard guard(device_);
    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    ensure_capacity(num_voxels);
    ensure_pool_slots(stream);  // Auto-grow pool if free slots are low
    config_.sim_thresh = (float)sim_thresh;
    config_.replace_thresh = (float)replace_thresh;
    config_.score_thresh = (float)score_thresh;
    
    int64_t H = config_.H, D = config_.D, P = config_.P, B = config_.B, V_alloc = config_.V_alloc;
    int64_t overflow_max = overflow_count_host_;  // Use actual count, not capacity
    overflow_dropped_last_dev_.zero_();
    
    // Handle VX shape and type - ensure int64 for detail::pack_valid_tokens
    torch::Tensor vx_flat = VX_new.dim() == 1 ? VX_new : VX_new.reshape({-1});
    if (vx_flat.scalar_type() != torch::kInt64) {
        vx_flat = vx_flat.to(torch::kInt64);
    }
    vx_flat = vx_flat.contiguous();
    int64_t E = vx_flat.size(0);
    bool vx_per_head = (VX_new.dim() == 2);
    
    if (E == 0) return;
    
    // Ensure input tensors are contiguous
    K_new = K_new.contiguous();
    V_new = V_new.contiguous();
    S_new = S_new.contiguous();
    
    int64_t E_combined_max = overflow_max + E;
    
    // Ensure workspace is large enough
    ensure_workspace(E_combined_max);
    backend::MergerWorkspace ws = get_workspace(E_combined_max);
    
    bool diag_on = merger_mem_profile_enabled();
    if (diag_on && ws.diag) {
        cudaMemsetAsync(ws.diag, 0, backend::DIAG_COUNT * sizeof(int32_t), stream);
    } else {
        ws.diag = nullptr;
    }
    
    // =========================================================================
    // Step 1: Pack valid tokens using detail::pack_valid_tokens
    // =========================================================================
    backend::detail::pack_valid_tokens(
        K_new.data_ptr(),
        V_new.data_ptr(),
        S_new.data_ptr<float>(),
        vx_flat.data_ptr<int64_t>(),
        E,
        D,
        H,
        V_alloc,
        num_voxels,
        vx_per_head,
        ws.packed_rows,
        ws.packed_K,
        ws.packed_V,
        ws.packed_S,
        ws.E_valid_dev,
        ws.packed_orig_idx,
        config_.dtype,
        stream);
    
    // =========================================================================
    // Step 2: Concat new tokens + overflow using device-side kernels
    // FIX: Order changed to match Hybrid (Python): new tokens FIRST, overflow SECOND
    // This ensures consistent token ordering when buffer_topb_update selects by score.
    // When scores are tied, different ordering can lead to different token retention,
    // which accumulates over frames causing accuracy drift.
    // =========================================================================
    // E_total = E_valid + overflow_count
    add_device_scalars_kernel<<<1, 1, 0, stream>>>(
        ws.E_valid_dev,
        views_.overflow_count,
        ws.E_total_dev);
    
    int blocks_combined = (E_combined_max + WRAPPER_THREADS_PER_BLOCK - 1) / WRAPPER_THREADS_PER_BLOCK;
    
    // FIX: Initialize combined_rows to -1 to ensure unwritten positions have sentinel values
    // This prevents group_by_row from processing uninitialized data (which may contain 0)
    // and incorrectly including row 0 in unique_rows when it shouldn't be there.
    fill_array_kernel<int64_t><<<blocks_combined, WRAPPER_THREADS_PER_BLOCK, 0, stream>>>(
        ws.combined_rows, static_cast<int64_t>(-1), E_combined_max);
    int blocks_combined_vec = (E_combined_max * D + WRAPPER_THREADS_PER_BLOCK - 1) / WRAPPER_THREADS_PER_BLOCK;
    
    // Concat order: NEW tokens first, then OVERFLOW (matches Hybrid Python implementation)
    if (config_.dtype == backend::DType::Float16) {
        concat_vectors_kernel<__half><<<blocks_combined_vec, WRAPPER_THREADS_PER_BLOCK, 0, stream>>>(
            reinterpret_cast<const __half*>(ws.packed_K),
            ws.E_valid_dev,
            reinterpret_cast<const __half*>(views_.overflow_K),
            views_.overflow_count,
            reinterpret_cast<__half*>(ws.combined_K),
            static_cast<int>(D), E, overflow_max);
        
        concat_vectors_kernel<__half><<<blocks_combined_vec, WRAPPER_THREADS_PER_BLOCK, 0, stream>>>(
            reinterpret_cast<const __half*>(ws.packed_V),
            ws.E_valid_dev,
            reinterpret_cast<const __half*>(views_.overflow_V),
            views_.overflow_count,
            reinterpret_cast<__half*>(ws.combined_V),
            static_cast<int>(D), E, overflow_max);
    } else {
        concat_vectors_kernel<__nv_bfloat16><<<blocks_combined_vec, WRAPPER_THREADS_PER_BLOCK, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(ws.packed_K),
            ws.E_valid_dev,
            reinterpret_cast<const __nv_bfloat16*>(views_.overflow_K),
            views_.overflow_count,
            reinterpret_cast<__nv_bfloat16*>(ws.combined_K),
            static_cast<int>(D), E, overflow_max);
        
        concat_vectors_kernel<__nv_bfloat16><<<blocks_combined_vec, WRAPPER_THREADS_PER_BLOCK, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(ws.packed_V),
            ws.E_valid_dev,
            reinterpret_cast<const __nv_bfloat16*>(views_.overflow_V),
            views_.overflow_count,
            reinterpret_cast<__nv_bfloat16*>(ws.combined_V),
            static_cast<int>(D), E, overflow_max);
    }
    
    concat_scalars_kernel<float><<<blocks_combined, WRAPPER_THREADS_PER_BLOCK, 0, stream>>>(
        ws.packed_S,
        ws.E_valid_dev,
        views_.overflow_S,
        views_.overflow_count,
        ws.combined_S,
        E, overflow_max);
    
    concat_scalars_kernel<int64_t><<<blocks_combined, WRAPPER_THREADS_PER_BLOCK, 0, stream>>>(
        ws.packed_rows,
        ws.E_valid_dev,
        views_.overflow_rows,
        views_.overflow_count,
        ws.combined_rows,
        E, overflow_max);
    
    // Build combined orig_idx for deterministic within-row ordering
    build_combined_orig_idx_kernel<<<blocks_combined, WRAPPER_THREADS_PER_BLOCK, 0, stream>>>(
        ws.packed_orig_idx,
        ws.E_valid_dev,
        views_.overflow_count,
        ws.combined_orig_idx,
        static_cast<int32_t>(E),
        E_combined_max);
    
    // Reset overflow count for this iteration
    zero_scalar_kernel<<<1, 1, 0, stream>>>(views_.overflow_count);
    
    
    // =========================================================================
    // Step 3: Group by row using detail::group_by_row
    // =========================================================================
    backend::detail::group_by_row(
        ws.combined_rows,
        ws.combined_K,
        ws.combined_V,
        ws.combined_S,
        E_combined_max,  // Use max size, kernel guards with device-side count
        D,
        ws.unique_rows,
        ws.row_offsets,
        ws.sorted_indices,
        ws.sorted_K,
        ws.sorted_V,
        ws.sorted_S,
        ws.G_dev,
        ws.cub_temp,
        ws.cub_temp_size,
        config_.dtype,
        stream,
        ws.combined_orig_idx);
    
    int64_t G_max = E_combined_max;
    
    // =========================================================================
    // Step 6 (before Step 4): One2one merge (CSR mode)
    // =========================================================================
    // Run one2one FIRST so we know per-row candidate counts. This allows
    // materialize_rows to skip rows with 0 candidates, eliminating
    // "empty-shell" buffer slots that waste pool memory.
    int32_t* cand_row_counts = ws.cand_row_offsets;
    
    backend::detail::one2one_merge(
        ws.unique_rows,
        ws.row_offsets,
        ws.sorted_K,
        ws.sorted_V,
        ws.sorted_S,
        ws.G_dev,
        views_.piv_pool_K,
        views_.piv_pool_V,
        views_.piv_pool_W,
        views_.piv_pool_S,
        views_.piv_pool_C,
        views_.piv_pool_K_seed,
        views_.piv_pool_M,
        views_.piv_row_ptr,
        views_.piv_row_state,
        ws.cand_K,
        ws.cand_V,
        ws.cand_S,
        ws.cand_rows,
        ws.cand_count_dev,
        config_.sim_thresh,
        config_.replace_thresh,
        config_.score_thresh,
        P,
        D,
        G_max,
        E_combined_max,
        config_.dtype,
        stream,
        cand_row_counts,
        ws.diag);
    
    // =========================================================================
    // Step 4 (after Step 6): Materialize buffer rows with candidate filter
    // =========================================================================
    // Only materialize rows that have cand_row_counts > 0. Rows where all
    // tokens were absorbed by pivots in one2one don't need a buffer slot.
    int32_t* buf_n_mat_dev = ws.cand_count_dev;  // Reuse workspace scalar
    backend::detail::materialize_rows(
        ws.unique_rows,
        ws.G_dev,
        views_.buf_row_ptr,
        views_.buf_row_state,
        views_.buf_row_count,
        views_.buf_pool_M,
        views_.buf_free_stack,
        views_.buf_free_top,
        B,
        G_max,
        buf_n_mat_dev,
        config_.S_tot,
        stream,
        cand_row_counts);
    
    // =========================================================================
    // Step 7b: Buffer top-B update (reads candidates from CSR layout)
    // Steps 6b (group_by_row) and 7a (re-materialize) are eliminated.
    // =========================================================================
    backend::detail::buffer_topb_update(
        ws.unique_rows,
        ws.row_offsets,
        ws.cand_K,
        ws.cand_V,
        ws.cand_S,
        ws.G_dev,
        views_.buf_row_ptr,
        views_.buf_row_state,
        views_.buf_row_count,
        views_.buf_pool_K,
        views_.buf_pool_V,
        views_.buf_pool_S,
        views_.buf_pool_M,
        ws.over_K,
        ws.over_V,
        ws.over_S,
        ws.over_rows,
        ws.over_count_dev,
        B,
        D,
        G_max,
        E_combined_max,
        config_.S_tot,
        config_.dtype,
        stream,
        cand_row_counts,
        ws.diag);
    
    
    // =========================================================================
    // Steps 8+8b+9+10 (fused): filter FULL → materialize pivot → all2one
    //                           merge → clean buffer, all in one kernel.
    // The fused kernel checks buf_row_state == FULL per-thread, so we can
    // always pass the superset (unique_rows / G_dev) without a host sync.
    // =========================================================================
    backend::detail::all2one_merge_fused(
        ws.unique_rows,
        ws.G_dev,
        views_.buf_pool_K,
        views_.buf_pool_V,
        views_.buf_pool_S,
        views_.buf_pool_M,
        views_.piv_pool_K,
        views_.piv_pool_V,
        views_.piv_pool_W,
        views_.piv_pool_S,
        views_.piv_pool_C,
        views_.piv_pool_K_seed,
        views_.piv_pool_S_seed,
        views_.piv_pool_M,
        views_.buf_row_ptr,
        views_.buf_row_state,
        views_.buf_row_count,
        views_.piv_row_ptr,
        views_.piv_row_state,
        views_.piv_row_count,
        views_.piv_free_stack,
        views_.piv_free_top,
        views_.buf_free_stack,
        views_.buf_free_top,
        B,
        P,
        D,
        G_max,
        config_.S_tot,
        config_.dtype,
        stream,
        ws.diag);
    
    // =========================================================================
    // Step 10.5: Overflow pivot absorption
    // =========================================================================
    copy_and_clamp_scalar_kernel<<<1, 1, 0, stream>>>(
        ws.over_count_dev,
        ws.over_count_dev,
        static_cast<int>(E_combined_max));
    
    int32_t over_count_host = 0;
    cudaMemcpyAsync(&over_count_host, ws.over_count_dev, sizeof(int32_t),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    bool ran_overflow_absorb = false;
    if (over_count_host > 0) {
        backend::detail::group_by_row(
            ws.over_rows, ws.over_K, ws.over_V, ws.over_S,
            over_count_host, D,
            ws.unique_rows, ws.row_offsets, ws.sorted_indices,
            ws.sorted_K, ws.sorted_V, ws.sorted_S, ws.G_dev,
            ws.cub_temp, ws.cub_temp_size,
            config_.dtype, stream, nullptr);
        
        int32_t* over_diag = ws.diag ? &ws.diag[backend::DIAG_OVER_O2O_ABSORBED] : nullptr;
        backend::detail::one2one_merge(
            ws.unique_rows, ws.row_offsets,
            ws.sorted_K, ws.sorted_V, ws.sorted_S, ws.G_dev,
            views_.piv_pool_K, views_.piv_pool_V, views_.piv_pool_W,
            views_.piv_pool_S, views_.piv_pool_C, views_.piv_pool_K_seed,
            views_.piv_pool_M, views_.piv_row_ptr, views_.piv_row_state,
            ws.cand_K, ws.cand_V, ws.cand_S, ws.cand_rows, ws.cand_count_dev,
            config_.sim_thresh, 10.0f, -1.0f,
            P, D, over_count_host, over_count_host,
            config_.dtype, stream, nullptr, over_diag);
        
        if (ws.diag) {
            subtract_device_scalars_kernel<<<1, 1, 0, stream>>>(
                ws.over_count_dev, ws.cand_count_dev,
                &ws.diag[backend::DIAG_OVER_ABSORBED]);
        }
        ran_overflow_absorb = true;
    }
    
    // =========================================================================
    // Step 10.5b: Second-pass overflow absorption (ping-pong cand → over)
    // =========================================================================
    bool ran_pass2 = false;
    if (ran_overflow_absorb) {
        int32_t pass1_cand_count = 0;
        cudaMemcpyAsync(&pass1_cand_count, ws.cand_count_dev, sizeof(int32_t),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        
        if (pass1_cand_count > 0) {
            backend::detail::group_by_row(
                ws.cand_rows, ws.cand_K, ws.cand_V, ws.cand_S,
                pass1_cand_count, D,
                ws.unique_rows, ws.row_offsets, ws.sorted_indices,
                ws.sorted_K, ws.sorted_V, ws.sorted_S, ws.G_dev,
                ws.cub_temp, ws.cub_temp_size,
                config_.dtype, stream, nullptr);
            
            backend::detail::one2one_merge(
                ws.unique_rows, ws.row_offsets,
                ws.sorted_K, ws.sorted_V, ws.sorted_S, ws.G_dev,
                views_.piv_pool_K, views_.piv_pool_V, views_.piv_pool_W,
                views_.piv_pool_S, views_.piv_pool_C, views_.piv_pool_K_seed,
                views_.piv_pool_M, views_.piv_row_ptr, views_.piv_row_state,
                ws.over_K, ws.over_V, ws.over_S, ws.over_rows, ws.over_count_dev,
                config_.sim_thresh, 10.0f, -1.0f,
                P, D, pass1_cand_count, pass1_cand_count,
                config_.dtype, stream, nullptr, nullptr);
            
            ran_pass2 = true;
        }
    }
    
    // =========================================================================
    // Step 11: Copy overflow to carry-out buffer (dynamic growth, no dropping)
    // =========================================================================
    // Pass2 output → over; pass1 only → cand; no absorb → over
    bool final_in_cand = ran_overflow_absorb && !ran_pass2;
    const int32_t* src_count_dev = final_in_cand
        ? ws.cand_count_dev : ws.over_count_dev;
    
    int32_t final_over_count = 0;
    cudaMemcpyAsync(&final_over_count, src_count_dev, sizeof(int32_t),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    if (final_over_count > overflow_cap_) {
        grow_overflow_if_needed(final_over_count, stream);
    }
    
    int32_t final_i32 = final_over_count;
    cudaMemcpyAsync(views_.overflow_count, &final_i32, sizeof(int32_t),
                    cudaMemcpyHostToDevice, stream);
    
    overflow_count_host_ = final_over_count;
    
    const void*    src_K    = final_in_cand ? ws.cand_K : ws.over_K;
    const void*    src_V    = final_in_cand ? ws.cand_V : ws.over_V;
    const float*   src_S    = final_in_cand ? ws.cand_S : ws.over_S;
    const int64_t* src_rows = final_in_cand ? ws.cand_rows : ws.over_rows;
    
    int64_t over_max = E_combined_max;
    int blocks_over = (over_max + WRAPPER_THREADS_PER_BLOCK - 1) / WRAPPER_THREADS_PER_BLOCK;
    int blocks_over_vec = (over_max * D + WRAPPER_THREADS_PER_BLOCK - 1) / WRAPPER_THREADS_PER_BLOCK;
    
    if (config_.dtype == backend::DType::Float16) {
        copy_vectors_guarded_kernel<__half><<<blocks_over_vec, WRAPPER_THREADS_PER_BLOCK, 0, stream>>>(
            reinterpret_cast<const __half*>(src_K),
            reinterpret_cast<__half*>(views_.overflow_K),
            views_.overflow_count,
            static_cast<int>(D), over_max);
        
        copy_vectors_guarded_kernel<__half><<<blocks_over_vec, WRAPPER_THREADS_PER_BLOCK, 0, stream>>>(
            reinterpret_cast<const __half*>(src_V),
            reinterpret_cast<__half*>(views_.overflow_V),
            views_.overflow_count,
            static_cast<int>(D), over_max);
    } else {
        copy_vectors_guarded_kernel<__nv_bfloat16><<<blocks_over_vec, WRAPPER_THREADS_PER_BLOCK, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(src_K),
            reinterpret_cast<__nv_bfloat16*>(views_.overflow_K),
            views_.overflow_count,
            static_cast<int>(D), over_max);
        
        copy_vectors_guarded_kernel<__nv_bfloat16><<<blocks_over_vec, WRAPPER_THREADS_PER_BLOCK, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(src_V),
            reinterpret_cast<__nv_bfloat16*>(views_.overflow_V),
            views_.overflow_count,
            static_cast<int>(D), over_max);
    }
    
    copy_scalars_guarded_kernel<float><<<blocks_over, WRAPPER_THREADS_PER_BLOCK, 0, stream>>>(
        src_S,
        views_.overflow_S,
        views_.overflow_count,
        over_max);
    
    copy_scalars_guarded_kernel<int64_t><<<blocks_over, WRAPPER_THREADS_PER_BLOCK, 0, stream>>>(
        src_rows,
        views_.overflow_rows,
        views_.overflow_count,
        over_max);
    
    if (diag_on && ws.diag) {
        int32_t diag_host[backend::DIAG_COUNT] = {};
        cudaMemcpyAsync(diag_host, ws.diag, backend::DIAG_COUNT * sizeof(int32_t),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        fprintf(stderr,
            "[DIAG] E_in=%ld | o2o: absorbed=%d reserved=%d no_pivot=%d low_sim=%d dropped=%d"
            " | buf: kept=%d over_noslot=%d over_excess=%d full_rows=%d"
            " | a2o: merged=%d piv=%d"
            " | over_absorbed=%d | overflow_out=%d\n",
            E_combined_max,
            diag_host[backend::DIAG_O2O_ABSORBED],
            diag_host[backend::DIAG_O2O_CAND_RESERVED],
            diag_host[backend::DIAG_O2O_CAND_NO_PIVOT],
            diag_host[backend::DIAG_O2O_CAND_LOW_SIM],
            diag_host[backend::DIAG_O2O_DROPPED],
            diag_host[backend::DIAG_BUF_KEPT],
            diag_host[backend::DIAG_BUF_OVER_NO_SLOT],
            diag_host[backend::DIAG_BUF_OVER_EXCESS],
            diag_host[backend::DIAG_BUF_ROWS_FULL],
            diag_host[backend::DIAG_A2O_ROWS_MERGED],
            diag_host[backend::DIAG_A2O_PIV_CREATED],
            diag_host[backend::DIAG_OVER_ABSORBED],
            final_over_count);
        fflush(stderr);
    }
    
    print_pool_stats(stream);
    update_views();
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
MergerWrapper::insert_and_merge_with_rows(
    torch::Tensor rows,
    torch::Tensor K_new,
    torch::Tensor V_new,
    torch::Tensor S_new,
    double sim_thresh,
    double replace_thresh,
    double score_thresh)
{
    c10::cuda::CUDAGuard guard(device_);
    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    
    config_.sim_thresh = (float)sim_thresh;
    config_.replace_thresh = (float)replace_thresh;
    config_.score_thresh = (float)score_thresh;
    
    int64_t H = config_.H, D = config_.D, P = config_.P, B = config_.B;
    
    // Validate inputs
    TORCH_CHECK(rows.is_cuda() && rows.dim() == 1, "rows must be 1D CUDA tensor");
    TORCH_CHECK(K_new.is_cuda() && K_new.dim() == 2, "K_new must be 2D CUDA tensor");
    TORCH_CHECK(V_new.is_cuda() && V_new.dim() == 2, "V_new must be 2D CUDA tensor");
    TORCH_CHECK(S_new.is_cuda() && S_new.dim() == 1, "S_new must be 1D CUDA tensor");
    
    int64_t E = rows.size(0);
    TORCH_CHECK(K_new.size(0) == E && K_new.size(1) == D, "K_new shape mismatch");
    TORCH_CHECK(V_new.size(0) == E && V_new.size(1) == D, "V_new shape mismatch");
    TORCH_CHECK(S_new.size(0) == E, "S_new shape mismatch");
    
    // Prepare output tensors for empty case
    auto opts_kv = torch::TensorOptions().dtype(dtype_).device(device_);
    auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(device_);
    auto opts_i64 = torch::TensorOptions().dtype(torch::kInt64).device(device_);
    
    if (E == 0) {
        return std::make_tuple(
            torch::empty({0, D}, opts_kv),
            torch::empty({0, D}, opts_kv),
            torch::empty({0}, opts_f32),
            torch::empty({0}, opts_i64));
    }
    
    // Ensure inputs are contiguous and correct dtype
    rows = rows.to(torch::kInt64).contiguous();
    K_new = K_new.to(dtype_).contiguous();
    V_new = V_new.to(dtype_).contiguous();
    S_new = S_new.to(torch::kFloat32).contiguous();
    
    // Auto-grow pool if free slots are low (before workspace allocation)
    ensure_pool_slots(stream);
    
    // Slim workspace: skip packed_* and combined_K/V/S/rows (input passed directly)
    int64_t npiv_cap = std::min(E, buf_pool_cap_ / B + 1024);
    if (merger_mem_profile_enabled()) {
        fprintf(stderr, "  [MEM-CUDA] insert_and_merge_with_rows: E=%ld, S_tot=%ld, buf_pool_cap=%ld, piv_pool_cap=%ld\n",
                E, config_.S_tot, buf_pool_cap_, piv_pool_cap_);
    }
    print_cuda_mem("pipeline:before-ensure-workspace");
    ensure_workspace(E, /*skip_packed=*/true, /*skip_combined_kv=*/true, npiv_cap);
    backend::MergerWorkspace ws = get_workspace(E, /*skip_packed=*/true, /*skip_combined_kv=*/true, npiv_cap);
    print_cuda_mem("pipeline:after-workspace");

    // Zero diagnostic counters when profiling is enabled
    bool diag_on = merger_mem_profile_enabled();
    if (diag_on && ws.diag) {
        cudaMemsetAsync(ws.diag, 0, backend::DIAG_COUNT * sizeof(int32_t), stream);
    } else {
        ws.diag = nullptr;
    }

    int64_t elem_size = (dtype_ == torch::kFloat16) ? sizeof(__half) : sizeof(__nv_bfloat16);

    // =========================================================================
    // Step 2.5: No copy needed — pass input tensors directly to group_by_row
    // =========================================================================
    int blocks_iota = (E + WRAPPER_THREADS_PER_BLOCK - 1) / WRAPPER_THREADS_PER_BLOCK;
    iota_kernel<<<blocks_iota, WRAPPER_THREADS_PER_BLOCK, 0, stream>>>(ws.combined_orig_idx, static_cast<int32_t>(E));
    
    // =========================================================================
    // Step 3: Group by row using detail::group_by_row (reads input directly)
    // =========================================================================
    backend::detail::group_by_row(
        rows.data_ptr<int64_t>(),
        K_new.data_ptr(),
        V_new.data_ptr(),
        S_new.data_ptr<float>(),
        E,
        D,
        ws.unique_rows,
        ws.row_offsets,
        ws.sorted_indices,
        ws.sorted_K,
        ws.sorted_V,
        ws.sorted_S,
        ws.G_dev,
        ws.cub_temp,
        ws.cub_temp_size,
        config_.dtype,
        stream,
        ws.combined_orig_idx);
    
    int64_t G_max = E;
    
    // =========================================================================
    // Step 6 (before Step 4): One2one merge (CSR mode)
    // =========================================================================
    int32_t* cand_row_counts = ws.cand_row_offsets;
    
    backend::detail::one2one_merge(
        ws.unique_rows,
        ws.row_offsets,
        ws.sorted_K,
        ws.sorted_V,
        ws.sorted_S,
        ws.G_dev,
        views_.piv_pool_K,
        views_.piv_pool_V,
        views_.piv_pool_W,
        views_.piv_pool_S,
        views_.piv_pool_C,
        views_.piv_pool_K_seed,
        views_.piv_pool_M,
        views_.piv_row_ptr,
        views_.piv_row_state,
        ws.cand_K,
        ws.cand_V,
        ws.cand_S,
        ws.cand_rows,
        ws.cand_count_dev,
        config_.sim_thresh,
        config_.replace_thresh,
        config_.score_thresh,
        P,
        D,
        G_max,
        E,
        config_.dtype,
        stream,
        cand_row_counts,
        ws.diag);
    
    // =========================================================================
    // Step 4 (after Step 6): Materialize buffer rows with candidate filter
    // =========================================================================
    int32_t* buf_n_mat_dev = ws.cand_count_dev;  // Reuse workspace scalar
    backend::detail::materialize_rows(
        ws.unique_rows,
        ws.G_dev,
        views_.buf_row_ptr,
        views_.buf_row_state,
        views_.buf_row_count,
        views_.buf_pool_M,
        views_.buf_free_stack,
        views_.buf_free_top,
        B,
        G_max,
        buf_n_mat_dev,
        config_.S_tot,
        stream,
        cand_row_counts);
    
    // =========================================================================
    // Step 7b: Buffer top-B update (reads candidates from CSR layout)
    // Steps 6b (group_by_row) and 7a (re-materialize) are eliminated.
    // =========================================================================
    backend::detail::buffer_topb_update(
        ws.unique_rows,
        ws.row_offsets,
        ws.cand_K,
        ws.cand_V,
        ws.cand_S,
        ws.G_dev,
        views_.buf_row_ptr,
        views_.buf_row_state,
        views_.buf_row_count,
        views_.buf_pool_K,
        views_.buf_pool_V,
        views_.buf_pool_S,
        views_.buf_pool_M,
        ws.over_K,
        ws.over_V,
        ws.over_S,
        ws.over_rows,
        ws.over_count_dev,
        B,
        D,
        G_max,
        E,
        config_.S_tot,
        config_.dtype,
        stream,
        cand_row_counts,
        ws.diag);
    
    // =========================================================================
    // Steps 8+8b+9+10 (fused): filter FULL → materialize pivot → all2one
    //                           merge → clean buffer, all in one kernel.
    // The fused kernel checks buf_row_state == FULL per-thread, so we can
    // always pass the superset (unique_rows / G_dev) without a host sync.
    // =========================================================================
    backend::detail::all2one_merge_fused(
        ws.unique_rows,
        ws.G_dev,
        views_.buf_pool_K,
        views_.buf_pool_V,
        views_.buf_pool_S,
        views_.buf_pool_M,
        views_.piv_pool_K,
        views_.piv_pool_V,
        views_.piv_pool_W,
        views_.piv_pool_S,
        views_.piv_pool_C,
        views_.piv_pool_K_seed,
        views_.piv_pool_S_seed,
        views_.piv_pool_M,
        views_.buf_row_ptr,
        views_.buf_row_state,
        views_.buf_row_count,
        views_.piv_row_ptr,
        views_.piv_row_state,
        views_.piv_row_count,
        views_.piv_free_stack,
        views_.piv_free_top,
        views_.buf_free_stack,
        views_.buf_free_top,
        B,
        P,
        D,
        G_max,
        config_.S_tot,
        config_.dtype,
        stream,
        ws.diag);
    
    // =========================================================================
    // Step 10.5: Overflow pivot absorption
    // =========================================================================
    // Clamp overflow count to E (workspace capacity), then read to host.
    copy_and_clamp_scalar_kernel<<<1, 1, 0, stream>>>(
        ws.over_count_dev,
        ws.over_count_dev,
        static_cast<int>(E));
    
    int32_t over_count_host = 0;
    cudaMemcpyAsync(&over_count_host, ws.over_count_dev, sizeof(int32_t),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    bool ran_overflow_absorb = false;
    if (over_count_host > 0) {
        backend::detail::group_by_row(
            ws.over_rows, ws.over_K, ws.over_V, ws.over_S,
            over_count_host, D,
            ws.unique_rows, ws.row_offsets, ws.sorted_indices,
            ws.sorted_K, ws.sorted_V, ws.sorted_S, ws.G_dev,
            ws.cub_temp, ws.cub_temp_size,
            config_.dtype, stream, nullptr);
        
        int32_t* over_diag = ws.diag ? &ws.diag[backend::DIAG_OVER_O2O_ABSORBED] : nullptr;
        backend::detail::one2one_merge(
            ws.unique_rows, ws.row_offsets,
            ws.sorted_K, ws.sorted_V, ws.sorted_S, ws.G_dev,
            views_.piv_pool_K, views_.piv_pool_V, views_.piv_pool_W,
            views_.piv_pool_S, views_.piv_pool_C, views_.piv_pool_K_seed,
            views_.piv_pool_M, views_.piv_row_ptr, views_.piv_row_state,
            ws.cand_K, ws.cand_V, ws.cand_S, ws.cand_rows, ws.cand_count_dev,
            config_.sim_thresh, 10.0f, -1.0f,
            P, D, over_count_host, over_count_host,
            config_.dtype, stream, nullptr, over_diag);
        
        if (ws.diag) {
            subtract_device_scalars_kernel<<<1, 1, 0, stream>>>(
                ws.over_count_dev, ws.cand_count_dev,
                &ws.diag[backend::DIAG_OVER_ABSORBED]);
        }
        ran_overflow_absorb = true;
    }
    
    // =========================================================================
    // Step 10.5b: Second-pass overflow absorption (ping-pong cand → over)
    // =========================================================================
    bool ran_pass2 = false;
    if (ran_overflow_absorb) {
        int32_t pass1_cand_count = 0;
        cudaMemcpyAsync(&pass1_cand_count, ws.cand_count_dev, sizeof(int32_t),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        
        if (pass1_cand_count > 0) {
            backend::detail::group_by_row(
                ws.cand_rows, ws.cand_K, ws.cand_V, ws.cand_S,
                pass1_cand_count, D,
                ws.unique_rows, ws.row_offsets, ws.sorted_indices,
                ws.sorted_K, ws.sorted_V, ws.sorted_S, ws.G_dev,
                ws.cub_temp, ws.cub_temp_size,
                config_.dtype, stream, nullptr);
            
            backend::detail::one2one_merge(
                ws.unique_rows, ws.row_offsets,
                ws.sorted_K, ws.sorted_V, ws.sorted_S, ws.G_dev,
                views_.piv_pool_K, views_.piv_pool_V, views_.piv_pool_W,
                views_.piv_pool_S, views_.piv_pool_C, views_.piv_pool_K_seed,
                views_.piv_pool_M, views_.piv_row_ptr, views_.piv_row_state,
                ws.over_K, ws.over_V, ws.over_S, ws.over_rows, ws.over_count_dev,
                config_.sim_thresh, 10.0f, -1.0f,
                P, D, pass1_cand_count, pass1_cand_count,
                config_.dtype, stream, nullptr, nullptr);
            
            ran_pass2 = true;
        }
    }
    
    // =========================================================================
    // Step 11: Extract overflow to output tensors
    // =========================================================================
    bool final_in_cand = ran_overflow_absorb && !ran_pass2;
    const int32_t* final_count_dev = final_in_cand
        ? ws.cand_count_dev : ws.over_count_dev;
    
    int32_t final_over_count = 0;
    int32_t diag_host[backend::DIAG_COUNT] = {};
    cudaMemcpyAsync(&final_over_count, final_count_dev, sizeof(int32_t),
                    cudaMemcpyDeviceToHost, stream);
    if (diag_on && ws.diag) {
        cudaMemcpyAsync(diag_host, ws.diag, backend::DIAG_COUNT * sizeof(int32_t),
                        cudaMemcpyDeviceToHost, stream);
    }
    cudaStreamSynchronize(stream);
    
    if (diag_on) {
        int total_in = (int)E;
        int o2o_absorbed   = diag_host[backend::DIAG_O2O_ABSORBED];
        int o2o_reserved   = diag_host[backend::DIAG_O2O_CAND_RESERVED];
        int o2o_no_pivot   = diag_host[backend::DIAG_O2O_CAND_NO_PIVOT];
        int o2o_low_sim    = diag_host[backend::DIAG_O2O_CAND_LOW_SIM];
        int o2o_dropped    = diag_host[backend::DIAG_O2O_DROPPED];
        int buf_kept       = diag_host[backend::DIAG_BUF_KEPT];
        int buf_over_noslot= diag_host[backend::DIAG_BUF_OVER_NO_SLOT];
        int buf_over_excess= diag_host[backend::DIAG_BUF_OVER_EXCESS];
        int buf_rows_full  = diag_host[backend::DIAG_BUF_ROWS_FULL];
        int a2o_merged     = diag_host[backend::DIAG_A2O_ROWS_MERGED];
        int a2o_piv        = diag_host[backend::DIAG_A2O_PIV_CREATED];
        int over_absorbed  = diag_host[backend::DIAG_OVER_ABSORBED];
        int ov_abs = diag_host[backend::DIAG_OVER_O2O_ABSORBED];
        int ov_res = diag_host[backend::DIAG_OVER_O2O_RESERVED];
        int ov_nop = diag_host[backend::DIAG_OVER_O2O_NO_PIVOT];
        int ov_low = diag_host[backend::DIAG_OVER_O2O_LOW_SIM];
        int ov_drp = diag_host[backend::DIAG_OVER_O2O_DROPPED];
        fprintf(stderr,
            "[DIAG] E_in=%d | o2o: absorbed=%d reserved=%d no_pivot=%d low_sim=%d dropped=%d"
            " | buf: kept=%d over_noslot=%d over_excess=%d full_rows=%d"
            " | a2o: merged=%d piv=%d"
            " | over_absorbed=%d | ov_detail: abs=%d res=%d nopiv=%d low=%d drop=%d"
            " | overflow_out=%d (pass2=%s)\n",
            total_in, o2o_absorbed, o2o_reserved, o2o_no_pivot, o2o_low_sim, o2o_dropped,
            buf_kept, buf_over_noslot, buf_over_excess, buf_rows_full,
            a2o_merged, a2o_piv, over_absorbed,
            ov_abs, ov_res, ov_nop, ov_low, ov_drp,
            final_over_count, ran_pass2 ? "yes" : "no");
        fflush(stderr);
    }
    
    print_pool_stats(stream);
    
    // Allocate output tensors
    torch::Tensor K_over_out = torch::empty({final_over_count, D}, opts_kv);
    torch::Tensor V_over_out = torch::empty({final_over_count, D}, opts_kv);
    torch::Tensor S_over_out = torch::empty({final_over_count}, opts_f32);
    torch::Tensor rows_over_out = torch::empty({final_over_count}, opts_i64);
    
    if (final_over_count > 0) {
        const void* src_K  = final_in_cand ? ws.cand_K : ws.over_K;
        const void* src_V  = final_in_cand ? ws.cand_V : ws.over_V;
        const float* src_S = final_in_cand ? ws.cand_S : ws.over_S;
        const int64_t* src_rows = final_in_cand ? ws.cand_rows : ws.over_rows;
        
        cudaMemcpyAsync(K_over_out.data_ptr(), src_K,
                        final_over_count * D * elem_size, cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(V_over_out.data_ptr(), src_V,
                        final_over_count * D * elem_size, cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(S_over_out.data_ptr<float>(), src_S,
                        final_over_count * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(rows_over_out.data_ptr<int64_t>(), src_rows,
                        final_over_count * sizeof(int64_t), cudaMemcpyDeviceToDevice, stream);
    }
    
    cudaStreamSynchronize(stream);
    
    return std::make_tuple(K_over_out, V_over_out, S_over_out, rows_over_out);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
MergerWrapper::retrieve(
    torch::Tensor voxel_ids,
    int64_t retrieve_size,
    int64_t used_voxel_limit) {
    c10::cuda::CUDAGuard guard(device_);
    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    
    TORCH_CHECK(voxel_ids.is_cuda() && voxel_ids.dim() == 1);
    
    int64_t Q = voxel_ids.size(0);
    int64_t H = config_.H;
    int64_t D = config_.D;
    int64_t P = config_.P;
    
    if (retrieve_size <= 0) retrieve_size = Q * P;
    
    auto opts_kv = torch::TensorOptions().dtype(dtype_).device(device_);
    auto opts_u8 = torch::TensorOptions().dtype(torch::kUInt8).device(device_);
    auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(device_);
    auto opts_i64 = torch::TensorOptions().dtype(torch::kInt64).device(device_);
    
    auto empty_out = [&](int64_t sz) {
        return std::make_tuple(
            torch::zeros({H, sz, D}, opts_kv),
            torch::zeros({H, sz, D}, opts_kv),
            torch::zeros({H, sz}, opts_u8),
            torch::full({H, sz}, -std::numeric_limits<float>::infinity(), opts_f32));
    };
    
    if (Q == 0) return empty_out(retrieve_size);
    
    torch::Tensor voxel_ids_i64 = voxel_ids;
    if (voxel_ids_i64.scalar_type() != torch::kInt64) {
        voxel_ids_i64 = voxel_ids_i64.to(torch::kInt64);
    }
    voxel_ids_i64 = voxel_ids_i64.contiguous();
    int64_t voxel_upper = used_voxel_limit;
    if (voxel_upper < 0) voxel_upper = config_.V_alloc;
    voxel_upper = std::min<int64_t>(voxel_upper, config_.V_alloc);
    auto valid_mask = (voxel_ids_i64 >= 0) & (voxel_ids_i64 < voxel_upper);
    voxel_ids_i64 = voxel_ids_i64.masked_select(valid_mask).contiguous();
    Q = voxel_ids_i64.size(0);
    if (Q == 0) return empty_out(retrieve_size);
    
    // Retrieve ALL Q*P pivots first (needed for correct W-based sorting)
    int64_t full_size = Q * P;
    
    auto K_full = torch::empty({H, full_size, D}, opts_kv);
    auto V_full = torch::empty({H, full_size, D}, opts_kv);
    auto M_full = torch::empty({H, full_size}, opts_u8);
    auto bias_full = torch::empty({H, full_size}, opts_f32);
    
    backend::RetrieveOutputs outputs;
    outputs.K_out = K_full.data_ptr();
    outputs.V_out = V_full.data_ptr();
    outputs.M_out = M_full.data_ptr<uint8_t>();
    outputs.bias_out = bias_full.data_ptr<float>();
    
    backend::merger_retrieve_fixed(
        config_, views_,
        voxel_ids_i64.data_ptr<int64_t>(), Q, full_size,
        outputs, stream);
    
    // Build W tensor [H, full_size] from pivot pool for weight-based sorting
    // (matches Python: torch.argsort(W_all_flat, dim=1, descending=True))
    auto heads_idx = torch::arange(H, opts_i64).unsqueeze(1);              // [H, 1]
    auto rows_2d = heads_idx * config_.V_alloc + voxel_ids_i64.unsqueeze(0); // [H, Q]
    auto rows_flat = rows_2d.reshape(-1);                                    // [H*Q]
    
    auto slots = piv_row_ptr_.index_select(0, rows_flat);                    // [H*Q] int32
    auto slot_valid = (slots >= 0);
    auto safe_slots = slots.clamp_min(0).to(torch::kInt64);
    
    auto W_gathered = piv_pool_W_.index_select(0, safe_slots);               // [H*Q, P]
    W_gathered.masked_fill_(~slot_valid.unsqueeze(1), 0.0f);
    auto W_full = W_gathered.reshape({H, full_size});                        // [H, Q*P]
    
    // Sort by W descending
    auto sorted_idx = std::get<1>(W_full.sort(/*dim=*/1, /*descending=*/true));
    
    // Reorder outputs by sorted indices
    auto idx_D = sorted_idx.unsqueeze(-1).expand({H, full_size, D});
    K_full = K_full.gather(1, idx_D);
    V_full = V_full.gather(1, idx_D);
    M_full = M_full.to(torch::kInt32).gather(1, sorted_idx).to(torch::kUInt8);
    bias_full = bias_full.gather(1, sorted_idx);
    
    // Truncate or pad to requested retrieve_size
    if (retrieve_size <= full_size) {
        return std::make_tuple(
            K_full.slice(1, 0, retrieve_size).contiguous(),
            V_full.slice(1, 0, retrieve_size).contiguous(),
            M_full.slice(1, 0, retrieve_size).contiguous(),
            bias_full.slice(1, 0, retrieve_size).contiguous());
    }
    auto K_out = torch::zeros({H, retrieve_size, D}, opts_kv);
    auto V_out = torch::zeros({H, retrieve_size, D}, opts_kv);
    auto M_out = torch::zeros({H, retrieve_size}, opts_u8);
    auto bias_out = torch::full({H, retrieve_size}, -std::numeric_limits<float>::infinity(), opts_f32);
    K_out.slice(1, 0, full_size).copy_(K_full);
    V_out.slice(1, 0, full_size).copy_(V_full);
    M_out.slice(1, 0, full_size).copy_(M_full);
    bias_out.slice(1, 0, full_size).copy_(bias_full);
    return std::make_tuple(K_out, V_out, M_out, bias_out);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
MergerWrapper::retrieve_buf(
    torch::Tensor voxel_ids,
    int64_t buf_retrieve_size,
    int64_t used_voxel_limit) {
    c10::cuda::CUDAGuard guard(device_);
    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    
    TORCH_CHECK(voxel_ids.is_cuda() && voxel_ids.dim() == 1);
    
    int64_t Q = voxel_ids.size(0);
    int64_t H = config_.H;
    int64_t D = config_.D;
    int64_t B = config_.B;
    
    // Default buf_retrieve_size = Q * B (all buffer slots from all queried voxels)
    if (buf_retrieve_size <= 0) buf_retrieve_size = Q * B;
    
    // Allocate output tensors (empty, not zeros -- kernel writes every position)
    auto opts_kv = torch::TensorOptions().dtype(dtype_).device(device_);
    auto opts_u8 = torch::TensorOptions().dtype(torch::kUInt8).device(device_);
    auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(device_);
    
    torch::Tensor K_out = torch::empty({H, buf_retrieve_size, D}, opts_kv);
    torch::Tensor V_out = torch::empty({H, buf_retrieve_size, D}, opts_kv);
    torch::Tensor M_out = torch::empty({H, buf_retrieve_size}, opts_u8);
    torch::Tensor bias_out = torch::empty({H, buf_retrieve_size}, opts_f32);
    
    // Early return if no queries
    if (Q == 0) {
        K_out.zero_();
        V_out.zero_();
        M_out.zero_();
        bias_out.fill_(-std::numeric_limits<float>::infinity());
        return std::make_tuple(K_out, V_out, M_out, bias_out);
    }
    
    // Ensure voxel_ids is int64 and contiguous
    torch::Tensor voxel_ids_i64 = voxel_ids;
    if (voxel_ids_i64.scalar_type() != torch::kInt64) {
        voxel_ids_i64 = voxel_ids_i64.to(torch::kInt64);
    }
    voxel_ids_i64 = voxel_ids_i64.contiguous();
    int64_t voxel_upper = used_voxel_limit;
    if (voxel_upper < 0) {
        voxel_upper = config_.V_alloc;
    }
    voxel_upper = std::min<int64_t>(voxel_upper, config_.V_alloc);
    auto valid_mask = (voxel_ids_i64 >= 0) & (voxel_ids_i64 < voxel_upper);
    voxel_ids_i64 = voxel_ids_i64.masked_select(valid_mask).contiguous();
    Q = voxel_ids_i64.size(0);
    if (Q == 0) {
        K_out.zero_();
        V_out.zero_();
        M_out.zero_();
        bias_out.fill_(-std::numeric_limits<float>::infinity());
        return std::make_tuple(K_out, V_out, M_out, bias_out);
    }
    
    // Setup output structure with raw pointers
    backend::RetrieveOutputs outputs;
    outputs.K_out = K_out.data_ptr();
    outputs.V_out = V_out.data_ptr();
    outputs.M_out = M_out.data_ptr<uint8_t>();
    outputs.bias_out = bias_out.data_ptr<float>();
    
    // Single-kernel retrieve for buffer pool
    backend::merger_retrieve_buf(
        config_,
        views_,
        voxel_ids_i64.data_ptr<int64_t>(),
        Q,
        buf_retrieve_size,
        outputs,
        stream);
    
    return std::make_tuple(K_out, V_out, M_out, bias_out);
}

std::tuple<bool, std::string> MergerWrapper::validate_pool_state(
    bool check_buffer,
    bool check_pivot) const
{
    c10::cuda::CUDAGuard guard(device_);
    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    
    std::string error_msg;
    bool all_valid = true;
    
    // Allocate device memory for validation results
    auto opts_result = torch::TensorOptions().dtype(torch::kUInt8).device(device_);
    torch::Tensor buf_result_tensor = torch::zeros({static_cast<int64_t>(sizeof(backend::detail::ValidationResult))}, opts_result);
    torch::Tensor piv_result_tensor = torch::zeros({static_cast<int64_t>(sizeof(backend::detail::ValidationResult))}, opts_result);
    
    auto* buf_result_dev = reinterpret_cast<backend::detail::ValidationResult*>(buf_result_tensor.data_ptr());
    auto* piv_result_dev = reinterpret_cast<backend::detail::ValidationResult*>(piv_result_tensor.data_ptr());
    
    // Validate buffer pool
    if (check_buffer) {
        backend::detail::validate_buffer_pool(
            views_.buf_row_ptr,
            views_.buf_row_state,
            views_.buf_row_count,
            views_.buf_pool_M,
            views_.buf_free_stack,
            views_.buf_free_top,
            buf_result_dev,
            config_.S_tot,
            buf_pool_cap_,
            config_.B,
            stream);
    }
    
    // Validate pivot pool
    if (check_pivot) {
        backend::detail::validate_pivot_pool(
            views_.piv_row_ptr,
            views_.piv_row_state,
            views_.piv_row_count,
            views_.piv_pool_M,
            views_.piv_pool_W,
            views_.piv_free_stack,
            views_.piv_free_top,
            piv_result_dev,
            config_.S_tot,
            piv_pool_cap_,
            config_.P,
            stream);
    }
    
    // Synchronize and copy results back to host
    cudaStreamSynchronize(stream);
    
    backend::detail::ValidationResult buf_result, piv_result;
    if (check_buffer) {
        cudaMemcpy(&buf_result, buf_result_dev, sizeof(backend::detail::ValidationResult), cudaMemcpyDeviceToHost);
    }
    if (check_pivot) {
        cudaMemcpy(&piv_result, piv_result_dev, sizeof(backend::detail::ValidationResult), cudaMemcpyDeviceToHost);
    }
    
    // Build error message
    if (check_buffer && buf_result.total_errors > 0) {
        all_valid = false;
        error_msg += "Buffer pool errors: ";
        if (buf_result.orphaned_slots > 0) {
            error_msg += std::to_string(buf_result.orphaned_slots) + " orphaned slots, ";
        }
        if (buf_result.invalid_row_ptrs > 0) {
            error_msg += std::to_string(buf_result.invalid_row_ptrs) + " invalid row pointers, ";
        }
        if (buf_result.state_count_mismatch > 0) {
            error_msg += std::to_string(buf_result.state_count_mismatch) + " state-count mismatches, ";
        }
        if (buf_result.state_mask_mismatch > 0) {
            error_msg += std::to_string(buf_result.state_mask_mismatch) + " state-mask mismatches, ";
        }
        if (buf_result.free_stack_errors > 0) {
            error_msg += std::to_string(buf_result.free_stack_errors) + " free stack errors, ";
        }
        error_msg += "total: " + std::to_string(buf_result.total_errors) + ". ";
    }
    
    if (check_pivot && piv_result.total_errors > 0) {
        all_valid = false;
        error_msg += "Pivot pool errors: ";
        if (piv_result.orphaned_slots > 0) {
            error_msg += std::to_string(piv_result.orphaned_slots) + " orphaned slots, ";
        }
        if (piv_result.invalid_row_ptrs > 0) {
            error_msg += std::to_string(piv_result.invalid_row_ptrs) + " invalid row pointers, ";
        }
        if (piv_result.state_count_mismatch > 0) {
            error_msg += std::to_string(piv_result.state_count_mismatch) + " state-count mismatches, ";
        }
        if (piv_result.state_mask_mismatch > 0) {
            error_msg += std::to_string(piv_result.state_mask_mismatch) + " state-mask mismatches, ";
        }
        if (piv_result.free_stack_errors > 0) {
            error_msg += std::to_string(piv_result.free_stack_errors) + " free stack errors, ";
        }
        error_msg += "total: " + std::to_string(piv_result.total_errors) + ". ";
    }
    
    if (all_valid) {
        error_msg = "Pool state validation passed.";
    }
    
    return std::make_tuple(all_valid, error_msg);
}

// =============================================================================
// =================== SEGMENTED POOL METHODS ================================
// =============================================================================

void MergerWrapper::set_seg_mode(bool enabled) {
    if (enabled && !use_seg_mode_) {
        use_seg_mode_ = true;
        if (seg_buf_cap_ == 0) {
            init_seg_pools();
        }
        update_seg_views();

        // Release contiguous pool data tensors (row_state/row_count are shared)
        buf_pool_K_ = torch::Tensor();
        buf_pool_V_ = torch::Tensor();
        buf_pool_S_ = torch::Tensor();
        buf_pool_M_ = torch::Tensor();
        buf_free_stack_ = torch::Tensor();
        buf_free_top_ = torch::Tensor();
        buf_row_ptr_ = torch::Tensor();
        piv_pool_K_ = torch::Tensor();
        piv_pool_V_ = torch::Tensor();
        piv_pool_W_ = torch::Tensor();
        piv_pool_S_ = torch::Tensor();
        piv_pool_C_ = torch::Tensor();
        piv_pool_M_ = torch::Tensor();
        piv_pool_K_seed_ = torch::Tensor();
        piv_pool_S_seed_ = torch::Tensor();
        piv_free_stack_ = torch::Tensor();
        piv_free_top_ = torch::Tensor();
        piv_row_ptr_ = torch::Tensor();
        buf_pool_cap_ = 0;
        piv_pool_cap_ = 0;
        config_.buf_pool_cap = 0;
        config_.piv_pool_cap = 0;
        config_.pool_cap = 0;

        if (merger_mem_profile_enabled()) {
            fprintf(stderr, "  [MEM-CUDA] set_seg_mode: released contiguous pool tensors\n");
            fflush(stderr);
        }
    }
    use_seg_mode_ = enabled;
}

void MergerWrapper::set_voxel_zones(torch::Tensor zones) {
    TORCH_CHECK(zones.is_cuda() && zones.dtype() == torch::kInt32,
                "set_voxel_zones: zones must be a CUDA int32 tensor");
    c10::cuda::CUDAGuard guard(device_);
    int64_t n = zones.size(0);
    int64_t V = config_.V_alloc;
    if (!voxel_zone_map_.defined() || voxel_zone_map_.size(0) < V) {
        auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32).device(device_);
        voxel_zone_map_ = torch::zeros({V}, opts_i32);
    }
    int64_t copy_len = std::min(n, V);
    int32_t Z = config_.piv_num_zones;
    voxel_zone_map_.narrow(0, 0, copy_len).copy_(
        zones.narrow(0, 0, copy_len).clamp(0, Z - 1));
    seg_views_.voxel_zone_map = voxel_zone_map_.data_ptr<int32_t>();
}

void MergerWrapper::init_seg_pools() {
    c10::cuda::CUDAGuard guard(device_);
    int64_t D = config_.D, P = config_.P, B = config_.B, S_tot = config_.S_tot;

    // Buffer: start with 2*S_tot, grows dynamically as before.
    seg_buf_cap_ = std::max(int64_t(8192), S_tot * 2);
    seg_buf_growth_ = std::max(int64_t(4096), seg_buf_cap_ / 4);

    // Pivot: configurable multiplier over theoretical max (S_tot * P).
    // Default 0.5x saves ~2 GB vs old 1.5x while leaving ~60% headroom over
    // observed peak usage.  Override: MERGER_PIV_POOL_MULT=0.75
    int32_t Z = config_.piv_num_zones;  // 512
    const char* piv_mult_env = std::getenv("MERGER_PIV_POOL_MULT");
    double piv_mult = piv_mult_env ? std::atof(piv_mult_env) : 0.5;
    seg_piv_cap_ = std::max(int64_t(16384), (int64_t)(S_tot * P * piv_mult));
    seg_piv_cap_ = ((seg_piv_cap_ + Z - 1) / Z) * Z;  // round up to multiple of Z
    seg_piv_growth_ = 0;  // no expansion for pivot pool
    config_.piv_zone_cap = (int32_t)(seg_piv_cap_ / Z);

    print_cuda_mem("init_seg_pools:before");
    if (merger_mem_profile_enabled()) {
        int elem_kv = (dtype_ == torch::kFloat16 || dtype_ == torch::kBFloat16) ? 2 : 4;
        double buf_mb = (double)seg_buf_cap_ * D * elem_kv * 2 / (1024.0 * 1024.0);
        double piv_mb = (double)seg_piv_cap_ * D * elem_kv * 4 / (1024.0 * 1024.0);
        double indir_mb = (double)S_tot * (B + P) * 4 / (1024.0 * 1024.0);
        fprintf(stderr, "  [MEM-CUDA] init_seg_pools: seg_buf_cap=%ld, seg_piv_cap=%ld, "
                "buf=%.1fMB, piv=%.1fMB, indirection=%.1fMB\n",
                seg_buf_cap_, seg_piv_cap_, buf_mb, piv_mb, indir_mb);
        fflush(stderr);
    }

    auto opts_kv = torch::TensorOptions().dtype(dtype_).device(device_);
    auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(device_);
    auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32).device(device_);
    auto opts_i8 = torch::TensorOptions().dtype(torch::kInt8).device(device_);

    // Buffer segments
    seg_buf_K_ = torch::zeros({seg_buf_cap_, D}, opts_kv);
    seg_buf_V_ = torch::zeros({seg_buf_cap_, D}, opts_kv);
    seg_buf_S_ = torch::zeros({seg_buf_cap_}, opts_f32);
    seg_buf_free_stack_ = torch::arange(seg_buf_cap_, opts_i32);
    seg_buf_free_top_ = torch::tensor({(int32_t)seg_buf_cap_}, opts_i32);
    buf_row_seg_ = torch::full({S_tot, B}, -1, opts_i32);

    // Pivot segments
    seg_piv_K_ = torch::zeros({seg_piv_cap_, D}, opts_kv);
    seg_piv_V_ = torch::zeros({seg_piv_cap_, D}, opts_kv);
    seg_piv_W_ = torch::zeros({seg_piv_cap_}, opts_f32);
    seg_piv_S_ = torch::zeros({seg_piv_cap_}, opts_f32);
    seg_piv_C_ = torch::zeros({seg_piv_cap_}, opts_f32);
    seg_piv_K_seed_ = torch::zeros({seg_piv_cap_, D}, opts_kv);
    seg_piv_S_seed_ = torch::zeros({seg_piv_cap_}, opts_f32);
    piv_row_seg_ = torch::full({S_tot, P}, -1, opts_i32);

    // Pivot zone-partitioned free stack:
    // Zone z owns sids [z*zone_cap, (z+1)*zone_cap).
    // Stack section for zone z: seg_piv_free_stack[z*zone_cap .. (z+1)*zone_cap - 1]
    int32_t zone_cap = config_.piv_zone_cap;
    seg_piv_free_stack_ = torch::empty({seg_piv_cap_}, opts_i32);
    for (int32_t z = 0; z < Z; z++) {
        seg_piv_free_stack_.narrow(0, (int64_t)z * zone_cap, zone_cap)
            .copy_(torch::arange((int64_t)z * zone_cap, (int64_t)(z + 1) * zone_cap, opts_i32));
    }
    piv_zone_top_ = torch::full({(int64_t)Z}, zone_cap, opts_i32);

    // Global top kept for backward compat (unused when zone mode is active)
    seg_piv_free_top_ = torch::tensor({(int32_t)seg_piv_cap_}, opts_i32);

    // Voxel zone map: default zone 0 until set_voxel_zones() is called
    voxel_zone_map_ = torch::zeros({config_.V_alloc}, opts_i32);

    // Reuse row_state and row_count from contiguous pool (already allocated)
    // They are buf_row_state_, buf_row_count_, piv_row_state_, piv_row_count_

    print_cuda_mem("init_seg_pools:after");
}

void MergerWrapper::update_seg_views() {
    auto& sv = seg_views_;
    sv.seg_buf_K = seg_buf_K_.data_ptr();
    sv.seg_buf_V = seg_buf_V_.data_ptr();
    sv.seg_buf_S = seg_buf_S_.data_ptr<float>();
    sv.buf_row_seg = buf_row_seg_.data_ptr<int32_t>();
    sv.buf_row_count = buf_row_count_.data_ptr<int32_t>();
    sv.buf_row_state = buf_row_state_.data_ptr<int8_t>();
    sv.seg_buf_free_stack = seg_buf_free_stack_.data_ptr<int32_t>();
    sv.seg_buf_free_top = seg_buf_free_top_.data_ptr<int32_t>();

    sv.seg_piv_K = seg_piv_K_.data_ptr();
    sv.seg_piv_V = seg_piv_V_.data_ptr();
    sv.seg_piv_W = seg_piv_W_.data_ptr<float>();
    sv.seg_piv_S = seg_piv_S_.data_ptr<float>();
    sv.seg_piv_C = seg_piv_C_.data_ptr<float>();
    sv.seg_piv_K_seed = seg_piv_K_seed_.data_ptr();
    sv.seg_piv_S_seed = seg_piv_S_seed_.data_ptr<float>();
    sv.piv_row_seg = piv_row_seg_.data_ptr<int32_t>();
    sv.piv_row_count = piv_row_count_.data_ptr<int32_t>();
    sv.piv_row_state = piv_row_state_.data_ptr<int8_t>();
    sv.seg_piv_free_stack = seg_piv_free_stack_.data_ptr<int32_t>();
    sv.seg_piv_free_top = seg_piv_free_top_.data_ptr<int32_t>();

    sv.seg_buf_cap = seg_buf_cap_;
    sv.seg_piv_cap = seg_piv_cap_;

    // Pivot zone allocator views
    if (piv_zone_top_.defined() && voxel_zone_map_.defined()) {
        sv.voxel_zone_map = voxel_zone_map_.data_ptr<int32_t>();
        sv.piv_zone_top = piv_zone_top_.data_ptr<int32_t>();
        sv.piv_num_zones = config_.piv_num_zones;
        sv.piv_zone_cap = config_.piv_zone_cap;
    } else {
        sv.voxel_zone_map = nullptr;
        sv.piv_zone_top = nullptr;
        sv.piv_num_zones = 0;
        sv.piv_zone_cap = 0;
    }
}

void MergerWrapper::expand_seg_buf_pool(int64_t old_cap, int64_t new_cap) {
    c10::cuda::CUDAGuard guard(device_);
    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    int64_t D = config_.D;

    auto opts_kv = torch::TensorOptions().dtype(dtype_).device(device_);
    auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(device_);
    auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32).device(device_);

    // Sequential replacement to avoid peak memory
    auto replace = [&](torch::Tensor& old_t, std::vector<int64_t> shape) {
        auto new_t = torch::zeros(shape, old_t.options());
        new_t.narrow(0, 0, old_cap).copy_(old_t.narrow(0, 0, old_cap));
        old_t = new_t;
    };
    replace(seg_buf_K_, {new_cap, D});
    cudaStreamSynchronize(stream);
    replace(seg_buf_V_, {new_cap, D});
    cudaStreamSynchronize(stream);
    replace(seg_buf_S_, {new_cap});

    // Expand free stack: new entries [old_cap..new_cap)
    auto new_stack = torch::empty({new_cap}, opts_i32);
    new_stack.narrow(0, 0, old_cap).copy_(seg_buf_free_stack_);
    seg_buf_free_stack_ = new_stack;

    // Push new slots onto free stack
    int32_t old_top_h = 0;
    cudaMemcpyAsync(&old_top_h, seg_buf_free_top_.data_ptr<int32_t>(),
                    sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    int64_t n_new = new_cap - old_cap;
    auto new_ids = torch::arange(old_cap, new_cap, opts_i32);
    seg_buf_free_stack_.narrow(0, old_top_h, n_new).copy_(new_ids);
    int32_t new_top = old_top_h + (int32_t)n_new;
    cudaMemcpyAsync(seg_buf_free_top_.data_ptr<int32_t>(), &new_top,
                    sizeof(int32_t), cudaMemcpyHostToDevice, stream);

    seg_buf_cap_ = new_cap;
}

void MergerWrapper::expand_seg_piv_pool(int64_t old_cap, int64_t new_cap) {
    c10::cuda::CUDAGuard guard(device_);
    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    int64_t D = config_.D;

    auto opts_kv = torch::TensorOptions().dtype(dtype_).device(device_);
    auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(device_);
    auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32).device(device_);

    auto replace = [&](torch::Tensor& old_t, std::vector<int64_t> shape) {
        auto new_t = torch::zeros(shape, old_t.options());
        new_t.narrow(0, 0, old_cap).copy_(old_t.narrow(0, 0, old_cap));
        old_t = new_t;
    };
    replace(seg_piv_K_, {new_cap, D});
    cudaStreamSynchronize(stream);
    replace(seg_piv_V_, {new_cap, D});
    cudaStreamSynchronize(stream);
    replace(seg_piv_W_, {new_cap});
    replace(seg_piv_S_, {new_cap});
    replace(seg_piv_C_, {new_cap});
    replace(seg_piv_K_seed_, {new_cap, D});
    cudaStreamSynchronize(stream);
    replace(seg_piv_S_seed_, {new_cap});

    auto new_stack = torch::empty({new_cap}, opts_i32);
    new_stack.narrow(0, 0, old_cap).copy_(seg_piv_free_stack_);
    seg_piv_free_stack_ = new_stack;

    int32_t old_top_h = 0;
    cudaMemcpyAsync(&old_top_h, seg_piv_free_top_.data_ptr<int32_t>(),
                    sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    int64_t n_new = new_cap - old_cap;
    auto new_ids = torch::arange(old_cap, new_cap, opts_i32);
    seg_piv_free_stack_.narrow(0, old_top_h, n_new).copy_(new_ids);
    int32_t new_top = old_top_h + (int32_t)n_new;
    cudaMemcpyAsync(seg_piv_free_top_.data_ptr<int32_t>(), &new_top,
                    sizeof(int32_t), cudaMemcpyHostToDevice, stream);

    seg_piv_cap_ = new_cap;
}

void MergerWrapper::expand_seg_row_metadata(int64_t old_S_tot, int64_t new_S_tot) {
    c10::cuda::CUDAGuard guard(device_);
    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    int64_t H = config_.H;
    int64_t B = config_.B, P = config_.P;
    int64_t old_alloc = old_S_tot / H;
    int64_t new_alloc = new_S_tot / H;
    auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32).device(device_);

    // buf_row_seg: [S_tot, B] — remap with V_alloc stride change
    {
        auto old_buf_rs = buf_row_seg_;
        buf_row_seg_ = torch::full({new_S_tot, B}, -1, opts_i32);
        if (old_S_tot > 0) {
            int64_t total = H * old_alloc * B;
            int blk = (total + 255) / 256;
            remap_seg_row_metadata_kernel<int32_t><<<blk, 256, 0, stream>>>(
                old_buf_rs.data_ptr<int32_t>(),
                buf_row_seg_.data_ptr<int32_t>(),
                H, old_alloc, new_alloc, B);
        }
    }

    // piv_row_seg: [S_tot, P] — same remapping
    {
        auto old_piv_rs = piv_row_seg_;
        piv_row_seg_ = torch::full({new_S_tot, P}, -1, opts_i32);
        if (old_S_tot > 0) {
            int64_t total = H * old_alloc * P;
            int blk = (total + 255) / 256;
            remap_seg_row_metadata_kernel<int32_t><<<blk, 256, 0, stream>>>(
                old_piv_rs.data_ptr<int32_t>(),
                piv_row_seg_.data_ptr<int32_t>(),
                H, old_alloc, new_alloc, P);
        }
    }
}

void MergerWrapper::ensure_seg_pool_slots(cudaStream_t stream) {
    // Check buffer segments
    int32_t buf_top_h = 0;
    cudaMemcpyAsync(&buf_top_h, seg_buf_free_top_.data_ptr<int32_t>(),
                    sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    int32_t piv_top_h = 0;
    cudaMemcpyAsync(&piv_top_h, seg_piv_free_top_.data_ptr<int32_t>(),
                    sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    int64_t buf_threshold = std::max(seg_buf_growth_, int64_t(2048));
    if (buf_top_h < buf_threshold) {
        int64_t new_cap = seg_buf_cap_ + seg_buf_growth_;
        push_diagnostic("  [SEG-EXPAND] buf_pool: %ld->%ld (free=%d, threshold=%ld)",
                        seg_buf_cap_, new_cap, buf_top_h, buf_threshold);
        print_cuda_mem("seg_expand_buf:before");
        expand_seg_buf_pool(seg_buf_cap_, new_cap);
        update_seg_views();
        print_cuda_mem("seg_expand_buf:after");
    }

    // Pivot pool: zone-mode, no expansion. Warn if critically low.
    if (piv_zone_top_.defined()) {
        int32_t piv_total_free = piv_zone_top_.sum().item<int32_t>();
        int64_t piv_threshold = std::max(int64_t(2048), seg_piv_cap_ / 8);
        if (piv_total_free < piv_threshold) {
            push_diagnostic("  [SEG-WARN] piv_zone_pool critically low: %d free of %ld (threshold=%ld). Increase init piv_pool_cap or reduce voxel count.",
                            piv_total_free, seg_piv_cap_, piv_threshold);
        }
    } else {
        int64_t piv_threshold = std::max(seg_piv_growth_, int64_t(2048));
        if (piv_top_h < piv_threshold) {
            int64_t new_cap = seg_piv_cap_ + seg_piv_growth_;
            push_diagnostic("  [SEG-EXPAND] piv_pool: %ld->%ld (free=%d, threshold=%ld)",
                            seg_piv_cap_, new_cap, piv_top_h, piv_threshold);
            print_cuda_mem("seg_expand_piv:before");
            expand_seg_piv_pool(seg_piv_cap_, new_cap);
            update_seg_views();
            print_cuda_mem("seg_expand_piv:after");
        }
    }
}

// =============================================================================
// insert_and_merge_with_rows_seg: Same pipeline as contiguous, using _seg kernels
// =============================================================================
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
MergerWrapper::insert_and_merge_with_rows_seg(
    torch::Tensor rows, torch::Tensor K_new, torch::Tensor V_new, torch::Tensor S_new,
    double sim_thresh, double replace_thresh, double score_thresh)
{
    c10::cuda::CUDAGuard guard(device_);
    auto stream = c10::cuda::getCurrentCUDAStream().stream();

    config_.sim_thresh = (float)sim_thresh;
    config_.replace_thresh = (float)replace_thresh;
    config_.score_thresh = (float)score_thresh;

    int64_t H = config_.H, D = config_.D, P = config_.P, B = config_.B;

    TORCH_CHECK(rows.is_cuda() && rows.dim() == 1);
    TORCH_CHECK(K_new.is_cuda() && K_new.dim() == 2 && K_new.size(1) == D);
    TORCH_CHECK(V_new.is_cuda() && V_new.dim() == 2 && V_new.size(1) == D);
    TORCH_CHECK(S_new.is_cuda() && S_new.dim() == 1);

    int64_t E = rows.size(0);
    auto opts_kv = torch::TensorOptions().dtype(dtype_).device(device_);
    auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(device_);
    auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32).device(device_);
    auto opts_i64 = torch::TensorOptions().dtype(torch::kInt64).device(device_);

    if (E == 0) {
        return std::make_tuple(
            torch::empty({0, D}, opts_kv), torch::empty({0, D}, opts_kv),
            torch::empty({0}, opts_f32), torch::empty({0}, opts_i64));
    }

    rows = rows.to(torch::kInt64).contiguous();
    K_new = K_new.to(dtype_).contiguous();
    V_new = V_new.to(dtype_).contiguous();
    S_new = S_new.to(torch::kFloat32).contiguous();

    // Ensure seg pool initialized
    if (seg_buf_cap_ == 0) {
        init_seg_pools();
        update_seg_views();
    }

    print_cuda_mem("seg_pipeline:enter");
    ensure_seg_pool_slots(stream);
    int64_t npiv_cap = std::min(E, seg_buf_cap_ / B + 1024);
    ensure_workspace(E, /*skip_packed=*/true, /*skip_combined_kv=*/true, npiv_cap);
    backend::MergerWorkspace ws = get_workspace(E, /*skip_packed=*/true, /*skip_combined_kv=*/true, npiv_cap);
    print_cuda_mem("seg_pipeline:after-workspace");

    bool diag_on = merger_mem_profile_enabled();
    if (diag_on && ws.diag) {
        cudaMemsetAsync(ws.diag, 0, backend::DIAG_COUNT * sizeof(int32_t), stream);
    } else {
        ws.diag = nullptr;
    }

    // Helper: check for CUDA errors after each kernel (when CUDA_LAUNCH_BLOCKING=1)
    auto check_cuda = [&](const char* label) {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "[SEG-CRASH] %s FAILED: %s (E=%ld, S_tot=%ld)\n",
                    label, cudaGetErrorString(err), E, config_.S_tot);
            fflush(stderr);
            TORCH_CHECK(false, "CUDA kernel failed at ", label, ": ", cudaGetErrorString(err));
        }
        if (merger_mem_profile_enabled()) {
            cudaStreamSynchronize(stream);
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                fprintf(stderr, "[SEG-CRASH] %s sync FAILED: %s\n", label, cudaGetErrorString(err));
                fflush(stderr);
                TORCH_CHECK(false, "CUDA sync failed at ", label, ": ", cudaGetErrorString(err));
            }
        }
    };

    // Validate input rows
    if (merger_mem_profile_enabled()) {
        auto rows_cpu = rows.to(torch::kCPU);
        int64_t* rp = rows_cpu.data_ptr<int64_t>();
        int64_t rmin = rp[0], rmax = rp[0];
        for (int64_t i = 1; i < E; ++i) { rmin = std::min(rmin, rp[i]); rmax = std::max(rmax, rp[i]); }
        fprintf(stderr, "[SEG-CHECK] E=%ld rows_min=%ld rows_max=%ld S_tot=%ld ws_E_max=%ld npiv_cap=%ld\n",
                E, rmin, rmax, config_.S_tot, workspace_E_max_, npiv_cap);
        fflush(stderr);
        if (rmax >= config_.S_tot || rmin < 0) {
            fprintf(stderr, "[SEG-CRASH] INPUT ROWS OUT OF BOUNDS: min=%ld max=%ld S_tot=%ld\n",
                    rmin, rmax, config_.S_tot);
            fflush(stderr);
            TORCH_CHECK(false, "Input rows out of bounds: max=", rmax, " >= S_tot=", config_.S_tot);
        }
    }

    // Step 2.5: No copy needed — pass input tensors directly
    int64_t elem_size = (dtype_ == torch::kFloat16) ? sizeof(__half) : sizeof(__nv_bfloat16);
    int blocks_iota = (E + WRAPPER_THREADS_PER_BLOCK - 1) / WRAPPER_THREADS_PER_BLOCK;
    iota_kernel<<<blocks_iota, WRAPPER_THREADS_PER_BLOCK, 0, stream>>>(ws.combined_orig_idx, static_cast<int32_t>(E));
    check_cuda("iota_kernel");

    // Step 3: Group by row (reads input tensors directly)
    backend::detail::group_by_row(
        rows.data_ptr<int64_t>(), K_new.data_ptr(), V_new.data_ptr(), S_new.data_ptr<float>(),
        E, D, ws.unique_rows, ws.row_offsets, ws.sorted_indices,
        ws.sorted_K, ws.sorted_V, ws.sorted_S, ws.G_dev,
        ws.cub_temp, ws.cub_temp_size, config_.dtype, stream, ws.combined_orig_idx);
    check_cuda("group_by_row");

    int64_t G_max = E;
    auto& sv = seg_views_;

    // Step 6: One2one merge (seg version, CSR mode)
    int32_t* cand_row_counts = ws.cand_row_offsets;
    backend::detail::one2one_merge_seg(
        ws.unique_rows, ws.row_offsets, ws.sorted_K, ws.sorted_V, ws.sorted_S, ws.G_dev,
        sv.seg_piv_K, sv.seg_piv_V, sv.seg_piv_W, sv.seg_piv_S, sv.seg_piv_C,
        sv.seg_piv_K_seed, sv.piv_row_seg, sv.piv_row_state, sv.piv_row_count,
        ws.cand_K, ws.cand_V, ws.cand_S, ws.cand_rows, ws.cand_count_dev,
        config_.sim_thresh, config_.replace_thresh, config_.score_thresh,
        P, D, G_max, E, config_.dtype, stream, cand_row_counts, ws.diag);
    check_cuda("one2one_merge_seg");

    // Step 4: Materialize buffer rows (seg version)
    backend::detail::materialize_rows_seg(
        ws.unique_rows, ws.G_dev, sv.buf_row_seg, sv.buf_row_state, sv.buf_row_count,
        sv.seg_buf_free_stack, sv.seg_buf_free_top,
        B, G_max, config_.S_tot, stream, cand_row_counts);
    check_cuda("materialize_rows_seg");

    // K1: buf_fill_seg — fill buffer, sort+evict for overflow rows, deferred free via to_free_*
    backend::detail::buf_fill_seg(
        ws.unique_rows, ws.row_offsets, ws.cand_K, ws.cand_V, ws.cand_S, ws.G_dev,
        sv.buf_row_seg, sv.buf_row_state, sv.buf_row_count,
        sv.seg_buf_K, sv.seg_buf_V, sv.seg_buf_S,
        sv.seg_buf_free_stack, sv.seg_buf_free_top,
        ws.over_K, ws.over_V, ws.over_S, ws.over_rows, ws.over_count_dev,
        ws.to_free_sids, ws.to_free_count_dev,
        B, D, G_max, E, config_.S_tot, config_.dtype, stream, cand_row_counts, ws.diag);
    check_cuda("buf_fill_seg");

    // K2: buf_merge_seg — merge FULL buffer rows into new pivots (no seg_buf free stack ops)
    backend::detail::buf_merge_seg(
        ws.unique_rows, ws.G_dev,
        sv.seg_buf_K, sv.seg_buf_V, sv.seg_buf_S,
        sv.buf_row_seg, sv.buf_row_state, sv.buf_row_count,
        ws.new_piv_K, ws.new_piv_V, ws.new_piv_W, ws.new_piv_S, ws.new_piv_C,
        ws.new_piv_Ks, ws.new_piv_Ss,
        ws.new_piv_rows, ws.new_piv_count_dev,
        ws.to_free_sids, ws.to_free_count_dev,
        ws.over_K, ws.over_V, ws.over_S, ws.over_rows, ws.over_count_dev,
        B, D, G_max, E, config_.S_tot, config_.dtype, stream, ws.diag);
    check_cuda("buf_merge_seg");

    // K3: buf_free_seg — push-only on seg_buf free stack (K1 evicts + K2 merges)
    backend::detail::buf_free_seg(
        ws.to_free_sids, ws.to_free_count_dev,
        sv.seg_buf_free_stack, sv.seg_buf_free_top,
        2 * G_max * B, stream);
    check_cuda("buf_free_seg");

    // Remerge: insert new pivots into pivot seg pool (zone-aware)
    backend::detail::remerge_fused_seg(
        ws.new_piv_K, ws.new_piv_V, ws.new_piv_W, ws.new_piv_S, ws.new_piv_C,
        ws.new_piv_Ks, ws.new_piv_Ss,
        ws.new_piv_rows, ws.new_piv_count_dev,
        sv.seg_piv_K, sv.seg_piv_V, sv.seg_piv_W, sv.seg_piv_S, sv.seg_piv_C,
        sv.seg_piv_K_seed, sv.seg_piv_S_seed,
        sv.piv_row_seg, sv.piv_row_state, sv.piv_row_count,
        sv.seg_piv_free_stack, sv.seg_piv_free_top,
        P, D, G_max, config_.S_tot, config_.dtype, stream,
        sv.voxel_zone_map, sv.piv_zone_top,
        sv.piv_zone_cap, sv.piv_num_zones, config_.V_alloc,
        ws.diag);
    check_cuda("remerge_fused_seg");

    // Step 10.5: Overflow absorption
    copy_and_clamp_scalar_kernel<<<1, 1, 0, stream>>>(
        ws.over_count_dev, ws.over_count_dev, static_cast<int>(E));

    int32_t over_count_host = 0;
    cudaMemcpyAsync(&over_count_host, ws.over_count_dev, sizeof(int32_t),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    bool ran_overflow_absorb = false;
    if (over_count_host > 0) {
        backend::detail::group_by_row(
            ws.over_rows, ws.over_K, ws.over_V, ws.over_S,
            over_count_host, D,
            ws.unique_rows, ws.row_offsets, ws.sorted_indices,
            ws.sorted_K, ws.sorted_V, ws.sorted_S, ws.G_dev,
            ws.cub_temp, ws.cub_temp_size, config_.dtype, stream, nullptr);

        int32_t* over_diag = ws.diag ? &ws.diag[backend::DIAG_OVER_O2O_ABSORBED] : nullptr;
        backend::detail::one2one_merge_seg(
            ws.unique_rows, ws.row_offsets,
            ws.sorted_K, ws.sorted_V, ws.sorted_S, ws.G_dev,
            sv.seg_piv_K, sv.seg_piv_V, sv.seg_piv_W, sv.seg_piv_S, sv.seg_piv_C,
            sv.seg_piv_K_seed, sv.piv_row_seg, sv.piv_row_state, sv.piv_row_count,
            ws.cand_K, ws.cand_V, ws.cand_S, ws.cand_rows, ws.cand_count_dev,
            config_.sim_thresh, 10.0f, -1.0f,
            P, D, over_count_host, over_count_host,
            config_.dtype, stream, nullptr, over_diag);

        if (ws.diag) {
            subtract_device_scalars_kernel<<<1, 1, 0, stream>>>(
                ws.over_count_dev, ws.cand_count_dev,
                &ws.diag[backend::DIAG_OVER_ABSORBED]);
        }
        ran_overflow_absorb = true;
    }

    // Step 11: Extract overflow
    const int32_t* final_count_dev = ran_overflow_absorb ? ws.cand_count_dev : ws.over_count_dev;
    int32_t final_over_count = 0;
    int32_t diag_host[backend::DIAG_COUNT] = {};
    cudaMemcpyAsync(&final_over_count, final_count_dev, sizeof(int32_t),
                    cudaMemcpyDeviceToHost, stream);
    if (diag_on && ws.diag) {
        cudaMemcpyAsync(diag_host, ws.diag, backend::DIAG_COUNT * sizeof(int32_t),
                        cudaMemcpyDeviceToHost, stream);
    }
    cudaStreamSynchronize(stream);

    if (diag_on) {
        int total_in = (int)E;
        int o2o_absorbed   = diag_host[backend::DIAG_O2O_ABSORBED];
        int o2o_reserved   = diag_host[backend::DIAG_O2O_CAND_RESERVED];
        int o2o_no_pivot   = diag_host[backend::DIAG_O2O_CAND_NO_PIVOT];
        int o2o_low_sim    = diag_host[backend::DIAG_O2O_CAND_LOW_SIM];
        int o2o_dropped    = diag_host[backend::DIAG_O2O_DROPPED];
        int buf_kept       = diag_host[backend::DIAG_BUF_KEPT];
        int buf_over_noslot= diag_host[backend::DIAG_BUF_OVER_NO_SLOT];
        int buf_over_excess= diag_host[backend::DIAG_BUF_OVER_EXCESS];
        int buf_rows_full  = diag_host[backend::DIAG_BUF_ROWS_FULL];
        int a2o_merged     = diag_host[backend::DIAG_A2O_ROWS_MERGED];
        int a2o_piv        = diag_host[backend::DIAG_A2O_PIV_CREATED];
        int over_absorbed  = diag_host[backend::DIAG_OVER_ABSORBED];
        int ov_abs = diag_host[backend::DIAG_OVER_O2O_ABSORBED];
        int ov_res = diag_host[backend::DIAG_OVER_O2O_RESERVED];
        int ov_nop = diag_host[backend::DIAG_OVER_O2O_NO_PIVOT];
        int ov_low = diag_host[backend::DIAG_OVER_O2O_LOW_SIM];
        int ov_drp = diag_host[backend::DIAG_OVER_O2O_DROPPED];
        int zone_total = diag_host[backend::DIAG_ZONE_ALLOC_TOTAL];
        int zone_hit   = diag_host[backend::DIAG_ZONE_ALLOC_HIT];
        int zone_spill = diag_host[backend::DIAG_ZONE_ALLOC_SPILL];
        int zone_sdist = diag_host[backend::DIAG_ZONE_ALLOC_SPILL_DIST];
        int zone_exhaust = diag_host[backend::DIAG_ZONE_ALLOC_EXHAUSTED];
        int zone_reuse = diag_host[backend::DIAG_ZONE_ALLOC_VICTIM_REUSE];
        float zone_hit_pct = zone_total > 0 ? 100.0f * zone_hit / zone_total : 0.0f;
        float zone_spill_pct = zone_total > 0 ? 100.0f * zone_spill / zone_total : 0.0f;
        float zone_avg_spill = zone_spill > 0 ? (float)zone_sdist / zone_spill : 0.0f;

        fprintf(stderr,
            "[DIAG-SEG] E_in=%d | o2o: absorbed=%d reserved=%d no_pivot=%d low_sim=%d dropped=%d"
            " | buf: kept=%d over_noslot=%d over_excess=%d full_rows=%d"
            " | a2o: merged=%d piv=%d"
            " | over_absorbed=%d | ov_detail: abs=%d res=%d nopiv=%d low=%d drop=%d"
            " | overflow_out=%d\n",
            total_in, o2o_absorbed, o2o_reserved, o2o_no_pivot, o2o_low_sim, o2o_dropped,
            buf_kept, buf_over_noslot, buf_over_excess, buf_rows_full,
            a2o_merged, a2o_piv, over_absorbed,
            ov_abs, ov_res, ov_nop, ov_low, ov_drp,
            final_over_count);
        if (zone_total > 0 || zone_reuse > 0) {
            fprintf(stderr,
                "[DIAG-ZONE] alloc_total=%d hit=%d(%.1f%%) spill=%d(%.1f%% avg_dist=%.1f)"
                " exhausted=%d victim_reuse=%d\n",
                zone_total, zone_hit, zone_hit_pct,
                zone_spill, zone_spill_pct, zone_avg_spill,
                zone_exhaust, zone_reuse);
        }
        fflush(stderr);
    }

    print_seg_pool_stats(stream);

    torch::Tensor K_over_out = torch::empty({final_over_count, D}, opts_kv);
    torch::Tensor V_over_out = torch::empty({final_over_count, D}, opts_kv);
    torch::Tensor S_over_out = torch::empty({final_over_count}, opts_f32);
    torch::Tensor rows_over_out = torch::empty({final_over_count}, opts_i64);

    if (final_over_count > 0) {
        const void* src_K = ran_overflow_absorb ? ws.cand_K : ws.over_K;
        const void* src_V = ran_overflow_absorb ? ws.cand_V : ws.over_V;
        const float* src_S = ran_overflow_absorb ? ws.cand_S : ws.over_S;
        const int64_t* src_rows = ran_overflow_absorb ? ws.cand_rows : ws.over_rows;

        cudaMemcpyAsync(K_over_out.data_ptr(), src_K, final_over_count * D * elem_size, cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(V_over_out.data_ptr(), src_V, final_over_count * D * elem_size, cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(S_over_out.data_ptr<float>(), src_S, final_over_count * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(rows_over_out.data_ptr<int64_t>(), src_rows, final_over_count * sizeof(int64_t), cudaMemcpyDeviceToDevice, stream);
    }
    cudaStreamSynchronize(stream);
    return std::make_tuple(K_over_out, V_over_out, S_over_out, rows_over_out);
}

// =============================================================================
// retrieve_seg / retrieve_buf_seg
// =============================================================================
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
MergerWrapper::retrieve_seg(
    torch::Tensor voxel_ids, int64_t retrieve_size, int64_t used_voxel_limit)
{
    c10::cuda::CUDAGuard guard(device_);
    auto stream = c10::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK(voxel_ids.is_cuda() && voxel_ids.dim() == 1);
    voxel_ids = voxel_ids.to(torch::kInt64).contiguous();

    int64_t H = config_.H, D = config_.D, P = config_.P;
    int64_t Q = voxel_ids.size(0);
    if (retrieve_size < 0) retrieve_size = Q * P;

    auto opts_kv = torch::TensorOptions().dtype(dtype_).device(device_);
    auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(device_);
    auto opts_u8 = torch::TensorOptions().dtype(torch::kUInt8).device(device_);
    auto opts_i64 = torch::TensorOptions().dtype(torch::kInt64).device(device_);

    auto empty_out = [&](int64_t sz) {
        return std::make_tuple(
            torch::zeros({H, sz, D}, opts_kv),
            torch::zeros({H, sz, D}, opts_kv),
            torch::zeros({H, sz}, opts_u8),
            torch::full({H, sz}, -std::numeric_limits<float>::max(), opts_f32));
    };

    if (Q == 0) return empty_out(retrieve_size);

    update_seg_views();
    auto& sv = seg_views_;

    // Retrieve ALL Q*P pivots first (needed for correct W-based sorting)
    int64_t full_size = Q * P;

    auto K_full = torch::zeros({H, full_size, D}, opts_kv);
    auto V_full = torch::zeros({H, full_size, D}, opts_kv);
    auto M_full = torch::zeros({H, full_size}, opts_u8);
    auto bias_full = torch::full({H, full_size}, -std::numeric_limits<float>::max(), opts_f32);

    backend::detail::retrieve_fixed_seg(
        voxel_ids.data_ptr<int64_t>(), Q,
        sv.seg_piv_K, sv.seg_piv_V, sv.seg_piv_C,
        sv.piv_row_seg, sv.piv_row_state,
        K_full.data_ptr(), V_full.data_ptr(),
        M_full.data_ptr<uint8_t>(), bias_full.data_ptr<float>(),
        H, config_.V_alloc, P, D, full_size,
        config_.dtype, stream);

    // Build W tensor [H, full_size] from seg pivot pool for weight-based sorting
    // (matches Python: torch.argsort(W_all_flat, dim=1, descending=True))
    auto heads_idx = torch::arange(H, opts_i64).unsqueeze(1);                // [H, 1]
    auto rows_2d = heads_idx * config_.V_alloc + voxel_ids.unsqueeze(0);     // [H, Q]
    auto rows_flat = rows_2d.reshape(-1);                                     // [H*Q]

    // piv_row_seg_: [S_tot, P] int32 — maps (row, p) -> segment ID
    auto seg_ids = piv_row_seg_.index_select(0, rows_flat);                   // [H*Q, P] int32
    auto seg_valid = (seg_ids >= 0);
    auto safe_segs = seg_ids.clamp_min(0).to(torch::kInt64).reshape(-1);     // [H*Q*P]

    // seg_piv_W_: [seg_piv_cap] float32 — per-segment weights
    auto W_flat = seg_piv_W_.index_select(0, safe_segs).reshape({H * Q, P}); // [H*Q, P]
    W_flat.masked_fill_(~seg_valid, 0.0f);
    auto W_full = W_flat.reshape({H, full_size});                             // [H, Q*P]

    // Sort by W descending
    auto sorted_idx = std::get<1>(W_full.sort(/*dim=*/1, /*descending=*/true));

    // Reorder outputs by sorted indices
    auto idx_D = sorted_idx.unsqueeze(-1).expand({H, full_size, D});
    K_full = K_full.gather(1, idx_D);
    V_full = V_full.gather(1, idx_D);
    M_full = M_full.to(torch::kInt32).gather(1, sorted_idx).to(torch::kUInt8);
    bias_full = bias_full.gather(1, sorted_idx);

    // Truncate or pad to requested retrieve_size
    if (retrieve_size <= full_size) {
        return std::make_tuple(
            K_full.slice(1, 0, retrieve_size).contiguous(),
            V_full.slice(1, 0, retrieve_size).contiguous(),
            M_full.slice(1, 0, retrieve_size).contiguous(),
            bias_full.slice(1, 0, retrieve_size).contiguous());
    }
    auto K_out = torch::zeros({H, retrieve_size, D}, opts_kv);
    auto V_out = torch::zeros({H, retrieve_size, D}, opts_kv);
    auto M_out = torch::zeros({H, retrieve_size}, opts_u8);
    auto bias_out = torch::full({H, retrieve_size}, -std::numeric_limits<float>::max(), opts_f32);
    K_out.slice(1, 0, full_size).copy_(K_full);
    V_out.slice(1, 0, full_size).copy_(V_full);
    M_out.slice(1, 0, full_size).copy_(M_full);
    bias_out.slice(1, 0, full_size).copy_(bias_full);
    return std::make_tuple(K_out, V_out, M_out, bias_out);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
MergerWrapper::retrieve_buf_seg(
    torch::Tensor voxel_ids, int64_t buf_retrieve_size, int64_t used_voxel_limit)
{
    c10::cuda::CUDAGuard guard(device_);
    auto stream = c10::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK(voxel_ids.is_cuda() && voxel_ids.dim() == 1);
    voxel_ids = voxel_ids.to(torch::kInt64).contiguous();

    int64_t H = config_.H, D = config_.D, B = config_.B;
    int64_t Q = voxel_ids.size(0);
    if (buf_retrieve_size < 0) buf_retrieve_size = Q * B;

    auto opts_kv = torch::TensorOptions().dtype(dtype_).device(device_);
    auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(device_);
    auto opts_u8 = torch::TensorOptions().dtype(torch::kUInt8).device(device_);

    auto K_out = torch::zeros({H, buf_retrieve_size, D}, opts_kv);
    auto V_out = torch::zeros({H, buf_retrieve_size, D}, opts_kv);
    auto M_out = torch::zeros({H, buf_retrieve_size}, opts_u8);
    auto bias_out = torch::full({H, buf_retrieve_size}, -std::numeric_limits<float>::max(), opts_f32);

    if (Q == 0) return std::make_tuple(K_out, V_out, M_out, bias_out);

    update_seg_views();
    auto& sv = seg_views_;

    backend::detail::retrieve_buf_seg(
        voxel_ids.data_ptr<int64_t>(), Q,
        sv.seg_buf_K, sv.seg_buf_V,
        sv.buf_row_seg, sv.buf_row_state,
        K_out.data_ptr(), V_out.data_ptr(),
        M_out.data_ptr<uint8_t>(), bias_out.data_ptr<float>(),
        H, config_.V_alloc, B, D, buf_retrieve_size,
        config_.dtype, stream);

    return std::make_tuple(K_out, V_out, M_out, bias_out);
}

}  // namespace causalvggt
