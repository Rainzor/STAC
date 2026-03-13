// Copyright (c) 2024-2026 CausalVGGT Authors
// SPDX-License-Identifier: Apache-2.0
//
// PyTorch/pybind11 bindings for merger-cuda extension
// 
// This module exposes the stateful C++ merger class:
// - MergerWrapper: Tensor-owning wrapper (recommended)

#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "common.h"
#include "merger_wrapper.h"

namespace py = pybind11;

// Forward declarations for test operations (implemented in stub_ops.cu)
namespace causalvggt {
namespace cuda {

torch::Tensor test_cuda_op(torch::Tensor input);

}  // namespace cuda
}  // namespace causalvggt

// Check if CUDA is available
bool is_available() {
#ifdef __CUDACC__
    return torch::cuda::is_available();
#else
    return torch::cuda::is_available();
#endif
}

// Get version string
std::string get_version() {
    return std::to_string(SPARSEVGGT_CUDA_VERSION_MAJOR) + "." +
           std::to_string(SPARSEVGGT_CUDA_VERSION_MINOR) + "." +
           std::to_string(SPARSEVGGT_CUDA_VERSION_PATCH);
}

// Get device info
std::map<std::string, std::string> get_device_info() {
    std::map<std::string, std::string> info;
    
    if (!torch::cuda::is_available()) {
        info["available"] = "false";
        return info;
    }
    
    info["available"] = "true";
    info["device_count"] = std::to_string(torch::cuda::device_count());
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    info["device_name"] = prop.name;
    info["compute_capability"] = std::to_string(prop.major) + "." + std::to_string(prop.minor);
    info["total_memory_gb"] = std::to_string(prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    
    return info;
}

// Python module definition
// Note: Module name must match the extension name in setup.py: "merger_cuda._ext"
// When built, this becomes _ext.cpython-XXX-YYY.so and is imported as: from . import _ext
PYBIND11_MODULE(_ext, m) {
    m.doc() = "merger-cuda: Stateful C++ Merger Classes for CausalVGGT";
    
    // ==========================================================================
    // Version and availability
    // ==========================================================================
    m.def("is_available", &is_available, "Check if CUDA is available");
    m.def("get_version", &get_version, "Get extension version string");
    m.def("get_device_info", &get_device_info, "Get CUDA device information");
    
    // Test operation (for debugging/testing)
    m.def("test_cuda_op", &causalvggt::cuda::test_cuda_op, 
          "Test CUDA operation (doubles input values)",
          py::arg("input"));
    
    // ==========================================================================
    // MergerWrapper: Tensor-owning wrapper (recommended)
    // ==========================================================================
    
    m.def("has_merger_wrapper", &causalvggt::has_merger_wrapper,
          "Check if MergerWrapper class is available");
    
    py::class_<causalvggt::MergerWrapper>(m, "MergerWrapper",
        R"doc(
        Tensor-owning wrapper for the merger pipeline (recommended).
        
        MergerWrapper follows the diff-gaussian-rasterization pattern:
        - Wrapper layer owns all torch::Tensor storage
        - Backend layer operates on raw pointers only
        - Pointer views are automatically refreshed after tensor reallocation
        
        This is the recommended interface for the KV merger pipeline.
        
        Example:
            wrapper = MergerWrapper(
                num_heads=16, head_dim=64,
                pivot_cap=4, budget_cap=8,
                init_voxels=1024,
                dtype=torch.float16, device=torch.device("cuda:0"))
            
            wrapper.insert_and_merge(K, V, S, VX, num_voxels)
            K_out, V_out, M_out, bias = wrapper.retrieve(voxel_ids)
        )doc")
        .def(py::init<int64_t, int64_t, int64_t, int64_t, int64_t, 
                      torch::Dtype, torch::Device, bool>(),
             py::arg("num_heads"), py::arg("head_dim"), py::arg("pivot_cap"),
             py::arg("budget_cap"), py::arg("init_voxels"), py::arg("dtype"), py::arg("device"),
             py::arg("seg_mode") = false)
        .def("insert_and_merge", &causalvggt::MergerWrapper::insert_and_merge,
             py::arg("K_new"), py::arg("V_new"), py::arg("S_new"), py::arg("VX_new"),
             py::arg("num_voxels"), py::arg("sim_thresh") = 0.75,
             py::arg("replace_thresh") = 0.5, py::arg("score_thresh") = 0.2)
        .def("insert_and_merge_with_rows", &causalvggt::MergerWrapper::insert_and_merge_with_rows,
             R"doc(
             Insert and merge with pre-computed row indices.
             
             Unlike insert_and_merge(), this method:
             - Takes pre-computed rows [E] (Python handles overflow concat + row computation)
             - Returns overflow tokens as output (Python manages overflow between calls)
             - Does NOT touch internal overflow state
             
             Args:
                 rows: Pre-computed row indices [E] int64, where row = head * V_alloc + voxel
                 K_new: Key vectors [E, D] in pool dtype (fp16/bf16)
                 V_new: Value vectors [E, D] in pool dtype
                 S_new: Scores [E] float32
                 sim_thresh: Similarity threshold for merging (default 0.75)
                 replace_thresh: Replace threshold (default 0.5)
                 score_thresh: Score threshold ratio (default 0.2)
             
             Returns:
                 tuple: (K_over, V_over, S_over, rows_over) - overflow tokens for Python to store
             )doc",
             py::arg("rows"), py::arg("K_new"), py::arg("V_new"), py::arg("S_new"),
             py::arg("sim_thresh") = 0.75,
             py::arg("replace_thresh") = 0.5, py::arg("score_thresh") = 0.2)
        .def("retrieve", &causalvggt::MergerWrapper::retrieve,
             py::arg("voxel_ids"),
             py::arg("retrieve_size") = -1,
             py::arg("used_voxel_limit") = -1)
        .def("retrieve_buf", &causalvggt::MergerWrapper::retrieve_buf,
             R"doc(
                Retrieve buffer (unmerged) tokens from the buffer pool.
                
                Args:
                    voxel_ids: [Q] voxel indices to retrieve
                    buf_retrieve_size: Output size per head (default Q*B)
                    used_voxel_limit: Upper bound on valid voxel indices
                    
                Returns:
                    tuple: (K_buf, V_buf, M_buf, bias_buf) where bias=0.0 for valid tokens
             )doc",
             py::arg("voxel_ids"),
             py::arg("buf_retrieve_size") = -1,
             py::arg("used_voxel_limit") = -1)
        .def("ensure_capacity", &causalvggt::MergerWrapper::ensure_capacity, py::arg("num_voxels"))
        .def("reset", &causalvggt::MergerWrapper::reset)
        .def_property_readonly("num_heads", &causalvggt::MergerWrapper::num_heads)
        .def_property_readonly("head_dim", &causalvggt::MergerWrapper::head_dim)
        .def_property_readonly("pivot_cap", &causalvggt::MergerWrapper::pivot_cap)
        .def_property_readonly("budget_cap", &causalvggt::MergerWrapper::budget_cap)
        .def_property_readonly("voxel_alloc", &causalvggt::MergerWrapper::voxel_alloc)
        .def_property_readonly("total_rows", &causalvggt::MergerWrapper::total_rows)
        .def_property_readonly("has_overflow", &causalvggt::MergerWrapper::has_overflow)
        .def_property_readonly("overflow_count", &causalvggt::MergerWrapper::overflow_count)
        .def_property_readonly("dropped_overflow_count", &causalvggt::MergerWrapper::dropped_overflow_count)
        .def_property_readonly("dropped_overflow_last", &causalvggt::MergerWrapper::dropped_overflow_last)
        .def_property_readonly("valid_pivot_count", &causalvggt::MergerWrapper::valid_pivot_count)
        .def("workspace_bytes", &causalvggt::MergerWrapper::workspace_bytes,
             "Return the current workspace buffer size in bytes.")
        .def("pool_stats", &causalvggt::MergerWrapper::pool_stats,
             R"doc(
             Return voxel pool storage statistics.

             Returns a dict with:
                 buf_data_count: actual buffer tokens stored (sum of row counts)
                 buf_alloc_count: total buffer token capacity
                 buf_used_slots: physical pool slots in use
                 piv_data_count: actual pivot tokens stored
                 piv_alloc_count: total pivot token capacity
                 piv_used_slots: physical pool slots in use
             )doc")
        .def("take_diagnostics", &causalvggt::MergerWrapper::take_diagnostics,
             R"doc(
             Return and clear diagnostic messages (e.g. [SEG-EXPAND], [SEG-WARN]).
             Call after insert_and_merge_with_rows_seg or ensure_capacity and log in Python.
             )doc")
        .def_property_readonly("buffer_pool_K", &causalvggt::MergerWrapper::buffer_pool_K)
        .def_property_readonly("buffer_pool_V", &causalvggt::MergerWrapper::buffer_pool_V)
        .def_property_readonly("buffer_pool_S", &causalvggt::MergerWrapper::buffer_pool_S)
        .def_property_readonly("buffer_pool_M", &causalvggt::MergerWrapper::buffer_pool_M)
        .def_property_readonly("pivot_pool_K", &causalvggt::MergerWrapper::pivot_pool_K)
        .def_property_readonly("pivot_pool_V", &causalvggt::MergerWrapper::pivot_pool_V)
        .def_property_readonly("pivot_pool_W", &causalvggt::MergerWrapper::pivot_pool_W)
        .def_property_readonly("pivot_pool_S", &causalvggt::MergerWrapper::pivot_pool_S)
        .def_property_readonly("pivot_pool_C", &causalvggt::MergerWrapper::pivot_pool_C)
        .def_property_readonly("pivot_pool_M", &causalvggt::MergerWrapper::pivot_pool_M)
        .def_property_readonly("buffer_row_ptr", &causalvggt::MergerWrapper::buffer_row_ptr)
        .def_property_readonly("buffer_row_state", &causalvggt::MergerWrapper::buffer_row_state)
        .def_property_readonly("buffer_row_count", &causalvggt::MergerWrapper::buffer_row_count)
        .def_property_readonly("pivot_row_ptr", &causalvggt::MergerWrapper::pivot_row_ptr)
        .def_property_readonly("pivot_row_state", &causalvggt::MergerWrapper::pivot_row_state)
        .def("validate_pool_state", &causalvggt::MergerWrapper::validate_pool_state,
             R"doc(
             Validate pool state consistency after operations.
             
             This method checks:
             - All slots in use have valid row pointers
             - Row states match pool occupancy
             - No orphaned data in pools
             - Free stack contains valid entries without duplicates
             
             Args:
                 check_buffer: If True, validate buffer pool (default: True)
                 check_pivot: If True, validate pivot pool (default: True)
             
             Returns:
                 tuple: (is_valid, error_message)
                     is_valid is True if all checks pass
                     error_message contains details about any errors found
             )doc",
             py::arg("check_buffer") = true, py::arg("check_pivot") = true)
        // Segmented pool methods
        .def("set_seg_mode", &causalvggt::MergerWrapper::set_seg_mode,
             "Enable or disable segmented pool storage mode.",
             py::arg("enabled"))
        .def_property_readonly("is_seg_mode", &causalvggt::MergerWrapper::is_seg_mode)
        .def("set_voxel_zones", &causalvggt::MergerWrapper::set_voxel_zones,
             "Set voxel-to-zone mapping for Morton-aware pivot allocation.",
             py::arg("zones"))
        .def("insert_and_merge_with_rows_seg", &causalvggt::MergerWrapper::insert_and_merge_with_rows_seg,
             R"doc(
             Segmented-pool variant of insert_and_merge_with_rows.
             Uses 1-token-per-segment storage to eliminate internal fragmentation.
             Returns (K_over, V_over, S_over, rows_over).
             )doc",
             py::arg("rows"), py::arg("K_new"), py::arg("V_new"), py::arg("S_new"),
             py::arg("sim_thresh") = 0.75,
             py::arg("replace_thresh") = 0.5, py::arg("score_thresh") = 0.2)
        .def("retrieve_seg", &causalvggt::MergerWrapper::retrieve_seg,
             "Retrieve pivots from segmented pool.",
             py::arg("voxel_ids"),
             py::arg("retrieve_size") = -1,
             py::arg("used_voxel_limit") = -1)
        .def("retrieve_buf_seg", &causalvggt::MergerWrapper::retrieve_buf_seg,
             "Retrieve buffer tokens from segmented pool.",
             py::arg("voxel_ids"),
             py::arg("buf_retrieve_size") = -1,
             py::arg("used_voxel_limit") = -1);
}
