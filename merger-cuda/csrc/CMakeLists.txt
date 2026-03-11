cmake_minimum_required(VERSION 3.18)
project(sparsevggt_cuda LANGUAGES CXX CUDA)

# ==============================================================================
# CausalVGGT CUDA Extension - CMake Build (Developer Use)
# ==============================================================================
# This CMakeLists.txt is for developer use only.
# The primary build method is: pip install . (uses setuptools)
#
# Developer usage:
#   mkdir build && cd build
#   cmake ..
#   cmake --build . --parallel
#
# The built extension will be placed in the parent directory (src/causalvggt/cuda/)
# ==============================================================================

option(BUILD_PYTHON_MODULE "Build Python extension module" ON)
option(USE_FAST_MATH "Enable fast math operations" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ==============================================================================
# Find Dependencies
# ==============================================================================
find_package(CUDAToolkit REQUIRED)

if(BUILD_PYTHON_MODULE)
    find_package(Python REQUIRED COMPONENTS Interpreter Development)
    
    # Find PyTorch
    execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
        OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE TORCH_FOUND
    )
    
    if(NOT TORCH_FOUND EQUAL 0)
        message(FATAL_ERROR "PyTorch not found. Please install PyTorch first.")
    endif()
    
    list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX}")
    find_package(Torch REQUIRED)
    
    # Get Python extension suffix
    execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"
        OUTPUT_VARIABLE PYTHON_EXT_SUFFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
endif()

# ==============================================================================
# CUDA Architecture Detection
# ==============================================================================
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # Auto-detect GPU architecture
    execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import torch; print(torch.cuda.get_device_capability()[0] * 10 + torch.cuda.get_device_capability()[1]) if torch.cuda.is_available() else print('75')"
        OUTPUT_VARIABLE DETECTED_ARCH
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    set(CMAKE_CUDA_ARCHITECTURES ${DETECTED_ARCH})
    message(STATUS "Auto-detected CUDA architecture: ${CMAKE_CUDA_ARCHITECTURES}")
endif()

# ==============================================================================
# Source Files (relative to this CMakeLists.txt location: csrc/)
# ==============================================================================
set(BACKEND_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/merger_pipeline.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/merger_kernels.cu
)

set(MERGER_WRAPPER_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/merger_wrapper.cu
)

set(EXTENSION_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/bindings.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/stub_ops.cu
    ${BACKEND_SOURCES}
    ${MERGER_WRAPPER_SOURCES}
)

# ==============================================================================
# Build Python Extension
# ==============================================================================
if(BUILD_PYTHON_MODULE)
    # Create shared library for Python extension
    add_library(_ext SHARED ${EXTENSION_SOURCES})
    
    # Output to parent directory (src/causalvggt/cuda/) as _ext.so
    # This matches the import: from . import _ext
    set_target_properties(_ext PROPERTIES
        PREFIX ""
        OUTPUT_NAME "_ext"
        SUFFIX "${PYTHON_EXT_SUFFIX}"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
    )
    
    target_include_directories(_ext PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${TORCH_INCLUDE_DIRS}
        ${Python_INCLUDE_DIRS}
        ${CUDAToolkit_INCLUDE_DIRS}
    )
    
    # Get torch library directory for python bindings
    execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import torch; print(torch.__path__[0] + '/lib')"
        OUTPUT_VARIABLE TORCH_LIB_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    
    target_link_libraries(_ext PRIVATE
        ${TORCH_LIBRARIES}
        Python::Python
        CUDA::cudart
        "${TORCH_LIB_DIR}/libtorch_python.so"
    )
    
    # Compiler flags
    target_compile_options(_ext PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:-U__CUDA_NO_HALF_OPERATORS__>
        $<$<COMPILE_LANGUAGE:CUDA>:-U__CUDA_NO_HALF_CONVERSIONS__>
        $<$<COMPILE_LANGUAGE:CUDA>:-U__CUDA_NO_HALF2_OPERATORS__>
    )
    
    if(USE_FAST_MATH)
        target_compile_options(_ext PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
        )
    endif()
    
    # Handle torch CXX11 ABI
    if(TORCH_CXX_FLAGS)
        target_compile_options(_ext PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:${TORCH_CXX_FLAGS}>
        )
    endif()
    
    message(STATUS "Python extension will be installed to: ${CMAKE_CURRENT_SOURCE_DIR}/..")
endif()

# ==============================================================================
# Summary
# ==============================================================================
message(STATUS "")
message(STATUS "=== CausalVGGT CUDA Extension Configuration ===")
message(STATUS "  CUDA Toolkit:       ${CUDAToolkit_VERSION}")
message(STATUS "  CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  Python:             ${Python_VERSION}")
message(STATUS "  PyTorch:            ${Torch_VERSION}")
message(STATUS "  Build Type:         ${CMAKE_BUILD_TYPE}")
message(STATUS "  Fast Math:          ${USE_FAST_MATH}")
message(STATUS "")
