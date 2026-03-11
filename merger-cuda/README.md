# merger-cuda: CUDA Merger Kernels for CausalVGGT

merger-cuda provides GPU-accelerated implementations of the KV merger pipeline for the CausalVGGT project.

## Features

- **MergerWrapper**: Stateful C++ class that owns all tensor storage and provides a zero-copy merger pipeline
- **pack_valid_tokens**: Filter and compact valid tokens by voxel index
- **group_by_row**: Token grouping using CUB radix sort + run-length encode
- **materialize_rows**: Slot allocation for new rows
- **one2one_merge**: Similarity-based merge into existing pivots
- **buffer_topb_update**: Buffer update with top-B selection by score
- **all2one_merge_fused**: Fused cluster merge for FULL buffers into pivots
- **retrieve**: Pivot retrieval with logit bias computation

## Environment Requirements

### Prerequisites

```bash
# 1. CUDA Toolkit (tested with CUDA 11.8)
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH

# 2. Conda environment with PyTorch
conda activate sparse-vggt

# 3. Verify CUDA is accessible
nvcc --version
python -c "import torch; print(torch.cuda.is_available())"
```

### Tested Configuration

| Component | Version |
|-----------|---------|
| CUDA Toolkit | 11.8 |
| PyTorch | 2.7.0+cu118 |
| Python | 3.10 |
| GPU | NVIDIA GeForce RTX 3090 (compute capability 8.6) |

## Installation

### Development Install (Recommended)

```bash
cd merger-cuda

export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH

# (Optional) Specify GPU architecture for faster compilation
export TORCH_CUDA_ARCH_LIST="8.6"  # Adjust based on your GPU

pip install -e ".[dev]"
```

### Standard Install

```bash
cd merger-cuda
pip install .

# Or build without isolation
pip install -e . --no-build-isolation
```

### Build Options

| Environment Variable | Description | Example |
|---------------------|-------------|---------|
| `CUDA_HOME` | Path to CUDA toolkit | `/usr/local/cuda-11.8` |
| `TORCH_CUDA_ARCH_LIST` | Target GPU architectures | `"8.6"` or `"7.5;8.0;8.6"` |
| `MAX_JOBS` | Parallel compilation jobs | `4` |

## Verification

```python
from merger_cuda import cuda_available, get_pipeline_status, get_device_info

print('CUDA available:', cuda_available())
print('Pipeline status:', get_pipeline_status())
print('Device info:', get_device_info())
```

## Usage

```python
import torch
from merger_cuda import has_merger_wrapper, create_merger_wrapper

if has_merger_wrapper():
    wrapper = create_merger_wrapper(
        num_heads=16, head_dim=64,
        pivot_cap=4, budget_cap=8, init_voxels=1024,
        dtype=torch.float16, device=torch.device("cuda:0"))

    wrapper.insert_and_merge(K, V, S, VX, num_voxels=1000)
    K_out, V_out, M_out, bias = wrapper.retrieve(voxel_ids)
```

## Running Tests

```bash
cd merger-cuda
pytest tests/ -v
```

## Troubleshooting

### ImportError: libc10.so not found

Ensure the PyTorch library path is in your `LD_LIBRARY_PATH`:

```bash
export LD_LIBRARY_PATH="$(python -c 'import torch; import os; print(os.path.dirname(torch.__file__))')/lib:$LD_LIBRARY_PATH"
```

### Compilation Errors

1. **CUDA not found**: Verify `CUDA_HOME` is set correctly
2. **Architecture mismatch**: Set `TORCH_CUDA_ARCH_LIST` to match your GPU
3. **Build failures**: Try cleaning and rebuilding:
   ```bash
   rm -rf build merger_cuda/_ext*.so
   pip install -e ".[dev]"
   ```

### CPU-only Mode

If CUDA is not available, the package will still import but CUDA operations will be unavailable:

```python
from merger_cuda import cuda_available
print(cuda_available())  # Returns False on CPU-only systems
```

## Project Structure

```
merger-cuda/
├── csrc/
│   ├── bindings.cpp            # PyBind11 bindings
│   ├── merger_wrapper.cu       # MergerWrapper (tensor-owning wrapper)
│   ├── merger_wrapper.h        # MergerWrapper header
│   ├── stub_ops.cu             # Test/stub operations
│   ├── include/
│   │   ├── common.h            # Version constants
│   │   ├── merger_types.h      # Type definitions and enums
│   │   └── merger_kernels.cuh  # Kernel function declarations
│   └── kernels/
│       ├── merger_pipeline.cu  # Pipeline orchestration (detail:: functions)
│       └── merger_kernels.cu   # Kernel implementations
├── merger_cuda/
│   └── __init__.py             # Python API
├── setup.py
├── pyproject.toml
└── README.md
```

## License

Apache-2.0
