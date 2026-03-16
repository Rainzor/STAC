# attn-cuda

CUDA extension for STAC flash attention forward pass with:

- optional vector bias
- optional column-sum scoring
- fp16 / bf16 inputs
- head dimension fixed to `D=64`

This package builds a PyTorch CUDA extension (`attn_cuda._ext`) from `csrc/`.

## Requirements

- Python 3.11+
- PyTorch with CUDA support
- CUDA toolkit (nvcc available)
- C++17 toolchain

The extension vendors CUTLASS headers under `third_party/cutlass/include` by default.
If needed, you can override with `CUTLASS_DIR`.

## Install

From repo root:

```bash
pip install -e ./attn-cuda --no-build-isolation
```

Or inside this folder:

```bash
cd attn-cuda
pip install -e . --no-build-isolation
```

## CUDA Architecture Selection

`setup.py` chooses `TORCH_CUDA_ARCH_LIST` in the following order:

1. `STAC_CUDA_ARCHS` (project-specific override)
2. `TORCH_CUDA_ARCH_LIST` (PyTorch standard override)
3. `STAC_BUILD_PROFILE=release` -> `8.0;8.6;8.9;9.0+PTX`
4. Current GPU capability (dev fallback)
5. Final fallback: `8.0;8.6`

### Common examples

Build for current machine only (fastest):

```bash
pip install -e ./attn-cuda --no-build-isolation
```

Build for multi-GPU release coverage:

```bash
STAC_BUILD_PROFILE=release pip install -e ./attn-cuda --no-build-isolation
```

Explicit architecture list:

```bash
STAC_CUDA_ARCHS="8.0;8.6;8.9;9.0+PTX" pip install -e ./attn-cuda --no-build-isolation
```

## Python API

```python
import torch
import attn_cuda

q = torch.randn(1, 1024, 16, 64, device="cuda", dtype=torch.float16)
k = torch.randn(1, 4096, 16, 64, device="cuda", dtype=torch.float16)
v = torch.randn(1, 4096, 16, 64, device="cuda", dtype=torch.float16)

out, lse = attn_cuda.flash_attn_bias_colsum(q, k, v)
out2, lse2, colsum = attn_cuda.flash_attn_bias_colsum(
    q, k, v, return_colsum=True, subsample_ratio=0.25
)
```

### Function signature

```python
flash_attn_bias_colsum(
    q, k, v,
    bias=None,
    softmax_scale=None,
    return_colsum=False,
    subsample_ratio=1.0
)
```

Input constraints:

- `q`, `k`, `v` are CUDA contiguous tensors
- shape: `q[B, M, H, D]`, `k[B, N, H, D]`, `v[B, N, H, D]`
- dtype: `float16` or `bfloat16`
- `D` must be `64`

If provided, `bias` supports:

- `[B, H, N]`
- `[B, H, 1, N]`
- `[1, H, 1, N]` (broadcast on batch)

## Notes

- This extension currently implements forward only.
- Build time depends heavily on selected CUDA architectures and host toolchain.
- If build fails due to missing CUTLASS headers, set `CUTLASS_DIR` explicitly.
