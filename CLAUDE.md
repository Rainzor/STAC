# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

STAC (Sparse Token Attention Cache) is a plug-and-play KV-cache management module for memory-efficient streaming 3D reconstruction over long video sequences. It compresses evicted KV-cache tokens into a 3D voxel pool and retrieves them on demand, compatible with any causal vision transformer backbone (STream3R, StreamVGGT).

## Setup

```bash
conda create -n stac python=3.11 cmake=3.14.0 -y
conda activate stac

# CUDA 11.8 or CUDA 12.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# CUDA KV merger (requires CUDA_HOME set)
export CUDA_HOME=/usr/local/cuda-12.8
pip install -e merger-cuda --no-build-isolation

# CUDA attention extension (attn-cuda)
pip install -e attn-cuda --no-build-isolation
```

Checkpoints go under `ckpt/{stream3r,streamvggt}/` as `model.safetensors`, `model.pt`, or `model.pth` (auto-detected by `model_wrapper`).

## Common Commands

```bash
# Single-scene / few scenes (--mode stac = window_chunk_merge + streaming + default STAC params)
python eval/long_recon/launch.py --output_dir eval_recon --dataset_type NRGBD --scene_name complete_kitchen \
    --model_name causalvggt --base_model stream3r --mode stac --save_tag stac

# Evaluation - 3D reconstruction
python eval/long_recon/launch.py --output_dir eval_recon --dataset_type NRGBD \
    --model_name causalvggt --base_model stream3r --mode stac --save_tag stac

# Evaluation - camera pose
python eval/cam_pose/launch.py --output_dir eval_cam_results --dataset_type tum \
    --model_name causalvggt --base_model stream3r --mode stac --tag stac

# Evaluation - video depth (two-step: predict then evaluate)
python eval/video_depth/launch.py --output_dir eval_depth --eval_dataset bonn \
    --model_name causalvggt --base_model stream3r --mode stac
python eval/video_depth/eval_depth.py --output_dir eval_depth --eval_dataset bonn --align scale

# Batch eval scripts
bash eval/long_recon/run.sh
bash eval/cam_pose/run.sh
bash eval/video_depth/run.sh

# Verbose KV cache stats
VERBOSE=1 python eval/long_recon/launch.py ...

# Enable CUDA attention backend in STAC decoding
python eval/long_recon/launch.py --attn_backend cuda ...

# Optional: colsum subsampling for sparse decode
python eval/long_recon/launch.py --attn_backend cuda --subsample 0.25 ...
```

## Architecture

### Two entry points

- **`main.py`** — Minimal inference example using the `model_wrapper` API. Supports `--streaming`, `--mode`, `--base_model`, all voxel/STAC params, and `--dtype auto|fp16|bf16`.
- **`model_wrapper.py`** — Unified `load_model(model_name, base_model)` / `run_model()` API used by evaluation scripts. Supports `model_name=causalvggt` with `StreamSession` from `stream_session.py` for streaming. Contains `STAC_DEFAULTS` dict that defines default parameter expansion when `mode="stac"`.

### Core components

- **`stream_session.py`** — `StreamSession` orchestrates frame-by-frame streaming: feeds frames, manages KV cache lifecycle (append → attention → prune → retrieve), accumulates predictions. Uses `rich` for live progress display. Handles two streaming modes: `window_kv`/`causal` (with `KVManager`) and `window_chunk_merge` (with `STACVoxelKV`).

- **`causalvggt/`** — Backbone-agnostic CausalVGGT adapter
  - `models/vggt.py` — `CausalVGGT` model class, wraps backbone weights selected by `base_model` param
  - `models/aggregator.py` — `CausalAggregator`, 24-layer ViT-L with `SparseAttention`
  - `layers/attention.py` — `SparseAttention` implementing all attention modes (full, causal, window, window_kv, window_chunk, window_merge, window_chunk_merge)
  - `layers/block.py` — Transformer block with RoPE
  - `heads/` — CameraHead (extrinsic+intrinsic), DPTHead (depth, point maps)

- **`stac/`** — STAC KV-cache management (plug-and-play, independent of backbone)
  - `kv_manager.py` — `KVManager` base class: window-based KV cache with recent+pinned token slots, GPU/CPU buffer split
  - `h2o.py` — `HeavyHittersKV(KVManager)`: adds H2O heavy-hitter selection using attention scores (token-level grouping)
  - `stac_voxel.py` — `STACVoxelKV(HeavyHittersKV)`: full STAC with 3D voxel pool for evicted KV merge + pivot retrieval
  - `voxel.py` — `BinaryVoxel`, `HashVoxel` spatial indexing structures
  - `merger.py` — `VoxelKVMerger` handles KV merge into voxels with slab/segment allocators
  - `allocator.py` — Slab and segment memory allocators for voxel pool
  - `flash_attn_triton.py` — Triton kernels for flash attention with column-sum scoring

### Inheritance chain for KV managers

`KVManager` → `HeavyHittersKV` → `STACVoxelKV`

Per-step lifecycle in streaming: `append_kv` (layer-wise) → `decode_sparse_attn` (layer-wise) → `append_positions` (all layers) → `prune_kv` (all layers) → `retrieve_kv` (all layers)

### Attention modes

| Mode | Key |
| :--- | :--- |
| `stac` | **Recommended** preset — expands to `window_chunk_merge` + streaming + default params (win=4, ck=4, hh=2, ret_sz=2, ret_buf) |
| `window_chunk_merge` | Chunked sliding window + voxel-based spatial KV merging (manual param tuning) |
| `window_kv` | Sliding window with KVManager (H2O pruning) |
| `causal` | Causal attention (window_size = num_frames, effectively full history) |
| `full` | Full bidirectional attention (non-streaming mask in aggregator) |
| `window` | Sliding window mask (block.py `create_attn_mask`) |
| `causal_full_causal` / `full_causal` / `causal_full` | Hybrid modes with full attention on specific layer ranges (layers 10-17 or 18-23) |

### Key arguments shorthand

`-win` = window_size, `-ck` = chunk_size, `-hh` = hh_size (heavy-hitter frames), `-ret_sz` = retrieval_size (voxel pivots), `-ret_buf` = include retrieved pivots in attention

### CUDA extension (`merger-cuda/`)

Optional GPU-accelerated voxel merging. Built with `torch.utils.cpp_extension.CUDAExtension`. Enabled via `--voxel_backend cuda`. Source in `csrc/` with C++17/CUDA kernels. Exposes `MergerWrapper` for tensor-owning stateful voxel merge operations (`insert_and_merge`, `retrieve`). Has `pyproject.toml` (build system) + `setup.py` (metadata + extension config).

### CUDA attention extension (`attn-cuda/`)

Optional CUDA flash-attention extension used by STAC decode path. Exposes
`flash_attn_bias_colsum` (forward + optional vector bias + optional colsum). Bundles `third_party/cutlass/` headers (CUTLASS GEMM/CUTE tensor abstractions).

- Install: `pip install -e attn-cuda --no-build-isolation`
- Enable at runtime: `--attn_backend cuda` (default)
- Optional colsum subsampling: `--subsample 0.25` (or other ratio in (0, 1])

Build architecture selection in `attn-cuda/setup.py`:

1. `STAC_CUDA_ARCHS` (project override)
2. `TORCH_CUDA_ARCH_LIST` (PyTorch standard override)
3. `STAC_BUILD_PROFILE=release` fallback -> `8.0;8.6;8.9;9.0+PTX`
4. Current GPU capability (dev fallback)
5. Final fallback -> `8.0;8.6`

## Data layout

- Input scenes: `<scene_dir>/images/*.{png,jpg,jpeg}`
- Evaluation datasets: symlinked under `data/` (7scenes, neural_rgbd, DTU, tum, scannet, sintel, bonn, kitti)
- Checkpoints: `ckpt/{stream3r,streamvggt}/` (gitignored)
- Eval outputs: `eval_recon*/`, `eval_cam_results/`, `eval_depth/` (gitignored)
- Python path: project root is in `sys.path`; imports use package names directly (e.g., `from causalvggt.models.vggt import CausalVGGT`)
- Input resolution: 518x392 (default), 512x384, or 224x224; controlled by `--size`

## Key conventions

- Batch size is always 1 (`B=1` asserted throughout KV managers and streaming session)
- KV cache tensors: `[L_eff, H, T, D]` where L_eff = number of managed layers, H = heads, T = tokens, D = head_dim
- Token counts are always multiples of `token_per_frame` (= image_patches + special_tokens)
- Uses `torch.bfloat16` on Ampere+ GPUs, `torch.float16` otherwise
- `torch.backends.cuda.matmul.allow_tf32 = True` in all eval scripts
- Eval scripts cap CPU threads to 1 via `OMP_NUM_THREADS` / `MKL_NUM_THREADS` env vars
- `rich` library used for live progress bars and stats display in streaming mode
- Chinese comments appear in some files (kv_manager.py, stac_voxel.py, etc.)
