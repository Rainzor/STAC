# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

STAC (Sparse Token Attention Cache) is a plug-and-play KV-cache management module for memory-efficient streaming 3D reconstruction over long video sequences. It compresses evicted KV-cache tokens into a 3D voxel pool and retrieves them on demand, compatible with any causal vision transformer backbone (STream3R, StreamVGGT, VGGT).

## Setup

```bash
conda create -n stac python=3.11 cmake=3.14.0 -y
conda activate stac
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# Optional: CUDA KV merger (requires CUDA_HOME set)
export CUDA_HOME=/usr/local/cuda-12.8
pip install -e merger-cuda --no-build-isolation
```

Checkpoints go under `ckpt/{stream3r,streamvggt,vggt}/` as `model.safetensors` or `model.pt` (auto-detected by `model_wrapper`).

## Common Commands

```bash
# Single-scene / few scenes: use eval launch with dataset_type + optional scene_name (main.py is legacy, no causalvggt)
python eval/long_recon/launch.py --output_dir eval_recon --dataset_type NRGBD --scene_name complete_kitchen \
    --model_name causalvggt --base_model stream3r --mode window_chunk_merge --streaming \
    -win 4 -ck 4 -hh 2 -ret_sz 2 -ret_buf --save_tag stac

# Evaluation - 3D reconstruction
python eval/long_recon/launch.py --output_dir eval_recon --dataset_type NRGBD \
    --model_name causalvggt --base_model stream3r --mode window_chunk_merge --streaming \
    -win 4 -ck 4 -hh 2 -ret_sz 2 -ret_buf --save_tag stac

# Evaluation - camera pose
python eval/cam_pose/launch.py --output_dir eval_cam_results --dataset_type tum \
    --model_name causalvggt --base_model stream3r --mode window_chunk_merge --streaming \
    -win 4 -ck 4 -hh 2 -ret_sz 2 -ret_buf --tag stac

# Evaluation - video depth (two-step: predict then evaluate)
python eval/video_depth/launch.py --output_dir eval_depth --eval_dataset bonn \
    --model_name causalvggt --base_model stream3r --mode window_chunk_merge --streaming \
    -win 4 -ck 4 -hh 2 -ret_sz 2
python eval/video_depth/eval_depth.py --output_dir eval_depth --eval_dataset bonn --align scale

# Batch eval scripts
bash eval/long_recon/run.sh
bash eval/cam_pose/run.sh
bash eval/video_depth/run.sh

# Verbose KV cache stats
VERBOSE=1 python eval/long_recon/launch.py ...

# Demos
python demo/app_stream3r.py          # Gradio web UI
python demo/demo_viser.py --scene_dir /path/to/scene  # 3D visualization
python demo/demo_colmap.py --scene_dir /path/to/scene --output_dir output/colmap
```

## Architecture

### Two entry points

- **`main.py`** — Legacy standalone CLI. Uses `stream3r` / `vggt` / `streamvggt` / `sparsevggt` only (no causalvggt); imports from `stream3r.stream_session` and backbone packages. For STAC + causalvggt use eval scripts or the Python API (see README).
- **`src/model_wrapper.py`** — Unified `load_model(model_name, base_model)` / `run_model()` API used by evaluation scripts. Supports `model_name=causalvggt` with `StreamSession` from `src/stream_session.py` for streaming.

### Core components (`src/`)

- **`src/stream_session.py`** — `StreamSession` orchestrates frame-by-frame streaming: feeds frames, manages KV cache lifecycle (append → attention → prune → retrieve), accumulates predictions

- **`src/causalvggt/`** — Backbone-agnostic CausalVGGT adapter
  - `models/vggt.py` — `CausalVGGT` model class, wraps backbone weights selected by `base_model` param
  - `models/aggregator.py` — `CausalAggregator`, 24-layer ViT-L with `SparseAttention`
  - `layers/attention.py` — `SparseAttention` implementing all attention modes (full, causal, window, window_kv, window_chunk, window_merge, window_chunk_merge)
  - `layers/block.py` — Transformer block with RoPE
  - `heads/` — CameraHead (extrinsic+intrinsic), DPTHead (depth, point maps)

- **`src/stac/`** — STAC KV-cache management (plug-and-play, independent of backbone)
  - `kv_manager.py` — `KVManager` base class: window-based KV cache with recent+pinned token slots, GPU/CPU buffer split
  - `h2o.py` — `HeavyHittersKV(KVManager)`: adds H2O heavy-hitter selection using attention scores
  - `stac_voxel.py` — `STACVoxelKV(HeavyHittersKV)`: full STAC with 3D voxel pool for evicted KV merge + pivot retrieval
  - `voxel.py` — `BinaryVoxel`, `HashVoxel` spatial indexing structures
  - `merger.py` — `VoxelKVMerger` handles KV merge into voxels with slab/segment allocators
  - `allocator.py` — Slab and segment memory allocators for voxel pool
  - `flash_attn_triton.py` — Triton kernels for flash attention with column-sum scoring

- **`src/vggt/`** — Original upstream VGGT reference implementation (read-only reference)

### Inheritance chain for KV managers

`KVManager` → `HeavyHittersKV` → `STACVoxelKV`

Per-step lifecycle in streaming: `append_kv` (layer-wise) → `decode_sparse_attn` (layer-wise) → `append_positions` (all layers) → `prune_kv` (all layers) → `retrieve_kv` (all layers)

### Attention modes

| Mode | Key |
|:---|:---|
| `window_chunk_merge` | **Recommended** — chunked sliding window + voxel-based spatial KV merging |
| `window_merge` | Window + voxel merging (no chunking) |
| `window` / `window_kv` | Sliding window only |
| `window_chunk` | Chunked window without merging |
| `full` / `causal` | Full or strictly causal attention |

### Key arguments shorthand

`-win` = window_size, `-ck` = chunk_size, `-hh` = hh_size (heavy-hitter frames), `-ret_sz` = retrieval_size (voxel pivots), `-ret_buf` = include retrieved pivots in attention

### CUDA extension (`merger-cuda/`)

Optional GPU-accelerated voxel merging. Built with `torch.utils.cpp_extension.CUDAExtension`. Enabled via `--voxel_backend cuda`. Source in `csrc/` with C++17/CUDA kernels.

## Data layout

- Input scenes: `<scene_dir>/images/*.png`
- Evaluation datasets: symlinked under `data/` (7scenes, neural_rgbd, DTU, tum, scannet, sintel, bonn, kitti)
- Python path: `src/` is added to `sys.path` at runtime; imports use package names directly (e.g., `from causalvggt.models.vggt import CausalVGGT`)

## Key conventions

- Batch size is always 1 (`B=1` asserted throughout KV managers)
- KV cache tensors: `[L_eff, H, T, D]` where L_eff = number of managed layers, H = heads, T = tokens, D = head_dim
- Token counts are always multiples of `token_per_frame`
- Uses `torch.bfloat16` on Ampere+ GPUs, `torch.float16` otherwise
- Chinese comments appear in some files (kv_manager.py, stac_voxel.py, etc.)
