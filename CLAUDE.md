# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

STAC (Sparse Token Attention Cache) is a plug-and-play KV-cache module for memory-efficient streaming 3D reconstruction over long videos. It compresses evicted KV tokens into a 3D voxel pool and retrieves them on demand, compatible with causal vision transformer backbones (STream3R, StreamVGGT).

## Environment Setup

```bash
conda create -n stac python=3.11 cmake=3.14.0 -y
conda activate stac
pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Optional CUDA extensions (build from repo root with `CUDA_HOME` set):
```bash
pip install -e merger-cuda --no-build-isolation   # faster voxel merging
pip install -e attn-cuda --no-build-isolation     # custom FlashAttention with colsum
```

## Running Inference

```bash
# Minimal (full attention, no streaming)
python main.py --scene_dir /path/to/scene

# STAC preset (recommended)
python main.py --scene_dir /path/to/scene --mode stac

# Explicit STAC config
python main.py --scene_dir /path/to/scene \
  --base_model stream3r --streaming \
  --mode window_chunk_merge \
  -win 4 -ck 4 -hh 2 -ret_sz 2 -ret_buf
```

## Evaluation

```bash
# 3D Reconstruction
python eval/long_recon/launch.py --dataset_type NRGBD --scene_name complete_kitchen \
  --model_name causalvggt --base_model stream3r --mode stac --streaming

# Camera Pose Estimation
python eval/cam_pose/launch.py --dataset_type tum \
  --model_name causalvggt --base_model stream3r --mode stac --streaming

# Video Depth Estimation (two-step)
python eval/video_depth/launch.py --eval_dataset sintel \
  --model_name causalvggt --base_model stream3r --mode stac --streaming
python eval/video_depth/eval_depth.py --align scale
```

Batch scripts: `eval/long_recon/run.sh`, `eval/cam_pose/run.sh`, `eval/video_depth/run.sh`

## Architecture

### Data Flow

1. Images `[S, 3, H, W]` (range `[0,1]`) enter `model_wrapper.run_model()`
2. `run_model()` expands `--mode stac` into `window_chunk_merge + streaming=True` with default STAC params
3. **Streaming path**: `StreamSession.pipeline()` processes frames in chunks, calling `model()` per chunk
4. **Non-streaming path**: `model()` called once on all frames

### Key Components

**`model_wrapper.py`** — Public API. `load_model()` loads `CausalVGGT` from `ckpt/{stream3r|streamvggt}/`. `run_model()` dispatches to `StreamSession` (streaming) or direct model call (non-streaming). The `stac` mode preset sets `window_size=4, chunk_size=4, hh_size=2, retrieval_size=2`.

**`stream_session.py` — `StreamSession`** — Manages chunk-by-chunk inference. Registers a `KVManager` or `STACVoxelKV` on the aggregator via `register_kv_mgr()`. Per-chunk loop: forward pass → camera head inference → get point map → update KV positions → retrieve pivots → prune/evict → merge into voxel pool.

**`causalvggt/models/vggt.py` — `CausalVGGT`** — Top-level model. Contains:
- `CausalAggregator` (24-layer ViT-L with alternating frame/global attention)
- `CameraHead` → pose encoding → extrinsic + intrinsic
- `DPTHead` (×2) → depth map + point map

**`causalvggt/models/aggregator.py` — `CausalAggregator`** — ViT-L backbone with alternating-attention (frame-level and global). Hosts `kv_manager` (registered externally by `StreamSession`). Exposes `register_kv_mgr()`, `prune_kv_mgr()`, `update_kv_mgr_pos()`, `retrieve_kv_mgr()`.

**`causalvggt/layers/attention.py` — `SparseAttention`** — Attention layer that hooks into the `KVManager`. When `kv_manager` is set, calls `append_kv()` then `decode_sparse_attn()` instead of standard SDPA.

**`stac/kv_manager.py` — `KVManager`** — Sliding window KV cache. Stores recent + pinned frames in a pre-allocated GPU tensor `[L, H, T_buffer, D]`. Prunes by keeping `recent_size` frames + pinned frame indices. Supports CPU offload for overflow.

**`stac/h2o.py` — `HeavyHittersKV`** — Extends `KVManager` with Heavy-Hitter Oracle (H2O) scoring. Prunes by keeping `recent_size` + top-`hh_size` scored frames + pinned frames.

**`stac/stac_voxel.py` — `STACVoxelKV`** — Extends `HeavyHittersKV`. On eviction, tokens are written to a `VoxelKVMerger` (3D voxel pool keyed by world coordinate). On retrieval, per-layer pivot tokens are fetched from the voxel pool back into the hot cache for the next attention step.

**`stac/merger.py` — `VoxelKVMerger`** — Manages the long-term voxel store. Merges incoming buffer tokens into pivot representations per voxel. Supports `python` and `cuda` backends.

**`stac/voxel.py` — `BinaryVoxel`** — Quantizes 3D world coordinates to voxel IDs using a configurable `voxel_size` grid.

**`stac/allocator.py`** — Memory allocators for the voxel store: `static`, `slab`, `segment` (default).

**`stac/flash_attn_triton.py`** — Triton-based FlashAttention with optional column-sum output (used for voxel importance scoring). Fallback when `attn-cuda` is unavailable.

### Attention Modes

| Mode | Description |
|------|-------------|
| `full` | Standard full attention (no KV management) |
| `causal` | Causal full-sequence KV cache via `KVManager` |
| `window_kv` | Sliding window via `KVManager` |
| `window_chunk_merge` | Sliding window + voxel pool via `STACVoxelKV` (**STAC**) |
| `stac` | Alias for `window_chunk_merge` with recommended defaults |

### Checkpoints and Data Layout

```
STAC/
├── ckpt/stream3r/model.safetensors      # or .pt / .pth
├── ckpt/streamvggt/model.safetensors
└── data/<dataset>/<scene>/images/*.png  # e.g. data/7scenes/chess/images/
```

Supported datasets: `7scenes`, `neural_rgbd` (NRGBD), `DTU`, `tum`, `scannet`, `sintel`, `bonn`, `kitti`.

## Environment Variables

- `VERBOSE=1` — print per-frame KV stats during streaming
- `MERGER_MEM_PROFILE=1` — report CUDA memory fragmentation during voxel cleanup
