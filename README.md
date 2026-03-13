# STAC: Sparse Token Attention Cache for Streaming 3D Reconstruction

**STAC** is a **plug-and-play** KV-cache module for memory-efficient streaming 3D reconstruction over long videos. It compresses evicted KV-cache tokens into a 3D voxel pool and retrieves them on demand — compatible with any causal vision transformer backbone.

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Data preparation](#data-preparation)
- [Evaluation](#evaluation)
- [Key arguments](#key-arguments)
- [Project structure](#project-structure)
- [Citation](#citation)

---

| Capability         | Full Attention | Causal / Window   | **STAC (Ours)**           |
| ------------------ | -------------- | ----------------- | ------------------------- |
| Attention          | All frames     | Sliding window    | Window + voxel retrieval  |
| Memory scaling     | O(N²)          | O(W) fixed window | O(W) + bounded voxel pool |
| Long-video support | ✗ (OOM)        | ✓ (no history)    | ✓ (with spatial memory)   |

**Supported backbones** (switch via `--base_model`): [STream3R](https://github.com/NIRVANALAN/STream3R) (`stream3r`) · [StreamVGGT](https://github.com/wzzheng/StreamVGGT) (`streamvggt`) · [VGGT](https://github.com/facebookresearch/vggt) (`vggt`)

## Overview

Feed-forward 3D models ([VGGT](https://github.com/facebookresearch/vggt), [STream3R](https://github.com/NIRVANALAN/STream3R), [StreamVGGT](https://github.com/wzzheng/StreamVGGT)) scale poorly on long videos (O(N²) memory); sliding-window attention avoids OOM but loses history. **STAC** merges evicted tokens into a 3D voxel pool by world coordinates and retrieves the most relevant *pivot* tokens at each step, keeping long-range spatial memory with bounded memory and compute.

### Key features

- **Plug-and-play** — any causal ViT backbone via `--base_model`; no code changes to switch.
- **Voxel KV merging** — evicted KV pairs merged into 3D voxels by world position; bounded memory.
- **On-demand pivot retrieval** — attention-score–guided selection of historical tokens per step.
- **H2O heavy-hitter selection** — keeps high-attention tokens in cache for quality.
- **Optional CUDA merger** — `--voxel_backend cuda` for faster merging (build `merger-cuda`).
- **StreamSession** — frame-by-frame inference with prediction accumulation.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  Backbone (interchangeable)                         │
│  ┌───────────┐  ┌───────────┐  ┌────────────────┐  │
│  │ STream3R  │  │StreamVGGT │  │ VGGT / others  │  │
│  └─────┬─────┘  └─────┬─────┘  └───────┬────────┘  │
│        └───────────────┼────────────────┘           │
│                        ▼                            │
│              CausalVGGT Adapter                     │
│        (src/causalvggt/models/vggt.py)              │
│  ┌─────────────────────────────────────────────┐    │
│  │ CausalAggregator  (24-layer ViT-L)          │    │
│  │   └─ SparseAttention (flex / KV-cache)      │    │
│  │ CameraHead   → extrinsic + intrinsic        │    │
│  │ DPTHead (×2) → depth map + point map        │    │
│  └─────────────────────────────────────────────┘    │
└────────────────────────┬────────────────────────────┘
                         │ KV pairs
                         ▼
┌─────────────────────────────────────────────────────┐
│  STAC KV-Cache  (src/stac/)     ← plug-and-play    │
│  ┌─────────────┐                                    │
│  │ KVManager   │  sliding window (recent + pinned)  │
│  │ ├ H2O       │  heavy-hitter selection            │
│  │ └ STACVoxel │  3D voxel pool: evict→merge→retri. │
│  └─────────────┘                                    │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│  StreamSession  (src/stream_session.py)             │
│  Frame-by-frame inference + prediction accumulation │
└─────────────────────────────────────────────────────┘
```

### Attention modes

| Mode                   | Streaming | Description                                      |
| ---------------------- | --------- | ------------------------------------------------ |
| `full`                 | No        | Full attention (memory ∝ N²)                     |
| `causal` / `window_kv` | Yes       | Sliding window KV cache                          |
| `window_chunk_merge`   | Yes       | Chunked window + voxel merge (**recommended**)   |


## Installation

```bash
git clone https://github.com/Rainzor/STAC.git
cd STAC
conda create -n stac python=3.11 cmake=3.14.0 -y
conda activate stac
```

Install [PyTorch](https://pytorch.org/get-started/locally/) for your CUDA (e.g. `cu128` or `cu118`), then dependencies:

```bash
# Example: CUDA 12.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

**Optional — CUDA KV merger** (faster `--voxel_backend cuda`): set `CUDA_HOME` to your CUDA root, then:

```bash
pip install -e merger-cuda --no-build-isolation
```

### Checkpoints

Place backbone weights under `ckpt/{stream3r|streamvggt|vggt}/` as `model.safetensors` or `model.pt` (auto-detected).

| Backbone   | Hugging Face |
| ---------- | -------------|
| STream3R   | [yslan/STream3R](https://huggingface.co/yslan/STream3R) (default) |
| StreamVGGT | [lch01/StreamVGGT](https://huggingface.co/lch01/StreamVGGT) |
| VGGT       | [facebook/VGGT-1B](https://huggingface.co/facebook/VGGT-1B) |

```bash
mkdir -p ckpt/stream3r && hf download yslan/STream3R --local-dir ckpt/stream3r
# Similarly for streamvggt, vggt. Use HF_ENDPOINT=https://hf-mirror.com for mirrors.
```

## Quick Start

### Python API

Run from repo root (or add `src` to `PYTHONPATH`). Example script:

```python
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__) or ".", "src")))

import torch
from model_wrapper import load_model, run_model

device = "cuda"

# Pick any supported backbone: "stream3r", "streamvggt", or "vggt"
model = load_model("causalvggt", base_model="stream3r", device=device)

# images: (N, 3, H, W) tensor, pixel values in [0, 1]
predictions = run_model(
    model, images, "causalvggt",
    mode="window_chunk_merge",
    streaming=True,
    window_size=4,
    chunk_size=4,
    hh_size=2,
    retrieval_size=2,
    return_buf=True,       # include retrieved pivots in attention
)
# predictions keys: extrinsic, intrinsic, depth, depth_conf,
#                   world_points, world_points_conf, timing, merger, ...
```

Switching backbones is a one-line change:

```python
# Use StreamVGGT backbone instead
model = load_model("causalvggt", base_model="streamvggt", device=device)

# Use VGGT backbone instead
model = load_model("causalvggt", base_model="vggt", device=device)
```

### Command line

Use the eval launch scripts for full STAC options; `main.py` is legacy (basic modes only). Scene dirs need an `images/` subfolder with `.png` files.

```bash
python eval/long_recon/launch.py --output_dir eval_recon --dataset_type NRGBD \
    --model_name causalvggt --base_model stream3r \
    --mode window_chunk_merge --streaming -win 4 -ck 4 -hh 2 -ret_sz 2 -ret_buf --save_tag stac
```

## Data preparation

Put datasets under `data/` (symlinks OK): `7scenes/`, `neural_rgbd/`, `DTU/`, `tum/`, `scannet/`, `sintel/`, `bonn/`, `kitti/`. Preprocessing follows [CUT3R](https://github.com/CUT3R/CUT3R/blob/main/docs/preprocess.md); pre-processed sets: [Hugging Face](https://huggingface.co/datasets/yslan/pointmap_regression_evalsets). Example: `ln -s /path/to/7scenes data/7scenes`.

## Evaluation

Use `--base_model` to switch backbones. Batch run:

| Task              | Batch script | Single-run entry |
| ----------------- | ------------ | ----------------- |
| 3D reconstruction | `bash eval/long_recon/run.sh` | `eval/long_recon/launch.py` (e.g. `--dataset_type NRGBD`) |
| Camera pose       | `bash eval/cam_pose/run.sh`    | `eval/cam_pose/launch.py` (e.g. `--dataset_type tum`) |
| Video depth       | `bash eval/video_depth/run.sh` | `eval/video_depth/launch.py` then `eval/video_depth/eval_depth.py --align scale` |

Common flags: `--model_name causalvggt --base_model stream3r --mode window_chunk_merge --streaming -win 4 -ck 4 -hh 2 -ret_sz 2 -ret_buf`. Eval scripts default to `--voxel_backend cuda` and `--allocator segment`.

## Demos

- **Gradio:** `python demo/app_stream3r.py`
- **Viser 3D:** `python demo/demo_viser.py --scene_dir /path/to/scene`
- **COLMAP:** `python demo/demo_colmap.py --scene_dir /path/to/scene --output_dir output/colmap`

## Key arguments

| Argument | Short | Default | Description |
| -------- | ----- | ------- | ----------- |
| `--model_name` | | `causalvggt` | Model variant |
| `--base_model` | | `stream3r` | Backbone: `stream3r`, `streamvggt`, `vggt` |
| `--mode` | | `full` | Attention mode ([table](#attention-modes)) |
| `--streaming` | | off | Frame-by-frame via StreamSession |
| `--window_size` | `-win` | 0 | Sliding KV window (frames) |
| `--chunk_size` | `-ck` | 1 | Frames per forward pass |
| `--hh_size` | `-hh` | 0 | Heavy-hitter frames (H2O) |
| `--retrieval_size` | `-ret_sz` | 0 | Voxel pivots per step; `-1` = all |
| `--retrieve_buf` | `-ret_buf` | off | Include retrieved pivots in buffer (API: `return_buf=True`) |
| `--pinned` | | 0 | Frame indices pinned in KV cache |
| `--voxel_size` | | 0.05 | Voxel grid resolution (m) |
| `--voxel_num` | | 4096 | Initial voxel pool size |
| `--voxel_buf_cap` | | 8 | Max KV entries per buffer voxel |
| `--voxel_piv_cap` | | 4 | Max KV entries per pivot voxel |
| `--voxel_backend` | | cuda (eval) | `python` or `cuda`; eval scripts default to `cuda` |
| `--allocator` | `-alloc` | segment (eval) | `static`, `slab`, `segment`; eval scripts default to `segment` |
| `--temperature` | | 0.9 | H2O score temperature |
| `--size` | | 518 | Input resolution |
| `--kf_every` | | 1 | Process every N-th frame |

**Env:** `VERBOSE=1` — per-frame KV stats; `MERGER_MEM_PROFILE=1` — CUDA memory fragmentation at cleanup.

## Project structure

| Path | Role |
| ---- | ---- |
| `main.py` | Legacy CLI; prefer eval scripts for full STAC |
| `src/model_wrapper.py` | Unified `load_model` / `run_model` for all backbones |
| `src/stream_session.py` | Frame-by-frame streaming and prediction accumulation |
| `src/causalvggt/` | CausalVGGT adapter: models, SparseAttention, heads |
| `src/stac/` | KV manager, H2O, voxel merge/retrieval, merger, allocator, Triton flash attn |
| `src/vggt/` | Upstream VGGT reference |
| `eval/long_recon/`, `cam_pose/`, `video_depth/` | 3D recon, pose, depth |
| `demo/` | Gradio app, Viser, COLMAP |
| `merger-cuda/` | Optional CUDA KV merger extension |

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{wang2025vggt,
  title     = {VGGT: Visual Geometry Grounded Transformer},
  author    = {Wang, Jianyuan and Chen, Minghao and Karaev, Nikita and Vedaldi, Andrea and Rupprecht, Christian and Novotny, David},
  booktitle = {CVPR},
  year      = {2025}
}
```

```bibtex
@article{stream3r2025,
  title     = {STream3R: Scalable Sequential 3D Reconstruction with Causal Transformer},
  author    = {Lan, Yushi and Luo, Yihang and Hong, Fangzhou and Zhou, Shangchen and Chen, Honghua and Lyu, Zhaoyang and Yang, Shuai and Dai, Bo and Loy, Chen Change and Pan, Xingang},
  booktitle = {arXiv preprint arXiv:2508.10893},
  year      = {2025}
}
```

```bibtex
@article{streamvggt2025,
  title     = {Streaming 4D Visual Geometry Transformer},
  author    = {Zhuo, Dong and Zheng, Wenzhao and Guo, Jiahe and Wu, Yuqi and Zhou, Jie and Lu, Jiwen},
  journal   = {arXiv preprint arXiv:2507.11539},
  year      = {2025}
}
```

## Acknowledgments

STAC builds upon the following excellent open-source projects:

[VGGT](https://github.com/facebookresearch/vggt) | [STream3R](https://github.com/NIRVANALAN/STream3R) | [StreamVGGT](https://github.com/wzzheng/StreamVGGT) | [CUT3R](https://github.com/CUT3R/CUT3R) | [Spann3R](https://github.com/HengyiWang/spann3r)

## License

Please refer to the licenses of the upstream projects ([VGGT](https://github.com/facebookresearch/vggt), [STream3R](https://github.com/NIRVANALAN/STream3R), [StreamVGGT](https://github.com/wzzheng/StreamVGGT)) for usage terms.