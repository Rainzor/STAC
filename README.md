# STAC: Sparse Token Attention Cache for Streaming 3D Reconstruction

**STAC** is a **plug-and-play** KV-cache module for memory-efficient streaming 3D reconstruction over long videos. It compresses evicted KV-cache tokens into a 3D voxel pool and retrieves them on demand — compatible with any causal vision transformer backbone.

- [Overview](#overview)
- [Installation](#installation)
- [CUDA attention extension (attn-cuda)](#cuda-attention-extension-attn-cuda)
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


**Supported backbones** (switch via `--base_model`): [STream3R](https://github.com/NIRVANALAN/STream3R) (`stream3r`) · [StreamVGGT](https://github.com/wzzheng/StreamVGGT) (`streamvggt`)

## Overview

Feed-forward 3D models ([STream3R](https://github.com/NIRVANALAN/STream3R), [StreamVGGT](https://github.com/wzzheng/StreamVGGT)) scale poorly on long videos (O(N²) memory); sliding-window attention avoids OOM but loses history. **STAC** merges evicted tokens into a 3D voxel pool by world coordinates and retrieves the most relevant *pivot* tokens at each step, keeping long-range spatial memory with bounded memory and compute.

### Key features

- **Plug-and-play** — any causal ViT backbone via `--base_model`; no code changes to switch.
- **Voxel KV merging** — evicted KV pairs merged into 3D voxels by world position; bounded memory.
- **On-demand pivot retrieval** — attention-score–guided selection of historical tokens per step.
- **H2O heavy-hitter selection** — keeps high-attention tokens in cache for quality.
- **Optional CUDA merger** — `--voxel_backend cuda` for faster merging (build `merger-cuda`).
- **Optional CUDA attention extension** — `attn-cuda` provides a custom FlashAttention forward (+ optional bias + colsum) backend used by STAC decoding.
- **StreamSession** — frame-by-frame inference with prediction accumulation.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  Backbone (interchangeable)                         │
│  ┌───────────┐  ┌───────────┐  ┌────────────────┐  │
│  │ STream3R  │  │StreamVGGT │  │  (others)  │       │
│  └─────┬─────┘  └─────┬─────┘  └─────┬────┘        │
│        └───────────────┼──────────────┘             │
│                        ▼                            │
│              CausalVGGT Adapter                     │
│        (causalvggt/models/vggt.py)                  │
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
│  STAC KV-Cache  (stac/)         ← plug-and-play    │
│  ┌─────────────┐                                    │
│  │ KVManager   │  sliding window (recent + pinned)  │
│  │ ├ H2O       │  heavy-hitter selection            │
│  │ └ STACVoxel │  3D voxel pool: evict→merge→retri. │
│  └─────────────┘                                    │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│  StreamSession  (stream_session.py)                 │
│  Frame-by-frame inference + prediction accumulation │
└─────────────────────────────────────────────────────┘
```

### Attention modes


| Mode                   | Streaming | Description                                                           |
| ---------------------- | --------- | --------------------------------------------------------------------- |
| `stac`                 | Yes       | **Recommended** preset (= `window_chunk_merge` + default STAC params) |
| `full`                 | No        | Full attention (memory ∝ N²)                                          |
| `causal` / `window_kv` | Yes       | Sliding window KV cache                                               |
| `window_chunk_merge`   | Yes       | Chunked window + voxel merge (manual param tuning)                    |


## Installation

> **Tested GPUs:** NVIDIA RTX 3090 (24 GB) and A100 (40 GB).

```bash
git clone https://github.com/Rainzor/STAC.git
cd STAC
conda create -n stac python=3.11 cmake=3.14.0 -y
conda activate stac
```

Install [PyTorch](https://pytorch.org/get-started/locally/) for your CUDA (e.g. `cu128` or `cu118`), then dependencies:

```bash
# Example: CUDA 12.8
pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 torchaudio==2.7.0+cu128 \
--index-url https://download.pytorch.org/whl/cu128 

pip install -r requirements.txt
```

**Install CUDA KV merger** (faster `--voxel_backend cuda`): set `CUDA_HOME` to your CUDA root, then:

```bash
pip install -e merger-cuda --no-build-isolation
```

## CUDA attention extension (`attn-cuda`)

`attn-cuda` is an optional CUDA extension used by STAC attention decoding. It provides:

- FlashAttention forward (`out`, `lse`)
- Optional vector bias (`[B,H,N]` / `[B,H,1,N]` / `[1,H,1,N]`)
- Optional column-sum (`colsum`) for retrieval scoring
- Optional colsum subsampling (`subsample_ratio`)

Build from repo root:

```bash
pip install -e attn-cuda --no-build-isolation
```

Runtime switches:

```bash
# Enable attn-cuda path in STAC
ATTN_CUDA=1 python eval/long_recon/launch.py ...

# Optional colsum subsampling ratio used by attn-cuda path
ATTN_CUDA=1 SUBSAMPLE=0.25 python eval/long_recon/launch.py ...
```
### Checkpoints

Place backbone weights under `ckpt/{stream3r|streamvggt}/` as `model.safetensors` or `model.pt` (auto-detected).


| Backbone   | Hugging Face                                                      |
| ---------- | ----------------------------------------------------------------- |
| STream3R   | [yslan/STream3R](https://huggingface.co/yslan/STream3R) (default) |
| StreamVGGT | [lch01/StreamVGGT](https://huggingface.co/lch01/StreamVGGT)       |


```bash
mkdir -p ckpt/stream3r && hf download yslan/STream3R --local-dir ckpt/stream3r
# Similarly for streamvggt. Use HF_ENDPOINT=https://hf-mirror.com for mirrors.
```

## Quick Start

### Python API

Run from repo root. Example script:

```python
import torch
from model_wrapper import load_model, run_model

device = "cuda"

# Pick any supported backbone: "stream3r" or "streamvggt"
model = load_model("causalvggt", base_model="stream3r", device=device)

# images: (N, 3, H, W) tensor, pixel values in [0, 1]
# mode="stac" auto-enables streaming + recommended STAC params
predictions = run_model(model, images, "causalvggt", mode="stac")
# predictions keys: extrinsic, intrinsic, depth, depth_conf,
#                   world_points, world_points_conf, timing, merger, ...
```

Switching backbones is a one-line change:

```python
# Use StreamVGGT backbone instead
model = load_model("causalvggt", base_model="streamvggt", device=device)
```

### Command line

`main.py` provides a minimal inference example on a scene folder; eval scripts add dataset loading and metrics. Scene dirs need an `images/` subfolder with `.png` or `.jpg` files.

```bash
# --mode stac = window_chunk_merge + streaming + default STAC params (win=4, ck=4, hh=2, ret_sz=2, ret_buf)
python eval/long_recon/launch.py --output_dir eval_recon --dataset_type NRGBD \
    --model_name causalvggt --base_model stream3r --mode stac --save_tag stac

# Override individual params if needed:
python eval/long_recon/launch.py --output_dir eval_recon --dataset_type NRGBD \
    --model_name causalvggt --base_model stream3r --mode stac -win 8 -hh 4
```

## Data preparation

Put datasets under `data/` (symlinks OK): `7scenes/`, `neural_rgbd/`, `DTU/`, `tum/`, `scannet/`, `sintel/`, `bonn/`, `kitti/`. Preprocessing follows [CUT3R](https://github.com/CUT3R/CUT3R/blob/main/docs/preprocess.md); pre-processed sets: [Hugging Face](https://huggingface.co/datasets/yslan/pointmap_regression_evalsets). Example: `ln -s /path/to/7scenes data/7scenes`.

## Evaluation

Use `--base_model` to switch backbones. Batch run:


| Task              | Batch script                   | Single-run entry                                                               |
| ----------------- | ------------------------------ | ------------------------------------------------------------------------------ |
| 3D reconstruction | `bash eval/long_recon/run.sh`  | `eval/long_recon/launch.py` (e.g. `--dataset_type NRGBD`)                      |
| Camera pose       | `bash eval/cam_pose/run.sh`    | `eval/cam_pose/launch.py` (e.g. `--dataset_type tum`)                          |
| Video depth       | `bash eval/video_depth/run.sh` | `eval/video_depth/launch.py` && `eval/video_depth/eval_depth.py --align scale` |


Common flags: `--model_name causalvggt --base_model stream3r --mode stac`. Eval scripts default to `--voxel_backend cuda` and `--allocator segment`.

## Demos

- **Gradio:** `python demo/app_stream3r.py`
- **Viser 3D:** `python demo/demo_viser.py --scene_dir /path/to/scene`
- **COLMAP:** `python demo/demo_colmap.py --scene_dir /path/to/scene --output_dir output/colmap`

## Key arguments

The arguments below are used by the evaluation and inference pipeline. **Not every script exposes all of them:** `eval/long_recon/launch.py` supports the full set; `eval/cam_pose/launch.py` and `eval/video_depth/launch.py` support a subset (model, mode, streaming, window/chunk/hh/retrieval, voxel_backend, size, etc.). For programmatic use, see `model_wrapper.run_model()` and the `stac` / `stream_session` APIs.

<details>
<summary><span style="font-weight: bold;">Command line arguments (eval scripts)</span></summary>

  #### --model_name
  Model variant, `causalvggt` by default.
  #### --base_model
  Backbone: `stream3r` or `streamvggt`.
  #### --mode
  Attention mode; see [Attention modes](#attention-modes). Default: `stac` (recommended preset).
  #### --streaming
  Enable frame-by-frame inference via StreamSession (off by default).
  #### --window_size / -win
  Sliding KV window size in frames. Default: `0`.
  #### --chunk_size / -ck
  Frames per forward pass. Default: `1`.
  #### --hh_size / -hh
  Heavy-hitter frames (H2O). Default: `0`.
  #### --retrieval_size / -ret_sz
  Voxel pivots per step; `-1` = all. Default: `0`.
  #### --retrieve_buf / -ret_buf
  Include retrieved pivots in buffer (API: `return_buf=True`). Off by default.
  #### --pinned
  Frame indices pinned in KV cache. Default: `0`.
  #### --voxel_size
  Voxel grid resolution in meters. Default: `0.05`.
  #### --voxel_num
  Initial voxel pool size. Default: `4096`.
  #### --voxel_buf_cap
  Max KV entries per buffer voxel. Default: `8`.
  #### --voxel_piv_cap
  Max KV entries per pivot voxel. Default: `4`.
  #### --voxel_backend
  `python` or `cuda`; eval scripts default to `cuda`.
  #### --allocator / -alloc
  `static`, `slab`, or `segment`; eval scripts default to `segment`.
  #### --temperature
  H2O score temperature. Default: `0.9`.
  #### --size
  Input resolution. Default: `518`.
</details>
<br>

**Env:** `VERBOSE=1` — per-frame KV stats; `MERGER_MEM_PROFILE=1` — CUDA memory fragmentation at cleanup; `ATTN_CUDA=1` — enable attn-cuda backend; `SUBSAMPLE=0.25` — colsum subsampling ratio for attn-cuda path.

## Project structure


| Path                                            | Role                                                                         |
| ----------------------------------------------- | ---------------------------------------------------------------------------- |
| `main.py`                                       | Minimal inference example (scene folder → predictions)                       |
| `model_wrapper.py`                              | Unified `load_model` / `run_model` for all backbones                         |
| `stream_session.py`                             | Frame-by-frame streaming and prediction accumulation                         |
| `causalvggt/`                                   | CausalVGGT adapter: models, SparseAttention, heads                           |
| `stac/`                                         | KV manager, H2O, voxel merge/retrieval, merger, allocator, Triton flash attn |
| `eval/long_recon/`, `cam_pose/`, `video_depth/` | 3D recon, pose, depth                                                        |
| `demo/`                                         | Gradio app, Viser, COLMAP                                                    |
| `merger-cuda/`                                  | CUDA KV merger extension                                                     |
| `attn-cuda/`                                    | CUDA attention extension (FlashAttention forward + bias + colsum)            |


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

STAC source code in this repository is licensed under the MIT License.
See the top-level `LICENSE` file.

Third-party components may use different licenses:

- Vendored CUTLASS/CuTe headers under `attn-cuda/third_party/cutlass/` are licensed by NVIDIA under BSD-3-Clause (see `attn-cuda/third_party/cutlass/LICENSE.txt` and `attn-cuda/third_party/cutlass/NOTICE`).
- Upstream model/backbone projects retain their own licenses and terms:
  [VGGT](https://github.com/facebookresearch/vggt),
  [STream3R](https://github.com/NIRVANALAN/STream3R),
  [StreamVGGT](https://github.com/wzzheng/StreamVGGT).