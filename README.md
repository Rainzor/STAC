# STAC: Sparse Token Attention Cache for Streaming 3D Reconstruction

**STAC** is a **plug-and-play** KV-cache management module that enables memory-efficient streaming 3D reconstruction over long video sequences.  
It compresses evicted KV-cache tokens into a 3D voxel pool and retrieves them on demand — compatible with any causal vision transformer backbone.


| Capability         | Full Attention | Causal / Window   | **STAC (Ours)**           |
| ------------------ | -------------- | ----------------- | ------------------------- |
| Attention          | All frames     | Sliding window    | Window + voxel retrieval  |
| Memory scaling     | O(N²)          | O(W) fixed window | O(W) + bounded voxel pool |
| Long-video support | ✗ (OOM)        | ✓ (no history)    | ✓ (with spatial memory)   |


### Supported Backbones

STAC is **model-agnostic** and can be plugged into any causal transformer that uses KV-cache based attention. Currently tested backbones:


| Backbone                                                | Source                    | `--base_model` |
| ------------------------------------------------------- | ------------------------- | -------------- |
| **[STream3R](https://github.com/NIRVANALAN/STream3R)**  | Lan et al., 2025          | `stream3r`     |
| **[StreamVGGT](https://github.com/wzzheng/StreamVGGT)** | Zhuo & Zheng et al., 2025 | `streamvggt`   |
| **[VGGT](https://github.com/facebookresearch/vggt)**    | Wang et al., CVPR 2025    | `vggt`         |


> Simply switch `--base_model` to use a different backbone — no code changes required.

## Overview

Modern feed-forward 3D reconstruction models (e.g. [VGGT](https://github.com/facebookresearch/vggt), [STream3R](https://github.com/NIRVANALAN/STream3R), [StreamVGGT](https://github.com/wzzheng/StreamVGGT)) achieve strong results but face quadratic memory growth when processing long videos. Simply truncating attention to a sliding window discards valuable history.

**STAC** (Sparse Token Attention Cache) bridges this gap: as new frames enter the sliding window, evicted tokens are *merged* into a persistent 3D voxel pool indexed by their world coordinates. At each step, the most relevant *pivot tokens* are retrieved from the pool and injected into attention, so the model retains long-range spatial memory with bounded compute and memory.

### Key Features

- **Plug-and-play** — works with any causal vision transformer backbone (STream3R, StreamVGGT, VGGT, etc.) via a unified `--base_model` interface.
- **Voxel-based KV merging** — evicted KV pairs are spatially merged into 3D voxels, preserving geometric structure while bounding memory.
- **On-demand pivot retrieval** — attention-score-guided selection of the most relevant historical tokens per frame.
- **H2O heavy-hitter selection** — retains high-attention tokens in the cache for better quality.
- **Optional CUDA merger** — custom CUDA kernels for faster voxel merging on GPU.
- **Drop-in streaming session** — `StreamSession` wraps the backbone model for frame-by-frame inference with full prediction accumulation.

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
│  StreamSession  (src/causalvggt/stream_session.py)  │
│  Frame-by-frame inference + prediction accumulation │
└─────────────────────────────────────────────────────┘
```

### Attention Modes


| Mode                   | Description                                      |
| ---------------------- | ------------------------------------------------ |
| `full`                 | Standard full attention (memory ∝ N²)            |
| `causal`               | Strictly causal attention                        |
| `window` / `window_kv` | Sliding window KV cache                          |
| `window_chunk`         | Chunked sliding window                           |
| `window_merge`         | Window + voxel-based spatial KV merging          |
| `window_chunk_merge`   | Chunked window + voxel merging (**recommended**) |


## Installation

### 1. Clone and create environment

```bash
git clone https://github.com/Rainzor/STAC.git
cd STAC

conda create -n stac python=3.11 cmake=3.14.0 -y
conda activate stac
```

### 2. Install PyTorch

Install [PyTorch](https://pytorch.org/get-started/locally/) matching your CUDA version.

**CUDA 12.8 (recommended):**

```bash
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
```

**CUDA 11.8:**

```bash
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu118
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. (Optional) Build CUDA KV Merger

For faster voxel merging (`--voxel_backend cuda`), build the CUDA extension:

```bash
pip install -e merger-cuda --no-build-isolation
```

> Requires `CUDA_HOME` to be set, matching your PyTorch CUDA version:
>
> - CUDA 12.8: `export CUDA_HOME=/usr/local/cuda-12.8`
> - CUDA 11.8: `export CUDA_HOME=/usr/local/cuda-11.8`

### Checkpoints

STAC reuses existing backbone weights — no additional training is needed. Download the weights for your chosen backbone and place them under `ckpt/`:

```
ckpt/
├── stream3r/model.safetensors     # STream3R weights
├── streamvggt/model.safetensors   # StreamVGGT weights
└── vggt/model.safetensors         # VGGT-1B weights
```

> Both `model.safetensors` and `model.pt` are supported; the loader auto-detects the format.

| Backbone   | Download                                                                   | Notes               |
| ---------- | -------------------------------------------------------------------------- | -------------------- |
| STream3R   | [Hugging Face (yslan/STream3R)](https://huggingface.co/yslan/STream3R)     | Recommended default  |
| StreamVGGT | [Hugging Face (lch01/StreamVGGT)](https://huggingface.co/lch01/StreamVGGT) |                      |
| VGGT       | [Hugging Face (facebook/VGGT-1B)](https://huggingface.co/facebook/VGGT-1B) | Original full-attention model |


Example download with `hf` (requires `huggingface_hub[cli]>=0.25.0`):

```bash
# STream3R
mkdir -p ckpt/stream3r
hf download yslan/STream3R --local-dir ckpt/stream3r

# StreamVGGT
mkdir -p ckpt/streamvggt
hf download lch01/StreamVGGT --local-dir ckpt/streamvggt

# VGGT
mkdir -p ckpt/vggt
hf download facebook/VGGT-1B --local-dir ckpt/vggt
```

> For faster downloads, install `hf-transfer` and set `HF_HUB_ENABLE_HF_TRANSFER=1`.  
> For mirrors (e.g. in China), prefix with `HF_ENDPOINT=https://hf-mirror.com`.

> **Note:** Checkpoints and public dataset download links will be updated. Stay tuned.

## Quick Start

### Python API

```python
import torch
from src.model_wrapper import load_model, run_model

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

### Command Line

> **Note:** `main.py` is the legacy entry point (supports `vggt`, `stream3r`, `streamvggt` with basic modes only).  
> For full STAC features use the evaluation launch scripts, which expose all KV-cache and voxel arguments.

```bash
# 3D reconstruction on a single scene (NRGBD dataset, STream3R backbone)
python eval/long_recon/launch.py \
    --output_dir eval_recon \
    --dataset_type NRGBD \
    --model_name causalvggt --base_model stream3r \
    --mode window_chunk_merge --streaming \
    -win 4 -ck 4 -hh 2 -ret_sz 2 -ret_buf \
    --save_tag stac
```

The `--scene_dir` for standalone inference should contain an `images/` subfolder with `.png` files.

## Data Preparation

Organize evaluation datasets under `data/` (symlinks are supported):

```
data/
├── 7scenes/          # 7-Scenes
├── neural_rgbd/      # Neural RGBD (NRGBD)
├── DTU/              # DTU MVS
├── tum/              # TUM RGB-D
├── scannet/          # ScanNet
├── sintel/           # MPI Sintel
├── bonn/             # Bonn RGB-D
└── kitti/            # KITTI
```

We follow [CUT3R](https://github.com/CUT3R/CUT3R/blob/main/docs/preprocess.md) for dataset preprocessing. For convenience, pre-processed evaluation datasets are available on [Hugging Face](https://huggingface.co/datasets/yslan/pointmap_regression_evalsets).

```bash
ln -s /path/to/7scenes data/7scenes
```

## Evaluation

All evaluation scripts support switching backbones via `--base_model`.

### 3D Reconstruction (NRGBD / 7-Scenes)

```bash
bash eval/long_recon/run.sh
```

Or run a single dataset:

```bash
python eval/long_recon/launch.py \
    --output_dir eval_recon \
    --dataset_type NRGBD \
    --model_name causalvggt --base_model stream3r \
    --mode window_chunk_merge --streaming \
    -win 4 -ck 4 -hh 2 -ret_sz 2 -ret_buf \
    --save_tag stac --vis_tag w4h2r2c4
```

### Camera Pose Estimation (TUM / ScanNet / Sintel)

```bash
bash eval/cam_pose/run.sh
```

Or run a single dataset:

```bash
python eval/cam_pose/launch.py \
    --output_dir eval_cam_results \
    --dataset_type tum \
    --model_name causalvggt --base_model stream3r \
    --mode window_chunk_merge --streaming \
    -win 4 -ck 4 -hh 2 -ret_sz 2 -ret_buf \
    --tag stac --vis_tag w4h2r2c4
```

### Video Depth Estimation (Bonn / Sintel / KITTI)

```bash
bash eval/video_depth/run.sh
```

Or run a single dataset:

```bash
python eval/video_depth/launch.py \
    --output_dir eval_depth \
    --eval_dataset bonn \
    --model_name causalvggt --base_model stream3r \
    --mode window_chunk_merge --streaming \
    -win 4 -ck 4 -hh 2 -ret_sz 2

python eval/video_depth/eval_depth.py \
    --output_dir eval_depth \
    --eval_dataset bonn \
    --align scale
```

## Demos

### Gradio Web UI

```bash
python demo/app_stream3r.py
```

### Viser 3D Visualization

```bash
python demo/demo_viser.py --scene_dir /path/to/scene
```

### COLMAP Export

```bash
python demo/demo_colmap.py --scene_dir /path/to/scene --output_dir output/colmap
```

## Key Arguments

**Command Line Arguments**

#### --model_name

  Model variant to use. `causalvggt` by default.

#### --base_model

  Backbone weights to load: `stream3r`, `streamvggt`, or `vggt`. `stream3r` by default.

#### --mode

  Attention mode for the aggregator. `full` by default. See the [Attention Modes](#attention-modes) table for available options.

#### --streaming

  Add this flag to enable frame-by-frame streaming inference via `StreamSession`.

#### -win / --window_size

  Sliding KV window size in frames. `0` by default (disabled).

#### -ck / --chunk_size

  Number of frames processed per forward pass. `1` by default.

#### -hh / --hh_size

  Number of heavy-hitter frames retained in the KV cache (H2O selection). `0` by default (disabled).

#### -ret_sz / --retrieval_size

  Number of voxel pivot tokens retrieved and injected into attention per step. `0` by default (disabled); `-1` = retrieve all available.

#### -ret_buf / --retrieve_buf

  Add this flag to include retrieved pivot tokens in the sliding KV buffer as well as attention.

> **Note:** When using the Python API directly, pass `return_buf=True` (not `retrieve_buf`).

#### --pinned

  Space-separated list of frame indices to permanently pin in the KV cache. `0` (first frame) by default.

#### --voxel_size

  3D voxel grid resolution in meters. `0.05` by default.

#### --voxel_num

  Initial voxel pool capacity (number of voxels pre-allocated). `4096` by default.

#### --voxel_buf_cap

  Maximum number of KV entries stored per buffer voxel. `8` by default.

#### --voxel_piv_cap

  Maximum number of KV entries stored per pivot voxel. `4` by default.

#### --voxel_conf

  Point confidence threshold for voxel position assignment. Not set by default (no filtering).

#### --voxel_backend

  Voxel KV merging backend: `python` or `cuda` (requires the optional CUDA extension). `python` by default.

#### -alloc / --allocator

  Voxel pool memory allocator strategy: `static`, `slab`, or `segment`. `slab` by default.

#### --temperature

  Attention score temperature for H2O heavy-hitter KV selection. `0.9` by default.

#### --size

  Input image resolution (height or shorter side). `518` by default.

#### --kf_every

  Keyframe sampling interval — process every N-th frame. `1` (every frame) by default.

**Environment variables:**

Set `VERBOSE=1` to print per-frame KV cache statistics:

```bash
VERBOSE=1 python eval/long_recon/launch.py ...
```

Set `MERGER_MEM_PROFILE=1` to log CUDA memory fragmentation details at every cache-cleanup step:

```bash
MERGER_MEM_PROFILE=1 python eval/long_recon/launch.py ...
```

## Project Structure

```
STAC/
├── main.py                       # Legacy CLI (basic modes only; use eval/ scripts for STAC)
├── eval.py                       # Legacy unified evaluation
├── requirements.txt              # Python dependencies
│
├── src/
│   ├── model_wrapper.py          # Unified load / run interface for all backbones
│   ├── causalvggt/               # CausalVGGT adapter (backbone-agnostic)
│   │   ├── models/               #   CausalVGGT, CausalAggregator
│   │   ├── layers/               #   SparseAttention, Block, RoPE
│   │   ├── heads/                #   CameraHead, DPTHead
│   │   ├── utils/                #   geometry, pose_enc, rotation
│   │   └── stream_session.py     #   Streaming inference session
│   ├── stac/                     # Sparse Token Attention Cache (plug-and-play)
│   │   ├── kv_manager.py         #   Base KV window manager
│   │   ├── h2o.py                #   Heavy-hitter KV selection
│   │   ├── stac_voxel.py         #   Voxel-based KV merging + retrieval
│   │   ├── voxel.py              #   BinaryVoxel / HashVoxel
│   │   ├── merger.py             #   KV merge with slab/segment allocator
│   │   └── allocator.py          #   Slab / segment memory allocators
│   └── vggt/                     # Original VGGT (upstream reference)
│
├── eval/
│   ├── long_recon/               # 3D reconstruction (NRGBD, 7-Scenes, DTU)
│   ├── cam_pose/                 # Camera pose (TUM, ScanNet, Sintel)
│   ├── video_depth/              # Video depth estimation
│   └── utils/                    # Shared evaluation utilities
│
├── demo/
│   ├── app_stream3r.py           # Gradio web demo
│   ├── demo_viser.py             # Viser 3D visualization
│   └── demo_colmap.py            # COLMAP export
│
└── merger-cuda/                  # Optional CUDA extension for fast KV merging
```

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