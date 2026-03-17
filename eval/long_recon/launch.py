import os
import sys
from copy import deepcopy

_file_dir = os.path.dirname(os.path.abspath(__file__))   # eval/long_recon/
root_dir  = os.path.dirname(os.path.dirname(_file_dir))  # project root
sys.path.insert(0, root_dir)

from datetime import datetime
import torch
import argparse
import numpy as np
import os.path as osp
import logging
import json

from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm
from scipy.spatial.transform import Rotation

from causalvggt.utils.geometry import unproject_depth_map_to_point_map
from causalvggt.utils.pose_enc import pose_encoding_to_extri_intri
from causalvggt.utils.helper import ImgNorm2Unit as ImgDust3r2Stream3r

from model_wrapper import load_model, run_model
from eval.long_recon.data import SevenScenes, NRGBD, DTU
from eval.long_recon.eval_utils import eval_scene, eval_depth, eval_traj


def _to_json_serializable(obj):
    """Recursively convert torch/numpy types to native Python for JSON dump."""
    if isinstance(obj, dict):
        return {k: _to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_serializable(x) for x in obj]
    if hasattr(obj, "item"):  # torch.Tensor / np.ndarray scalar
        return obj.item()
    if hasattr(obj, "tolist"):  # torch.Tensor / np.ndarray
        return obj.tolist()
    return obj


torch.backends.cuda.matmul.allow_tf32 = True

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
torch.set_num_threads(1)

logger = logging.getLogger("EvalLogger")
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def get_args_parser():
    parser = argparse.ArgumentParser("Reconstruction Evaluation", add_help=False)

    # model
    parser.add_argument("--model_name", type=str, default="causalvggt")
    parser.add_argument("--base_model", type=str, default="stream3r",
                        choices=["stream3r", "streamvggt"],
                        help="Base model for CausalVGGT")

    # output
    parser.add_argument("--output_dir", type=str, default="eval_results/recon")
    parser.add_argument("--save_tag", "--tag", type=str, default=None)
    parser.add_argument("--vis_tag", type=str, default=None)

    # dataset
    parser.add_argument("--dataset_type", type=str, required=True,
                        choices=["7scenes", "NRGBD", "DTU"],
                        help="Dataset type to evaluate")
    parser.add_argument("--scene_name", nargs="*", default=[],
                        help="Specific scene(s) to evaluate (default: all)")
    parser.add_argument("--size", type=int, default=518)
    parser.add_argument("--kf_every", type=int, default=1,
                        help="Keyframe interval")
    parser.add_argument("--num_frames", type=int, default=-1,
                        help="Number of frames to use (-1 for all, NRGBD only)")
    parser.add_argument("--start_frame", type=int, default=0,
                        help="Starting frame index (NRGBD only)")

    # inference mode
    parser.add_argument("--mode", type=str, default="stac",
                        help="Processing mode")
    parser.add_argument("--streaming", action="store_true",
                        help="Use streaming mode")

    # KV cache
    parser.add_argument("--window_size", "-win", type=int, default=0)
    parser.add_argument("--chunk_size", "-ck", type=int, default=1)
    parser.add_argument("--hh_size", "-hh", type=int, default=0)
    parser.add_argument("--retrieval_size", "-ret_sz", type=int, default=0)
    parser.add_argument("--retrieve_buf", "-ret_buf", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--attn_backend", type=str, default="cuda",
                        choices=["cuda", "triton"],
                        help="Attention backend for sparse decode: cuda or triton")
    parser.add_argument("--subsample", type=float, default=1.0,
                        help="Colsum subsampling ratio in (0, 1]")
    parser.add_argument("--pinned", type=int, default=[0], nargs="+")

    # voxel
    parser.add_argument("--voxel_size", type=float, default=0.05)
    parser.add_argument("--voxel_num", type=int, default=4096)
    parser.add_argument("--voxel_conf", type=float, default=None,
                        help="Confidence threshold filter for voxel merging. ")
    parser.add_argument("--voxel_buf_cap", type=int, default=8)
    parser.add_argument("--voxel_piv_cap", type=int, default=4)
    parser.add_argument("--voxel_backend", type=str, default="cuda",
                        choices=["cuda", "python"])
    parser.add_argument("--allocator", "-alloc", type=str, default="segment",
                        choices=["static", "slab", "segment"])

    # eval
    parser.add_argument("--eval_cpu", action="store_true",
                        help="Evaluate on CPU (default: CUDA)")
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "fp16", "bf16"],
        help="Autocast dtype. auto=bf16 on Ampere+ else fp16",
    )
    parser.add_argument("--eval_depth", action="store_true",
                        help="Evaluate depth map metrics (print only)")
    parser.add_argument("--eval_cam", action="store_true",
                        help="Evaluate camera trajectory metrics (print only)")
    parser.add_argument("--no_recon", action="store_true",
                        help="Disable reconstruction evaluation")
    return parser


def find_scene_index(dataset, scene_name):
    scene_ids = []
    if hasattr(dataset, 'scene_list'):
        for name in scene_name:
            try:
                scene_ids.append(dataset.scene_list.index(name))
            except ValueError:
                logger.error(f"Scene '{name}' not found. Available: {dataset.scene_list}")
                return None
    else:
        logger.error("Dataset doesn't have scene_list attribute")
        return None
    return scene_ids


def run(images, model, dtype, device, args):
    frame_num = images.shape[0]
    logger.info(f"Input images shape: {images.shape}")

    model_kwargs = {
        "tag": args.vis_tag,
        "cam_cache_update": False,
        "window_size": args.window_size,
        "hh_size": args.hh_size,
        "retrieval_size": args.retrieval_size,
        "return_buf": args.retrieve_buf,
        "temperature": args.temperature,
        "attn_backend": args.attn_backend,
        "subsample_ratio": args.subsample,
        "voxel_size": args.voxel_size,
        "voxel_num": args.voxel_num,
        "conf_threshold": args.voxel_conf,
        "voxel_buf_cap": args.voxel_buf_cap,
        "voxel_piv_cap": args.voxel_piv_cap,
        "voxel_backend": args.voxel_backend,
        "chunk_size": args.chunk_size,
        "allocator": args.allocator,
        "pinned_frame_indices": args.pinned,
    }

    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=dtype):
            predictions = run_model(model, images,
                                    args.model_name,
                                    streaming=args.streaming,
                                    mode=args.mode,
                                    dtype=dtype,
                                    device=device,
                                    **model_kwargs)
            args.mode = predictions.get("mode", args.mode)
            args.streaming = predictions.get("streaming", args.streaming)
            predictions.pop("mode", None)
            predictions.pop("streaming", None)
    effective_config = predictions.pop("effective_config", None)

    logger.info(f"Model: {args.model_name}, Mode: {args.mode}, Streaming: {args.streaming}")

    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"], predictions["images"].shape[-2:]
    )
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)
    predictions['pose_enc_list'] = None

    depth_map = predictions["depth"]
    predictions["world_points_from_depth"] = unproject_depth_map_to_point_map(
        depth_map, predictions["extrinsic"], predictions["intrinsic"]
    )

    model_stats = {
        "name": args.model_name,
        "base_model": args.base_model,
        "mode": args.mode,
        "streaming": args.streaming,
    }
    if effective_config is not None:
        model_stats.update(effective_config)
    elif "window" in args.mode:
        model_stats["window_size"] = args.window_size
        if args.streaming:
            model_stats["hh_size"] = args.hh_size
            model_stats["retrieval_size"] = args.retrieval_size
            model_stats["chunk_size"] = args.chunk_size
            model_stats["pinned_frame_indices"] = args.pinned
            model_stats["temperature"] = args.temperature
            model_stats["attn_backend"] = args.attn_backend
            model_stats["subsample_ratio"] = args.subsample
            model_stats["voxel_size"] = args.voxel_size
            model_stats["voxel_num"] = args.voxel_num
            model_stats["conf_threshold"] = args.voxel_conf
            if "merge" in args.mode:
                model_stats["allocator"] = args.allocator
                model_stats["return_buf"] = args.retrieve_buf
                model_stats["voxel_buf_cap"] = args.voxel_buf_cap
                model_stats["voxel_piv_cap"] = args.voxel_piv_cap
                model_stats["voxel_backend"] = args.voxel_backend

    out = {"model": model_stats}
    if "timing" in predictions:
        out["Time(ms)"] = predictions["timing"]
    if "merger" in predictions and predictions["merger"]:
        out["Merger"] = _to_json_serializable(predictions["merger"])
    torch.cuda.empty_cache()
    return predictions, out


def _poses_to_tum_traj(poses_c2w):
    """Convert Nx4x4 camera-to-world poses to [xyz+qwqxqyqz, timestamps]."""
    xyz = poses_c2w[:, :3, 3]
    quat_xyzw = Rotation.from_matrix(poses_c2w[:, :3, :3]).as_quat()
    quat_wxyz = quat_xyzw[:, [3, 0, 1, 2]]
    traj = np.concatenate([xyz, quat_wxyz], axis=1).astype(np.float64)
    timestamps = np.arange(traj.shape[0], dtype=np.float64)
    return [traj, timestamps]


def _depth_metric(depth_metrics, *candidate_keys):
    """Read depth metric with backward-compatible key aliases."""
    for key in candidate_keys:
        if key in depth_metrics:
            return depth_metrics[key]
    return float("nan")

def main(args):
    if args.size == 518:  # keep (518, 392) aligned with SparseVGGT for same token count / memory
        resolution = (518, 392)
    elif args.size == 512:
        resolution = (512, 384)
    elif args.size == 224:
        resolution = (224, 224)
    else:
        raise NotImplementedError(f"Unsupported size: {args.size}")

    if args.dataset_type == "7scenes":
        dataset = SevenScenes(
            split="test", ROOT='./data/7scenes',
            resolution=resolution, num_seq=1,
            full_video=True, kf_every=args.kf_every,
        )
        args.voxel_conf = 2.0 if args.voxel_conf is None else args.voxel_conf
    elif args.dataset_type == "NRGBD":
        dataset = NRGBD(
            split="test", ROOT='./data/neural_rgbd',
            resolution=resolution,
            start_frame=args.start_frame,
            num_frames=args.num_frames,
            num_seq=1, full_video=True, kf_every=args.kf_every,
        )
        args.voxel_conf = 4.0 if args.voxel_conf is None else args.voxel_conf
    elif args.dataset_type == "DTU":
        dataset = DTU(
            split="test", ROOT='./data/DTU',
            resolution=resolution, num_seq=1,
            full_video=True, kf_every=1,
        )
        args.voxel_conf = 2.0 if args.voxel_conf is None else args.voxel_conf
    else:
        raise ValueError(f"Unknown dataset type: {args.dataset_type}")

    logger.info(f"Dataset {args.dataset_type} loaded with {len(dataset.scene_list)} scenes")
    logger.info(f"Available scenes: {dataset.scene_list}")

    if args.scene_name == []:
        data_idx = range(len(dataset))
    else:
        data_idx = find_scene_index(dataset, args.scene_name)
        if data_idx is None:
            return
        logger.info(f"Found scene(s) {args.scene_name} at index {data_idx}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "Evaluation currently only supports CUDA device"
    if args.dtype == "bf16":
        dtype = torch.bfloat16
    elif args.dtype == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    logger.info("Using autocast dtype: %s", str(dtype).replace("torch.", ""))

    model = load_model(args.model_name, args.base_model, device)
    os.makedirs(args.output_dir, exist_ok=True)

    all_metrics = {}
    for name_idx in tqdm(data_idx, desc="Evaluating scenes"):
        try:
            basic_metrics = {}
            scene_name = dataset.scene_list[name_idx]
            save_scene_name = f"{scene_name}/{args.save_tag}" if args.save_tag else scene_name
            save_dir = osp.join(args.output_dir, args.dataset_type, args.base_model, save_scene_name)
            os.makedirs(save_dir, exist_ok=True)

            timedelta = datetime.now().strftime("%Y-%m-%d-%H-%M")
            vis_tag = args.vis_tag if args.vis_tag is not None else timedelta

            logger.info(f"Evaluating scene '{scene_name}' [{args.dataset_type}]")
            basic_metrics["clock"]   = timedelta
            basic_metrics["dataset"] = args.dataset_type
            basic_metrics["scene"]   = scene_name
            basic_metrics["kf_every"] = args.kf_every
            batch = default_collate([dataset[name_idx]])
            images = torch.cat([item['img'] for item in batch])
            images = ImgDust3r2Stream3r(images)

            basic_metrics["num_frames"] = images.shape[0]
            predictions, model_metrics = run(images, model, dtype, device, args)
            basic_metrics.update(model_metrics)

            # Reconstruction evaluation (GPU pointmap from depth)
            if not args.no_recon:
                logger.info("📌 Evaluating Reconstruction")
                metrics_data = eval_scene(batch, predictions, args.dataset_type,
                                          save_dir=save_dir, revisit=1, use_gpu=not args.eval_cpu)
                basic_metrics["reconstruction"] = metrics_data

            if args.eval_depth:
                logger.info("📌 Evaluating Depth")
                gt_depth = torch.cat([view["depthmap"] for view in batch], dim=0)  # (S, H, W)
                pred_depth = predictions["depth"]
                if isinstance(pred_depth, np.ndarray):
                    pred_depth = torch.from_numpy(pred_depth)
                if pred_depth.ndim == 4 and pred_depth.shape[-1] == 1:
                    pred_depth = pred_depth[..., 0]
                if pred_depth.ndim == 2:
                    pred_depth = pred_depth.unsqueeze(0)

                _, h_gt, w_gt = gt_depth.shape
                if pred_depth.shape[1:] != (h_gt, w_gt):
                    pred_depth = pred_depth.unsqueeze(1)
                    pred_depth = torch.nn.functional.interpolate(
                        pred_depth, size=(h_gt, w_gt), mode="bicubic", align_corners=False
                    )
                    pred_depth = pred_depth.squeeze(1)

                max_depth = 100.0
                depth_metrics, _, _, _ = eval_depth(
                    pred_depth,
                    gt_depth,
                    max_depth=max_depth,
                    align_with_lad2=True,
                    use_gpu=(device == "cuda"),
                    post_clip_max=max_depth,
                    verbose=False,
                )
                basic_metrics["depth"] = depth_metrics
                abs_rel = _depth_metric(depth_metrics, "abs_rel", "Abs Rel")
                rmse = _depth_metric(depth_metrics, "rmse", "RMSE")
                log_rmse = _depth_metric(depth_metrics, "log_rmse", "Log RMSE")
                d1 = _depth_metric(depth_metrics, "threshold_1", "delta < 1.25")
                logger.info(
                    "🎯 Depth AbsRel: %.4f, RMSE: %.4f, LogRMSE: %.4f, d1: %.4f",
                    abs_rel,
                    rmse,
                    log_rmse,
                    d1,
                )

            if args.eval_cam:
                logger.info("📌 Evaluating Camera Trajectory")
                extrinsic = predictions["extrinsic"]
                if extrinsic.ndim == 2:
                    extrinsic = extrinsic[None, ...]

                pred_c2w = []
                for i in range(extrinsic.shape[0]):
                    pred_w2c = np.eye(4, dtype=np.float64)
                    pred_w2c[:3, :4] = extrinsic[i]
                    pred_c2w.append(np.linalg.inv(pred_w2c))
                pred_c2w = np.stack(pred_c2w, axis=0)

                gt_c2w = []
                for view in batch:
                    gt_pose = view["camera_pose"].squeeze().cpu().numpy()
                    gt_c2w.append(gt_pose)
                gt_c2w = np.stack(gt_c2w, axis=0).astype(np.float64)

                traj_metrics = eval_traj(
                    _poses_to_tum_traj(pred_c2w),
                    _poses_to_tum_traj(gt_c2w),
                )
                basic_metrics["trajectory"] = traj_metrics
                logger.info(
                    "🎯 Traj ATE-RMSE: %.4f, RPE-Trans-RMSE: %.4f, RPE-Rot-RMSE: %.4f",
                    traj_metrics["ate"]["rmse"],
                    traj_metrics["rpe_trans"]["rmse"],
                    traj_metrics["rpe_rot"]["rmse"],
                )

            print_only_mode = args.eval_depth or args.eval_cam
            if not print_only_mode:
                # Save per-scene metrics
                metrics_dir = osp.join(save_dir, "metrics")
                os.makedirs(metrics_dir, exist_ok=True)
                metrics_file = osp.join(metrics_dir, f"metrics_{vis_tag}.json")
                if os.path.exists(metrics_file):
                    import random, string
                    rand_hash = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
                    metrics_file = metrics_file.replace(".json", f"_{rand_hash}.json")
                with open(metrics_file, "w") as f:
                    json.dump(basic_metrics, f, indent=2)
                logger.info(f"📌 Saved metrics to \"{metrics_file}\"")

            all_metrics[scene_name] = basic_metrics

        except Exception as e:
            timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
            os.makedirs("./logs/error", exist_ok=True)
            with open(f"./logs/error/log_{timestamp}.txt", "a") as f:
                f.write(f"Exception in scene {scene_name}: {str(e)}\n")
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                logger.warning(f"OOM in scene {scene_name}, skipping.")
            else:
                raise e

        torch.cuda.empty_cache()

    # Print summary table
    # display columns and their source keys in metrics_data
    col_keys = [
        ("Acc_mean",  "accuracy"),
        ("Comp_mean", "completion"),
        ("NC_mean",   None),           # derived: (nc1 + nc2) / 2
        ("Acc_med",   "accuracy_median"),
        ("Comp_med",  "completion_median"),
        ("NC_med",    None),           # derived: (nc1_med + nc2_med) / 2
    ]
    display_cols = [c for c, _ in col_keys]

    def _get_row_vals(recon):
        row = {}
        for col, key in col_keys:
            if key is not None:
                row[col] = recon.get(key)
            elif col == "NC_mean":
                nc1 = recon.get("normal_consistency_1")
                nc2 = recon.get("normal_consistency_2")
                row[col] = (nc1 + nc2) / 2 if (nc1 is not None and nc2 is not None) else None
            elif col == "NC_med":
                nc1 = recon.get("normal_consistency_1_median")
                nc2 = recon.get("normal_consistency_2_median")
                row[col] = (nc1 + nc2) / 2 if (nc1 is not None and nc2 is not None) else None
        return row

    if not args.no_recon:
        accum = {c: [] for c in display_cols}
        scene_rows = []
        for scene, info in all_metrics.items():
            recon = info.get("reconstruction", {})
            row = {"scene": scene, **_get_row_vals(recon)}
            for c in display_cols:
                if row[c] is not None:
                    accum[c].append(row[c])
            scene_rows.append(row)

        def _fmt(v):
            return f"{v:.4f}" if isinstance(v, float) else (str(v) if v is not None else "-")

        header_cols = ["scene"] + display_cols
        col_w = {c: max(len(c), 8) for c in header_cols}
        for row in scene_rows:
            for c in header_cols:
                col_w[c] = max(col_w[c], len(_fmt(row.get(c))))

        sep = "  "
        header_str = sep.join(c.ljust(col_w[c]) for c in header_cols)
        hline = sep.join("-" * col_w[c] for c in header_cols)
        mean_row = {"scene": f"MEAN({len(scene_rows)})"}
        for c in display_cols:
            vals = accum[c]
            mean_row[c] = sum(vals) / len(vals) if vals else float("nan")

        table_lines = [
            "",
            "📊 Reconstruction Evaluation Summary",
            header_str,
            hline,
        ]
        for row in scene_rows:
            table_lines.append(sep.join(_fmt(row.get(c)).ljust(col_w[c]) for c in header_cols))
        table_lines.append(hline)
        table_lines.append(sep.join(_fmt(mean_row.get(c)).ljust(col_w[c]) for c in header_cols))
        logger.info("\n".join(table_lines))

    if args.eval_depth:
        depth_rows = []
        for scene, info in all_metrics.items():
            depth = info.get("depth")
            if depth is None:
                continue
            depth_rows.append(
                (
                    scene,
                    _depth_metric(depth, "abs_rel", "Abs Rel"),
                    _depth_metric(depth, "rmse", "RMSE"),
                    _depth_metric(depth, "log_rmse", "Log RMSE"),
                    _depth_metric(depth, "threshold_1", "delta < 1.25"),
                )
            )
        if depth_rows:
            logger.info("\n📊 Depth Summary (scene, abs_rel, rmse, log_rmse, d1)")
            for row in depth_rows:
                logger.info("%s: %.4f, %.4f, %.4f, %.4f", row[0], row[1], row[2], row[3], row[4])

    if args.eval_cam:
        traj_rows = []
        for scene, info in all_metrics.items():
            traj = info.get("trajectory")
            if traj is None:
                continue
            traj_rows.append((scene, traj["ate"]["rmse"], traj["rpe_trans"]["rmse"], traj["rpe_rot"]["rmse"]))
        if traj_rows:
            logger.info("\n📊 Trajectory Summary (scene, ate_rmse, rpe_trans_rmse, rpe_rot_rmse)")
            for row in traj_rows:
                logger.info("%s: %.4f, %.4f, %.4f", row[0], row[1], row[2], row[3])

    if len(list(data_idx)) > 1 and not (args.eval_depth or args.eval_cam):
        overall_metrics_dir = osp.join(args.output_dir, args.dataset_type, args.base_model,
                                       "overall_metrics", f"kf_{args.kf_every}")
        os.makedirs(overall_metrics_dir, exist_ok=True)
        suffix = f"_{args.vis_tag}" if args.vis_tag else ""
        overall_metrics_file = osp.join(overall_metrics_dir,
                                        f"overall_metrics{suffix}_{datetime.now().strftime('%Y%m%d_%H%M')}.json")
        with open(overall_metrics_file, "w") as f:
            json.dump(all_metrics, f, indent=2)
        logger.info(f"📌 Saved overall metrics to {overall_metrics_file}")


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join("./logs", args.dataset_type)
    os.makedirs(log_dir, exist_ok=True)
    scene_name = "all" if args.scene_name == [] else args.scene_name[0].replace("/", "-")
    if args.vis_tag:
        scene_name += f"_{args.vis_tag}"
    this_log = os.path.join(log_dir, f"{scene_name}_{timestamp}.txt")
    with open(this_log, "w") as f:
        f.write("python " + " ".join(sys.argv) + "\n")
    logger.info("Logging to %s", this_log)
    main(args)
