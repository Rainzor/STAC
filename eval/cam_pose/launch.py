import os
import sys

_file_dir = os.path.dirname(os.path.abspath(__file__))          # eval/cam_pose/
root_dir  = os.path.dirname(os.path.dirname(_file_dir))         # project root
sys.path.insert(0, root_dir)

from datetime import datetime
import torch
import argparse
import numpy as np
import os.path as osp
import logging

from tqdm import tqdm

from eval.utils.geometry import inv
from eval.utils.image import load_images_for_eval as load_images
from eval.utils.device import collate_with_cat


from eval.cam_pose.utils import get_tum_poses
from eval.cam_pose.evo_utils import *
from eval.cam_pose.metadata import dataset_metadata

from causalvggt.utils.geometry import unproject_depth_map_to_point_map
from causalvggt.utils.pose_enc import pose_encoding_to_extri_intri, extri_intri_to_pose_encoding
from causalvggt.utils.helper import ImgNorm2Unit as ImgDust3r2Stream3r

from model_wrapper import load_model, run_model

import json

torch.backends.cuda.matmul.allow_tf32 = True

# avoid high cpu usage
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
torch.set_num_threads(1)
# ===========================================
logger = logging.getLogger("EvalLogger")
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)



# clean up GPU memory
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

def get_args_parser():
    parser = argparse.ArgumentParser("Evaluation of CausalVGGT",
                                     add_help=False)
    parser.add_argument("--model_name", type=str, default="causalvggt")
    parser.add_argument("--base_model", type=str, default="stream3r", choices=["stream3r", "streamvggt"], help="Base model for CausalVGGT")
    parser.add_argument("--voxel_conf",
                        type=float,
                        default=2.0,
                        help="voxel confidence threshold")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="eval_results/all",
        help="value for outdir",
    )
    parser.add_argument("--size", type=int, default=518 ,
                        choices=[224, 512, 518],
                        help="Image load size: long side for 512/518, short side for 224")

    # scene / dataset
    parser.add_argument("--dataset_type", type=str, required=True,
                        choices=["sintel", "scannet", "tum"],
                        help="Dataset type to evaluate")
    parser.add_argument("--scene_name", nargs="*", default=[],
                        help="Specific scene name to evaluate")
    parser.add_argument("--save_tag", "--tag", type=str, default=None,
                        help="Tag for saving results under different settings")
    parser.add_argument("--vis_tag", type=str, default=None)

    parser.add_argument("--mode", type=str, default="stac",
                        help="Processing mode")
    parser.add_argument("--pose_eval_stride",
                        default=1,
                        type=int,
                        help="stride for pose evaluation")

    # streaming mode
    parser.add_argument("--streaming", action="store_true",
                        help="Use streaming mode (sequential processing)")
    parser.add_argument("--chunk_size", '-ck',type=int, default=1,
                        help="Chunk size for chunked processing modes")
    parser.add_argument("--window_size", '-win', type=int, default=0,
                        help="Window size for windowed processing modes")
    parser.add_argument("--hh_size",'-hh', type=int, default=0,
                        help="Number of heavy-hitter patches to keep in H2O attention (0 to disable H2O)")
    parser.add_argument("--retrieval_size",'-ret_sz', type=int, default=0,
                        help="Number of frames to retrieve from in SASA attention (0 to disable retrieval)")
    parser.add_argument("--retrieve_buf",'-ret_buf', action="store_true",
                        help="Whether to return retrieved K/V pairs from the buffer")
    parser.add_argument("--temperature", type=float, default=0.9,
                        help="Temperature for softmax in attention-based retrieval")
    parser.add_argument("--attn_backend", type=str, default="cuda",
                        choices=["cuda", "triton"],
                        help="Attention backend for sparse decode: cuda or triton")
    parser.add_argument("--subsample", type=float, default=1.0,
                        help="Colsum subsampling ratio in (0, 1]")
    parser.add_argument("--voxel_size", type=float, default=0.05,
                        help="Voxel size for VoxelSasa KV cache management")
    parser.add_argument("--voxel_num", type=int, default=4096,
                        help="Initial number of voxels for VoxelSasa KV cache management")
    parser.add_argument("--voxel_buf_cap", type=int, default=8,
                        help="Evicted buffer capacity for VoxelSasa KV cache management")
    parser.add_argument("--voxel_piv_cap", type=int, default=4,
                        help="Pivot capacity for VoxelSasa KV cache management")
    parser.add_argument("--voxel_backend", type=str, default="cuda",
                        choices=["cuda", "python"],
                        help="Backend type for voxel KV cache management")
    parser.add_argument("--allocator","-alloc", type=str, default="segment",
                        choices=["static", "slab", "segment"],
                        help="Allocator type for VoxelSasa Merge KV cache")
    parser.add_argument("--pinned", type=int, default=[0], nargs="+",
                        help="List of pinned frame indices (default: [0])")
    return parser


def run(images, model, dtype, device, args):
    # images: [S, 3, H, W]

    assert len(images.shape) == 4
    assert images.shape[1] == 3
    frame_num = images.shape[0]

    # with torch.autocast('cuda', enabled=False):

    logger.info("📌 Inference Summary")
    logger.info(f"Input images shape: {images.shape}")

    model_kwargs = {
        "tag": args.vis_tag,
        "window_size": args.window_size,
        "hh_size": args.hh_size,
        "retrieval_size": args.retrieval_size,
        "return_buf": args.retrieve_buf,
        "temperature": args.temperature,
        "attn_backend": args.attn_backend,
        "subsample_ratio": args.subsample,
        "voxel_size": args.voxel_size,
        "voxel_num": args.voxel_num,
        "voxel_buf_cap": args.voxel_buf_cap,
        "voxel_piv_cap": args.voxel_piv_cap,
        "conf_threshold": args.voxel_conf,
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
                                    **model_kwargs
                                    )
            # update mode and streaming if changed
            args.mode = predictions.get("mode", args.mode)
            args.streaming = predictions.get("streaming", args.streaming)
            predictions.pop("mode", None)
            predictions.pop("streaming", None)

    logger.info(f"Model: {args.model_name}, Mode: {args.mode}, Streaming: {args.streaming}")

    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"], predictions["images"].shape[-2:]
    )

    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    # Convert tensors to numpy
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)  # remove batch dimension
    predictions['pose_enc_list'] = None # remove pose_enc_list

    # Generate world points from depth map
    logger.info("Computing world points from depth map...")
    depth_map = predictions["depth"]  # (S, H, W, 1)
    world_points = unproject_depth_map_to_point_map(depth_map, predictions["extrinsic"], predictions["intrinsic"])
    predictions["world_points_from_depth"] = world_points

    model_stats = {
        "name": args.model_name,
        "base_model": args.base_model,
        "mode": args.mode,
        "streaming": args.streaming,
    }

    if "window" in args.mode:
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

    metrics = {
        "model": model_stats,
    }

    merger_metrics = predictions.get("merger", {})
    if merger_metrics:
        metrics["merger"] = merger_metrics
    # Clean up
    torch.cuda.empty_cache()
    return predictions, metrics


def main(args):
    # Create dataset based on specified type
    if args.dataset_type in ["sintel", "scannet", "tum"]:
        metadata = dataset_metadata.get(args.dataset_type)
        anno_path = metadata.get("anno_path", None)
        img_path = metadata["img_path"]
        mask_path = metadata["mask_path"]
        seq_list = None
        if seq_list is None:
            if metadata.get("full_seq", False):
                args.full_seq = True
            else:
                seq_list = metadata.get("seq_list", [])
            if args.full_seq:
                seq_list = os.listdir(img_path)
                seq_list = [
                    seq for seq in seq_list
                    if os.path.isdir(os.path.join(img_path, seq))
                ]
            seq_list = sorted(seq_list)

    else:
        raise ValueError(f"Unknown dataset type: {args.dataset_type}")

    logger.info(f"Dataset {args.dataset_type} loaded with {seq_list} scenes")
    logger.info(f"Available scenes: {seq_list}")

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device=="cuda", "Evaluation currently only supports CUDA device"

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    model_name = args.model_name
    model = load_model(model_name, args.base_model, device)
    
    os.makedirs(args.output_dir, exist_ok=True)
    all_matrics = {}
    for seq in tqdm(seq_list, desc="Evaluating scenes"):
        try:
            basic_metrics = {}
            dir_path = metadata["dir_path_func"](img_path, seq)
            scene_name = seq
            if args.save_tag is not None:
                save_dir = osp.join(args.output_dir, args.dataset_type, args.base_model, scene_name, args.save_tag)
            else:
                save_dir = osp.join(args.output_dir, args.dataset_type, args.base_model, scene_name)
            skip_condition = metadata.get("skip_condition", None)
            if skip_condition is not None and skip_condition(save_dir, seq):
                continue
            filelist = [
                    os.path.join(dir_path, name)
                    for name in os.listdir(dir_path)
                ]
            filelist.sort()
            filelist = filelist[::args.pose_eval_stride]
            os.makedirs(save_dir, exist_ok=True)
            timedelta = datetime.now().strftime("%Y-%m-%d-%H-%M")
            if args.vis_tag is not None:
                vis_tag = args.vis_tag
            else:
                vis_tag = timedelta

            logger.info(f"Evaluating scene '{scene_name}' from {args.dataset_type} dataset")
            basic_metrics["clock"] = timedelta
            basic_metrics["dataset"] = args.dataset_type
            basic_metrics["scene"] = scene_name


            # Load images (resolution controlled by --size: 518/512 for long side, 224 for short side)
            images = load_images(
                    filelist,
                    size=args.size,
                    verbose=False,
                    crop=False,
                )

            images = collate_with_cat([tuple(images)])
            images = torch.stack([view["img"] for view in images], dim=1)
            images = ImgDust3r2Stream3r(images).to(device)
            images = images.squeeze(0)  # remove batch dimension
            logger.info("Loaded Images shape: %s", images.shape)



            if args.model_name in ["causalvggt"]:
                logger.info(f"📌 Running inference on {args.model_name} for scene {scene_name} ")

                #! Start of Inference
                basic_metrics["num_frames"] = images.shape[0]
                predictions, model_metrics = run(images, model, dtype, device, args)
                basic_metrics.update(model_metrics)
                #! End of inference
                
                logger.info(f"📌 Evaluating Camera Trajectory")
                pr_poses = []
                extrinsic = predictions["extrinsic"]  # (S, 3, 4) numpy array
                for i in range(extrinsic.shape[0]):
                    pr_poses.append(inv(np.vstack([extrinsic[i], np.array([0, 0, 0, 1])])))  # get camera-to-world
                pred_traj = get_tum_poses(pr_poses)  # [x,y,z,qw,qx,qy,qz]

                gt_traj_file = metadata["gt_traj_func"](img_path, anno_path, seq)
                traj_format = metadata.get("traj_format", None)

                if args.dataset_type == "sintel":
                    gt_traj = load_traj(gt_traj_file=gt_traj_file,
                                        stride=args.pose_eval_stride)
                elif traj_format is not None:
                    gt_traj = load_traj(
                        gt_traj_file=gt_traj_file,
                        traj_format=traj_format,
                        stride=args.pose_eval_stride,
                    )
                else:
                    gt_traj = None

                traj_save_dir = osp.join(save_dir, "trajectory")
                os.makedirs(traj_save_dir, exist_ok=True)

                if gt_traj is not None:
                    metrics_data = eval_metrics(
                        pred_traj,
                        gt_traj,
                        seq=seq,
                        filename=f"{traj_save_dir}/eval_metric_{vis_tag}.txt",
                    )
                    plot_trajectory(pred_traj,
                                    gt_traj,
                                    title=seq,
                                    filename=f"{traj_save_dir}/{vis_tag}.png")
                else:
                    metrics_data = {}

                basic_metrics["trajectory"] = metrics_data
                # Save overall metrics
                metrics_dir = osp.join(save_dir, "metrics")
                os.makedirs(metrics_dir, exist_ok=True)
                if args.vis_tag is None:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
                    metrics_file = osp.join(metrics_dir, f"metrics_{timestamp}.json")
                else:
                    metrics_file = osp.join(metrics_dir, f"metrics_{args.vis_tag}.json")

                if os.path.exists(metrics_file):
                    # random hash to avoid overwriting
                    import random, string
                    rand_hash = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
                    metrics_file = metrics_file.replace(".json", f"_{rand_hash}.json")
                    
                with open(metrics_file, "w") as f:
                    json.dump(basic_metrics, f, indent=2)
                logger.info(f"📌 Saved metrics to \"{metrics_file}\"")

            else:
                raise NotImplementedError(f"Model {args.model_name} not implemented")
            all_matrics[scene_name] = basic_metrics
        except Exception as e:
            timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
            os.makedirs("./logs/error", exist_ok=True)
            error_log_path = f"./logs/error/log_{timestamp}.txt"
            if "out of memory" in str(e):
                # Handle OOM
                torch.cuda.empty_cache()  # Clear the CUDA memory
                with open(error_log_path, "a") as f:
                    f.write(
                        f"OOM error in sequence {seq}, skipping this sequence.\n"
                    )
                logger.warning("OOM error in sequence %s, skipping...", seq)
            elif "Degenerate covariance rank" in str(
                    e) or "Eigenvalues did not converge" in str(e):
                # Handle Degenerate covariance rank exception and Eigenvalues did not converge exception
                with open(error_log_path, "a") as f:
                    f.write(f"Exception in sequence {seq}: {str(e)}\n")
                logger.warning("Traj evaluation error in sequence %s, skipping.", seq)
            else:
                raise e  # Rethrow if it's not an expected exception

            
    # Print per-scene and averaged trajectory metrics
    traj_metrics = ["ate", "rpe_rot", "rpe_trans"]
    traj_stats   = ["mean", "rmse"]
    accum = {m: {s: [] for s in traj_stats} for m in traj_metrics}

    header_cols = ["scene"] + [f"{m}.{s}" for m in traj_metrics for s in traj_stats]
    col_w = {c: max(len(c), 8) for c in header_cols}
    scene_rows = []
    for scene, info in all_matrics.items():
        traj = info.get("trajectory", {})
        row = {"scene": scene}
        for m in traj_metrics:
            for s in traj_stats:
                v = traj.get(m, {}).get(s)
                row[f"{m}.{s}"] = v
                if v is not None:
                    accum[m][s].append(v)
        scene_rows.append(row)
        for c in header_cols:
            v = row.get(c)
            col_w[c] = max(col_w[c], len(f"{v:.4f}" if isinstance(v, float) else str(v) if v else "-"))

    def _fmt(v):
        return f"{v:.4f}" if isinstance(v, float) else (str(v) if v is not None else "-")

    sep = "  "
    header_str = sep.join(c.ljust(col_w[c]) for c in header_cols)
    hline = sep.join("-" * col_w[c] for c in header_cols)
    mean_row = {"scene": f"MEAN({len(scene_rows)})"}
    for m in traj_metrics:
        for s in traj_stats:
            vals = accum[m][s]
            mean_row[f"{m}.{s}"] = sum(vals) / len(vals) if vals else float("nan")

    table_lines = [
        "",
        "📊 Trajectory Evaluation Summary",
        header_str,
        hline,
    ]
    for row in scene_rows:
        table_lines.append(sep.join(_fmt(row.get(c)).ljust(col_w[c]) for c in header_cols))
    table_lines.append(hline)
    table_lines.append(sep.join(_fmt(mean_row.get(c)).ljust(col_w[c]) for c in header_cols))
    logger.info("\n".join(table_lines))

    if len(seq_list) > 1:
        # Save overall metrics for all scenes
        overall_metrics_dir = osp.join(args.output_dir, args.dataset_type, args.base_model, f"overall_metrics")
        os.makedirs(overall_metrics_dir, exist_ok=True)
        if args.vis_tag is None:
            overall_metrics_file = osp.join(overall_metrics_dir, f"overall_metrics_{datetime.now().strftime('%Y%m%d_%H%M')}.json")
        else:
            overall_metrics_file = osp.join(overall_metrics_dir, f"overall_metrics_{args.vis_tag}_{datetime.now().strftime('%Y%m%d_%H%M')}.json")
        with open(overall_metrics_file, "w") as f:
                json.dump(all_matrics, f, indent=2)
        logger.info(f"📌 Saved overall metrics to {overall_metrics_file}")

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = "./logs"
    dataset = args.dataset_type
    log_dir = os.path.join(log_dir, dataset)
    os.makedirs(log_dir, exist_ok=True)
    if args.scene_name == []:
        scene_name = "all"
    else:
        scene_name = args.scene_name[0]
        # replace "/" to "-"
        scene_name = scene_name.replace("/", "-")
    if args.vis_tag is not None:
        scene_name += f"_{args.vis_tag}"
    this_log = os.path.join(log_dir, f"{scene_name}_{timestamp}.txt")
    args.full_seq = False
    args.no_crop = False
    cmd = "python " + " ".join(sys.argv)
    with open(this_log, "w") as f:
        f.write(cmd + "\n")
    logger.info("Logging to %s", this_log)
    main(args)