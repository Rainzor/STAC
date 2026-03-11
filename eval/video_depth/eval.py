import os
import sys
import glob
import json
import argparse

import numpy as np
import torch
import cv2
from tqdm import tqdm
from PIL import Image
import imageio.v2 as iio

from accelerate import PartialState  # 如果不想多卡分布式，可以换成单卡逻辑
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, root_dir)
# Ensure eval package is properly accessible
eval_dir = os.path.join(root_dir, 'eval')
sys.path.insert(0, eval_dir)
src_dir = os.path.join(root_dir, 'src')
sys.path.insert(0, src_dir)
# ---- STream3R & 数据工具 ----
from stream3r.models.stream3r import STream3R
from stream3r.stream_session import StreamSession
from utils.image import load_images_for_eval as load_images
from utils.device import collate_with_cat
from stream3r.utils.utils import ImgDust3r2Stream3r
from demo.model_wrapper import load_model, run_model

# ---- depth eval 工具 ----
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from eval.video_depth.metadata import dataset_metadata
from eval.video_depth.utils import colorize
from eval.video_depth.tools import depth_evaluation, group_by_directory


# ================== 全局设置 ==================
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = True

# 避免高 CPU 占用
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
torch.set_num_threads(1)

# ================== 统一参数解析 ==================
def get_gpu_memory_usage():
    """Get current GPU memory usage in MB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024**2)  # Convert bytes to MB
    return 0.0

def reset_peak_memory():
    """Reset peak memory stats"""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

def get_peak_memory_usage():
    """Get peak GPU memory usage in MB since last reset"""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024**2)  # Convert bytes to MB
    return 0.0
def get_args_parser():
    parser = argparse.ArgumentParser()

    # 预测相关
    parser.add_argument("--device", type=str, default="cuda", help="pytorch device")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="directory to save visualization & logs",
    )
    parser.add_argument(
        "--no_crop", type=bool, default=True, help="whether to crop input data"
    )
    parser.add_argument(
        "--eval_dataset",
        type=str,
        default="sintel",
        choices=list(dataset_metadata.keys()),
    )
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument(
        "--pose_eval_stride", default=1, type=int, help="frame stride for evaluation"
    )
    parser.add_argument(
        "--full_seq",
        action="store_true",
        default=False,
        help="use full sequence list from dataset",
    )
    parser.add_argument(
        "--seq_list",
        nargs="+",
        default=None,
        help="manually specify sequence list",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="",
        help="path to the checkpoint directory (unused if using from_pretrained)",
    )
    parser.add_argument(
        "--save_viz",
        action="store_true",
        default=False,
        help="save colorized depth pngs for visualization",
    )
    parser.add_argument(
        "--save_pred_npy",
        action="store_true",
        default=False,
        help="also save per-frame .npy (not recommended if只想减少IO)",
    )

    # depth evaluation 相关
    parser.add_argument(
        "--align",
        type=str,
        default="scale&shift",
        choices=["scale&shift", "scale", "metric"],
    )

    return parser


# ================== 深度 GT 读取函数 ==================
def depth_read_sintel(filename):
    TAG_FLOAT = 202021.25
    with open(filename, "rb") as f:
        check = np.fromfile(f, dtype=np.float32, count=1)[0]
        assert (
            check == TAG_FLOAT
        ), f"depth_read:: Wrong tag in flow file (should be: {TAG_FLOAT}, is: {check})."
        width = np.fromfile(f, dtype=np.int32, count=1)[0]
        height = np.fromfile(f, dtype=np.int32, count=1)[0]
        size = width * height
        assert (
            width > 0 and height > 0 and 1 < size < 100000000
        ), f"depth_read:: Wrong input size (w={width}, h={height})."
        depth = np.fromfile(f, dtype=np.float32, count=-1).reshape((height, width))
    return depth


def depth_read_bonn(filename):
    depth_png = np.asarray(Image.open(filename))
    assert np.max(depth_png) > 255
    depth = depth_png.astype(np.float64) / 5000.0
    depth[depth_png == 0] = -1.0
    return depth


def depth_read_kitti(filename):
    img_pil = Image.open(filename)
    depth_png = np.array(img_pil, dtype=int)
    assert np.max(depth_png) > 255
    depth = depth_png.astype(float) / 256.0
    depth[depth_png == 0] = -1.0
    return depth


# ================== 预测阶段：获得每个 seq 的深度结果 ==================
def run_inference(args, model):
    """
    返回:
        pred_depths: dict[str, np.ndarray]，每个 seq -> [T, H, W]
        used_seqs:   实际处理的 seq 列表
    """
    metadata = dataset_metadata.get(args.eval_dataset)
    img_path = metadata["img_path"]
    mask_path = metadata["mask_path"]

    # 确定 sequence list
    seq_list = args.seq_list
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

    pred_depths = {}

    # 如果你不想用多卡，可以把下面 PartialState 换成简单的 for-loop
    distributed_state = PartialState()
    model.to(distributed_state.device)
    device_local = distributed_state.device

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    with distributed_state.split_between_processes(seq_list) as seqs:
        for seq in tqdm(seqs, desc="Running STream3R inference"):
            try:
                dir_path = metadata["dir_path_func"](img_path, seq)

                # 可选 skip_condition
                skip_condition = metadata.get("skip_condition", None)
                if skip_condition is not None and skip_condition(args.output_dir, seq):
                    continue

                mask_path_seq_func = metadata.get(
                    "mask_path_seq_func", lambda mask_path, seq: None
                )
                _ = mask_path_seq_func(mask_path, seq)  # 保持接口一致，暂不使用 mask

                filelist = [
                    os.path.join(dir_path, name)
                    for name in os.listdir(dir_path)
                ]
                filelist.sort()
                filelist = filelist[:: args.pose_eval_stride]

                images = load_images(
                    filelist,
                    size=args.size,
                    verbose=True,
                    crop=not args.no_crop,
                    patch_size=14,
                )

                images = collate_with_cat([tuple(images)])
                images = torch.stack([view["img"] for view in images], dim=1)
                images = ImgDust3r2Stream3r(images).to(device_local)

                with torch.no_grad():
                    session = StreamSession(model, mode="causal")
                    for i in range(images.shape[1]):
                        image = images[:, i : i + 1]
                        predictions = session.forward_stream(image)

                depth_pred = predictions["depth"].squeeze().cpu()  # [T, H, W]
                pred_depths[seq] = depth_pred.numpy()

                # # 可选保存彩色 PNG（极少量 IO）
                # if args.output_dir and args.save_viz:
                #     save_dir_seq = os.path.join(args.output_dir, seq)
                #     save_depth_visualizations(
                #         depth_pred,
                #         save_dir_seq,
                #         conf_self=None,
                #         save_npy=args.save_pred_npy,
                #     )

            except Exception as e:
                # 简单 OOM 处理，避免炸掉整个 eval
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    if args.output_dir:
                        error_log_path = os.path.join(
                            args.output_dir,
                            f"_error_log_{distributed_state.process_index}.txt",
                        )
                        with open(error_log_path, "a") as f:
                            f.write(f"OOM error in sequence {seq}, skipping.\n")
                    print(f"OOM error in sequence {seq}, skipping...")
                else:
                    raise e

    # 只保留当前进程处理的 seq
    return pred_depths


# ================== 评估阶段：直接使用内存中的 pred_depths ==================
def evaluate_depth_sintel(args, pred_depths):
    # 确定是否 full（这里主要影响你想评哪些 seq，
    # 真正路径我们用 glob 从 data/sintel/training/depth/ 里取）
    if len(pred_depths) == 0:
        print("No predictions to evaluate for Sintel.")
        return

    # 构建 GT depth 路径列表
    # full 情况：所有 depth 文件；否则只取部分 seq
    # 这里沿用原始脚本的 seq_list 定义
    # 但真正匹配时，我们只对 pred_depths 中存在的 seq 做评估
    depth_pathes = glob.glob("data/sintel/training/depth/*/*.dpt")
    depth_pathes = sorted(depth_pathes)
    grouped_gt_depth = group_by_directory(depth_pathes)

    gathered_depth_metrics = []

    for key in tqdm(grouped_gt_depth.keys(), desc="Evaluating Sintel depth"):
        # key 是形如 data/sintel/training/depth/ALLEY_2 这样的目录
        seq_name = os.path.basename(key)
        if seq_name not in pred_depths:
            # 说明该 seq 没有预测，跳过
            continue

        gt_pathes = grouped_gt_depth[key]
        # 与预测阶段一样的 stride
        gt_pathes = gt_pathes[:: args.pose_eval_stride]

        gt_depth = np.stack(
            [depth_read_sintel(gt_path) for gt_path in gt_pathes], axis=0
        )  # [T, H, W]

        pr_depth = pred_depths[seq_name]  # [T, H', W']

        # 空间分辨率对齐
        if pr_depth.shape[1:] != gt_depth.shape[1:]:
            pr_depth_resized = np.stack(
                [
                    cv2.resize(
                        pr,
                        (gt_depth.shape[2], gt_depth.shape[1]),
                        interpolation=cv2.INTER_CUBIC,
                    )
                    for pr in pr_depth
                ],
                axis=0,
            )
        else:
            pr_depth_resized = pr_depth

        # 对齐方式，同时限定 max_depth=70，与原始脚本一致
        if args.align == "scale&shift":
            depth_results, error_map, depth_predict, depth_gt = depth_evaluation(
                pr_depth_resized,
                gt_depth,
                max_depth=70,
                align_with_lad2=True,
                use_gpu=True,
                post_clip_max=70,
            )
        elif args.align == "scale":
            depth_results, error_map, depth_predict, depth_gt = depth_evaluation(
                pr_depth_resized,
                gt_depth,
                max_depth=70,
                align_with_scale=True,
                use_gpu=True,
                post_clip_max=70,
            )
        else:  # metric
            depth_results, error_map, depth_predict, depth_gt = depth_evaluation(
                pr_depth_resized,
                gt_depth,
                max_depth=70,
                metric_scale=True,
                use_gpu=True,
                post_clip_max=70,
            )

        gathered_depth_metrics.append(depth_results)

    if not gathered_depth_metrics:
        print("No Sintel sequences matched between prediction and GT.")
        return

    depth_log_path = os.path.join(args.output_dir, f"result_{args.align}.json")
    average_metrics = {
        key: np.average(
            [metrics[key] for metrics in gathered_depth_metrics],
            weights=[metrics["valid_pixels"] for metrics in gathered_depth_metrics],
        )
        for key in gathered_depth_metrics[0].keys()
        if key != "valid_pixels"
    }
    print("Average Sintel depth evaluation metrics:", average_metrics)
    with open(depth_log_path, "w") as f:
        json.dump(average_metrics, f)


def evaluate_depth_bonn(args, pred_depths):
    if len(pred_depths) == 0:
        print("No predictions to evaluate for Bonn.")
        return

    seq_list = ["balloon2", "crowd2", "crowd3", "person_tracking2", "synchronous"]

    # GT depth path
    depth_pathes_folder = [
        f"data/bonn/rgbd_bonn_dataset/rgbd_bonn_{seq}/depth_110/*.png"
        for seq in seq_list
    ]
    depth_pathes = []
    for depth_pathes_folder_i in depth_pathes_folder:
        depth_pathes += glob.glob(depth_pathes_folder_i)
    depth_pathes = sorted(depth_pathes)

    grouped_gt_depth = group_by_directory(depth_pathes, idx=-2)
    gathered_depth_metrics = []

    for key in tqdm(grouped_gt_depth.keys(), desc="Evaluating Bonn depth"):
        # key 类似 rgbd_bonn_balloon2
        seq_name = key.split("_")[-1]
        if seq_name not in pred_depths:
            continue
        gt_pathes = grouped_gt_depth[key]
        gt_pathes = gt_pathes[:: args.pose_eval_stride]

        gt_depth = np.stack(
            [depth_read_bonn(gt_path) for gt_path in gt_pathes], axis=0
        )
        pr_depth = pred_depths[seq_name]

        if pr_depth.shape[1:] != gt_depth.shape[1:]:
            pr_depth_resized = np.stack(
                [
                    cv2.resize(
                        pr,
                        (gt_depth.shape[2], gt_depth.shape[1]),
                        interpolation=cv2.INTER_CUBIC,
                    )
                    for pr in pr_depth
                ],
                axis=0,
            )
        else:
            pr_depth_resized = pr_depth

        if args.align == "scale&shift":
            depth_results, error_map, depth_predict, depth_gt = depth_evaluation(
                pr_depth_resized,
                gt_depth,
                max_depth=70,
                align_with_lad2=True,
                use_gpu=True,
            )
        elif args.align == "scale":
            depth_results, error_map, depth_predict, depth_gt = depth_evaluation(
                pr_depth_resized,
                gt_depth,
                max_depth=70,
                align_with_scale=True,
                use_gpu=True,
            )
        else:  # metric
            depth_results, error_map, depth_predict, depth_gt = depth_evaluation(
                pr_depth_resized,
                gt_depth,
                max_depth=70,
                metric_scale=True,
                use_gpu=True,
            )
        gathered_depth_metrics.append(depth_results)

    if not gathered_depth_metrics:
        print("No Bonn sequences matched between prediction and GT.")
        return

    depth_log_path = os.path.join(args.output_dir, f"result_{args.align}.json")
    average_metrics = {
        key: np.average(
            [metrics[key] for metrics in gathered_depth_metrics],
            weights=[metrics["valid_pixels"] for metrics in gathered_depth_metrics],
        )
        for key in gathered_depth_metrics[0].keys()
        if key != "valid_pixels"
    }
    print("Average Bonn depth evaluation metrics:", average_metrics)
    with open(depth_log_path, "w") as f:
        json.dump(average_metrics, f)


def evaluate_depth_kitti(args, pred_depths):
    if len(pred_depths) == 0:
        print("No predictions to evaluate for KITTI.")
        return

    depth_pathes = glob.glob(
        "data/kitti/depth_selection/val_selection_cropped/groundtruth_depth_gathered/*/*.png"
    )
    depth_pathes = sorted(depth_pathes)
    grouped_gt_depth = group_by_directory(depth_pathes)

    gathered_depth_metrics = []

    for key in tqdm(grouped_gt_depth.keys(), desc="Evaluating KITTI depth"):
        # key 类似 .../2011_10_03_drive_0027_sync
        seq_name = os.path.basename(key)
        if seq_name not in pred_depths:
            continue

        gt_pathes = grouped_gt_depth[key]
        gt_pathes = gt_pathes[:: args.pose_eval_stride]

        gt_depth = np.stack(
            [depth_read_kitti(gt_path) for gt_path in gt_pathes], axis=0
        )
        pr_depth = pred_depths[seq_name]

        if pr_depth.shape[1:] != gt_depth.shape[1:]:
            pr_depth_resized = np.stack(
                [
                    cv2.resize(
                        pr,
                        (gt_depth.shape[2], gt_depth.shape[1]),
                        interpolation=cv2.INTER_CUBIC,
                    )
                    for pr in pr_depth
                ],
                axis=0,
            )
        else:
            pr_depth_resized = pr_depth

        if args.align == "scale&shift":
            depth_results, error_map, depth_predict, depth_gt = depth_evaluation(
                pr_depth_resized,
                gt_depth,
                max_depth=None,
                align_with_lad2=True,
                use_gpu=True,
            )
        elif args.align == "scale":
            depth_results, error_map, depth_predict, depth_gt = depth_evaluation(
                pr_depth_resized,
                gt_depth,
                max_depth=None,
                align_with_scale=True,
                use_gpu=True,
            )
        else:  # metric
            depth_results, error_map, depth_predict, depth_gt = depth_evaluation(
                pr_depth_resized,
                gt_depth,
                max_depth=None,
                metric_scale=True,
                use_gpu=True,
            )
        gathered_depth_metrics.append(depth_results)

    if not gathered_depth_metrics:
        print("No KITTI sequences matched between prediction and GT.")
        return

    depth_log_path = os.path.join(args.output_dir, f"result_{args.align}.json")
    average_metrics = {
        key: np.average(
            [metrics[key] for metrics in gathered_depth_metrics],
            weights=[metrics["valid_pixels"] for metrics in gathered_depth_metrics],
        )
        for key in gathered_depth_metrics[0].keys()
        if key != "valid_pixels"
    }
    print("Average KITTI depth evaluation metrics:", average_metrics)
    with open(depth_log_path, "w") as f:
        json.dump(average_metrics, f)


def evaluate_depth(args, pred_depths):
    if args.eval_dataset == "sintel":
        evaluate_depth_sintel(args, pred_depths)
    elif args.eval_dataset == "bonn":
        evaluate_depth_bonn(args, pred_depths)
    elif args.eval_dataset == "kitti":
        evaluate_depth_kitti(args, pred_depths)
    else:
        print(f"Depth evaluation for dataset {args.eval_dataset} is not implemented.")

# ================== 主入口 ==================
def main():
    args = get_args_parser().parse_args()

    if args.eval_dataset == "sintel":
        args.full_seq = True
    args.no_crop = True

    model = STream3R.from_pretrained("yslan/STream3R").to(args.device)
    model.eval()

    # 一次脚本完成：预测 + 评估
    pred_depths = run_inference(args, model)
    evaluate_depth(args, pred_depths)


if __name__ == "__main__":
    main()
