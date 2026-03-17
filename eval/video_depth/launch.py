import os
import sys
import logging
import numpy as np
import torch
import argparse
from accelerate import PartialState
from tqdm import tqdm
from PIL import Image
import imageio.v2 as iio

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, root_dir)

logger = logging.getLogger("EvalLogger")
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
from eval.utils.image import load_images_for_eval as load_images
from eval.utils.device import collate_with_cat
from causalvggt.utils.helper import ImgNorm2Unit as ImgDust3r2Stream3r

from eval.video_depth.metadata import dataset_metadata
from eval.video_depth.utils import colorize
from model_wrapper import load_model, run_model

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.backends.cuda.matmul.allow_tf32 = True

# avoid high cpu usage
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
torch.set_num_threads(1)
# ===========================================


def save_depth_maps(pts3ds_self, path, conf_self=None, depth_maps=None,append_cbar=True):
    if depth_maps is None:
        depth_maps = torch.stack([pts3d_self[..., -1] for pts3d_self in pts3ds_self], 0)
    min_depth = depth_maps.min()  # float(torch.quantile(out, 0.01))
    max_depth = depth_maps.max()  # float(torch.quantile(out, 0.99))
    colored_depth = colorize(
        depth_maps,
        cmap_name="Spectral_r",
        range=(min_depth, max_depth),
        append_cbar=append_cbar,
    )
    images = []

    if conf_self is not None:
        conf_selfs = torch.concat(conf_self, 0)
        min_conf = torch.log(conf_selfs.min())  # float(torch.quantile(out, 0.01))
        max_conf = torch.log(conf_selfs.max())  # float(torch.quantile(out, 0.99))
        colored_conf = colorize(
            torch.log(conf_selfs),
            cmap_name="jet",
            range=(min_conf, max_conf),
            append_cbar=append_cbar,
        )

    for i, depth_map in enumerate(colored_depth):
        # Apply color map to depth map
        img_path = f"{path}/frame_{(i):04d}.png"
        if conf_self is None:
            to_save = (depth_map * 255).detach().cpu().numpy().astype(np.uint8)
        else:
            to_save = torch.cat([depth_map, colored_conf[i]], dim=1)
            to_save = (to_save * 255).detach().cpu().numpy().astype(np.uint8)
        iio.imwrite(img_path, to_save)
        images.append(Image.open(img_path))
        np.save(f"{path}/frame_{(i):04d}.npy", depth_maps[i].detach().cpu().numpy())

    return depth_maps


def get_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default="cuda", help="pytorch device")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="value for outdir",
    )
    parser.add_argument(
        "--eval_dataset",
        type=str,
        default="sintel",
        choices=list(dataset_metadata.keys()),
    )
    parser.add_argument("--size", type=int, default=518,
                        help="Image load size: long side for 512/518, short side for 224 (same as cam_pose/long_recon)")

    parser.add_argument(
        "--pose_eval_stride", default=1, type=int, help="stride for pose evaluation"
    )
    parser.add_argument(
        "--seq_list",
        nargs="+",
        default=None,
        help="list of sequences for pose evaluation",
    )
    parser.add_argument("--model_name", type=str, default="causalvggt")
    parser.add_argument("--base_model", type=str, default="stream3r", choices=["stream3r", "streamvggt"], help="Base model for CausalVGGT")

    parser.add_argument("--mode", type=str, default="stac",
                        help="Processing mode")

    parser.add_argument("--use_cam_cache", action="store_true",
                        help="Enable camera cache update during streaming")


    # streaming mode
    parser.add_argument("--streaming", action="store_true",
                        help="Use streaming mode (sequential processing)")
    
    # KV Cache
    parser.add_argument("--pinned", type=int, default=[0], nargs="+",
                        help="List of pinned frame indices (default: [0])")
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
    
    # Voxel
    parser.add_argument("--voxel_size", type=float, default=0.05,
                        help="Voxel size for VoxelSasa KV cache management")
    parser.add_argument("--voxel_num", type=int, default=4096,
                        help="Initial number of voxels for VoxelSasa KV cache management")
    parser.add_argument("--voxel_conf",
                        type=float,
                        default=2.0,
                        help="voxel confidence threshold")
    parser.add_argument("--voxel_buf_cap", type=int, default=8,
                        help="Evicted Buffer capacity for VoxelSasa KV cache management")
    parser.add_argument("--voxel_piv_cap", type=int, default=4,
                        help="Pivot capacity for VoxelSasa KV cache management")
    parser.add_argument("--voxel_backend", type=str, default="cuda",
                        choices=["cuda", "python"],
                        help="Backend type for VoxelSasa KV cache management")
    parser.add_argument("--allocator","-alloc", type=str, default="segment",
                        choices=["static", "slab", "segment"],
                        help="Allocator type for VoxelSasa Merge KV cache")
    return parser



def run(images, model, dtype, device, args):
    # images: [S, 3, H, W]
    if len(images.shape) == 5:
        images = images.squeeze(0)  # remove batch dim
    assert len(images.shape) == 4, f"Expected images to have 4 dimensions [S, 3, H, W], but got {images.shape}"
    assert images.shape[1] == 3
    frame_num = images.shape[0]
    
    model_kwargs = {
        "cam_cache_update": args.use_cam_cache,
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
        with torch.amp.autocast(
            device_type="cuda", dtype=dtype
        ):  
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
    torch.cuda.empty_cache()
    return predictions


def eval_pose_estimation(args, model, save_dir=None):
    metadata = dataset_metadata.get(args.eval_dataset)
    img_path = metadata["img_path"]
    mask_path = metadata["mask_path"]

    ate_mean, rpe_trans_mean, rpe_rot_mean = eval_pose_estimation_dist(
        args, model, save_dir=save_dir, img_path=img_path, mask_path=mask_path)
    return ate_mean, rpe_trans_mean, rpe_rot_mean


def eval_pose_estimation_dist(args,
                              model,
                              img_path,
                              save_dir=None,
                              mask_path=None):
    metadata = dataset_metadata.get(args.eval_dataset)
    model.eval()

    seq_list = args.seq_list

    if seq_list is None:
        full_seq = metadata.get("full_seq", False)
        if full_seq:
            seq_list = os.listdir(img_path)
            seq_list = [
                seq for seq in seq_list
                if os.path.isdir(os.path.join(img_path, seq))
            ]
        else:
            seq_list = metadata.get("seq_list", [])
        seq_list = sorted(seq_list)

    if save_dir is None:
        save_dir = args.output_dir

    distributed_state = PartialState()
    model.to(distributed_state.device)
    device = distributed_state.device

    with distributed_state.split_between_processes(seq_list) as seqs:
        error_log_path = f"{save_dir}/_error_log_{distributed_state.process_index}.txt"  # Unique log file per process
        for seq in tqdm(seqs):
            try:
                dir_path = metadata["dir_path_func"](img_path, seq)

                # Handle skip_condition
                skip_condition = metadata.get("skip_condition", None)
                if skip_condition is not None and skip_condition(
                        save_dir, seq):
                    continue

                mask_path_seq_func = metadata.get("mask_path_seq_func",
                                                  lambda mask_path, seq: None)
                mask_path_seq = mask_path_seq_func(mask_path, seq)

                filelist = [
                    os.path.join(dir_path, name)
                    for name in os.listdir(dir_path)
                ]
                filelist.sort()
                filelist = filelist[::args.pose_eval_stride]

                images = load_images(
                    filelist,
                    size=args.size,
                    verbose=False,
                    crop=False,
                )

                images = collate_with_cat([tuple(images)])
                images = torch.stack([view["img"] for view in images], dim=1)
                images = ImgDust3r2Stream3r(images).to(device)
                predictions = run(images, model, dtype=torch.float16, device=device, args=args)

                logger.info("Finished depth estimation of %d images", len(filelist))

                os.makedirs(f"{save_dir}/{seq}", exist_ok=True)
                save_depth_maps(None,
                                f"{save_dir}/{seq}",
                                conf_self=None, 
                                depth_maps=predictions['depth'].squeeze().cpu())

            except Exception as e:
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
    return None, None, None


def main():
    args = get_args_parser()
    args = args.parse_args()

    model = load_model(args.model_name, args.base_model, args.device)
    save_dir = os.path.join(args.output_dir, args.base_model)
    eval_pose_estimation(args, model, save_dir=save_dir)


if __name__ == "__main__":
    main()
