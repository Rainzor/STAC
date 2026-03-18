
import os
import re
from copy import deepcopy
from pathlib import Path
import logging

import evo.main_ape as main_ape
import evo.main_rpe as main_rpe
import matplotlib.pyplot as plt
import numpy as np
from evo.core import sync
from evo.core.metrics import PoseRelation, Unit
from evo.core.trajectory import PoseTrajectory3D
import torch

import cupoch as cph
import open3d as o3d
from .utils import *

# add time to the logger
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("EvalLogger")
logger.setLevel(logging.INFO)

def eval_depth(
    predicted_depth_original,
    ground_truth_depth_original,
    max_depth=80,
    custom_mask=None,
    post_clip_min=None,
    post_clip_max=None,
    pre_clip_min=None,
    pre_clip_max=None,
    align_with_lstsq=False,
    align_with_lad=False,
    align_with_lad2=False,
    metric_scale=False,
    lr=1e-4,
    max_iters=1000,
    use_gpu=False,
    align_with_scale=False,
    disp_input=False,
    verbose=True,
):
    """
    Evaluate the depth map using various metrics and return a depth error parity map, with an option for least squares alignment.

    Args:
        predicted_depth (numpy.ndarray or torch.Tensor): The predicted depth map. (S,H,W)
        ground_truth_depth (numpy.ndarray or torch.Tensor): The ground truth depth map. (S,H,W)
        max_depth (float): The maximum depth value to consider. Default is 80 meters.
        align_with_lstsq (bool): If True, perform least squares alignment of the predicted depth with ground truth.

    Returns:
        dict: A dictionary containing the evaluation metrics.
        torch.Tensor: The depth error parity map.
    """
    if isinstance(predicted_depth_original, np.ndarray):
        predicted_depth_original = torch.from_numpy(predicted_depth_original)
    if isinstance(ground_truth_depth_original, np.ndarray):
        ground_truth_depth_original = torch.from_numpy(ground_truth_depth_original)
    if custom_mask is not None and isinstance(custom_mask, np.ndarray):
        custom_mask = torch.from_numpy(custom_mask)

    # if the dimension is 3, flatten to 2d along the batch dimension
    if predicted_depth_original.dim() == 3:
        _, h, w = predicted_depth_original.shape
        predicted_depth_original = predicted_depth_original.view(-1, w)
        ground_truth_depth_original = ground_truth_depth_original.view(-1, w)
        if custom_mask is not None:
            custom_mask = custom_mask.view(-1, w)

    # put to device
    if use_gpu:
        predicted_depth_original = predicted_depth_original.cuda()
        ground_truth_depth_original = ground_truth_depth_original.cuda()
    else:
        predicted_depth_original = predicted_depth_original.cpu()
        ground_truth_depth_original = ground_truth_depth_original.cpu()

    # Filter out depths greater than max_depth
    if max_depth is not None:
        mask = (ground_truth_depth_original > 0) & (
            ground_truth_depth_original < max_depth
        )
    else:
        mask = ground_truth_depth_original > 0
    predicted_depth = predicted_depth_original[mask]
    ground_truth_depth = ground_truth_depth_original[mask]

    # Clip the depth values
    if pre_clip_min is not None:
        predicted_depth = torch.clamp(predicted_depth, min=pre_clip_min)
    if pre_clip_max is not None:
        predicted_depth = torch.clamp(predicted_depth, max=pre_clip_max)

    if disp_input:  # align the pred to gt in the disparity space
        real_gt = ground_truth_depth.clone()
        ground_truth_depth = 1 / (ground_truth_depth + 1e-8)

    # various alignment methods
    if metric_scale:
        predicted_depth = predicted_depth
    elif align_with_lstsq:
        # Convert to numpy for lstsq
        predicted_depth_np = predicted_depth.cpu().numpy().reshape(-1, 1)
        ground_truth_depth_np = ground_truth_depth.cpu().numpy().reshape(-1, 1)

        # Add a column of ones for the shift term
        A = np.hstack([predicted_depth_np, np.ones_like(predicted_depth_np)])

        # Solve for scale (s) and shift (t) using least squares
        result = np.linalg.lstsq(A, ground_truth_depth_np, rcond=None)
        s, t = result[0][0], result[0][1]

        # convert to torch tensor
        s = torch.tensor(s, device=predicted_depth_original.device)
        t = torch.tensor(t, device=predicted_depth_original.device)

        # Apply scale and shift
        predicted_depth = s * predicted_depth + t
    elif align_with_lad:
        s, t = absolute_value_scaling(
            predicted_depth,
            ground_truth_depth,
            s=torch.median(ground_truth_depth) / torch.median(predicted_depth),
        )
        predicted_depth = s * predicted_depth + t
    elif align_with_lad2:
        s_init = (
            torch.median(ground_truth_depth) / torch.median(predicted_depth)
        ).item()
        s, t = absolute_value_scaling2(
            predicted_depth,
            ground_truth_depth,
            s_init=s_init,
            lr=lr,
            max_iters=max_iters,
        )
        predicted_depth = s * predicted_depth + t
    elif align_with_scale:
        # Compute initial scale factor 's' using the closed-form solution (L2 norm)
        dot_pred_gt = torch.nanmean(ground_truth_depth)
        dot_pred_pred = torch.nanmean(predicted_depth)
        s = dot_pred_gt / dot_pred_pred

        # Iterative reweighted least squares using the Weiszfeld method
        for _ in range(10):
            # Compute residuals between scaled predictions and ground truth
            residuals = s * predicted_depth - ground_truth_depth
            abs_residuals = (
                residuals.abs() + 1e-8
            )  # Add small constant to avoid division by zero

            # Compute weights inversely proportional to the residuals
            weights = 1.0 / abs_residuals

            # Update 's' using weighted sums
            weighted_dot_pred_gt = torch.sum(
                weights * predicted_depth * ground_truth_depth
            )
            weighted_dot_pred_pred = torch.sum(weights * predicted_depth**2)
            s = weighted_dot_pred_gt / weighted_dot_pred_pred

        # Optionally clip 's' to prevent extreme scaling
        s = s.clamp(min=1e-3)

        # Detach 's' if you want to stop gradients from flowing through it
        s = s.detach()

        # Apply the scale factor to the predicted depth
        predicted_depth = s * predicted_depth

    else:
        # Align the predicted depth with the ground truth using median scaling
        scale_factor = torch.median(ground_truth_depth) / torch.median(predicted_depth)
        predicted_depth *= scale_factor

    if disp_input:
        # convert back to depth
        ground_truth_depth = real_gt
        predicted_depth = depth2disparity(predicted_depth)

    # Clip the predicted depth values
    if post_clip_min is not None:
        predicted_depth = torch.clamp(predicted_depth, min=post_clip_min)
    if post_clip_max is not None:
        predicted_depth = torch.clamp(predicted_depth, max=post_clip_max)

    if custom_mask is not None:
        assert custom_mask.shape == ground_truth_depth_original.shape
        mask_within_mask = custom_mask.cpu()[mask]
        predicted_depth = predicted_depth[mask_within_mask]
        ground_truth_depth = ground_truth_depth[mask_within_mask]

    # Calculate the metrics
    error_depth = torch.abs(predicted_depth - ground_truth_depth) / ground_truth_depth
    # filiter out the outliers where abs_rel > 1
    # in case there is no valid pixel, set the metrics to 0
    mask_inlier = error_depth < 1.0
    predicted_depth = predicted_depth[mask_inlier]
    ground_truth_depth = ground_truth_depth[mask_inlier]
    abs_rel = torch.mean(
        torch.abs(predicted_depth - ground_truth_depth) / ground_truth_depth
    ).item()
    sq_rel = torch.mean(
        ((predicted_depth - ground_truth_depth) ** 2) / ground_truth_depth
    ).item()

    # Correct RMSE calculation
    rmse = torch.sqrt(torch.mean((predicted_depth - ground_truth_depth) ** 2)).item()

    # Clip the depth values to avoid log(0)
    predicted_depth = torch.clamp(predicted_depth, min=1e-5)
    log_rmse = torch.sqrt(
        torch.mean((torch.log(predicted_depth) - torch.log(ground_truth_depth)) ** 2)
    ).item()

    # Calculate the accuracy thresholds
    max_ratio = torch.maximum(
        predicted_depth / ground_truth_depth, ground_truth_depth / predicted_depth
    )
    threshold_0 = torch.mean((max_ratio < 1.0).float()).item()
    threshold_1 = torch.mean((max_ratio < 1.25).float()).item()
    threshold_2 = torch.mean((max_ratio < 1.25**2).float()).item()
    threshold_3 = torch.mean((max_ratio < 1.25**3).float()).item()

    # Compute the depth error parity map
    if metric_scale:
        predicted_depth_original = predicted_depth_original
        if disp_input:
            predicted_depth_original = depth2disparity(predicted_depth_original)
        depth_error_parity_map = (
            torch.abs(predicted_depth_original - ground_truth_depth_original)
            / ground_truth_depth_original
        )
    elif align_with_lstsq or align_with_lad or align_with_lad2:
        predicted_depth_original = predicted_depth_original * s + t
        if disp_input:
            predicted_depth_original = depth2disparity(predicted_depth_original)
        depth_error_parity_map = (
            torch.abs(predicted_depth_original - ground_truth_depth_original)
            / ground_truth_depth_original
        )
    elif align_with_scale:
        predicted_depth_original = predicted_depth_original * s
        if disp_input:
            predicted_depth_original = depth2disparity(predicted_depth_original)
        depth_error_parity_map = (
            torch.abs(predicted_depth_original - ground_truth_depth_original)
            / ground_truth_depth_original
        )
    else:
        predicted_depth_original = predicted_depth_original * scale_factor
        if disp_input:
            predicted_depth_original = depth2disparity(predicted_depth_original)
        depth_error_parity_map = (
            torch.abs(predicted_depth_original - ground_truth_depth_original)
            / ground_truth_depth_original
        )

    # Reshape the depth_error_parity_map back to the original image size
    depth_error_parity_map_full = torch.zeros_like(ground_truth_depth_original)
    depth_error_parity_map_full = torch.where(
        mask, depth_error_parity_map, depth_error_parity_map_full
    )

    predict_depth_map_full = predicted_depth_original
    gt_depth_map_full = torch.zeros_like(ground_truth_depth_original)
    gt_depth_map_full = torch.where(
        mask, ground_truth_depth_original, gt_depth_map_full
    )

    num_valid_pixels = (
        torch.sum(mask).item()
        if custom_mask is None
        else torch.sum(mask_within_mask).item()
    )
    if num_valid_pixels == 0:
        (
            abs_rel,
            sq_rel,
            rmse,
            log_rmse,
            threshold_0,
            threshold_1,
            threshold_2,
            threshold_3,
        ) = (0, 0, 0, 0, 0, 0, 0, 0)

    results = {
        "Abs Rel": abs_rel,
        "Sq Rel": sq_rel,
        "RMSE": rmse,
        "Log RMSE": log_rmse,
        "delta < 1.": threshold_0,
        "delta < 1.25": threshold_1,
        "delta < 1.25^2": threshold_2,
        "delta < 1.25^3": threshold_3,
        "valid_pixels": num_valid_pixels,
    }
    if verbose:
        logger.info(f"🎯 Abs Rel: {abs_rel:.4f}, Sq Rel: {sq_rel:.4f}, RMSE: {rmse:.4f}, Log RMSE: {log_rmse:.4f}")
        logger.info(f"🎯 Thresholds: <1: {threshold_0:.4f}, <1.25: {threshold_1:.4f}, <1.25^2: {threshold_2:.4f}, <1.25^3: {threshold_3:.4f}")

    return (
        results,
        depth_error_parity_map_full,
        predict_depth_map_full,
        gt_depth_map_full,
    )

def eval_scene(batch_data, predictions, dataset_type, 
               save_dir=None, revisit=1, 
               use_gpu=True, mode="from_depth"):
    from eval.utils.geometry import geotrf
    from eval.long_recon.criterion import Regr3D_t_ScaleShiftInv, L21
    
    criterion = Regr3D_t_ScaleShiftInv(L21, norm_mode=False, gt_scale=True)
    batch_cpu = [
                    {
                        k: v.to('cpu') if isinstance(v, torch.Tensor) else v for k, v in sample.items()
                    } for sample in batch_data
                ]
    gts = batch_cpu
    if mode == "from_depth":
        world_points = torch.from_numpy(predictions["world_points_from_depth"]).unsqueeze(0)
        confs = torch.from_numpy(predictions["depth_conf"]).unsqueeze(0)
    else:
        world_points = torch.from_numpy(predictions["world_points"]).unsqueeze(0)
        confs = torch.from_numpy(predictions["world_points_conf"]).unsqueeze(0)

    preds = []
    for idx in range(world_points.shape[1]):
        preds.append(
            {
                "pts3d": world_points[0][idx : idx + 1].cpu(),
                "conf": confs[0][idx : idx + 1],
            }
        )

    valid_length = len(preds)//revisit
    preds = preds[-valid_length:]
    batch = batch_data[-valid_length:]

    # Evaluation
    gt_pts, pred_pts, gt_factor, pr_factor, masks, monitoring = (
        criterion.get_all_pts3d_t(gts, preds)
    )
    pred_scale, gt_scale, pred_shift_z, gt_shift_z = (
        monitoring["pred_scale"],
        monitoring["gt_scale"],
        monitoring["pred_shift_z"],
        monitoring["gt_shift_z"],
    )

    in_camera1 = None
    pts_all = []
    pts_gt_all = []
    images_all = []
    masks_all = []
    conf_all = []

    for j, view in enumerate(batch):
        if in_camera1 is None:
            in_camera1 = view["camera_pose"][0].cpu()

        image = view["img"].permute(0, 2, 3, 1).cpu().numpy()[0]
        mask = view["valid_mask"].cpu().numpy()[0]

        pts = pred_pts[j].cpu().numpy()[0]
        conf = preds[j]["conf"].cpu().data.numpy()[0]

        pts_gt = gt_pts[j].detach().cpu().numpy()[0]

        H, W = image.shape[:2]
        cx = W // 2
        cy = H // 2
        l, t = cx - 112, cy - 112
        r, b = cx + 112, cy + 112
        image = image[t:b, l:r]
        mask = mask[t:b, l:r]
        pts = pts[t:b, l:r]
        pts_gt = pts_gt[t:b, l:r]

        #### Align predicted 3D points to the ground truth
        pts[..., -1] += gt_shift_z.cpu().numpy().item()
        pts = geotrf(in_camera1, pts)

        pts_gt[..., -1] += gt_shift_z.cpu().numpy().item()
        pts_gt = geotrf(in_camera1, pts_gt)

        images_all.append((image[None, ...] + 1.0) / 2.0)
        pts_all.append(pts[None, ...])
        pts_gt_all.append(pts_gt[None, ...])
        masks_all.append(mask[None, ...])
        conf_all.append(conf[None, ...])

    images_all = np.concatenate(images_all, axis=0)
    pts_all = np.concatenate(pts_all, axis=0)
    pts_gt_all = np.concatenate(pts_gt_all, axis=0)
    masks_all = np.concatenate(masks_all, axis=0)

    save_params = {}
    save_params["images_all"] = images_all
    save_params["pts_all"] = pts_all
    save_params["pts_gt_all"] = pts_gt_all
    save_params["masks_all"] = masks_all



    if "DTU" in dataset_type:
        threshold = 100
    else:
        threshold = 0.1
    pts_all_masked = pts_all[masks_all > 0]
    pts_gt_all_masked = pts_gt_all[masks_all > 0]
    images_all_masked = images_all[masks_all > 0]
    logger.info(f"Number of points: {pts_all_masked.shape[0]}")
    if not use_gpu:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_all_masked.reshape(-1, 3))
        pcd.colors = o3d.utility.Vector3dVector(images_all_masked.reshape(-1, 3))


        pcd_gt = o3d.geometry.PointCloud()
        pcd_gt.points = o3d.utility.Vector3dVector(pts_gt_all_masked.reshape(-1, 3))
        pcd_gt.colors = o3d.utility.Vector3dVector(images_all_masked.reshape(-1, 3))

        trans_init = np.eye(4)
        if save_dir is not None and os.path.exists(f"{save_dir}/points3d.npy"):
            logger.info(f"Loading existing 3D points from {save_dir}/points3d.npy")
            loaded = np.load(f"{save_dir}/points3d.npy", allow_pickle=True).item()
            if "transformation" in loaded:
                trans_init = loaded["transformation"]

        logger.info(f"Running ICP with threshold {threshold}...")
        reg_p2p = o3d.pipelines.registration.registration_icp(
            pcd,
            pcd_gt,
            threshold,
            trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        )

        transformation = reg_p2p.transformation
        save_params["transformation"] = transformation
        pcd = pcd.transform(transformation)
        pcd.estimate_normals()
        pcd_gt.estimate_normals()

        gt_normal = np.asarray(pcd_gt.normals)
        rec_normal = np.asarray(pcd.normals)

        gt_points = np.asarray(pcd_gt.points)
        rec_points = np.asarray(pcd.points)
    # acc, acc_med, nc1, nc1_med = accuracy(
    #     pcd_gt.points, pcd.points, gt_normal, rec_normal
    # )
    # comp, comp_med, nc2, nc2_med = completion(
    #     pcd_gt.points, pcd.points, gt_normal, rec_normal
    # )
    else:

        # 构造 GPU 点云
        pcd = cph.geometry.PointCloud()
        pcd.points = cph.utility.Vector3fVector(pts_all_masked.reshape(-1, 3).astype(np.float32))
        pcd.colors = cph.utility.Vector3fVector(images_all_masked.reshape(-1, 3).astype(np.float32))

        pcd_gt = cph.geometry.PointCloud()
        pcd_gt.points = cph.utility.Vector3fVector(pts_gt_all_masked.reshape(-1, 3).astype(np.float32))
        pcd_gt.colors = cph.utility.Vector3fVector(images_all_masked.reshape(-1, 3).astype(np.float32))

        trans_init = np.eye(4, dtype=np.float32)
        pcd.estimate_normals()
        pcd_gt.estimate_normals()


        logger.info(f"Running GPU ICP with threshold {threshold}...")
        reg_p2p = cph.registration.registration_icp(
            pcd,
            pcd_gt,
            threshold,
            trans_init,
            cph.registration.TransformationEstimationPointToPlane(),
            cph.registration.ICPConvergenceCriteria(max_iteration=50),
        )

        transformation = reg_p2p.transformation
        save_params["transformation"] = transformation
        pcd.transform(transformation)

        # 估计法向量 (GPU)
        pcd.estimate_normals()
        pcd_gt.estimate_normals()

        # 转 numpy
        gt_points = np.asarray(pcd_gt.points.cpu())
        rec_points = np.asarray(pcd.points.cpu())
        gt_normal = np.asarray(pcd_gt.normals.cpu())
        rec_normal = np.asarray(pcd.normals.cpu())

    logger.info("Calculating metrics ...")
    acc, acc_med, nc1, nc1_med = accuracy(gt_points, rec_points, gt_normal, rec_normal)
    comp, comp_med, nc2, nc2_med = completion(gt_points, rec_points, gt_normal, rec_normal)


    logger.info(f"🎯 Acc: {acc:.4f}, Comp: {comp:.4f}, NC1: {nc1:.4f}, NC2: {nc2:.4f}")
    logger.info(f"🎯 Acc_med: {acc_med:.4f}, Comp_med: {comp_med:.4f}, NC1_med: {nc1_med:.4f}, NC2_med: {nc2_med:.4f}")

    # Save metrics as JSON
    metrics_data = {
        "accuracy": float(acc),
        "completion": float(comp),
        "normal_consistency_1": float(nc1),
        "normal_consistency_2": float(nc2),
        "accuracy_median": float(acc_med),
        "completion_median": float(comp_med),
        "normal_consistency_1_median": float(nc1_med),
        "normal_consistency_2_median": float(nc2_med),
        "use_gpu": use_gpu,
    }

    # if save_dir is not None:
    #     os.makedirs(save_dir, exist_ok=True)
    #     np.save(f"{save_dir}/points3d.npy", save_params)
    #     logger.info(f"Saved 3D points to {save_dir}/points3d.npy")

    # release cuda memory
    torch.cuda.empty_cache()

    return metrics_data


def make_traj(args) -> PoseTrajectory3D:
    if isinstance(args, tuple) or isinstance(args, list):
        traj, tstamps = args
        return PoseTrajectory3D(
            positions_xyz=traj[:, :3],
            orientations_quat_wxyz=traj[:, 3:],
            timestamps=tstamps,
        )
    assert isinstance(args, PoseTrajectory3D), type(args)
    return deepcopy(args)


def eval_traj(pred_traj, gt_traj=None, seq="", filename="", sample_stride=1):

    if sample_stride > 1:
        pred_traj[0] = pred_traj[0][::sample_stride]
        pred_traj[1] = pred_traj[1][::sample_stride]
        if gt_traj is not None:
            updated_gt_traj = []
            updated_gt_traj.append(gt_traj[0][::sample_stride])
            updated_gt_traj.append(gt_traj[1][::sample_stride])
            gt_traj = updated_gt_traj

    pred_traj = make_traj(pred_traj)

    if gt_traj is not None:
        gt_traj = make_traj(gt_traj)

        if pred_traj.timestamps.shape[0] == gt_traj.timestamps.shape[0]:
            pred_traj.timestamps = gt_traj.timestamps
        else:
            print(pred_traj.timestamps.shape[0], gt_traj.timestamps.shape[0])

        gt_traj, pred_traj = sync.associate_trajectories(gt_traj, pred_traj)

    metrics = {}

    # ATE
    traj_ref = gt_traj
    traj_est = pred_traj

    ate_result = main_ape.ape(
        traj_ref,
        traj_est,
        est_name="traj",
        pose_relation=PoseRelation.translation_part,
        align=True,
        correct_scale=True,
    )

    ate = ate_result.stats["rmse"]
    metrics["seq"] = seq
    metrics["ate"] = {
        "max": ate_result.stats["max"],
        "min": ate_result.stats["min"],
        "mean": ate_result.stats["mean"],
        "median": ate_result.stats["median"],
        "rmse": ate_result.stats["rmse"],
        "sse": ate_result.stats["sse"],
        "std": ate_result.stats["std"],
    }
    # print(ate_result.np_arrays['error_array'])
    # exit()

    # RPE rotation and translation
    delta_list = [1]
    metrics["rpe_rot"] = {
        "max": [],
        "min": [],
        "mean": [],
        "median": [],
        "rmse": [],
        "sse": [],
        "std": [],
    }
    for delta in delta_list:
        rpe_rots_result = main_rpe.rpe(
            traj_ref,
            traj_est,
            est_name="traj",
            pose_relation=PoseRelation.rotation_angle_deg,
            align=True,
            correct_scale=True,
            delta=delta,
            delta_unit=Unit.frames,
            rel_delta_tol=0.01,
            all_pairs=True,
        )

        for k in metrics["rpe_rot"].keys():
            if k in rpe_rots_result.stats.keys():
                metrics["rpe_rot"][k].append(rpe_rots_result.stats[k])
    
    for k in metrics["rpe_rot"].keys():
        metrics["rpe_rot"][k] = np.mean(metrics["rpe_rot"][k])

    metrics["rpe_trans"] = {
        "max": [],
        "min": [],
        "mean": [],
        "median": [],
        "rmse": [],
        "sse": [],
        "std": [],
    }
    for delta in delta_list:
        rpe_transs_result = main_rpe.rpe(
            traj_ref,
            traj_est,
            est_name="traj",
            pose_relation=PoseRelation.translation_part,
            align=True,
            correct_scale=True,
            delta=delta,
            delta_unit=Unit.frames,
            rel_delta_tol=0.01,
            all_pairs=True,
        )

        for k in metrics["rpe_trans"].keys():
            if k in rpe_transs_result.stats.keys():
                metrics["rpe_trans"][k].append(rpe_transs_result.stats[k])

    for k in metrics["rpe_trans"].keys():
        metrics["rpe_trans"][k] = np.mean(metrics["rpe_trans"][k])

    metrics["means"] = {
        "max": [],
        "min": [],
        "mean": [],
        "median": [],
        "rmse": [],
        "sse": [],
        "std": [],
    }
    for k in metrics["means"].keys():
        for super_k in ["ate","rpe_rot","rpe_trans"]:
            metrics["means"][k].append(metrics[super_k][k])
        metrics["means"][k] = np.mean(metrics["means"][k])
    

    if filename != "":
        with open(filename, "w+") as f:
            f.write(f"Seq: {seq} \n\n")
            f.write(f"{ate_result}")
            f.write(f"{rpe_rots_result}")
            f.write(f"{rpe_transs_result}")

        print(f"Save results to {filename}")

    return metrics


def eval_metrics_first_pose_align_last_pose(
    pred_traj, gt_traj=None, seq="", filename="", figpath="", sample_stride=1
):
    if sample_stride > 1:
        pred_traj[0] = pred_traj[0][::sample_stride]
        pred_traj[1] = pred_traj[1][::sample_stride]
        if gt_traj is not None:
            gt_traj = [gt_traj[0][::sample_stride], gt_traj[1][::sample_stride]]
    pred_traj = make_traj(pred_traj)
    if gt_traj is not None:
        gt_traj = make_traj(gt_traj)

        if pred_traj.timestamps.shape[0] == gt_traj.timestamps.shape[0]:
            pred_traj.timestamps = gt_traj.timestamps
        else:
            print(
                "Different number of poses:",
                pred_traj.timestamps.shape[0],
                gt_traj.timestamps.shape[0],
            )

        gt_traj, pred_traj = sync.associate_trajectories(gt_traj, pred_traj)

    if gt_traj is not None and pred_traj is not None:
        if len(gt_traj.poses_se3) > 0 and len(pred_traj.poses_se3) > 0:
            first_gt_pose = gt_traj.poses_se3[0]
            first_pred_pose = pred_traj.poses_se3[0]
            # T = (first_gt_pose) * inv(first_pred_pose)
            T = first_gt_pose @ np.linalg.inv(first_pred_pose)

            # Apply T to every predicted pose
            aligned_pred_poses = []
            for pose in pred_traj.poses_se3:
                aligned_pred_poses.append(T @ pose)
            aligned_pred_traj = PoseTrajectory3D(
                poses_se3=aligned_pred_poses,
                timestamps=np.array(pred_traj.timestamps),
                # optionally copy other fields if your make_traj object has them
            )
            pred_traj = aligned_pred_traj  # .poses_se3 = aligned_pred_poses
        plot_trajectory_2d(
            pred_traj,
            gt_traj,
            title=seq,
            filename=figpath,
            align=False,
            correct_scale=False,
        )

    if gt_traj is not None and len(gt_traj.poses_se3) > 0:
        gt_traj = PoseTrajectory3D(
            poses_se3=[gt_traj.poses_se3[-1]], timestamps=[gt_traj.timestamps[-1]]
        )
    if pred_traj is not None and len(pred_traj.poses_se3) > 0:
        pred_traj = PoseTrajectory3D(
            poses_se3=[pred_traj.poses_se3[-1]], timestamps=[pred_traj.timestamps[-1]]
        )

    ate_result = main_ape.ape(
        gt_traj,
        pred_traj,
        est_name="traj",
        pose_relation=PoseRelation.translation_part,
        align=False,  # <-- important
        correct_scale=False,  # <-- important
    )
    ate = ate_result.stats["rmse"]
    with open(filename, "w+") as f:
        f.write(f"Seq: {seq}\n\n")
        f.write(f"{ate_result}")

    print(f"Save results to {filename}")

    return ate


def plot_trajectory_2d(
    pred_traj, gt_traj=None, title="", save_dir="", 
    align=True, correct_scale=True,
    tag=None
):
    # === 构造 Trajectory 对象 ===
    pred_traj = make_traj(pred_traj)

    if gt_traj is not None:
        gt_traj = make_traj(gt_traj)
        if pred_traj.timestamps.shape[0] == gt_traj.timestamps.shape[0]:
            pred_traj.timestamps = gt_traj.timestamps
        else:
            print("WARNING", pred_traj.timestamps.shape[0], gt_traj.timestamps.shape[0])

        # 时间同步
        gt_traj, pred_traj = sync.associate_trajectories(gt_traj, pred_traj)

        # 对齐
        if align:
            pred_traj.align(gt_traj, correct_scale=correct_scale)

    # === Matplotlib 绘图 ===
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.grid(True)

    # Ground Truth 轨迹
    if gt_traj is not None:
        ax.plot(
            gt_traj.positions_xyz[:, 0],
            gt_traj.positions_xyz[:, 2],
            "--",
            color="gray",
            label="Ground Truth"
        )
        # 起点和终点
        xyz_start = gt_traj.positions_xyz[0]
        xyz_end = gt_traj.positions_xyz[-1]
        ax.scatter(xyz_start[0], xyz_start[2], c="green", marker="o", s=100, label="GT Start")
        ax.scatter(xyz_end[0], xyz_end[2], c="red", marker="x", s=100, label="GT End")

    # Predicted 轨迹
    ax.plot(
        pred_traj.positions_xyz[:, 0],
        pred_traj.positions_xyz[:, 2],
        "-",
        color="blue",
        label="Predicted"
    )
    ax.scatter(
        pred_traj.positions_xyz[0, 0],
        pred_traj.positions_xyz[0, 2],
        c="cyan", marker="o", s=80, label="Pred Start"
    )
    ax.scatter(
        pred_traj.positions_xyz[-1, 0],
        pred_traj.positions_xyz[-1, 2],
        c="magenta", marker="x", s=80, label="Pred End"
    )

    ax.legend()
    plt.tight_layout()

    # === 保存或显示 ===
    if save_dir != "":
        save_path = os.path.join(save_dir, f'traj2d_error_{tag}.png') if tag is not None else os.path.join(save_dir, 'traj2d_error.png')
        plt.savefig(save_path, dpi=200)
        logger.info(f"Saved trajectory to {save_path}")
    else:
        plt.show()

    plt.close(fig)

import os
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from evo.core import sync
from evo.core.trajectory import PoseTrajectory3D
from scipy.spatial.transform import Rotation as R

def make_traj(args) -> PoseTrajectory3D:
    if isinstance(args, (tuple, list)):
        traj, tstamps = args
        return PoseTrajectory3D(
            positions_xyz=traj[:, :3],
            orientations_quat_wxyz=traj[:, 3:],
            timestamps=tstamps,
        )
    assert isinstance(args, PoseTrajectory3D), type(args)
    return deepcopy(args)


def plot_trajectory2d_cam(
    pred_traj,
    gt_traj=None,
    title="Trajectory with Camera (X-Z plane)",
    save_dir="",
    tag=None,
    align=True,
    correct_scale=True,
    max_cam_arrows=50,
    arrow_scale=0.2,
):
    """
    在 X–Z 平面上绘制预测轨迹与GT轨迹，并显示两者相机朝向（箭头）
    - pred_traj: (traj, timestamps) 或 PoseTrajectory3D
    - gt_traj: 同上，可为 None
    - arrow_scale: 控制箭头长度
    """

    # === 构造 Trajectory 对象 ===
    pred_traj = make_traj(pred_traj)

    if gt_traj is not None:
        gt_traj = make_traj(gt_traj)
        if pred_traj.timestamps.shape[0] == gt_traj.timestamps.shape[0]:
            pred_traj.timestamps = gt_traj.timestamps
        else:
            print("WARNING", pred_traj.timestamps.shape[0], gt_traj.timestamps.shape[0])
        gt_traj, pred_traj = sync.associate_trajectories(gt_traj, pred_traj)
        if align:
            pred_traj.align(gt_traj, correct_scale=correct_scale)

    # === 绘制 ===
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.grid(True)
    ax.axis("equal")

    # === Ground Truth ===
    if gt_traj is not None:
        ax.plot(gt_traj.positions_xyz[:, 0], gt_traj.positions_xyz[:, 2],
                "--", color="gray", label="Ground Truth")
        ax.scatter(gt_traj.positions_xyz[0, 0], gt_traj.positions_xyz[0, 2],
                   c="green", marker="o", s=100, label="GT Start")
        ax.scatter(gt_traj.positions_xyz[-1, 0], gt_traj.positions_xyz[-1, 2],
                   c="red", marker="x", s=100, label="GT End")

    # === Predicted Trajectory ===
    ax.plot(pred_traj.positions_xyz[:, 0], pred_traj.positions_xyz[:, 2],
            "-", color="blue", label="Predicted")
    ax.scatter(pred_traj.positions_xyz[0, 0], pred_traj.positions_xyz[0, 2],
               c="cyan", marker="o", s=80, label="Pred Start")
    ax.scatter(pred_traj.positions_xyz[-1, 0], pred_traj.positions_xyz[-1, 2],
               c="magenta", marker="x", s=80, label="Pred End")

    # === 绘制相机朝向箭头 ===
    def draw_orient_arrows(traj, color, alpha=1.0, label=None, zorder=3):
        num_frames = traj.positions_xyz.shape[0]
        idx_step = max(1, num_frames // max_cam_arrows)
        arrow_indices = np.arange(0, num_frames, idx_step)

        positions = traj.positions_xyz[arrow_indices]
        quats_wxyz = traj.orientations_quat_wxyz[arrow_indices]

        # wxyz → xyzw
        r = R.from_quat(quats_wxyz[:, [1, 2, 3, 0]])
        forward_vectors = r.apply(np.array([0, 0, 1]))  # 相机朝向（前向z轴）

        for i, (p, v) in enumerate(zip(positions, forward_vectors)):
            x0, z0 = p[0], p[2]
            dx, dz = v[0] * arrow_scale, v[2] * arrow_scale
            ax.arrow(
                x0, z0, dx, dz,
                head_width=arrow_scale * 0.4,
                head_length=arrow_scale * 0.6,
                fc=color, ec=color, alpha=alpha,
                length_includes_head=True,
                lw=1.2 if alpha > 0.9 else 0.8,
                zorder=zorder
            )

    # === 绘制两种朝向 ===
    if gt_traj is not None:
        draw_orient_arrows(gt_traj, color="lightgray", alpha=0.5, label="GT Orient", zorder=1)
    draw_orient_arrows(pred_traj, color="orange", alpha=0.9, label="Pred Orient", zorder=2)

    ax.legend(loc="best")
    plt.tight_layout()

    # === 保存或显示 ===
    if save_dir != "":
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"traj_xz_cam_{tag or 'result'}.png")
        plt.savefig(save_path, dpi=200)
        print(f"[Saved] Trajectory with orientation comparison → {save_path}")
    else:
        plt.show()

    plt.close(fig)

def plot_trajectory_dualplane(
    pred_traj, gt_traj=None,
    title="", save_dir="",
    align=True, correct_scale=True,
    tag=None
):
    # === 构造 Trajectory 对象 ===
    pred_traj = make_traj(pred_traj)

    if gt_traj is not None:
        gt_traj = make_traj(gt_traj)
        if pred_traj.timestamps.shape[0] == gt_traj.timestamps.shape[0]:
            pred_traj.timestamps = gt_traj.timestamps
        else:
            print("WARNING", pred_traj.timestamps.shape[0], gt_traj.timestamps.shape[0])

        # 时间同步
        gt_traj, pred_traj = sync.associate_trajectories(gt_traj, pred_traj)

        # 对齐
        if align:
            pred_traj.align(gt_traj, correct_scale=correct_scale)

    # === 创建并列图像 (X-Z 与 X-Y) ===
    fig, axs = plt.subplots(1, 2, figsize=(14, 7))
    planes = [("X–Z plane", (0, 2)), ("X–Y plane", (0, 1))]

    for ax, (plane_name, (ix, iy)) in zip(axs, planes):
        ax.set_title(f"{plane_name}")
        ax.set_xlabel(["X","Y","Z"][ix])
        ax.set_ylabel(["X","Y","Z"][iy])
        ax.grid(True)
        ax.axis("equal")

        # --- Ground Truth ---
        if gt_traj is not None:
            ax.plot(
                gt_traj.positions_xyz[:, ix],
                gt_traj.positions_xyz[:, iy],
                "--", color="gray", label="Ground Truth"
            )
            ax.scatter(gt_traj.positions_xyz[0, ix], gt_traj.positions_xyz[0, iy],
                       c="green", marker="o", s=100, label="GT Start")
            ax.scatter(gt_traj.positions_xyz[-1, ix], gt_traj.positions_xyz[-1, iy],
                       c="red", marker="x", s=100, label="GT End")

        # --- Prediction ---
        ax.plot(
            pred_traj.positions_xyz[:, ix],
            pred_traj.positions_xyz[:, iy],
            "-", color="blue", label="Predicted"
        )
        ax.scatter(pred_traj.positions_xyz[0, ix], pred_traj.positions_xyz[0, iy],
                   c="cyan", marker="o", s=80, label="Pred Start")
        ax.scatter(pred_traj.positions_xyz[-1, ix], pred_traj.positions_xyz[-1, iy],
                   c="magenta", marker="x", s=80, label="Pred End")

        ax.legend(loc="best")

    plt.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # === 保存或显示 ===
    if save_dir != "":
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"traj_dualplane_{tag or 'result'}.png")
        plt.savefig(save_path, dpi=200)
        print(f"[Saved] Trajectory comparison to {save_path}")
    else:
        plt.show()

    plt.close(fig)

def plot_trajectory_3d(
    pred_traj, gt_traj=None, 
    title="", save_dir="", 
    align=True, correct_scale=True,
    tag=None
):
    pred_traj = make_traj(pred_traj)

    if gt_traj is not None:
        gt_traj = make_traj(gt_traj)
        if pred_traj.timestamps.shape[0] == gt_traj.timestamps.shape[0]:
            pred_traj.timestamps = gt_traj.timestamps
        else:
            print("WARNING", pred_traj.timestamps.shape[0], gt_traj.timestamps.shape[0])

        gt_traj, pred_traj = sync.associate_trajectories(gt_traj, pred_traj)

        if align:
            pred_traj.align(gt_traj, correct_scale=correct_scale)

    # 开始绘制3D轨迹
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title)

    # Ground truth 轨迹
    if gt_traj is not None:
        xs, ys, zs = gt_traj.positions_xyz[:, 0], gt_traj.positions_xyz[:, 1], gt_traj.positions_xyz[:, 2]
        ax.plot(xs, ys, zs, "--", color="gray", label="Ground Truth")
        # 起点和终点
        ax.scatter(xs[0], ys[0], zs[0], c="green", marker="o", s=100, label="GT Start")
        ax.scatter(xs[-1], ys[-1], zs[-1], c="red", marker="x", s=100, label="GT End")

    # Predicted 轨迹
    xs, ys, zs = pred_traj.positions_xyz[:, 0], pred_traj.positions_xyz[:, 1], pred_traj.positions_xyz[:, 2]
    ax.plot(xs, ys, zs, "-", color="blue", label="Predicted")
    ax.scatter(xs[0], ys[0], zs[0], c="lime", marker="o", s=100, label="Pred Start")
    ax.scatter(xs[-1], ys[-1], zs[-1], c="darkred", marker="x", s=100, label="Pred End")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()

    plt.tight_layout()
    if save_dir != "":
        save_path = os.path.join(save_dir, f'traj3d_error_{tag}.png') if tag is not None else os.path.join(save_dir, 'traj3d_error.png')
        plt.savefig(save_path, dpi=200)
        logger.info(f"Saved trajectory to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def save_trajectory_tum_format(traj, filename):
    traj = make_traj(traj)
    tostr = lambda a: " ".join(map(str, a))
    with Path(filename).open("w") as f:
        for i in range(traj.num_poses):
            f.write(
                f"{traj.timestamps[i]} {tostr(traj.positions_xyz[i])} {tostr(traj.orientations_quat_wxyz[i][[0,1,2,3]])}\n"
            )
    print(f"Saved trajectory to {filename}")


def extract_metrics(file_path):
    with open(file_path, "r") as file:
        content = file.read()

    # Extract metrics using regex
    ate_match = re.search(
        r"APE w.r.t. translation part \(m\).*?rmse\s+([0-9.]+)", content, re.DOTALL
    )
    rpe_trans_match = re.search(
        r"RPE w.r.t. translation part \(m\).*?rmse\s+([0-9.]+)", content, re.DOTALL
    )
    rpe_rot_match = re.search(
        r"RPE w.r.t. rotation angle in degrees \(deg\).*?rmse\s+([0-9.]+)",
        content,
        re.DOTALL,
    )

    ate = float(ate_match.group(1)) if ate_match else 0.0
    rpe_trans = float(rpe_trans_match.group(1)) if rpe_trans_match else 0.0
    rpe_rot = float(rpe_rot_match.group(1)) if rpe_rot_match else 0.0

    return ate, rpe_trans, rpe_rot


def process_directory(directory):
    results = []
    for root, _, files in os.walk(directory):
        if files is not None:
            files = sorted(files)
        for file in files:
            if file.endswith("_metric.txt"):
                file_path = os.path.join(root, file)
                seq_name = file.replace("_eval_metric.txt", "")
                ate, rpe_trans, rpe_rot = extract_metrics(file_path)
                results.append((seq_name, ate, rpe_trans, rpe_rot))

    return results




import os
import glob
from tqdm import tqdm

# Define the merged dataset metadata dictionary
dataset_metadata = {
    "davis": {
        "img_path": "data/davis/DAVIS/JPEGImages/480p",
        "mask_path": "data/davis/DAVIS/masked_images/480p",
        "dir_path_func": lambda img_path, seq: os.path.join(img_path, seq),
        "gt_traj_func": lambda img_path, anno_path, seq: None,
        "traj_format": None,
        "seq_list": None,
        "full_seq": True,
        "mask_path_seq_func": lambda mask_path, seq: os.path.join(mask_path, seq),
        "skip_condition": None,
        "process_func": None,  # Not used in mono depth estimation
    },
    "kitti": {
        "img_path": "data/kitti/depth_selection/val_selection_cropped/image_gathered",  # Default path
        "mask_path": None,
        "dir_path_func": lambda img_path, seq: os.path.join(img_path, seq),
        "gt_traj_func": lambda img_path, anno_path, seq: None,
        "traj_format": None,
        "seq_list": None,
        "full_seq": True,
        "mask_path_seq_func": lambda mask_path, seq: None,
        "skip_condition": None,
        "process_func": lambda args, img_path: process_kitti(args, img_path),
    },
    "bonn": {
        "img_path": "data/bonn/rgbd_bonn_dataset",
        "mask_path": None,
        "dir_path_func": lambda img_path, seq: os.path.join(
            img_path, f"rgbd_bonn_{seq}", "rgb_110"
        ),
        "gt_traj_func": lambda img_path, anno_path, seq: os.path.join(
            img_path, f"rgbd_bonn_{seq}", "groundtruth_110.txt"
        ),
        "traj_format": "tum",
        "seq_list": ["balloon2", "crowd2", "crowd3", "person_tracking2", "synchronous"],
        "full_seq": False,
        "mask_path_seq_func": lambda mask_path, seq: None,
        "skip_condition": None,
        "process_func": lambda args, img_path: process_bonn(args, img_path),
    },
    "nyu": {
        "img_path": "data/nyu-v2/val/nyu_images",
        "mask_path": None,
        "process_func": lambda args, img_path: process_nyu(args, img_path),
    },
    "scannet": {
        "img_path": "data/scannetv2",
        "mask_path": None,
        "dir_path_func": lambda img_path, seq: os.path.join(img_path, seq, "color_90"),
        "gt_traj_func": lambda img_path, anno_path, seq: os.path.join(
            img_path, seq, "pose_90.txt"
        ),
        "traj_format": "replica",
        "seq_list": None,
        "full_seq": True,
        "mask_path_seq_func": lambda mask_path, seq: None,
        "skip_condition": None,  # lambda save_dir, seq: os.path.exists(os.path.join(save_dir, seq)),
        "process_func": lambda args, img_path: process_scannet(args, img_path),
    },
    "scannet-257": {
        "img_path": "data/scannetv2_3_257",
        "mask_path": None,
        "dir_path_func": lambda img_path, seq: os.path.join(img_path, seq, "color_90"),
        "gt_traj_func": lambda img_path, anno_path, seq: os.path.join(
            img_path, seq, "pose_90.txt"
        ),
        "traj_format": "replica",
        "seq_list": None,
        "full_seq": True,
        "mask_path_seq_func": lambda mask_path, seq: None,
        "skip_condition": None,  # lambda save_dir, seq: os.path.exists(os.path.join(save_dir, seq)),
        "process_func": lambda args, img_path: process_scannet(args, img_path),
    },
    "scannet-129": {
        "img_path": "data/scannetv2_3_129",
        "mask_path": None,
        "dir_path_func": lambda img_path, seq: os.path.join(img_path, seq, "color_90"),
        "gt_traj_func": lambda img_path, anno_path, seq: os.path.join(
            img_path, seq, "pose_90.txt"
        ),
        "traj_format": "replica",
        "seq_list": None,
        "full_seq": True,
        "mask_path_seq_func": lambda mask_path, seq: None,
        "skip_condition": None,  # lambda save_dir, seq: os.path.exists(os.path.join(save_dir, seq)),
        "process_func": lambda args, img_path: process_scannet(args, img_path),
    },
    "scannet-65": {
        "img_path": "data/scannetv2_3_65",
        "mask_path": None,
        "dir_path_func": lambda img_path, seq: os.path.join(img_path, seq, "color_90"),
        "gt_traj_func": lambda img_path, anno_path, seq: os.path.join(
            img_path, seq, "pose_90.txt"
        ),
        "traj_format": "replica",
        "seq_list": None,
        "full_seq": True,
        "mask_path_seq_func": lambda mask_path, seq: None,
        "skip_condition": None,  # lambda save_dir, seq: os.path.exists(os.path.join(save_dir, seq)),
        "process_func": lambda args, img_path: process_scannet(args, img_path),
    },
    "scannet-33": {
        "img_path": "data/scannetv2_3_33",
        "mask_path": None,
        "dir_path_func": lambda img_path, seq: os.path.join(img_path, seq, "color_90"),
        "gt_traj_func": lambda img_path, anno_path, seq: os.path.join(
            img_path, seq, "pose_90.txt"
        ),
        "traj_format": "replica",
        "seq_list": None,
        "full_seq": True,
        "mask_path_seq_func": lambda mask_path, seq: None,
        "skip_condition": None,  # lambda save_dir, seq: os.path.exists(os.path.join(save_dir, seq)),
        "process_func": lambda args, img_path: process_scannet(args, img_path),
    },
    "tum": {
        "img_path": "data/tum",
        "mask_path": None,
        "dir_path_func": lambda img_path, seq: os.path.join(img_path, seq, "rgb_90"),
        "gt_traj_func": lambda img_path, anno_path, seq: os.path.join(
            img_path, seq, "groundtruth_90.txt"
        ),
        "traj_format": "tum",
        "seq_list": None,
        "full_seq": True,
        "mask_path_seq_func": lambda mask_path, seq: None,
        "skip_condition": None,
        "process_func": None,
    },
    "sintel": {
        "img_path": "data/sintel/training/final",
        "anno_path": "data/sintel/training/camdata_left",
        "mask_path": None,
        "dir_path_func": lambda img_path, seq: os.path.join(img_path, seq),
        "gt_traj_func": lambda img_path, anno_path, seq: os.path.join(anno_path, seq),
        "traj_format": None,
        "seq_list": [
            "alley_2",
            "ambush_4",
            "ambush_5",
            "ambush_6",
            "cave_2",
            "cave_4",
            "market_2",
            "market_5",
            "market_6",
            "shaman_3",
            "sleeping_1",
            "sleeping_2",
            "temple_2",
            "temple_3",
        ],
        "full_seq": False,
        "mask_path_seq_func": lambda mask_path, seq: None,
        "skip_condition": None,
        "process_func": lambda args, img_path: process_sintel(args, img_path),
    },
}


# Define processing functions for each dataset
def process_kitti(args, img_path):
    for dir in tqdm(sorted(glob.glob(f"{img_path}/*"))):
        filelist = sorted(glob.glob(f"{dir}/*.png"))
        save_dir = f"{args.output_dir}/{os.path.basename(dir)}"
        yield filelist, save_dir


def process_bonn(args, img_path):
    if args.full_seq:
        for dir in tqdm(sorted(glob.glob(f"{img_path}/*/"))):
            filelist = sorted(glob.glob(f"{dir}/rgb/*.png"))
            save_dir = f"{args.output_dir}/{os.path.basename(os.path.dirname(dir))}"
            yield filelist, save_dir
    else:
        seq_list = (
            ["balloon2", "crowd2", "crowd3", "person_tracking2", "synchronous"]
            if args.seq_list is None
            else args.seq_list
        )
        for seq in tqdm(seq_list):
            filelist = sorted(glob.glob(f"{img_path}/rgbd_bonn_{seq}/rgb_110/*.png"))
            save_dir = f"{args.output_dir}/{seq}"
            yield filelist, save_dir


def process_nyu(args, img_path):
    filelist = sorted(glob.glob(f"{img_path}/*.png"))
    save_dir = f"{args.output_dir}"
    yield filelist, save_dir


def process_scannet(args, img_path):
    seq_list = sorted(glob.glob(f"{img_path}/*"))
    for seq in tqdm(seq_list):
        filelist = sorted(glob.glob(f"{seq}/color_90/*.jpg"))
        save_dir = f"{args.output_dir}/{os.path.basename(seq)}"
        yield filelist, save_dir


def process_sintel(args, img_path):
    if args.full_seq:
        for dir in tqdm(sorted(glob.glob(f"{img_path}/*/"))):
            filelist = sorted(glob.glob(f"{dir}/*.png"))
            save_dir = f"{args.output_dir}/{os.path.basename(os.path.dirname(dir))}"
            yield filelist, save_dir
    else:
        seq_list = [
            "alley_2",
            "ambush_4",
            "ambush_5",
            "ambush_6",
            "cave_2",
            "cave_4",
            "market_2",
            "market_5",
            "market_6",
            "shaman_3",
            "sleeping_1",
            "sleeping_2",
            "temple_2",
            "temple_3",
        ]
        for seq in tqdm(seq_list):
            filelist = sorted(glob.glob(f"{img_path}/{seq}/*.png"))
            save_dir = f"{args.output_dir}/{seq}"
            yield filelist, save_dir

