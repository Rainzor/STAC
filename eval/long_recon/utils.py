import numpy as np
from scipy.spatial import cKDTree as KDTree
import torch
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize


def accuracy(gt_points, rec_points, gt_normals=None, rec_normals=None):
    gt_points_kd_tree = KDTree(gt_points)
    distances, idx = gt_points_kd_tree.query(rec_points, workers=8)
    acc = np.mean(distances)
    acc_median = np.median(distances)
    if gt_normals is not None and rec_normals is not None:
        normal_dot = np.sum(gt_normals[idx] * rec_normals, axis=-1)
        normal_dot = np.abs(normal_dot)
        return acc, acc_median, np.mean(normal_dot), np.median(normal_dot)
    return acc, acc_median


def completion(gt_points, rec_points, gt_normals=None, rec_normals=None):
    gt_points_kd_tree = KDTree(rec_points)
    distances, idx = gt_points_kd_tree.query(gt_points, workers=8)
    comp = np.mean(distances)
    comp_median = np.median(distances)
    if gt_normals is not None and rec_normals is not None:
        normal_dot = np.sum(gt_normals * rec_normals[idx], axis=-1)
        normal_dot = np.abs(normal_dot)
        return comp, comp_median, np.mean(normal_dot), np.median(normal_dot)
    return comp, comp_median


def depth2disparity(depth, return_mask=False):
    if isinstance(depth, torch.Tensor):
        disparity = torch.zeros_like(depth)
    elif isinstance(depth, np.ndarray):
        disparity = np.zeros_like(depth)
    non_negtive_mask = depth > 0
    disparity[non_negtive_mask] = 1.0 / depth[non_negtive_mask]
    if return_mask:
        return disparity, non_negtive_mask
    return disparity


def _absolute_error_loss(params, predicted_depth, ground_truth_depth):
    s, t = params
    predicted_aligned = s * predicted_depth + t
    return np.sum(np.abs(predicted_aligned - ground_truth_depth))


def absolute_value_scaling(predicted_depth, ground_truth_depth, s=1, t=0):
    predicted_depth_np = predicted_depth.cpu().numpy().reshape(-1)
    ground_truth_depth_np = ground_truth_depth.cpu().numpy().reshape(-1)
    result = minimize(
        _absolute_error_loss, [s, t],
        args=(predicted_depth_np, ground_truth_depth_np),
    )
    s, t = result.x
    return s, t


def absolute_value_scaling2(
    predicted_depth, ground_truth_depth,
    s_init=1.0, t_init=0.0, lr=1e-4, max_iters=1000, tol=1e-6,
):
    s = torch.tensor([s_init], requires_grad=True,
                     device=predicted_depth.device, dtype=predicted_depth.dtype)
    t = torch.tensor([t_init], requires_grad=True,
                     device=predicted_depth.device, dtype=predicted_depth.dtype)
    optimizer = torch.optim.Adam([s, t], lr=lr)
    prev_loss = None
    for _ in range(max_iters):
        optimizer.zero_grad()
        loss = torch.sum(torch.abs(s * predicted_depth + t - ground_truth_depth))
        loss.backward()
        optimizer.step()
        if prev_loss is not None and torch.abs(prev_loss - loss) < tol:
            break
        prev_loss = loss.item()
    return s.detach().item(), t.detach().item()



def completion_ratio(gt_points, rec_points, dist_th=0.05):
    gen_points_kd_tree = KDTree(rec_points)
    distances, _ = gen_points_kd_tree.query(gt_points)
    comp_ratio = np.mean((distances < dist_th).astype(np.float32))
    return comp_ratio


def compute_iou(pred_vox, target_vox):
    # Get voxel indices
    v_pred_indices = [voxel.grid_index for voxel in pred_vox.get_voxels()]
    v_target_indices = [voxel.grid_index for voxel in target_vox.get_voxels()]

    # Convert to sets for set operations
    v_pred_filled = set(tuple(np.round(x, 4)) for x in v_pred_indices)
    v_target_filled = set(tuple(np.round(x, 4)) for x in v_target_indices)

    # Compute intersection and union
    intersection = v_pred_filled & v_target_filled
    union = v_pred_filled | v_target_filled

    # Compute IoU
    iou = len(intersection) / len(union)
    return iou
