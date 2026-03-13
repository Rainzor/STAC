import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from causalvggt.models.aggregator import CausalAggregator
from causalvggt.heads.camera_head import CameraHead
from causalvggt.heads.dpt_head import DPTHead
from causalvggt.heads.track_head import TrackHead

import logging

logger = logging.getLogger("CausalVGGT")


class CausalVGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024,
                 enable_camera=True, enable_point=True, enable_depth=True, enable_track=False,
                 base_model='stream3r'
                 ):
        super().__init__()

        self.aggregator: CausalAggregator = CausalAggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)

        self.camera_head: CameraHead = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.point_head: DPTHead = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1") if enable_point else None
        self.depth_head: DPTHead = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1") if enable_depth else None
        self.track_head: TrackHead = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size) if enable_track else None

        self.enable_camera = enable_camera
        self.enable_point = enable_point
        self.enable_depth = enable_depth
        self.enable_track = enable_track

    def set_camhead(self, status: bool):
        if self.camera_head is not None:
            self.enable_camera = status

    def set_depthhead(self, status: bool):
        if self.depth_head is not None:
            self.enable_depth = status

    def set_pointhead(self, status: bool):
        if self.point_head is not None:
            self.enable_point = status

    def set_trackhead(self, status: bool):
        if self.track_head is not None:
            self.enable_track = status

    def forward(self, images: torch.Tensor,
                query_points: torch.Tensor = None,
                mode: str = 'full',
                streaming: bool = False,
                **kwargs):
        """
        Forward pass of the CausalVGGT model.

        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
            query_points (torch.Tensor, optional): Query points for tracking, in pixel coordinates.
                Shape: [N, 2] or [B, N, 2]. Default: None
            mode (str): Attention mode ('full', 'causal', 'window', etc.). Default: 'full'.
            streaming (bool): If True, use streaming inference with kv_manager. Default: False.

        Returns:
            dict: A dictionary containing the following predictions:
                - pose_enc (torch.Tensor): Camera pose encoding with shape [B, S, 9] (from the last iteration)
                - depth (torch.Tensor): Predicted depth maps with shape [B, S, H, W, 1]
                - depth_conf (torch.Tensor): Confidence scores for depth predictions with shape [B, S, H, W]
                - world_points (torch.Tensor): 3D world coordinates for each pixel with shape [B, S, H, W, 3]
                - world_points_conf (torch.Tensor): Confidence scores for world points with shape [B, S, H, W]
                - images (torch.Tensor): Original input images, preserved for visualization

                If query_points is provided, also includes:
                - track (torch.Tensor): Point tracks with shape [B, S, N, 2] (from the last iteration), in pixel coordinates
                - vis (torch.Tensor): Visibility scores for tracked points with shape [B, S, N]
                - conf (torch.Tensor): Confidence scores for tracked points with shape [B, S, N]

                If return_qkv_cache is True, also includes:
                - aggregator_kv_cache_list (list): KV cache for the aggregator, list of [key, value] tensors for each layer, where each tensor has shape [B, num_heads, T, head_dim].
                - camera_head_kv_cache_list (list): KV cache for the camera head, list of list of [key, value] tensors for each iteration and layer. Each tensor has shape [B, num_heads, T, head_dim].
                - aggregated_tokens_list (list): List of aggregated tokens from each layer of the aggregator, each with shape [B, T, 2*embed_dim].
        """
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        if streaming:
            return self._forward_streaming(images, mode=mode, query_points=query_points, **kwargs)

        # Standard forward (training + non-streaming inference), aligned with VGGT.forward()
        debug_time = not self.training and kwargs.get("timing", True)
        if debug_time:
            time_start = torch.cuda.Event(enable_timing=True)
            time_end = torch.cuda.Event(enable_timing=True)

        aggregator_out = self.aggregator(images, mode=mode, **kwargs)
        aggregated_tokens_list = aggregator_out['output_list']
        patch_start_idx = aggregator_out['patch_start_idx']

        predictions = {}
        cam_time = depth_time = point_time = 0.0

        with torch.amp.autocast('cuda', enabled=False):
            if self.camera_head is not None and self.enable_camera:
                if debug_time:
                    time_start.record()
                pose_enc_list, _ = self.camera_head(aggregated_tokens_list, mode=mode)
                predictions["pose_enc"] = pose_enc_list[-1]
                predictions["pose_enc_list"] = pose_enc_list
                if debug_time:
                    time_end.record()
                    torch.cuda.synchronize()
                    cam_time = time_start.elapsed_time(time_end)

            if self.depth_head is not None and self.enable_depth:
                if debug_time:
                    time_start.record()
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf
                if debug_time:
                    time_end.record()
                    torch.cuda.synchronize()
                    depth_time = time_start.elapsed_time(time_end)

            if self.point_head is not None and self.enable_point:
                if debug_time:
                    time_start.record()
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf
                if debug_time:
                    time_end.record()
                    torch.cuda.synchronize()
                    point_time = time_start.elapsed_time(time_end)

        if self.track_head is not None and query_points is not None and self.enable_track:
            track_list, vis, conf = self.track_head(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, query_points=query_points
            )
            predictions["track"] = track_list[-1]
            predictions["vis"] = vis
            predictions["conf"] = conf

        predictions["images"] = images
        if debug_time:
            predictions["timing"] = {
                "aggregator_embed_time": aggregator_out.get('embed_time', 0.0),
                "aggregator_infer_time": aggregator_out.get('infer_time', 0.0),
                "camera_head_time": cam_time,
                "point_head_time": point_time,
                "depth_head_time": depth_time,
            }

        return predictions

    def _forward_streaming(self,
                           images: torch.Tensor,
                           mode: str = "full",
                           query_points: torch.Tensor = None,
                           **kwargs
                           ):
        """
        Streaming inference with kv_manager for the aggregator and optional KV cache for camera head.

        Args:
            images (torch.Tensor): Input images with shape [B, S, 3, H, W], in range [0, 1].
            mode (str): Attention mode. Default: 'full'.
            query_points (torch.Tensor, optional): Query points for tracking.

        Returns:
            dict: Predictions including pose_enc, depth, world_points, images,
                  camera_head_kv_cache_list, aggregated_tokens_list, and timing.
        """
        camera_head_kv_cache_list = kwargs.pop("camera_head_kv_cache_list", None)
        assert mode in ["causal", "full", "test"], "Mode must be 'causal', 'full', or 'test'."
        self.eval()

        debug_time = kwargs.get("timing", False)
        if debug_time:
            time_start = torch.cuda.Event(enable_timing=True)
            time_end = torch.cuda.Event(enable_timing=True)

        aggregator_out = self.aggregator.inference(images, mode=mode, **kwargs)
        agg_embed_time = aggregator_out.get('embed_time', 0.0)
        agg_infer_time = aggregator_out.get('infer_time', 0.0)
        aggregated_tokens_list = aggregator_out['output_list']
        patch_start_idx = aggregator_out['patch_start_idx']

        predictions = {}
        cam_time = depth_time = point_time = 0.0
        with torch.amp.autocast('cuda', enabled=False):
            if self.camera_head is not None and self.enable_camera:
                if debug_time:
                    time_start.record()
                pose_enc_list, camera_head_kv_cache_list_new = self.camera_head.inference(
                    aggregated_tokens_list,
                    mode=mode,
                    kv_cache_list=camera_head_kv_cache_list)
                if debug_time:
                    time_end.record()
                    torch.cuda.synchronize()
                    cam_time = time_start.elapsed_time(time_end)
                predictions["pose_enc"] = pose_enc_list[-1]
            else:
                camera_head_kv_cache_list_new = camera_head_kv_cache_list

            if self.point_head is not None and self.enable_point:
                if debug_time:
                    time_start.record()
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                if debug_time:
                    time_end.record()
                    torch.cuda.synchronize()
                    point_time = time_start.elapsed_time(time_end)
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

            if self.depth_head is not None and self.enable_depth:
                if debug_time:
                    time_start.record()
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                if debug_time:
                    time_end.record()
                    torch.cuda.synchronize()
                    depth_time = time_start.elapsed_time(time_end)
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

        predictions["images"] = images
        predictions["camera_head_kv_cache_list"] = camera_head_kv_cache_list_new
        predictions["aggregated_tokens_list"] = aggregated_tokens_list
        predictions["timing"] = {
            "aggregator_embed_time": agg_embed_time,
            "aggregator_infer_time": agg_infer_time,
            "camera_head_time": cam_time,
            "point_head_time": point_time,
            "depth_head_time": depth_time,
        }

        return predictions
