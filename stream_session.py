from typing import List, Optional, Sequence, Tuple, Union

import torch
import numpy as np
from causalvggt.models.vggt import CausalVGGT
from stac.kv_manager import KVManager
from stac.stac_voxel import STACVoxelKV
from causalvggt.utils.geometry import unproject_depth_map_to_point_map
from causalvggt.utils.pose_enc import pose_encoding_to_extri_intri

import psutil, os
import logging
import json
from copy import deepcopy
from tqdm import tqdm
import time

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger("StreamSession")

VERBOSE = os.environ.get("VERBOSE", "0").strip().lower() in ("1", "true", "yes")


def print_mem(tag=""):
    # ---- GPU Memory ----
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
    else:
        allocated = reserved = 0.0

    # ---- CPU Memory ----
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    rss = mem_info.rss / 1024**2   # (Resident Set Size)
    vms = mem_info.vms / 1024**2   # (Virtual Memory Size)

    if VERBOSE:
        print(
            f"[{tag}] "
            f"GPU allocated={allocated:.2f}MB, reserved={reserved:.2f}MB | "
            f"CPU RSS={rss:.2f}MB, VMS={vms:.2f}MB"
        )


class StreamSession:
    """
    A causal streaming inference session with KV cache management for CausalVGGT.
    """

    def __init__(
        self,
        model: CausalVGGT,
        cam_cache_update: bool = False,
        device: torch.device = torch.device("cuda"),
    ):
        self.model = model.to(device)
        self.device = device
        self.aggregator_kv_cache_depth = model.aggregator.depth
        self.camera_head_kv_cache_depth = model.camera_head.trunk_depth if model.camera_head is not None else 0
        self.camera_head_iterations = 4 if model.camera_head is not None else 0
        self.cam_cache_update = cam_cache_update
        self.pose_tokens_list = []
        # Prediction keys to track, where the element of prediction shape like [B, S, ...]
        self.predictions_keys = ["pose_enc", "world_points", "world_points_conf", "depth", "depth_conf", "images"]

        self._processed_frames = 0
        self.init()

    def init(self):
        self._processed_frames = 0
        self.predictions = {k: [] for k in self.predictions_keys}
        self.pose_tokens_list = []
        self.benchmark_metrics = {}
        self.stats = {}

    def clear(self):
        self._clear_predictions()
        self.model.aggregator.clear_kv_mgr()
        torch.cuda.empty_cache()
        self.pose_tokens_list = []
        self._processed_frames = 0
        self.benchmark_metrics = {}
        self.stats = {}

    # ======== Prediction management methods ========
    def _clear_predictions(self):
        for k in self.predictions:
            for i in reversed(range(len(self.predictions[k]))):
                tensor = self.predictions[k][i]
                if isinstance(tensor, torch.Tensor):
                    del tensor
                elif isinstance(tensor, list):
                    for j in reversed(range(len(tensor))):
                        if isinstance(tensor[j], torch.Tensor):
                            del tensor[j]
                    del tensor
        self.predictions = {k: [] for k in self.predictions_keys}

    def _update_predictions(self, predictions: dict, device: str = 'cpu'):
        for k in predictions:
            if k in self.predictions:
                if predictions[k] is None:
                    continue
                B,S = predictions[k].shape[0], predictions[k].shape[1]
                for i in range(B):
                    for j in range(S):
                        self.predictions[k].append(predictions[k][i:i+1, j:j+1].to(device=device))

    def get_all_predictions(self, device='cpu'):
        # return self.predictions
        all_predictions = dict()
        for key in self.predictions_keys:
            if key in self.predictions:
                if isinstance(self.predictions[key], torch.Tensor):
                    all_predictions[key] = self.predictions[key].to(device=device)
                    continue
                if self._processed_frames != len(self.predictions[key]):
                    raise ValueError(f"Processed frames {self._processed_frames} != stored predictions {len(self.predictions[key])} for key {key}")
                if isinstance(self.predictions[key][0], torch.Tensor):
                    all_predictions[key] = torch.cat(self.predictions[key], dim=1)
                elif isinstance(self.predictions[key][0], list):
                    prediction_list = []
                    for layer_idx in range(len(self.predictions[key][0])):
                        layer_predictions = []
                        for frame_idx in range(len(self.predictions[key])):
                            layer_predictions.append(self.predictions[key][frame_idx][layer_idx].to(device=device))
                        prediction_list.append(torch.cat(layer_predictions, dim=1))
                    all_predictions[key] = prediction_list # list of tensors
                else:
                    raise ValueError(f"Unsupported prediction type for key {key}: {type(self.predictions[key][0])}")
        return all_predictions

    def get_last_prediction(self):
        last_predictions = dict()

        for k in self.predictions_keys:
            if k in self.predictions:
                last_predictions[k] = self.predictions[k][-1]
        return last_predictions

    def pop_first_prediction(self):
        first_predictions = dict()
        for k in self.predictions_keys:
            if k in self.predictions and len(self.predictions[k]) > 0:
                first_predictions[k] = self.predictions[k].pop(0)
        return first_predictions

    def pushback_prediction(self, predictions, device='cpu'):
        self._update_predictions(predictions, device=device)
    
    def _update_benchmark(self, metrics: dict):
        if not self.benchmark_metrics:
            self.benchmark_metrics = metrics
        else:
            for k in metrics:
                if k in self.benchmark_metrics:
                    self.benchmark_metrics[k] += metrics[k]
                else:
                    self.benchmark_metrics[k] = metrics[k]
            
    def get_benchmark(self):
        return self.benchmark_metrics
    
    def get_stats(self):
        return self.stats


    # ======== Inference methods ========

    def camera_head_inference(
            self,
            agg_token_lists,
    ):
        time_start = torch.cuda.Event(enable_timing=True)
        time_end = torch.cuda.Event(enable_timing=True)
        time_start.record()
        pose_tokens = agg_token_lists[-1][:, :, 0].detach()
        self.pose_tokens_list.append(pose_tokens)
        pose_token_cache = torch.cat(self.pose_tokens_list, dim=1)
        with torch.amp.autocast("cuda", enabled=False):
            pose_enc_list, _ = self.model.camera_head.inference(                                            
                    aggregated_tokens_list=None,
                    pose_token_cache=pose_token_cache,
                    mode="full",
                    kv_cache_list=None,
                )
            self.predictions["pose_enc"] = pose_enc_list[-1]
            outputs = {"pose_enc": pose_enc_list[-1]}
        time_end.record()
        torch.cuda.synchronize()
        elapsed = time_start.elapsed_time(time_end)
        self._update_benchmark({"camera_head_time": elapsed})
        return outputs
    
    def get_pointmap(self, outputs, conf_threshold=1.0, special_tokens_size=0, 
                     pose_enc = None, images=None,
                     prediction_mode="pointmap"):
        """ 
        Process the point cloud from model outputs and update KV cache positions.
        """
        # Update KV cache positions
        if (prediction_mode == "pointmap"):
            pts3d = outputs.get("world_points", None) # [B,S,H,W,3]
            pts3d_conf = outputs.get("world_points_conf", None) # [B,S,H,W]
        else:
            depth_map = outputs.get("depth", None) # [B,S,H,W,1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(
                    pose_enc, images.shape[-2:]
                )
            pts3d_conf = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)
            depths_conf = outputs.get("depth_conf", None) # [B,S,H,W]
            
        assert pts3d is not None and pts3d_conf is not None, "World points and confidence must be provided outputs."
        B, S, H, W, C = pts3d.shape
        assert B==1, "Batch size must be 1."
        pts3d_conf = pts3d_conf.unsqueeze(-1) # [B,S,H,W,1]
        pts3d = pts3d.permute(0, 1, 4, 2, 3).view(-1, C, H, W) # [S, 3, H, W]
        pts3d_conf = pts3d_conf.permute(0, 1, 4, 2, 3).view(-1, 1, H, W) # [S, 1, H, W]
        # downsample to patch level
        ds_patch = self.model.point_head.patch_size
        ds_size = (max(1, H // ds_patch), max(1, W // ds_patch))
        pts3d = torch.nn.functional.interpolate(
            pts3d, size=ds_size, mode='bilinear', align_corners=False
        )
        pts3d_conf = torch.nn.functional.interpolate(
            pts3d_conf, size=ds_size, mode='bilinear', align_corners=False
        ) # [S, 1, H', W']
        # reshape to [S, H'*W', 3]
        H2, W2 = pts3d.shape[-2:]
        pts3d = pts3d.permute(0, 2, 3, 1).reshape(S, -1, C) # [S, H'*W', 3]
        pts_special = pts3d.new_zeros((S, special_tokens_size, C))
        pts3d = torch.cat((pts_special, pts3d), dim=1) # [S, special+H'*W', 3]
        
        pts3d_conf = pts3d_conf.permute(0, 2, 3, 1).reshape(S,-1) # [S, H'*W']
        valid_mask = pts3d_conf > conf_threshold # [S, H'*W']
        valid_special = valid_mask.new_zeros((S, special_tokens_size)).bool()
        valid_mask = torch.cat((valid_special, valid_mask), dim=1) # [S, special+H'*W']
        return pts3d, valid_mask

    def pipeline(self, 
                 images: torch.Tensor, 
                 mode="causal", 
                 **kwargs
                    ) -> dict:
        self.clear()
        # [S, 3, H, W]
        num_frames = images.shape[0]
        device = kwargs.get("device", self.device)
        dtype = kwargs.get("dtype", torch.float16)
        logger.info("Streaming Pipeline Warming up the model...")
        for _ in range(1):
            self.model(
                images=images[0:1].to(device=device, dtype=dtype),
                mode="full",
                camera_head_kv_cache_list=None,
                streaming=True,
                is_anchor_exist=True,
            )  # warmup

        if mode in ["window_kv","causal"]:
            # Use H2O attention to maintain a heavy-hitter + recent KV cache for the aggregator.
            window_size = kwargs.get("window_size", 0)
            if mode == "causal":
                window_size = num_frames
            if window_size < 0:
                logger.warning("Switching to causal attention mode.")
                window_size = num_frames  # effectively causal
            kv_kwargs = deepcopy(kwargs)
            kv_kwargs.update({
                "register_layers": None,
                "window_size": window_size,
            })
            kv_kwargs = self.register_kv_mgr(mode, images, KVManager, **kv_kwargs)
            self.cam_cache_update = False
            self.model.set_camhead(self.cam_cache_update)
            debug_timing = kwargs.get("timing", True)
            kv_manager_infos = []
            memory_stats_infos = []
            with tqdm(total=num_frames, desc="Window mode", dynamic_ncols=True) as pbar:
                for i in range(num_frames):
                    image = images[i : i + 1].to(device=self.device)
                    outputs = self.model(
                        images=image,
                        mode="full",
                        camera_head_kv_cache_list=None,
                        streaming=True,
                        is_anchor_exist=i==0,
                        timing=debug_timing,
                    )
                    timing = outputs.get("timing", {})
                    if not self.cam_cache_update and self.model.camera_head is not None:
                        self.camera_head_inference(outputs["aggregated_tokens_list"])
                    
                    prune_time = self.model.aggregator.prune_kv_mgr(timing=debug_timing)
                    timing["kv_pruning_time"] = prune_time

                    self.pushback_prediction(outputs)
                    self._update_benchmark(outputs.get("timing", {}))

                    info = {"frame_idx": i}
                    kvcache_info = self.model.aggregator.get_kv_mgr_info()
                    info.update(kvcache_info)
                    kv_manager_infos.append(info)

                    pbar.update(1)
                    kvcache_size = kvcache_info["kvcache_size"][0]
                    kvcache_mem = kvcache_info["kvcache_used"]

                    stats = {
                        "frame_idx": i,
                        "kvcache_size": kvcache_size,
                        "kvcache_mem": kvcache_mem,
                    }
                    total_time = 0.0
                    for k,v in timing.items():
                        stats[k] = v
                        total_time += v
                    stats["total_time"] = total_time
                    memory_stats_infos.append(stats)

                    if torch.cuda.is_available():
                        allocated = torch.cuda.memory_allocated() / 1024**2
                        reserved = torch.cuda.memory_reserved() / 1024**2
                    else:
                        allocated = reserved = 0.0
                    agg_time = timing.get("aggregator_infer_time", 0)
                    prune_time = timing.get("kv_pruning_time", 0)
                    pbar.set_postfix({
                        "Time(agg/prune)": f"{agg_time:.2f}/{prune_time:.2f}",
                        "KV used": f"{kvcache_size}",
                        "GPU(KV/A/R)": f"{kvcache_mem:.0f}/{allocated:.0f}/{reserved:.0f}MB",
                    })

                    self._processed_frames += 1
            if VERBOSE:
                print("Window mode Done!")

        elif mode in ["window_chunk_merge"]:
            # Use Voxel attention to maintain a voxel + recent KV cache for the aggregator.
            voxel_size = kwargs.get("voxel_size", 0.05)
            dist_thres = 2.0 * voxel_size
            kv_kwargs = deepcopy(kwargs)
            chunk_size = kwargs.get("chunk_size", 1)
            window_size = kwargs.get("window_size", 0)
            if chunk_size < 1:
                raise ValueError(f"chunk_size must be >= 1, got {chunk_size}.")
            if window_size > 0 and chunk_size > window_size:
                logger.warning(
                    f"chunk_size ({chunk_size}) > window_size ({window_size}): "
                    "retrieval and print will trigger every chunk."
                )
            debug_timing = kwargs.get("timing", True)

            merge_layers = None

            sim_threshold = kwargs.get("sim_threshold", 0.8)
            merger_kwargs = {
                        "voxel_size": voxel_size,
                        "voxelize_layers": merge_layers,
                        "init_voxels": kwargs.get("voxel_num", 4096),
                        "voxel_buf_cap": kwargs.get("voxel_buf_cap", 8),
                        "voxel_piv_cap": kwargs.get("voxel_piv_cap", 4),
                        "voxel_backend": kwargs.get("voxel_backend", "python"),
                        "sim_threshold": sim_threshold,
                        "replace_threshold": sim_threshold,
                        "score_threshold": 0.2,
                        "slab_growth": 1024,
                        "slab_cap": 10000,
                        "seg_size": 1,
                        "retrieval_size": kwargs.get("retrieval_size", -1),
                        "allocator": kwargs.get("allocator", "slab"),
                        # CPU offload parameters
                        "enable_alloc_cpu": kwargs.get("enable_alloc_cpu", False),
                        "gpu_threshold_gb": kwargs.get("gpu_threshold_gb", 10.0),
                        "cold_frame_threshold": kwargs.get("cold_frame_threshold", 5),
            }
            kv_kwargs.update(merger_kwargs)
            kv_kwargs = self.register_kv_mgr(mode, images, STACVoxelKV, **kv_kwargs)
            kv_manager = self.model.aggregator.kv_manager

            window_size = kv_kwargs.get("recent_size", 0)
            ret_size = kv_kwargs.get("retrieval_size", -1)
            buffer_size = kv_kwargs.get("buffer_size", 16)

            self.cam_cache_update = False
            self.model.set_camhead(self.cam_cache_update)

            conf_threshold = kwargs.get("conf_threshold", 2.0)
            logger.info(f"VoxelSasaMerge chunk mode with chunk_size {chunk_size} and window_size {window_size}, conf_threshold {conf_threshold}.")
            special_tokens_size = self.model.aggregator.patch_start_idx

            export_metrics = kwargs.get("export_metrics", False)
            export_metrics_dir = kwargs.get("export_metrics_dir", "./eval_results/step_metrics")
            
            kv_manager_infos = []
            memory_stats_infos = []
            step_metrics_list = []  # For JSON export
            with tqdm(total=num_frames, desc=f"{mode} mode") as pbar:
                for frame_idx in range(0, num_frames, chunk_size):
                    frame_buffer = images[frame_idx : min(frame_idx + chunk_size, num_frames)].to(device=self.device)
                    frame_buffer_size = frame_buffer.shape[0]
                    outputs = self.model(
                        images=frame_buffer,
                        mode="full",
                        camera_head_kv_cache_list=None,
                        streaming=True,
                        is_anchor_exist=frame_idx==0,
                        timing=debug_timing,
                    )
                    timing = outputs.get("timing", {})
                    if not self.cam_cache_update and self.model.camera_head is not None:
                        cam_output = self.camera_head_inference(outputs["aggregated_tokens_list"])
                        pose_enc = cam_output["pose_enc"]
                    else:
                        pose_enc = outputs["pose_enc"]

                    # update kv manager position with confident points
                    pts3d, valid_mask = self.get_pointmap(outputs, conf_threshold=conf_threshold, 
                                                          special_tokens_size=special_tokens_size,
                                                          pose_enc = pose_enc, images=frame_buffer
                                                          )
                    kv_pos_time = self.model.aggregator.update_kv_mgr_pos(pts3d, valid_mask, timing=debug_timing)
                    timing["kv_position_time"] = kv_pos_time

                    # retrieve from voxel grid
                    retrieval_time = 0.0
                    if frame_idx > max(buffer_size, 16):
                        if ret_size > 0:
                            chunks_per_window = max(1, window_size // chunk_size)
                            if (frame_idx // chunk_size + 1) % chunks_per_window == 0:
                                retrieval_time = self.model.aggregator.retrieve_kv_mgr(timing=debug_timing, verbose=False,
                                                                                       dist_thres=dist_thres,
                                                                                       return_buf=kwargs.get("return_buf", False))
                        elif ret_size == -1:
                            retrieval_time = self.model.aggregator.retrieve_kv_mgr(timing=debug_timing, verbose=False,
                                                                                   dist_thres=dist_thres,
                                                                                   return_buf=kwargs.get("return_buf", False))

                    timing["kv_retrieval_time"] = retrieval_time

                    # prune kv cache according to heavy-hitter + recent scheme
                    prune_merge_time = self.model.aggregator.prune_kv_mgr(timing=debug_timing)
                    timing["kv_prune_merge_time"] = prune_merge_time

                    # Reclaim fragmented reserved memory periodically to prevent OOM.
                    # The merge pipeline creates large transient fp32/bf16 tensors that
                    # fragment the CUDA caching allocator; without periodic cleanup,
                    # reserved memory grows unbounded (~5 GB on a 24 GB GPU).
                    if frame_idx % (chunk_size * 4) == 0 or frame_idx >= num_frames - chunk_size:
                        _mem_profile = os.environ.get("MERGER_MEM_PROFILE", "0") == "1"
                        if _mem_profile:
                            torch.cuda.synchronize()
                            a_before = torch.cuda.memory_allocated() / (1024**2)
                            r_before = torch.cuda.memory_reserved() / (1024**2)
                            frag_before = r_before - a_before
                        torch.cuda.empty_cache()
                        if _mem_profile:
                            r_after = torch.cuda.memory_reserved() / (1024**2)
                            frag_after = r_after - a_before
                            freed = r_before - r_after
                            print(f"  [MEM-FRAG] frame={frame_idx} | "
                                  f"alloc={a_before:.0f}MB, "
                                  f"res_before={r_before:.0f}MB, res_after={r_after:.0f}MB, "
                                  f"frag_before={frag_before:.0f}MB, frag_after={frag_after:.0f}MB, "
                                  f"freed_by_empty_cache={freed:.0f}MB", flush=True)

                    self.pushback_prediction(outputs)
                    self._update_benchmark(timing)

                    # logging kv manager info
                    info = {"frame_idx": frame_idx}
                    kvcache_info = self.model.aggregator.get_kv_mgr_info()
                    info.update(kvcache_info)
                    kv_manager_infos.append(info)

                    merger_stat = kv_manager.get_merger_info()
                    merger_stat["frame_idx"] = frame_idx
                    total_time = 0.0
                    for key, value in timing.items():
                        merger_stat[key] = value / frame_buffer_size
                        total_time += value
                    merger_stat["total_time"] = total_time / frame_buffer_size

                    memory_stats_infos.append(merger_stat)

                    if export_metrics:
                        memory_details = kv_manager.get_memory_details()
                        step_metric = {
                            "step": frame_idx // chunk_size,
                            "frame_idx": frame_idx,
                            "chunk_size": frame_buffer_size,
                            "timing": {
                                "aggregator_infer_time": timing.get("aggregator_infer_time", 0) / frame_buffer_size,
                                "kv_position_time": timing.get("kv_position_time", 0) / frame_buffer_size,
                                "kv_retrieval_time": timing.get("kv_retrieval_time", 0) / frame_buffer_size,
                                "kv_prune_merge_time": timing.get("kv_prune_merge_time", 0) / frame_buffer_size,
                                "total_time": total_time / frame_buffer_size,
                            },
                            # Memory usage (MB) - detailed breakdown
                            "memory": {
                                # Hot cache components
                                "pinned_memory": float(memory_details.get("pinned_memory", 0)),
                                "window_memory": float(memory_details.get("window_memory", 0)),
                                "heavy_hitters_memory": float(memory_details.get("heavy_hitters_memory", 0)),
                                "retrieval_memory": float(memory_details.get("retrieval_memory", 0)),
                                # Voxel store components
                                "voxel_buffer_usage": float(memory_details.get("voxel_buffer_usage", 0)),
                                "voxel_buffer_alloc": float(memory_details.get("voxel_buffer_alloc", 0)),
                                "voxel_pivot_usage": float(memory_details.get("voxel_pivot_usage", 0)),
                                "voxel_pivot_alloc": float(memory_details.get("voxel_pivot_alloc", 0)),
                                # Hot cache totals
                                "hot_cache_usage": float(memory_details.get("hot_cache_usage", 0)),
                                "hot_cache_alloc": float(memory_details.get("hot_cache_alloc", 0)),
                                # Grand totals
                                "total_usage": float(memory_details.get("total_usage", 0)),
                                "total_alloc": float(memory_details.get("total_alloc", 0)),
                            },
                            # KV cache info
                            "kv_cache": {
                                "kvcache_size": kvcache_info.get("kvcache_size", [0])[0] if isinstance(kvcache_info.get("kvcache_size", [0]), list) else kvcache_info.get("kvcache_size", 0),
                                "retrieval_size": int(np.mean(kvcache_info.get("retrieval_size", [0]))) if kvcache_info.get("retrieval_size") else 0,
                            },
                            # Merger stats (flatten for readability)
                            "merger": {
                                "num_voxels": merger_stat.get("used_voxels", 0),
                                "token_count": merger_stat.get("token_count", 0),
                                "best_compress_ratio": merger_stat.get("best_compress_ratio", 1.0),
                                "real_compress_ratio": merger_stat.get("real_compress_ratio", 1.0),
                                "pivot_pool_used": merger_stat.get("pivot_pool_used", 0),
                                "buffer_pool_used": merger_stat.get("buffer_pool_used", 0),
                            },
                        }
                        step_metrics_list.append(step_metric)
                    

                    kvcache_size = kvcache_info["kvcache_size"][0]
                    retrieval_size_list = kvcache_info.get("retrieval_size", None)
                    retrieval_size = 0
                    if retrieval_size_list is not None:
                        retrieval_size = int(np.mean(retrieval_size_list))
                    if torch.cuda.is_available():
                        allocated = torch.cuda.memory_allocated() / 1024**2
                        reserved = torch.cuda.memory_reserved() / 1024**2
                    else:
                        allocated = reserved = 0.0
                    agg_time = timing.get("aggregator_infer_time", 0) / frame_buffer_size
                    kv_pos_time = kv_pos_time / frame_buffer_size
                    prune_merge_time = prune_merge_time / frame_buffer_size
                    retrieval_time = retrieval_time / frame_buffer_size

                    mem_details = kv_manager.get_memory_details()
                    hot_mem = mem_details.get("hot_cache_usage", 0)
                    voxel_mem = mem_details.get("voxel_buffer_usage", 0) + mem_details.get("voxel_pivot_usage", 0)
                    ret_mem = mem_details.get("retrieval_memory", 0)

                    pbar_info = f"Time(agg/pos/prune&merge/ret):{agg_time:.2f}/{kv_pos_time:.2f}/{prune_merge_time:.2f}/{retrieval_time:.2f}, KV(hot/ret)={kvcache_size}/{retrieval_size},Mem(H/V/R/A/R)={hot_mem:.0f}/{voxel_mem:.0f}/{ret_mem:.0f}/{allocated:.0f}/{reserved:.0f}MB"
                    if VERBOSE and ((frame_idx // chunk_size + 1) % max(1, window_size // chunk_size) == 0 or (frame_idx + frame_buffer_size) >= num_frames):
                        print(f"==== VoxelSasaMerge: Frame {frame_idx + frame_buffer_size}, {pbar_info} =========================")
                    pbar.set_postfix_str(pbar_info)
                    pbar.update(frame_buffer_size)
                    self._processed_frames += frame_buffer_size
            if VERBOSE:
                print("Chunk VoxelSasaMerge mode Done!")

            tag = kwargs.get("tag", "voxel_merge")
            # Export step-level metrics to JSON
            if export_metrics and step_metrics_list:
                os.makedirs(export_metrics_dir, exist_ok=True)
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                json_filename = f"{tag}_f{num_frames}_step_metrics_{timestamp}.json"
                json_path = os.path.join(export_metrics_dir, json_filename)
                
                # Build summary statistics
                timing_keys = ["aggregator_infer_time", "kv_position_time", "kv_retrieval_time", "kv_prune_merge_time", "total_time"]
                memory_keys = [
                    # Hot cache components
                    "pinned_memory", "window_memory", "heavy_hitters_memory", "retrieval_memory",
                    # Voxel store components
                    "voxel_buffer_usage", "voxel_buffer_alloc", "voxel_pivot_usage", "voxel_pivot_alloc",
                    # Hot cache totals
                    "hot_cache_usage", "hot_cache_alloc",
                    # Grand totals
                    "total_usage", "total_alloc",
                ]
                
                timing_summary = {}
                for k in timing_keys:
                    values = [s["timing"][k] for s in step_metrics_list]
                    timing_summary[k] = {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                        "min": float(np.min(values)),
                        "max": float(np.max(values)),
                    }
                
                memory_summary = {}
                for k in memory_keys:
                    values = [s["memory"][k] for s in step_metrics_list]
                    memory_summary[k] = {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                        "min": float(np.min(values)),
                        "max": float(np.max(values)),
                    }
                
                export_data = {
                    "metadata": {
                        "tag": tag,
                        "num_frames": num_frames,
                        "chunk_size": chunk_size,
                        "window_size": window_size,
                        "voxel_size": voxel_size,
                        "sim_threshold": sim_threshold,
                        "conf_threshold": conf_threshold,
                        "timestamp": timestamp,
                    },
                    "hyperparameters": merger_kwargs,
                    "summary": {
                        "timing": timing_summary,
                        "memory": memory_summary,
                        "total_steps": len(step_metrics_list),
                    },
                    "steps": step_metrics_list,
                }
                
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False)
                logger.info(f"Exported step-level metrics to {json_path}")

            if merge_layers is None:
                layer_list = list(range(self.aggregator_kv_cache_depth))
            else:
                layer_list = merge_layers
            
            voxel_merge_stats = {}
            kv_mgr = self.model.aggregator.kv_manager
            if hasattr(kv_mgr, "get_merger_info"):
                merge_info = self.model.aggregator.kv_manager.get_merger_info()
                voxel_merge_stats[0] = merge_info
                metrics = {}
                metrics["hyperparameters"] = merger_kwargs
                metrics["stats"] = voxel_merge_stats
                self.stats = metrics

    def register_kv_mgr(self, mode,
                            images, 
                            kv_manager,
                            **kwargs):
            
            default_kwargs = {
                "chunk_size": kwargs.get("chunk_size", 1),
                "recent_size": kwargs.get("window_size", 2),
                "pinned_idx": kwargs.get("pinned_frame_indices", [0]),
                "hh_size": 0,
                "persist_size": 0,
                "temperature": 0.9,
                "device": self.device,
                "dtype": kwargs.get("dtype", torch.float16),
            }

            kwargs_kv = default_kwargs.copy()
            kwargs_kv.update(kwargs)

            recent_size = kwargs_kv["recent_size"]
            assert recent_size >= 1, "window_size must be at least 1."
            pinned_frame_indices = kwargs_kv["pinned_frame_indices"]
            hh_size = kwargs["hh_size"]
            chunk_size = kwargs["chunk_size"]
            pinned_size = len(pinned_frame_indices)
            buffer_size = chunk_size + pinned_size + recent_size + hh_size
            if buffer_size > 300:
                logger.warning(f"Buffer size {buffer_size} is large, may cause OOM issues; part memory offload to CPU device.")
            logger.info(f"Using {mode} mode: processing frames in windows of size {recent_size} with {kv_manager.__name__}-manager")

            if len(images.shape) == 4:
                S, C, H, W = images.shape
            else:
                B, S, C, H, W = images.shape
                assert B == 1, "Batch size must be 1 when input is 5D."
            vit_patch_size = self.model.aggregator.patch_embed.patch_size
            img_tokens = (H // vit_patch_size) * (W // vit_patch_size)
            cam_tokens = self.model.aggregator.patch_start_idx
            token_per_frame = img_tokens + cam_tokens

            kwargs_kv.update({
                "token_per_frame": token_per_frame,
                "buffer_size": buffer_size,
            })
            self.model.aggregator.register_kv_mgr(kv_manager=kv_manager,
                                                **kwargs_kv
                                                )

            return kwargs_kv
