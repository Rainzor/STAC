# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import numpy as np
from typing import Optional

from causalvggt.layers import PatchEmbed
from causalvggt.layers.block import Block, create_attn_mask
from causalvggt.layers.rope import RotaryPositionEmbedding2D, PositionGetter
from causalvggt.layers.vision_transformer import vit_small, vit_base, vit_large, vit_giant2
from causalvggt.layers.attention import SparseAttention, Attention
from stac.h2o import HeavyHittersKV
logger = logging.getLogger('sparse aggregator')

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]

effect_layers = [4,11,17,23]

class CausalAggregator(nn.Module):
    """
    The CausalAggregator applies alternating-attention over input frames,
    as described in VGGT: Visual Geometry Grounded Transformer.

    Remember to set model.train() to enable gradient checkpointing to reduce memory usage.

    Args:
        img_size (int): Image size in pixels.
        patch_size (int): Size of each patch for PatchEmbed.
        embed_dim (int): Dimension of the token embeddings.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim.
        num_register_tokens (int): Number of register tokens.
        block_fn (nn.Module): The block type used for attention (Block by default).
        qkv_bias (bool): Whether to include bias in QKV projections.
        proj_bias (bool): Whether to include bias in the output projection.
        ffn_bias (bool): Whether to include bias in MLP layers.
        patch_embed (str): Type of patch embed. e.g., "conv" or "dinov2_vitl14_reg".
        aa_order (list[str]): The order of alternating attention, e.g. ["frame", "global"].
        aa_block_size (int): How many blocks to group under each attention type before switching. If not necessary, set to 1.
        qk_norm (bool): Whether to apply QK normalization.
        rope_freq (int): Base frequency for rotary embedding. -1 to disable.
        init_values (float): Init scale for layer scale.
    """

    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        block_fn=Block,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        patch_embed="dinov2_vitl14_reg",
        aa_order=["frame", "global"],
        aa_block_size=1,
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
    ):
        super().__init__()

        self.__build_patch_embed__(patch_embed, img_size, patch_size, num_register_tokens, embed_dim=embed_dim)

        # Initialize rotary position embedding if frequency > 0
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None

        self.frame_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, 
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                    attn_class=Attention
                )
                for _ in range(depth)
            ]
        )

        self.global_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                    attn_class=SparseAttention
                )
                for _ in range(depth)
            ]
        )

        self.depth = depth
        self.aa_order = aa_order
        self.patch_size = patch_size
        self.aa_block_size = aa_block_size
        self.embed_dim = embed_dim
        self.nums_heads = num_heads

        # Validate that depth is divisible by aa_block_size
        if self.depth % self.aa_block_size != 0:
            raise ValueError(f"depth ({depth}) must be divisible by aa_block_size ({aa_block_size})")

        self.aa_block_num = self.depth // self.aa_block_size

        # Note: We have two camera tokens, one for the first frame and one for the rest
        # The same applies for register tokens
        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
        #   The register tokens likely serve as "working memory" during the
        #   attention process to improve representations, similar to their
        #   use in vision transformers, but don't directly contribute to
        #   final outputs.
        self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim))

        # The patch tokens start after the camera and register tokens
        self.patch_start_idx = 1 + num_register_tokens

        # Initialize parameters with small values
        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)

        # Register normalization constants as buffers
        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False)

        self.use_reentrant = False # hardcoded to False

        self.mask = None
        self.kv_manager = None

    def register_kv_mgr(self, kv_manager, **kwargs):
        """
        Enable Heavy-Hitter Only Attention (H2O) in all global attention blocks.
        """
        if self.kv_manager is None:
            for idx, block in enumerate(self.global_blocks):
                block.attn.set_layer_idx(idx)
            
            self.kv_manager = kv_manager(
                num_layers=len(self.global_blocks),
                num_heads=self.nums_heads,
                head_dim=self.embed_dim // self.nums_heads,
                **kwargs
            )
            self.kv_manager.reset() # initialize the kv_manager
    
    def clear_kv_mgr(self):
        """
        Clear the kv_manager cache.
        This should be called at the beginning of each new sequence.
        """
        if self.kv_manager is not None:
            self.kv_manager.free()
            del self.kv_manager
            self.kv_manager = None
    
    def reset_kv_mgr(self):
        """
        Reset the kv_manager cache without freeing memory.
        This should be called at the beginning of each new sequence.
        """
        if self.kv_manager is not None:
            self.kv_manager.reset()

    def prune_kv_mgr(self, timing=False):
        """
        Prune the kv_manager to remove low-importance tokens.
        This should be called after each inference step to maintain a manageable cache size.
        """
        if self.kv_manager is not None:
            if timing:
                time_start = torch.cuda.Event(enable_timing=True)
                time_end = torch.cuda.Event(enable_timing=True)
                time_start.record()
            self.kv_manager.prune_kv()
            if timing:
                time_end.record()
                torch.cuda.synchronize()
                prune_time = time_start.elapsed_time(time_end)
                return prune_time
            else:
                return 0
        else:
            raise ValueError("kv_manager is not registered. Please call register_kv_mgr() first.")  
        
    def get_kv_mgr_info(self):
        """
        Get the H2O info (e.g., scores, allocated size) from all global attention layers.

        Returns:
            Dict: A dictionary containing H2O info for each global attention layer.
        """
        kv_info = {}
        if self.kv_manager is not None:
            kv_info["layer_indices"] = self.kv_manager.get_layers()
            kv_info["kvcache_used"], kv_info["kvcache_alloc"] = self.kv_manager.get_memory_usage()
            kv_info["kvcache_size"] = self.kv_manager.get_offset()
            # When CPU offload is used, report CPU and total so stats reflect full context
            if hasattr(self.kv_manager, "get_offset_cpu") and hasattr(self.kv_manager, "get_cpu_memory_usage"):
                offset_cpu = self.kv_manager.get_offset_cpu()
                cpu_used, cpu_alloc = self.kv_manager.get_cpu_memory_usage()
                if offset_cpu and (cpu_used > 0 or any(offset_cpu)):
                    gpu_sizes = kv_info["kvcache_size"]
                    kv_info["kvcache_size_cpu"] = offset_cpu
                    kv_info["kvcache_used_cpu"] = cpu_used
                    kv_info["kvcache_size_total"] = [g + c for g, c in zip(gpu_sizes, offset_cpu)]
                    kv_info["kvcache_used_total"] = kv_info["kvcache_used"] + cpu_used
            kv_info["token_indices"] = self.kv_manager.get_token_indices().detach().cpu().numpy()
            kv_info["scores"] = self.kv_manager.get_scores().detach().cpu().numpy() if hasattr(self.kv_manager, 'get_scores') else None
            
            kv_info["pool_size"] = self.kv_manager.get_pool_size() if hasattr(self.kv_manager, 'get_pool_size') else None
            kv_info["pos_size"] = self.kv_manager.get_pos_size() if hasattr(self.kv_manager, 'get_pos_size') else None
            kv_info["pool_indices"] = self.kv_manager.get_pool_indices().detach().cpu().numpy() if hasattr(self.kv_manager, 'get_pool_indices') else None
            kv_info["voxel_num"] = self.kv_manager.get_voxel_num() if hasattr(self.kv_manager, 'get_voxel_num') else None

            if hasattr(self.kv_manager, 'get_retrieval_indices'):
                retrieval_indices = self.kv_manager.get_retrieval_indices()
                if isinstance(retrieval_indices, list):
                    kv_info["retrieval_indices"] = [ri.detach().cpu().numpy() for ri in retrieval_indices]
                else:
                    kv_info["retrieval_indices"] = retrieval_indices.detach().cpu().numpy()
            else:
                kv_info["retrieval_indices"] = None
            
            if hasattr(self.kv_manager, 'get_retrieval_scores'):
                retrieval_scores = self.kv_manager.get_retrieval_scores()
                if isinstance(retrieval_scores, list):
                    kv_info["retrieval_scores"] = [rs.detach().cpu().numpy() for rs in retrieval_scores]
                else:
                    kv_info["retrieval_scores"] = retrieval_scores.detach().cpu().numpy()
            else:
                kv_info["retrieval_scores"] = None
            
            if hasattr(self.kv_manager, 'get_retrieval_size'):
                kv_info["retrieval_size"] = self.kv_manager.get_retrieval_size()
        return kv_info
    
    def retrieve_kv_mgr(self, timing=False, verbose=True, **kwargs):
        """
        Retrieve tokens into the kv_manager based on their importance scores.
        This should be called after each inference step to maintain a diverse cache.

        Args:
            timing (bool): Whether to measure and return the time taken for retrieval.
        """
        if self.kv_manager is not None:
            if timing:
                time_start = torch.cuda.Event(enable_timing=True)
                time_end = torch.cuda.Event(enable_timing=True)
                time_start.record()

            total_rets = []
            num_heads = self.nums_heads
            num_layers = len(self.global_blocks)
            for layer_idx in range(num_layers):
                max_wrote, total_wrote = self.kv_manager.retrieve_kv(layer_idx, **kwargs)
                total_rets.append(total_wrote)
            if timing:
                time_end.record()
                torch.cuda.synchronize()
                retrieve_time = time_start.elapsed_time(time_end)
                
                # ---- print retrieval stats ----
                if verbose:
                    print("===========[KV Maneger] Retrieval stats:===========")
                    print("Time taken for retrieval (ms): {:.3f}".format(retrieve_time))
                    sum_total_rets = sum(total_rets)
                    occupancy = np.array(total_rets) / (self.kv_manager.retrieval_token_size * num_heads)
                    occupancy = np.round(occupancy, 3).tolist()
                    expected_rets = self.kv_manager.retrieval_token_size * num_layers* num_heads
                    ratio = sum_total_rets / expected_rets if expected_rets >0 else 0
                    print(f"[KV Maneger] Retrieval KV: total retrieval {sum_total_rets}/{expected_rets}({ratio:.3f})")
                    print(f"[KV Maneger] per-layer retrieval : {total_rets}")
                    print(f"[KV Maneger] per-layer occupancy : {occupancy}")
                return retrieve_time
            else:
                return 0
        else:
            raise ValueError("kv_manager is not registered. Please call register_kv_mgr() first.")
    
    def update_kv_mgr_pos(self, new_positions, new_mask, timing=False, **kwargs):
        """
        Update the positions of tokens in the kv_manager.
        This should be called when the positions of tokens change, e.g., during inference with sliding window.

        Args:
            new_positions (torch.Tensor): New positions of shape (..., P, 3).
            new_mask (torch.Tensor): Mask indicating valid tokens of shape (..., P).
        """
        if self.kv_manager is not None and hasattr(self.kv_manager, 'append_positions'):
            assert new_positions.size(-1) == 3, "Last dimension of new_positions must be 3 (x, y, z)."
            new_positions = new_positions.view(-1, 3)  # Flatten to (N, 3)
            new_mask = new_mask.view(-1).bool()  # Flatten to (N,)
            if timing:
                time_start = torch.cuda.Event(enable_timing=True)
                time_end = torch.cuda.Event(enable_timing=True)
                time_start.record()
                self.kv_manager.append_positions(new_positions, new_mask, **kwargs)
                time_end.record()
                torch.cuda.synchronize()
                kv_pos_time = time_start.elapsed_time(time_end)
                return kv_pos_time
            else:
                self.kv_manager.append_positions(new_positions, new_mask, **kwargs)
                return 0
        else:
            raise ValueError("kv_manager is not registered. Please call register_kv_mgr() first.")
            

    def __build_patch_embed__(
        self,
        patch_embed,
        img_size,
        patch_size,
        num_register_tokens,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        block_chunks=0,
        init_values=1.0,
        embed_dim=1024,
    ):
        """
        Build the patch embed layer. If 'conv', we use a
        simple PatchEmbed conv layer. Otherwise, we use a vision transformer.
        """

        if "conv" in patch_embed:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
        else:
            vit_models = {
                "dinov2_vitl14_reg": vit_large,
                "dinov2_vitb14_reg": vit_base,
                "dinov2_vits14_reg": vit_small,
                "dinov2_vitg2_reg": vit_giant2,
            }

            self.patch_embed = vit_models[patch_embed](
                img_size=img_size,
                patch_size=patch_size,
                num_register_tokens=num_register_tokens,
                interpolate_antialias=interpolate_antialias,
                interpolate_offset=interpolate_offset,
                block_chunks=block_chunks,
                init_values=init_values,
            )

            # Disable gradient updates for mask token
            if hasattr(self.patch_embed, "mask_token"):
                self.patch_embed.mask_token.requires_grad_(False)

    def forward(self, 
                images: torch.Tensor, 
                mode: str = "full",
                **kwargs
                ) -> dict:
        """
        Args:
            images (torch.Tensor): Input images with shape [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width

        Returns:
            dict with keys 'output_list', 'patch_start_idx', 'embed_time', 'infer_time'.
        """
        B, S, C_in, H, W = images.shape

        assert C_in == 3, f"Expected 3 input channels, got {C_in}"

        time_start = torch.cuda.Event(enable_timing=True)
        time_end = torch.cuda.Event(enable_timing=True)
        time_start.record()

        # Normalize images and reshape for patch embed
        images = (images - self._resnet_mean) / self._resnet_std

        # Reshape to [B*S, C, H, W] for patch embedding
        images = images.view(B * S, C_in, H, W)
        patch_tokens = self.patch_embed(images)

        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        _, P, C = patch_tokens.shape

        # Expand camera and register tokens to match batch size and sequence length
        camera_token = slice_expand_and_flatten(self.camera_token, B, S, True)
        register_token = slice_expand_and_flatten(self.register_token, B, S, True)

        # Concatenate special tokens with patch tokens
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=images.device)

        if self.patch_start_idx > 0:
            pos = pos + 1
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2).to(images.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        time_end.record()
        torch.cuda.synchronize()
        embed_time = time_start.elapsed_time(time_end)

        # update P because we added special tokens
        _, P, C = tokens.shape

        if self.mask is None:
            self.mask = create_attn_mask(S, P, mode, 
                                             tokens.dtype, tokens.device, 
                                             window_size=kwargs.get("window_size", 3),
                                             look_ahead_size=kwargs.get("look_ahead_size", 0)
                                             )
        attn_mask = self.mask

        time_start.record()
        frame_idx = 0
        global_idx = 0
        output_list = []
        for layer_idx in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos
                    )
                elif attn_type == "global":
                    if mode == "causal_full_causal" and global_idx in [10, 11, 12, 13, 14, 15, 16, 17]:
                        attn_mask = None
                    elif mode == "full_causal" and global_idx in [10, 11, 12, 13, 14, 15, 16, 17]:
                        attn_mask = None
                    elif mode == "causal_full" and global_idx in [18, 19, 20, 21, 22, 23]:
                        attn_mask = None
                    else:
                        attn_mask = self.mask

                    tokens, global_idx, global_intermediates = self._process_global_attention(
                        tokens, B, S, P, C, global_idx, pos=pos, attn_mask=attn_mask
                    )
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")
                
            for i in range(len(frame_intermediates)):
                concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                output_list.append(concat_inter)
        time_end.record()
        torch.cuda.synchronize()
        infer_time = time_start.elapsed_time(time_end)

        del concat_inter
        del frame_intermediates
        del global_intermediates

        return {
            'output_list': output_list,
            'patch_start_idx': self.patch_start_idx,
            'embed_time': embed_time,
            'infer_time': infer_time,
        }
    
    @torch.no_grad()
    def inference(self, 
                images: torch.Tensor, 
                mode: str = "full",
                **kwargs
                ) -> dict:
        """
        Streaming inference path. Uses kv_manager (if registered) for sparse attention.

        Args:
            images (torch.Tensor): Input images with shape [B, S, 3, H, W], in range [0, 1].

        Returns:
            dict with keys 'output_list', 'patch_start_idx', 'embed_time', 'infer_time'.
        """
        B, S, C_in, H, W = images.shape
        assert C_in == 3, f"Expected 3 input channels, got {C_in}"

        time_start = torch.cuda.Event(enable_timing=True)
        time_end = torch.cuda.Event(enable_timing=True)
        time_start.record()
        # Normalize images and reshape for patch embed
        images = (images - self._resnet_mean) / self._resnet_std

        # Reshape to [B*S, C, H, W] for patch embedding
        images = images.view(B * S, C_in, H, W)
        patch_tokens = self.patch_embed(images)

        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        _, P, C = patch_tokens.shape

        # Expand camera and register tokens to match batch size and sequence length
        # the first time call these tokens are empty (None) in kv_cache_list
        # is_anchor_exist = kv_cache_list is None or kv_cache_list[0][0] is None
        is_anchor_exist = kwargs.get("is_anchor_exist", True)
        camera_token = slice_expand_and_flatten(self.camera_token, B, S, is_anchor_exist=is_anchor_exist)
        register_token = slice_expand_and_flatten(self.register_token, B, S, is_anchor_exist=is_anchor_exist)


        # Concatenate special tokens with patch tokens
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=images.device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2).to(images.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        # update P because we added special tokens
        _, P, C = tokens.shape
        time_end.record()
        torch.cuda.synchronize()
        embed_time = time_start.elapsed_time(time_end)

        # create mask
        if self.mask is None:
            if mode == "test":
                self.mask = create_attn_mask(S, P, "test",
                                                    tokens.dtype, tokens.device, 
                                                    window_size=kwargs.get("window_size", 0),
                                                    )
            else:
                self.mask = None
        attn_mask = self.mask

        time_start.record()
        frame_idx = 0
        global_idx = 0
        output_list = []
        for layer_idx in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos
                    )
                elif attn_type == "global":
                    tokens, global_idx, global_intermediates = self._process_global_attention(
                        tokens, B, S, P, C, global_idx, pos=pos, attn_mask=attn_mask
                    )
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            for i in range(len(frame_intermediates)):
                concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                output_list.append(concat_inter)

        time_end.record()
        torch.cuda.synchronize()
        infer_time = time_start.elapsed_time(time_end)

        del concat_inter
        del frame_intermediates
        del global_intermediates

        return {
            'output_list': output_list,
            'patch_start_idx': self.patch_start_idx,
            'embed_time': embed_time,
            'infer_time': infer_time,
        }

    def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None):
        """
        Process frame attention blocks. We keep tokens in shape (B*S, P, C).
        """
        # If needed, reshape tokens or positions:
        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).view(B * S, P, C)

        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.view(B, S, P, 2).view(B * S, P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            if self.training:
                tokens = checkpoint(self.frame_blocks[frame_idx], tokens, pos, use_reentrant=self.use_reentrant)
            else:
                tokens = self.frame_blocks[frame_idx](tokens, pos=pos)
            if frame_idx in effect_layers:
                intermediates.append(tokens.view(B, S, P, C))
            else:
                intermediates.append(tokens.new_empty((B,S,0,C)))  # placeholder for layers without effect
            frame_idx += 1

        return tokens, frame_idx, intermediates

    def _process_global_attention(self, tokens, B, S, P, C, global_idx, pos=None, attn_mask=None):
        """
        Process global attention blocks. We keep tokens in shape (B, S*P, C).
        """
        if tokens.shape != (B, S * P, C):
            tokens = tokens.view(B, S, P, C).view(B, S * P, C)

        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.view(B, S, P, 2).view(B, S * P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            if self.training:
                tokens = checkpoint(self.global_blocks[global_idx], tokens, pos,
                                    attn_mask=attn_mask, use_reentrant=self.use_reentrant)
            else:
                #! use kv_manager for sparse attention
                tokens = self.global_blocks[global_idx](
                    tokens, pos=pos, attn_mask=attn_mask, kv_manager=self.kv_manager
                )
            if global_idx in effect_layers:
                intermediates.append(tokens.view(B, S, P, C))
            else:
                intermediates.append(tokens.new_empty((B, S, 0, C)))
            global_idx += 1

        return tokens, global_idx, intermediates
        
def slice_expand_and_flatten(token_tensor, B, S, is_anchor_exist=False):
    """
    Processes specialized tokens with shape (1, 2, X, C) for multi-frame processing:
    1) Uses the first position (index=0) for the first frame only
    2) Uses the second position (index=1) for all remaining frames (S-1 frames)
    3) Expands both to match batch size B
    4) Concatenates to form (B, S, X, C) where each sequence has 1 first-position token
       followed by (S-1) second-position tokens
    5) Flattens to (B*S, X, C) for processing

    Returns:
        torch.Tensor: Processed tokens with shape (B*S, X, C)
    """

    # Slice out the "query" tokens => shape (1, 1, ...)
    if is_anchor_exist: 
        # for the first time call when kv_cache is None,
        # the first token (index=0) is desigened only for the first frame
        query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
    else:
        query = token_tensor[:, 1:, ...].expand(B, 1, *token_tensor.shape[2:])
    # Slice out the "other" tokens => shape (1, S-1, ...)
    # when S=1, this results in an empty tensor with shape (1, 0, ...)
    others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
    # Concatenate => shape (B, S, ...)
    combined = torch.cat([query, others], dim=1)

    # Finally flatten => shape (B*S, ...)
    combined = combined.view(B * S, *combined.shape[2:])
    return combined