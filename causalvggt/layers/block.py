# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/patch_embed.py

import logging
import os
from typing import Callable, List, Any, Tuple, Dict
import warnings

import torch
from torch import nn, Tensor

from .attention import Attention
from .drop_path import DropPath
from .layer_scale import LayerScale
from .mlp import Mlp
from torch.nn.attention.flex_attention import create_block_mask

logger = logging.getLogger("AttentionBlock")

XFORMERS_AVAILABLE = False

def create_attn_mask(S: int, P: int, mode: str, 
                     dtype: torch.dtype, device: torch.device,
                     look_ahead_size: int = 0,
                     window_size: int = 2,
                     ) -> torch.Tensor:
    """
    Create attention mask for block-wise attention.
    
    Args:
        S (int): Number of frames in the sequence.
        P (int): Number of patches per frame.
        mode (str): Attention mode, could be "causal", "window", "full", "causal_full_causal","full_causal","causal_full".
        dtype (torch.dtype): Data type of the mask.
        device (torch.device): Device to place the mask on.

    Returns:
        torch.Tensor or List[torch.Tensor]: Attention mask tensor(s) with shape [S*P, S*P] or a list of such tensors.
    """

    total_length = S * P
    mask = None
    # print(f"Sequence length: {S}, Patches per frame: {P}, Total length: {total_length}")
    # print(f"Window size: {window_size}, Lookahead size: {look_ahead_size}")
    # print(f"Mode: {mode}")
    if mode == "causal" or "causal" in mode:
        logger.info("Using block-wise causal attention mask.")
        # for i in range(S):
        #     curr_view_start = i * P
        #     curr_view_end = (i + 1) * P
        #     mask[curr_view_start:curr_view_end, curr_view_end:] = float('-inf')
        ends = torch.zeros(total_length, device=device, dtype=torch.long)

        # Set up block-wise causal mask: each frame attends to all previous frames + current
        frame_indices = torch.arange(start=0, end=total_length, step=P, device=device)

        for frame_idx, frame_start in enumerate(frame_indices):
            frame_end = frame_start + P
            context_end = min((frame_idx + 1 + look_ahead_size) * P, total_length)
            ends[frame_start:frame_end] = context_end
        
        def block_causal_mask(b, h, q_idx, kv_idx):
            _ = b, h  # Suppress unused parameter warnings
            return (kv_idx < ends[q_idx])
        
        mask0 = torch.compile(create_block_mask)(
                                    block_causal_mask,
                                    B=None, H=None,
                                    Q_LEN=total_length,
                                    KV_LEN=total_length , 
                                    device=device)
        return mask0
    elif mode == "window":
        logger.info(f"Using block-wise sliding window attention mask with window size {window_size}, lookahead size {look_ahead_size}.")
        ends = torch.zeros(total_length, device=device, dtype=torch.long)
        starts = torch.zeros(total_length, device=device, dtype=torch.long)
        frame_indices = torch.arange(start=0, end=total_length, step=P, device=device)

        for frame_idx, frame_start in enumerate(frame_indices):
            frame_end = frame_start + P
            context_end = min((frame_idx + 1 + look_ahead_size) * P,  total_length)
            ends[frame_start:frame_end] = context_end
            start_frame = max(0, frame_idx - window_size)
            context_start = start_frame * P
            starts[frame_start:frame_end] = context_start

        def block_window_mask(b, h, q_idx, kv_idx):
            _ = b, h
            return ((starts[q_idx] <= kv_idx) & (kv_idx < ends[q_idx])) |  (kv_idx < P)
        
        mask = torch.compile(create_block_mask)(
                                    block_window_mask,
                                    B=None, H=None,
                                    Q_LEN=total_length,
                                    KV_LEN=total_length,
                                    device=device)
    elif mode == "test":
        logger.info(f"Testing: Using block-wise fixed window attention mask with window size {window_size}")
        ends = torch.zeros(total_length, device=device, dtype=torch.long)
        starts = torch.zeros(total_length, device=device, dtype=torch.long)
        frame_indices = torch.arange(start=0, end=total_length, step=P, device=device)
        window_start = window_size * P
        for frame_idx, frame_start in enumerate(frame_indices):
            frame_end = frame_start + P
            context_end = min((frame_idx + 1) * P,  total_length) + window_start
            ends[frame_start:frame_end] = context_end
            context_start = frame_idx * P + window_start
            starts[frame_start:frame_end] = context_start

        def block_window_mask(b, h, q_idx, kv_idx):
            _ = b, h
            return ((starts[q_idx] <= kv_idx) & (kv_idx < ends[q_idx])) |  (kv_idx < window_start)
        
        mask = torch.compile(create_block_mask)(
                                    block_window_mask,
                                    B=None, H=None,
                                    Q_LEN=total_length,
                                    KV_LEN=total_length + window_start,
                                    device=device)
    elif mode == "full":
        mask = None
    else:
        raise NotImplementedError(f"Unknown attention mode: {mode}")
    return mask

class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = Attention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None,
    ) -> None:
        super().__init__()

        self.norm1 = norm_layer(dim)

        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            qk_norm=qk_norm,
            fused_attn=fused_attn,
            rope=rope,
        )

        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, bias=ffn_bias
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.sample_drop_ratio = drop_path


    def forward(self, x: Tensor, pos=None, attn_mask=None, **kwargs) -> Tensor:
        def attn_residual_func(x: Tensor, pos=None, attn_mask=None, **kwargs) -> Tensor:
            # Only pass extra args when present; MemEffAttention (DINOv2) does not accept them
            extra = {}
            if attn_mask is not None:
                extra["attn_mask"] = attn_mask
            extra.update(kwargs)
            return self.ls1(self.attn(self.norm1(x), pos=pos, **extra))

        def ffn_residual_func(x: Tensor) -> Tensor:
            return self.ls2(self.mlp(self.norm2(x)))

        if self.training and self.sample_drop_ratio > 0.1:
            x = drop_add_residual_stochastic_depth(
                x,
                pos=pos,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
        elif self.training and self.sample_drop_ratio > 0.0:
            x = x + self.drop_path1(attn_residual_func(x, pos=pos, attn_mask=attn_mask))
            x = x + self.drop_path1(ffn_residual_func(x))  # FIXME: drop_path2
        else:
            x = x + attn_residual_func(x, pos=pos, attn_mask=attn_mask, **kwargs)
            x = x + ffn_residual_func(x)
        return x


def drop_add_residual_stochastic_depth(
    x: Tensor, residual_func: Callable[[Tensor], Tensor], sample_drop_ratio: float = 0.0, pos=None
) -> Tensor:
    # 1) extract subset using permutation
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    x_subset = x[brange]

    # 2) apply residual_func to get residual
    if pos is not None:
        # if necessary, apply rope to the subset
        pos = pos[brange]
        residual = residual_func(x_subset, pos=pos)
    else:
        residual = residual_func(x_subset)

    x_flat = x.flatten(1)
    residual = residual.flatten(1)

    residual_scale_factor = b / sample_subset_size

    # 3) add the residual
    x_plus_residual = torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
    return x_plus_residual.view_as(x)


def get_branges_scales(x, sample_drop_ratio=0.0):
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    residual_scale_factor = b / sample_subset_size
    return brange, residual_scale_factor


def add_residual(x, brange, residual, residual_scale_factor, scaling_vector=None):
    if scaling_vector is None:
        x_flat = x.flatten(1)
        residual = residual.flatten(1)
        x_plus_residual = torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
    else:
        x_plus_residual = scaled_index_add(
            x, brange, residual.to(dtype=x.dtype), scaling=scaling_vector, alpha=residual_scale_factor
        )
    return x_plus_residual


attn_bias_cache: Dict[Tuple, Any] = {}


def get_attn_bias_and_cat(x_list, branges=None):
    """
    this will perform the index select, cat the tensors, and provide the attn_bias from cache
    """
    batch_sizes = [b.shape[0] for b in branges] if branges is not None else [x.shape[0] for x in x_list]
    all_shapes = tuple((b, x.shape[1]) for b, x in zip(batch_sizes, x_list))
    if all_shapes not in attn_bias_cache.keys():
        seqlens = []
        for b, x in zip(batch_sizes, x_list):
            for _ in range(b):
                seqlens.append(x.shape[1])
        attn_bias = fmha.BlockDiagonalMask.from_seqlens(seqlens)
        attn_bias._batch_sizes = batch_sizes
        attn_bias_cache[all_shapes] = attn_bias

    if branges is not None:
        cat_tensors = index_select_cat([x.flatten(1) for x in x_list], branges).view(1, -1, x_list[0].shape[-1])
    else:
        tensors_bs1 = tuple(x.reshape([1, -1, *x.shape[2:]]) for x in x_list)
        cat_tensors = torch.cat(tensors_bs1, dim=1)

    return attn_bias_cache[all_shapes], cat_tensors


def drop_add_residual_stochastic_depth_list(
    x_list: List[Tensor],
    residual_func: Callable[[Tensor, Any], Tensor],
    sample_drop_ratio: float = 0.0,
    scaling_vector=None,
) -> Tensor:
    # 1) generate random set of indices for dropping samples in the batch
    branges_scales = [get_branges_scales(x, sample_drop_ratio=sample_drop_ratio) for x in x_list]
    branges = [s[0] for s in branges_scales]
    residual_scale_factors = [s[1] for s in branges_scales]

    # 2) get attention bias and index+concat the tensors
    attn_bias, x_cat = get_attn_bias_and_cat(x_list, branges)

    # 3) apply residual_func to get residual, and split the result
    residual_list = attn_bias.split(residual_func(x_cat, attn_bias=attn_bias))  # type: ignore

    outputs = []
    for x, brange, residual, residual_scale_factor in zip(x_list, branges, residual_list, residual_scale_factors):
        outputs.append(add_residual(x, brange, residual, residual_scale_factor, scaling_vector).view_as(x))
    return outputs


class NestedTensorBlock(Block):
    def forward_nested(self, x_list: List[Tensor]) -> List[Tensor]:
        """
        x_list contains a list of tensors to nest together and run
        """
        assert isinstance(self.attn, MemEffAttention)

        if self.training and self.sample_drop_ratio > 0.0:

            def attn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.attn(self.norm1(x), attn_bias=attn_bias)

            def ffn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.mlp(self.norm2(x))

            x_list = drop_add_residual_stochastic_depth_list(
                x_list,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                scaling_vector=(self.ls1.gamma if isinstance(self.ls1, LayerScale) else None),
            )
            x_list = drop_add_residual_stochastic_depth_list(
                x_list,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                scaling_vector=(self.ls2.gamma if isinstance(self.ls1, LayerScale) else None),
            )
            return x_list
        else:

            def attn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.ls1(self.attn(self.norm1(x), attn_bias=attn_bias))

            def ffn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.ls2(self.mlp(self.norm2(x)))

            attn_bias, x = get_attn_bias_and_cat(x_list)
            x = x + attn_residual_func(x, attn_bias=attn_bias)
            x = x + ffn_residual_func(x)
            return attn_bias.split(x)

    def forward(self, x_or_x_list):
        if isinstance(x_or_x_list, Tensor):
            return super().forward(x_or_x_list)
        elif isinstance(x_or_x_list, list):
            if not XFORMERS_AVAILABLE:
                raise AssertionError("xFormers is required for using nested tensors")
            return self.forward_nested(x_or_x_list)
        else:
            raise AssertionError
