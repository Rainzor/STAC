# Copyright (c) 2025 STAC Authors. All rights reserved.

import gc
import logging
import torch
import torch.nn.functional as F
import warnings
from typing import List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class KVManager:
    """
    Window-based Key-Value Manager with Recent + Pinned token caching.
    K, V shapes: [B, H, T, D]; T = (#frames * token_per_frame).

    Usage:
        1) append_kv: append key/value tokens as usual. (layer-wise).
        2) attention: attention is computed over hot cache and update scores (layer-wise).
        3) prune_kv: prune hot cache based on recent + pinned, for all layers at once.

    """
    def __init__(self, 
                 num_layers: int,
                 num_heads: int,
                 head_dim: int,
                 token_per_frame: int,
                 register_layers: Optional[List[int]] = None, 
                 chunk_size: int = 1,        # frames per step; used for buffer sizing and chunk_token_size
                 buffer_size: int = 8,  
                 recent_size: int = 1,        # recent "frame-equivalent" count -> converted to tokens
                 pinned_idx: Optional[List[int]] = None,  # frame indices to always pin (entire frames)
                 dtype: torch.dtype = torch.float16,  
                 device: Optional[torch.device] = None,  
                 debug: bool = False,
                 gpu_limit: int = 250,
                 **kwargs
                 ):
        
        # Metadata
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.token_per_f = int(token_per_frame)
        self.reserved_buffer_size = int(buffer_size)


        if register_layers is None:
            managed = list(range(self.num_layers))   # manage all layers
        else:
            managed = sorted({l for l in register_layers if 0 <= l < self.num_layers})
        if len(managed) == 0:
            raise ValueError("register_layers is empty after filtering valid indices.")

        # layer -> slot (unmanaged layers -1), slot -> layer (original layer index for managed)
        self._managed_layers: List[int] = managed                     # length L_eff
        self._L_eff = len(managed)
        
        self._layer2slot = torch.full((self.num_layers,), -1, dtype=torch.long)
        for s, l in enumerate(managed):
            self._layer2slot[l] = s
        self._slot2layer = torch.tensor(managed, dtype=torch.long)   # [L_eff]

        self.recent_size = int(recent_size)
        self.cache_size = self.recent_size
        pinned_idx = torch.tensor(pinned_idx or [], dtype=torch.long)
        self.pinned_size = len(pinned_idx)
        self.chunk_size = int(chunk_size)
        self.hot_size = self.cache_size + self.pinned_size
        if self.reserved_buffer_size < self.hot_size + self.chunk_size:
            warnings.warn(f"Warning: buffer_size {self.reserved_buffer_size} is smaller than hh_size+recent_size+pinned ({self.hot_size}). This may lead to no pruning.")
            self.reserved_buffer_size = self.hot_size + self.chunk_size

        # # When buffer is very large, cap GPU usage to reduce OOM risk
        # effective_gpu_limit = min(gpu_limit, 200) if self.reserved_buffer_size > 320 else gpu_limit
        effective_gpu_limit = gpu_limit
        # Split total buffer frames into GPU part and CPU part
        self.reserved_buffer_size_cpu = self.reserved_buffer_size - effective_gpu_limit
        if self.reserved_buffer_size_cpu < 0:
            self.reserved_buffer_size_cpu = 0
        # GPU buffer is capped by effective_gpu_limit
        self.reserved_buffer_size = min(self.reserved_buffer_size, effective_gpu_limit)
        
        assert self.token_per_f > 0, "token_per_frame must be positive."
        assert self.recent_size >= 0, "recent_size and hh_size must be non-negative."
        assert self.cache_size > 0, "At least one of hh_size, recent_size, pinned_idx must be non-zero."

        # Derived sizes (token-level)
        self.recent_token_size  = int(self.recent_size * self.token_per_f)
        self.cache_token_size   = int(self.cache_size * self.token_per_f)
        self.reserved_buffer_token_size = int(self.reserved_buffer_size * self.token_per_f)
        self.reserved_buffer_token_size_cpu = int(self.reserved_buffer_size_cpu * self.token_per_f)
        self.hot_token_size     = int(self.hot_size * self.token_per_f)
        self.chunk_token_size   = int(self.chunk_size * self.token_per_f)

        if self.pinned_size > 0:
            pinned_tok = []
            for idx in pinned_idx.tolist():
                if idx < 0:
                    continue
                start = idx * self.token_per_f
                end   = start + self.token_per_f
                pinned_tok.extend(range(start, end))
            self._pinned_token_index = torch.tensor(pinned_tok, dtype=torch.long)
        else:
            self._pinned_token_index = torch.tensor([], dtype=torch.long)

        self.dtype = dtype
        self.device = device if device is not None else torch.device("cpu")
        self.debug = debug

        self.key_cache_hot: Optional[torch.Tensor] = None
        self.value_cache_hot: Optional[torch.Tensor] = None
        self._token_indices_hot: Optional[torch.Tensor] = None
        self._offset_hot = None

        self.key_cache_hot_cpu: Optional[torch.Tensor] = None
        self.value_cache_hot_cpu: Optional[torch.Tensor] = None
        self._token_indices_hot_cpu: Optional[torch.Tensor] = None
        self._offset_hot_cpu = None
        self.use_cpu = False

        if type(self) is KVManager:
            self._log_registration()

    def _log_registration(self) -> None:
        """Override in subclasses to log registration. Called from __init__ only when type(self) is the concrete class."""
        pinned = []
        if self._pinned_token_index.numel() > 0:
            pinned = sorted(set((self._pinned_token_index // self.token_per_f).tolist()))
        logger.info(
            "[WindowKV] layers=%d/%d  tok/frame=%d  window=%dx%d  pinned=%s  kv_gpu_cap=%dx%d  kv_cpu_cap=%dx%d",
            self._L_eff, self.num_layers, self.token_per_f,
            self.recent_size, self.token_per_f,
            pinned,
            self.reserved_buffer_size, self.token_per_f,
            self.reserved_buffer_size_cpu, self.token_per_f,
        )

    #----- Utility functions -----#

    def is_managed(self, layer_idx: int) -> bool:
        s = int(self._layer2slot[layer_idx].item())
        return s >= 0

    def _to_slot(self, layer_idx: int, *, strict: bool = True) -> int:
        s = int(self._layer2slot[layer_idx].item())
        if s < 0 and strict:
            raise ValueError(f"Layer {layer_idx} is not managed by HeavyHittersKV.")
        return s

    def _layers_to_slots(self, layers: Union[List[int], torch.Tensor]) -> torch.Tensor:
        if isinstance(layers, list):
            slots = [int(self._layer2slot[l].item()) for l in layers if int(self._layer2slot[l].item()) >= 0]
            return torch.tensor(slots, dtype=torch.long, device=self.device)
        
        return self._layer2slot[layers].to(self.device)

    #----- Reset / Free -----#
    def reset(self):
        self._processed_frames = 0
        # per-layer [H, 0] LongTensor, initialize with -1
        self.key_cache_hot = torch.empty(self._L_eff, self.num_heads, self.reserved_buffer_token_size, self.head_dim,
                                    dtype=self.dtype, device=self.device)
        self.value_cache_hot = torch.empty_like(self.key_cache_hot)
        self._token_indices_hot = torch.full((self._L_eff, self.num_heads, self.reserved_buffer_token_size), -1, dtype=torch.long, device=self.device)
        self._offset_hot = [0] * self._L_eff
        
        self.key_cache_hot_cpu = torch.empty(self._L_eff, self.num_heads, self.reserved_buffer_token_size_cpu, self.head_dim, dtype=self.dtype, device="cpu")
        self.value_cache_hot_cpu = torch.empty_like(self.key_cache_hot_cpu)
        self._offset_hot_cpu = [0] * self._L_eff
    
    def free(self):
        # 1) Clear metadata
        self._processed_frames = 0
        # 2) Clear caches
        self.key_cache_hot      = None
        self.value_cache_hot    = None
        self._token_indices_hot = None
        self._offset_hot        = None

        #cpu cache
        self.key_cache_hot_cpu  = None
        self.value_cache_hot_cpu = None
        self._token_indices_hot_cpu = None
        self._offset_hot_cpu = None
        # 3) Garbage collect
        gc.collect()
        if isinstance(self.device, torch.device) and self.device.type == "cuda":
            with torch.cuda.device(self.device):
                torch.cuda.synchronize(self.device)
                torch.cuda.empty_cache()

    def prefill(self, 
                query_states: torch.Tensor, 
                key_states: torch.Tensor, 
                value_states: torch.Tensor, 
                layer_idx: int):
        """
        Prefill K, V cache and compute initial scores.
        query_states: [B, T, H, D]
        key_states, value_states: [B, T, H, D]
        """
        B, T, H, D = query_states.shape
        assert H == self.num_heads and D == self.head_dim, "Incompatible head or head_dim."
        assert B == 1, "Batch size > 1 not supported in prefill()."
        assert T % self.token_per_f == 0, "T must be a multiple of token_per_frame in prefill()."
        # Fill K, V
        self.reset()
        slot_idx = self._to_slot(layer_idx, strict=True)
        self.key_cache_hot[slot_idx, :, :T, :].copy_(key_states[0])
        self.value_cache_hot[slot_idx, :, :T, :].copy_(value_states[0])
        self._offset_hot[slot_idx] = T
        # Compute initial scores
        attn_output = self.decode_sparse_attn(query_states, layer_idx)  # [B, T, H, D]
        F = T // self.token_per_f
        self._processed_frames = F
        self._token_indices_hot[slot_idx, :, :T].copy_(
            torch.arange(F, device=self.device).unsqueeze(1) * self.token_per_f
            + torch.arange(self.token_per_f, device=self.device).unsqueeze(0)
        )  # [H, T]

        return attn_output


    def append_kv(self, key_states, value_states, layer_idx: int):
        """
        Append new K, V to the cache.
        key_states, value_states: [B, H, T_new, D]
        GPU has a hard cap (gpu_limit frames); overflow tokens are stored on CPU.
        """
        B, H, new_T, D = key_states.shape
        assert H == self.num_heads and D == self.head_dim, "Incompatible head or head_dim."
        assert B == 1, "Batch size > 1 not supported in append_kv()."
        assert new_T % self.token_per_f == 0, "T_new must be a multiple of token_per_frame in append_kv()."

        slot_idx = self._to_slot(layer_idx, strict=True)

        old_T_gpu = self._offset_hot[slot_idx]
        gpu_cap = self.reserved_buffer_token_size

        # How many new tokens can still go to GPU?
        gpu_free = max(gpu_cap - old_T_gpu, 0)
        to_gpu = min(new_T, gpu_free)
        to_cpu = new_T - to_gpu

        if to_cpu > 0:
            warnings.warn(
                f"Exceeding GPU buffer token size {self.reserved_buffer_token_size} in append_kv(). "
                f"{to_cpu} tokens will be stored on CPU."
            )
            self.use_cpu = True

        # Split new tokens: first to_cpu -> CPU, last to_gpu -> GPU (keep latest on GPU)
        k_new = key_states[0]  # [H, new_T, D]
        v_new = value_states[0]

        k_cpu_part = k_new[:, :to_cpu, :] if to_cpu > 0 else None
        v_cpu_part = v_new[:, :to_cpu, :] if to_cpu > 0 else None
        k_gpu_part = k_new[:, new_T - to_gpu:, :] if to_gpu > 0 else None
        v_gpu_part = v_new[:, new_T - to_gpu:, :] if to_gpu > 0 else None

        # ---- Append to CPU ring buffer ----
        if to_cpu > 0 and self.reserved_buffer_token_size_cpu > 0:
            k_cpu_part = k_cpu_part.to("cpu")
            v_cpu_part = v_cpu_part.to("cpu")

            old_T_cpu = self._offset_hot_cpu[slot_idx]
            cpu_cap = self.reserved_buffer_token_size_cpu
            total_cpu_after = old_T_cpu + to_cpu

            if total_cpu_after <= cpu_cap:
                # Just append
                self.key_cache_hot_cpu[slot_idx, :, :total_cpu_after, :].copy_(
                    torch.cat([self.key_cache_hot_cpu[slot_idx, :, :old_T_cpu, :], k_cpu_part], dim=1)
                )
                self.value_cache_hot_cpu[slot_idx, :, :total_cpu_after, :].copy_(
                    torch.cat([self.value_cache_hot_cpu[slot_idx, :, :old_T_cpu, :], v_cpu_part], dim=1)
                )
                self._offset_hot_cpu[slot_idx] = total_cpu_after
            else:
                raise RuntimeError("CPU buffer overflow in append_kv(). Increase gpu_limit or buffer_size.")
        # ---- Append to GPU buffer ----
        if to_gpu > 0:
            start_gpu = old_T_gpu
            end_gpu = old_T_gpu + to_gpu
            assert end_gpu <= gpu_cap, "GPU buffer overflow in append_kv."
            self.key_cache_hot[slot_idx, :, start_gpu:end_gpu, :].copy_(k_gpu_part.to(self.device))
            self.value_cache_hot[slot_idx, :, start_gpu:end_gpu, :].copy_(v_gpu_part.to(self.device))
            self._offset_hot[slot_idx] = end_gpu

        # ---- Update token indices (only for GPU part, CPU indices unused) ----
        F_new = new_T // self.token_per_f
        base = torch.arange(self._processed_frames,
                            self._processed_frames + F_new,
                            device=self.device)
        base = base.unsqueeze(1) * self.token_per_f                     # [F_new, 1]
        offsets = torch.arange(self.token_per_f, device=self.device).unsqueeze(0)  # [1, token_per_f]
        frame_token_ids = (base + offsets).reshape(-1)                  # [F_new * token_per_f]
        new_token_indices = frame_token_ids.expand(H, -1)               # [H, new_T]

        if to_gpu > 0:
            gpu_idx_part = new_token_indices[:, new_T - to_gpu:]
            self._token_indices_hot[slot_idx, :, old_T_gpu:old_T_gpu + to_gpu].copy_(gpu_idx_part)

        # We keep _processed_frames as global frame counter using the last slot
        if slot_idx == self._L_eff - 1:
            self._processed_frames += F_new

    @torch.no_grad()
    def decode_sparse_attn(self, query_states: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        Compute attention with the current sparse K, V cache.
        query_states: [B, Tq, H, D]
        Returns: attn_output: [B, Tq, H, D]
        """
        B, Tq, H, D = query_states.shape
        assert H == self.num_heads and D == self.head_dim, "Incompatible head or head_dim."
        assert B == 1, "Batch size > 1 not supported in decode_sparse_attn()."


        slot_idx = self._to_slot(layer_idx, strict=True)

        T_gpu = self._offset_hot[slot_idx]
        T_cpu = self._offset_hot_cpu[slot_idx]

        dev = query_states.device

        # Get GPU part
        k_gpu = self.key_cache_hot[slot_idx, :, :T_gpu, :].to(dev)
        v_gpu = self.value_cache_hot[slot_idx, :, :T_gpu, :].to(dev)

        # Optionally concat CPU part on the left (older context)
        if self.use_cpu and T_cpu > 0:
            k_cpu = self.key_cache_hot_cpu[slot_idx, :, :T_cpu, :].to(dev)
            v_cpu = self.value_cache_hot_cpu[slot_idx, :, :T_cpu, :].to(dev)
            k = torch.cat([k_gpu,k_cpu], dim=1)  # [H, T_cpu + T_gpu, D]
            v = torch.cat([v_gpu,v_cpu], dim=1)
        else:
            k = k_gpu
            v = v_gpu


        assert query_states.is_cuda and k.is_cuda and v.is_cuda, "triton kernel requires CUDA tensors"
        assert query_states.dtype in (torch.float16, torch.bfloat16), "q must be fp16/bf16"
        assert k.dtype in (torch.float16, torch.bfloat16) and v.dtype in (torch.float16, torch.bfloat16)

        q = query_states.transpose(1, 2)  # [B, H, Tq, D]
        k = k.unsqueeze(0)                # [B, H, Tkv, D]
        v = v.unsqueeze(0)                # [B, H, Tkv, D]
        out = F.scaled_dot_product_attention(q, k, v).transpose(1, 2)  # [B, Tq, H, D]

        # Release temporary GPU tensors (incl. CPU->GPU copies) so memory can be reclaimed
        # and not accumulate across layers/steps (reduces OOM risk).
        del q, k, v, k_gpu, v_gpu
        if T_cpu > 0:
            del k_cpu, v_cpu

        return out  # [B, Tq, H, D]

    @torch.no_grad()
    def prune_kv(self):
        """
        Perform pruning on all registered layers.
        """
        for slot_idx in range(self._L_eff):
            self._prune_kv(slot_idx)

    @torch.no_grad()
    def _prune_kv(self, slot_idx: int):
        """
        Perform pruning on the given slot_idx:
        Keep tokens belonging to (pinned ∪ recent ∪ per-head heavy-hitter),
        and compact K/V/score/token_indices to the first T' tokens.
        """
        #TODO: reduce the SORT operation cost by using a selection algorithm
        assert 0 <= slot_idx < self._L_eff, "Invalid slot_idx in _prune_kv()."
        T_tokens = self._offset_hot[slot_idx]
        if T_tokens == 0:
            return

        device = self.device
        H = self.num_heads
        # If current token length does not exceed pruning threshold, return early
        target_budget = self.hot_token_size  # Only recent + HH (pinned are additionally included)
        if T_tokens <= max(target_budget, self.recent_token_size):
            return

        # 1) Recent tokens
        recent_keep = min(self.recent_token_size, T_tokens)
        base_recent = torch.arange(T_tokens - recent_keep, T_tokens, device=device)  # [R]

        # 2) Merge pinned tokens into base
        pinned_tok = self._pinned_token_index.to(device)  # [P]
        if pinned_tok.numel() > 0:
            pinned_tok = pinned_tok[(pinned_tok >= 0) & (pinned_tok < T_tokens)]
            base = torch.unique(torch.cat([pinned_tok, base_recent], dim=0)).sort()[0]
        else:
            base = base_recent
        base_size = base.numel()  # [P + R]

        # 3) Prefix: tokens not in base
        if base_size == T_tokens:
            # All tokens are within base, no pruning needed
            return

        # 4) Determine the number of tokens to keep
        # Total number of tokens to keep
        all_keep_tokens = base.unsqueeze(0).expand(H, -1)  # [H, T']
        # 7) Compact K/V/scores/token_indices to the first T' tokens
        self._apply_keep_and_compact(slot_idx, all_keep_tokens)

    def _apply_keep_and_compact(self, slot_idx: int, keep_idx_h: torch.Tensor):
        """
        keep_idx_h: [H, T'] kept token indices per head.
        Parallel gather version; no per-head loop.
        """
        H, Tprime = keep_idx_h.shape
        K = self.key_cache_hot[slot_idx]          # [H, T, D]
        V = self.value_cache_hot[slot_idx]        # [H, T, D]
        TI = self._token_indices_hot[slot_idx]    # [H, T]


        D = K.shape[-1]
        # ---- expand index for batched gather ----
        idx_exp = keep_idx_h.unsqueeze(-1).expand(-1, -1, D)  # [H, T', D]

        # ---- gather in parallel ----
        K_sel  = torch.gather(K, 1, idx_exp)  # [H, T', D]
        V_sel  = torch.gather(V, 1, idx_exp)
        TI_sel = torch.gather(TI, 1, keep_idx_h)   # [H, T']

        # ---- copy back (in place) ----
        K[:, :Tprime, :].copy_(K_sel)
        V[:, :Tprime, :].copy_(V_sel)
        TI[:, :Tprime].copy_(TI_sel)
        # ---- clear out the tail ----
        TI[:, Tprime:].fill_(-1)

        self._offset_hot[slot_idx] = Tprime

    @torch.no_grad()
    def retrieve_kv(self, layer_idx: int, **kwargs) -> Tuple[int, int]:
        raise NotImplementedError("retrieve_kv() is not implemented for HeavyHittersKV.")

    @torch.no_grad()
    def append_positions(self, new_positions: torch.Tensor, new_pos_mask: torch.Tensor):
        raise NotImplementedError("append_positions() is not implemented for HeavyHittersKV.")

    # --- Public API ---
    def get_token_indices(self) -> torch.Tensor:
        """
        Get the original token indices (w.r.t. the entire video) of the currently kept tokens,
        for each head.
        """
        return self._token_indices_hot.clone().cpu()  # [L_eff, H, T']
    
    def get_layers(self) -> List[int]:
        """
        Get the list of layers managed by this KVManager.
        """
        return self._managed_layers.copy()

    def get_kv(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the currently kept K, V for all layers.
        Returns:
            key_cache_hot: [L_eff, H, T', D]
            value_cache_hot: [L_eff, H, T', D]
        """
        return self.key_cache_hot.clone().cpu(), self.value_cache_hot.clone().cpu()

    def get_offset(self) -> List[int]:
        """
        Get the current allocated size (number of tokens) for each layer (GPU part only).
        """
        return self._offset_hot.copy()

    def get_offset_cpu(self) -> List[int]:
        """
        Get the current CPU cache token count per layer. Returns zeros when use_cpu is False.
        """
        if getattr(self, "_offset_hot_cpu", None) is None:
            return [0] * self._L_eff
        return self._offset_hot_cpu.copy()

    def get_cpu_memory_usage(self) -> Tuple[float, float]:
        """
        Get CPU cache memory usage (used MB, allocated MB). Returns (0, 0) when not using CPU offload.
        """
        if not getattr(self, "use_cpu", False) or getattr(self, "_offset_hot_cpu", None) is None:
            return 0.0, 0.0
        usage = 0.0
        alloc = 0.0
        elem_bytes = torch.finfo(self.dtype).bits // 8
        for l in range(self._L_eff):
            T = self._offset_hot_cpu[l]
            used_mem_bytes = 2 * self.num_heads * T * self.head_dim * elem_bytes
            usage += used_mem_bytes / (1024.0 ** 2)
            alloc_mem_bytes = 2 * self.reserved_buffer_token_size_cpu * self.num_heads * self.head_dim * elem_bytes
            alloc += alloc_mem_bytes / (1024.0 ** 2)
        return usage, alloc

    def get_memory_usage(self) -> Tuple[float, float]:
        """
        Get the GPU cache memory usage (used MB, allocated MB) for each layer.
        """
        usage = 0.0
        alloc = 0.0
        for l in range(self._L_eff):
            T = self._offset_hot[l]
            used_mem_bytes = 2 * self.num_heads * T * self.head_dim * torch.tensor(torch.finfo(self.dtype).bits // 8)
            usage += used_mem_bytes.item() / (1024.0 **2)
            alloc_mem_bytes = 2 * (self.reserved_buffer_token_size) * self.num_heads * self.head_dim * torch.tensor(torch.finfo(self.dtype).bits // 8)
            alloc += alloc_mem_bytes.item() / (1024.0 **2)
        return usage, alloc

    def get_memory_details(self) -> dict:
        """Return dict compatible with eval/compare (total_usage, temporal_cache_usage, spatial_cache_usage in MB)."""
        used, alloc = self.get_memory_usage()
        cpu_used, cpu_alloc = self.get_cpu_memory_usage()
        total_usage = used + cpu_used
        return {
            "total_usage": total_usage,
            "total_alloc": alloc + cpu_alloc,
            "temporal_cache_usage": used,
            "temporal_cache_alloc": alloc,
            "spatial_cache_usage": 0.0,
        }