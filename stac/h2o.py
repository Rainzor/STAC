# Copyright (c) 2025 STAC Authors. All rights reserved.

import logging
from typing import List, Optional, Tuple
import torch
import torch.nn.functional as F
import numpy as np
import warnings
from .kv_manager import KVManager

from .flash_attn_triton import fa_forward_colsum_fast, fa_forward_colsum_fast_sub

try:
    import attn_cuda as _attn_cuda
    _ATTN_CUDA_AVAILABLE = getattr(_attn_cuda, "is_available", lambda: True)()
except Exception:
    _ATTN_CUDA_AVAILABLE = False

logger = logging.getLogger(__name__)


class HeavyHittersKV(KVManager):
    """
    H2O-style KV selector with **token-level** grouping.
    K, V shapes: [B, H, T, D]; T = (#frames * token_per_frame).

    Usage:
        1) append_kv: append key/value tokens as usual. (layer-wise).
        2) attention: attention is computed over hot cache and update scores (layer-wise).
        3) prune_kv: prune hot cache based on heavy-hitter + recent + pinned, for all layers at once.

    """
    def __init__(self, 
                 *args,
                 hh_size: int = 0,            # heavy-hitter "frame-equivalent" count -> converted to tokens
                 temperature: float = 1.0,
                 attn_backend: str = "cuda",
                 subsample_ratio: float = 1.0,
                 **kwargs
                 ):
        super().__init__(*args,
                         **kwargs)
        self.attn_backend = str(attn_backend).strip().lower()
        if self.attn_backend not in ("cuda", "triton"):
            raise ValueError(f"attn_backend must be 'cuda' or 'triton', got '{attn_backend}'.")
        self.subsample_ratio = float(subsample_ratio)
        if not (0.0 < self.subsample_ratio <= 1.0):
            raise ValueError(f"subsample_ratio must be in (0, 1], got {self.subsample_ratio}.")
        self.use_attn_cuda = (self.attn_backend == "cuda")

        # Metadata Update
        self.hh_size = int(hh_size)
        self.cache_size = self.hh_size + self.recent_size
        self.hot_size = self.cache_size + self.pinned_size
        if self.reserved_buffer_size <= self.hot_size:
            warnings.warn(f"Warning: buffer_size {self.reserved_buffer_size} is smaller than hh_size+recent_size+pinned ({self.hot_size}). This may lead to no pruning.")
            self.reserved_buffer_size = self.hot_size + 1

        self.temperature = float(temperature)
        assert self.hh_size >= 0, "hh_size must be non-negative."
        assert self.temperature > 0, "temperature must be positive."

        # Derived sizes (token-level)
        self.hh_token_size      = int(self.hh_size * self.token_per_f)
        self.cache_token_size   = int(self.cache_size * self.token_per_f)
        self.hot_token_size     = int(self.hot_size * self.token_per_f)

        self._scores_hot: Optional[torch.Tensor] =  None  # List of [H, buffer_T] tensors
        self._scores_hot_count: Optional[torch.Tensor] = None  # List of [H, buffer_T] tensors
        self._last_score_offset = None # List of int
        self._last_query_offset = None

        if type(self) is HeavyHittersKV:
            self._log_registration()

    def _log_registration(self) -> None:
        super()._log_registration()
        logger.info("[H2OKV] anchor tok=%dx%d  temperature=%.2f",
                    self.hh_size, self.token_per_f, self.temperature)
        use_cuda = bool(self.use_attn_cuda and _ATTN_CUDA_AVAILABLE)
        logger.info(
            "[ATTENTION] backend=%s (cuda_available=%s, effective_cuda=%s)  subsample_ratio=%.3f",
            self.attn_backend,
            _ATTN_CUDA_AVAILABLE,
            use_cuda,
            self.subsample_ratio,
        )

    #------ Utility functions ------
    def _estimate_scores(self, slot_idx: int, T_live: int) -> torch.Tensor:
        """
        Get the estimated scores for the first T_live tokens in the given slot.
        Returns a [H, T_live] tensor.
        """
        assert 0 <= slot_idx < self._L_eff
        assert T_live <= self._offset_hot[slot_idx], "T_live exceeds allocated size."
        scores = self._scores_hot[slot_idx, :, :T_live]  # [H, T_live]
        return scores
    
    #------ Core functions ------
    def reset(self):
        super().reset()
        self._processed_frames = 0
        # per-layer [H, 0] LongTensor, initialize with -1
        self._scores_hot = torch.zeros((self._L_eff, self.num_heads, self.reserved_buffer_token_size), dtype=torch.float32, device=self.device)
        self._scores_hot_count = torch.zeros((self._L_eff, self.num_heads, self.reserved_buffer_token_size), dtype=torch.float32, device=self.device)
        # track scored length
        self._last_score_offset = [0] * self._L_eff
        self._last_query_offset = [0] * self._L_eff

    def free(self):
        self._scores_hot = None
        self._scores_hot_count = None
        self._last_score_offset = None
        self._last_query_offset = None

        super().free()

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
        super().prefill(query_states, key_states, value_states, layer_idx)


    def append_kv(self, key_states, value_states, layer_idx: int):
        """
        Append new K, V to the cache.
        key_states, value_states: [B, H, T_new, D]
        """
        super().append_kv(key_states, value_states, layer_idx)

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
        slot_idx = self._to_slot(layer_idx)
        k = self.key_cache_hot[slot_idx, :, :self._offset_hot[slot_idx], :]    # [H, T, D]
        v = self.value_cache_hot[slot_idx, :, :self._offset_hot[slot_idx], :]  # [H, T, D]

        k = k.unsqueeze(0).transpose(1, 2).contiguous()  # [B, T, H, D]
        v = v.unsqueeze(0).transpose(1, 2).contiguous()  # [B, T, H, D]
        q = query_states.contiguous()               # [B, Tq, H, D]


        assert q.is_cuda and k.is_cuda and v.is_cuda, "triton kernel requires CUDA tensors"
        assert q.dtype in (torch.float16, torch.bfloat16), "q must be fp16/bf16"
        assert k.dtype in (torch.float16, torch.bfloat16) and v.dtype in (torch.float16, torch.bfloat16)

        subsample = getattr(self, "subsample_ratio", 1.0)
        if self.use_attn_cuda and _ATTN_CUDA_AVAILABLE:
            out, _, col_sum = _attn_cuda.flash_attn_bias_colsum(
                q, k, v, bias=None, return_colsum=True, subsample_ratio=subsample)
        else:
            out, _, col_sum = fa_forward_colsum_fast_sub(
                q, k, v, write_o=True, subsample_ratio=subsample)
        scores = col_sum.unsqueeze(2)  # [B, H, 1, T]

        self._last_query_offset[slot_idx] = Tq
        self._update_scores(scores, slot_idx)

        return out


    def _update_scores(self, scores: Optional[torch.Tensor]=None, slot_idx: int=0):
        """
        Efficient GPU-optimized version (no view/copy syncs)
        scores: [B, H, 1, T_live] —— already summed over query dim
        """
        if scores is None:
            return
        assert 0 <= slot_idx < self._L_eff, "Invalid slot_idx in _update_scores()."
        # Expriment Scores
        s = scores.sum(dim=(0, 2)).to(torch.float32)  # [H, T_live]
        H, T_live = s.shape
        assert H == self.num_heads, "Head mismatch in _update_scores"

        #~ Expriment Scores
        # mean over query dim and scale to buffer size
        s = s * T_live /self._last_query_offset[slot_idx]  

        # update hot scores
        old_T = self._last_score_offset[slot_idx]
        alloc_T = self._offset_hot[slot_idx]
        assert T_live <= alloc_T, f"T_live({T_live}) > allocated({alloc_T})"
        assert T_live <= self.reserved_buffer_token_size, f"T_live({T_live}) > reserved_buffer_token_size({self.reserved_buffer_token_size})"

        scores_hot = self._scores_hot[slot_idx]  # [H, buffer_T]
        # ---- 1. new tail (persist mask + direct write) ----
        new_T = T_live - old_T
        s_scale = self.temperature
        scores_count = self._scores_hot_count[slot_idx]  # [H, buffer_T]

        if new_T > 0:  # self-attention produced scores for new tokens
            start = old_T
            end = T_live
            scores_hot[:, start:end].copy_(s[:, start:end])
        # ---- 2. old prefix accumulation (in-place fused op) ----
        if old_T > 0:
            scores_hot[:, :old_T].mul_(s_scale).add_(s[:, :old_T])
        scores_count[:, :T_live].add_(1.0)

        # ---- 3. update boundary ----
        self._last_score_offset[slot_idx] = T_live

    # def prune_kv(self):
    #     """
    #     Perform pruning on all registered layers.
    #     """
    #     for slot_idx in range(self._L_eff):
    #         self._prune_kv(slot_idx)

    def prune_kv(self):
        """
        Public prune API.

        Try a fully parallel prune/compact across all slots when the following
        invariants hold:
            1) All slots share the same _offset_hot (T_live).
            2) Base/recent/prefix partition is shared across slots.
        Otherwise, fall back to the parent implementation (per-slot loop).
        """
        # Try fast parallel path; if it returns False, use slow loop mode.
        try:
            self._parallel_prune_kv()
        except RuntimeError as e:
            logger.warning("Parallel prune failed: %s. Falling back to per-slot prune.", e)
            raise RuntimeError("Falling back to per-slot prune is not supported anymore.")
            # for slot_idx in range(self._L_eff):
            #     self._prune_kv(slot_idx)

    @torch.no_grad()
    def _parallel_prune_kv(self):
        """
        Parallel pruning over all slots.

        Returns:
            True  - parallel path handled pruning or decided no-op.
            False - conditions not satisfied; caller should fall back.
        """
        L = self._L_eff
        H = self.num_heads
        if L == 0:
            return

        device = self.device

        # ---- 1. Check offset invariants ----
        offsets = np.array(self._offset_hot[:L])  # [L]
        # If any slot is empty, or offsets not all equal, we don't do parallel pruning.
        if (offsets <= 0).any():
            raise RuntimeError("Parallel prune failed: some slots are empty.")

        T_tokens = int(offsets[0])
        if not np.all(offsets == T_tokens):
            # Parallel mode requires identical live lengths.
            raise RuntimeError(f"Parallel prune failed: slots have different live lengths: {offsets}.")

        # ---- 2. Pruning threshold check (same as per-slot logic) ----
        target_budget = self.hot_token_size  # hh + recent + pinned
        if T_tokens <= max(target_budget, self.recent_token_size):
            # Nothing to prune; but conditions are valid, so we handled it.
            return

        # ---- 3. Build shared base & prefix (must be identical across slots) ----
        # Recent tokens
        recent_keep = min(self.recent_token_size, T_tokens)
        base_recent = torch.arange(
            T_tokens - recent_keep,
            T_tokens,
            device=device,
            dtype=torch.long,
        )  # [R]

        # Pinned tokens (global)
        pinned_tok = self._pinned_token_index.to(device=device, dtype=torch.long)
        if pinned_tok.numel() > 0:
            pinned_tok = pinned_tok[(pinned_tok >= 0) & (pinned_tok < T_tokens)]
            if pinned_tok.numel() > 0:
                base = torch.unique(torch.cat([pinned_tok, base_recent], dim=0)).sort()[0]
            else:
                base = base_recent
        else:
            base = base_recent

        base_size = base.numel()
        if base_size == 0:
            # Degenerate; no base, fall back for safety.
            raise RuntimeError("Parallel prune failed: no base tokens.")

        if base_size == T_tokens:
            # All tokens are in base; no pruning, but invariants hold.
            # Still ensure offsets are consistent.
            return

        # Prefix: candidates for heavy-hitter selection (shared across all slots)
        prefix_mask = torch.ones(T_tokens, dtype=torch.bool, device=device)
        prefix_mask[base] = False
        prefix_idx = prefix_mask.nonzero(as_tuple=False).squeeze(1)  # [N_prefix]

        if prefix_idx.numel() == 0:
            # No prefix left; just compact base to the front for all slots.
            keep = base.view(1, 1, base_size).expand(L, self.num_heads, base_size)
            return self._apply_keep_and_compact_parallel(keep)

        # ---- 4. Decide heavy-hitter budget ----
        desired_hh_k = min(self.hh_token_size, prefix_idx.numel())
        Tprime = base_size + desired_hh_k

        # ---- 5. Build keep indices [L, H, T'] ----
        if desired_hh_k == 0:
            # Only base tokens, shared across all layers/heads.
            keep_idx_lh = base.view(1, 1, base_size).expand(L, self.num_heads, base_size)
        elif desired_hh_k == prefix_idx.numel():
            # All prefix tokens are kept as HH; no top-k needed.
            hh_idx = prefix_idx.view(1, 1, -1).expand(L, self.num_heads, -1)  # [L, H, k]
            base_exp = base.view(1, 1, base_size).expand(L, self.num_heads, base_size)
            merged = torch.cat([base_exp, hh_idx], dim=-1)                   # [L, H, T']
            keep_idx_lh, _ = torch.sort(merged, dim=-1)
        else:
            # Per-(layer, head) top-k on shared prefix region (fully vectorized).
            scores_all = self._scores_hot[:L, :, :T_tokens]                  # [L, H, T]
            scores_prefix = scores_all.index_select(dim=2, index=prefix_idx) # [L, H, N_prefix]
            N_prefix = prefix_idx.numel()
            _, idx_in_prefix = torch.topk(
                scores_prefix,
                k=desired_hh_k,
                dim=-1
            )  # [L, H, desired_hh_k]

            # Map back to original token indices
            hh_idx = prefix_idx[idx_in_prefix]                               # [L, H, desired_hh_k]

            base_exp = base.view(1, 1, base_size).expand(L, self.num_heads, base_size)
            merged = torch.cat([base_exp, hh_idx], dim=-1)                   # [L, H, T']
            keep_idx_lh, _ = torch.sort(merged, dim=-1)

        assert keep_idx_lh.shape == (L, self.num_heads, Tprime)
        # prefix/base are constructed once and shared, satisfying your parallel constraint.

        # ---- 6. Apply compaction in parallel ----
        return self._apply_keep_and_compact_parallel(keep_idx_lh)

    @torch.no_grad()
    def _apply_keep_and_compact_parallel(self, keep_idx_lh: torch.Tensor) -> bool:
        """
        Parallel compact for all slots.

        keep_idx_lh: LongTensor [L_eff, H, T']
            Per-layer, per-head kept token indices into the current [0, T_live) range.

        This assumes:
            - All slots share the same T_live = _offset_hot[0].
            - base/prefix partition is shared (constructed once).
        """
        L = self._L_eff
        if L == 0:
            return

        keep_idx_lh = keep_idx_lh.to(device=self.device, dtype=torch.long)

        L_k, H, Tprime = keep_idx_lh.shape
        assert L_k == L, f"L mismatch in parallel compact: {L_k} vs {L}"
        assert H == self.num_heads, "Head mismatch in parallel compact"

        T_live = int(self._offset_hot[0])
        #^ debug
        if not np.all(np.array(self._offset_hot[:L]) == T_live):
            # Invariants broken; let caller fall back.
            raise RuntimeError(f"Parallel compact failed: slots have different live lengths: {self._offset_hot[:L]}.")

        if Tprime == 0:
            # Everything pruned: clear structures.
            self.key_cache_hot[:L].zero_()
            self.value_cache_hot[:L].zero_()
            self._token_indices_hot[:L].fill_(-1)
            self._scores_hot[:L].zero_()
            self._scores_hot_count[:L].zero_()
            self._offset_hot = [0] * self._L_eff
            self._last_score_offset[:L].zero_()
            return

        D = self.head_dim

        # Expand indices for K/V gather over token dimension (dim=2)
        idx_exp = keep_idx_lh.unsqueeze(-1).expand(-1, -1, -1, D)  # [L, H, T', D]

        # Restrict to live region [0, T_live)
        K = self.key_cache_hot[:L, :, :T_live, :]           # [L, H, T_live, D]
        V = self.value_cache_hot[:L, :, :T_live, :]
        TI = self._token_indices_hot[:L, :, :T_live]        # [L, H, T_live]
        S = self._scores_hot[:L, :, :T_live]                # [L, H, T_live]
        SC = self._scores_hot_count[:L, :, :T_live]         # [L, H, T_live]

        # ---- gather in parallel ----
        K_sel = torch.gather(K, 2, idx_exp)                 # [L, H, T', D]
        V_sel = torch.gather(V, 2, idx_exp)
        TI_sel = torch.gather(TI, 2, keep_idx_lh)           # [L, H, T']
        S_sel = torch.gather(S, 2, keep_idx_lh)             # [L, H, T']
        SC_sel = torch.gather(SC, 2, keep_idx_lh)           # [L, H, T']

        # ---- write back compacted prefix ----
        self.key_cache_hot[:L, :, :Tprime, :].copy_(K_sel)
        self.value_cache_hot[:L, :, :Tprime, :].copy_(V_sel)
        self._token_indices_hot[:L, :, :Tprime].copy_(TI_sel)
        self._scores_hot[:L, :, :Tprime].copy_(S_sel)
        self._scores_hot_count[:L, :, :Tprime].copy_(SC_sel)

        # ---- clear tail region [T', buffer) ----
        if Tprime < self.reserved_buffer_token_size:
            self._token_indices_hot[:L, :, Tprime:].fill_(-1)
            self._scores_hot[:L, :, Tprime:].zero_()
            self._scores_hot_count[:L, :, Tprime:].fill_(0.0)

        # ---- update metadata ----
        new_T = Tprime
        for slot in range(L):            
            self._offset_hot[slot] = new_T
            self._last_score_offset[slot] = min(
                self._last_score_offset[slot],
                new_T
            )

        return True

 
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

        scores_now = self._estimate_scores(slot_idx, T_tokens)  # [H, T_tokens]

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
        prefix_mask = torch.ones(T_tokens, dtype=torch.bool, device=device)
        prefix_mask[base] = False
        prefix_idx = prefix_mask.nonzero(as_tuple=False).squeeze(1)  # [N_prefix]
        if prefix_idx.numel() == 0:
            # No available prefix tokens for heavy-hitter selection
            # Compact only the base tokens to the front
            keep = base.unsqueeze(0).expand(H, -1)  # [H, base_size]
            self._apply_keep_and_compact(slot_idx, keep)
            return

        # 4) Determine the number of HH tokens to keep
        desired_hh_k = min(self.hh_token_size, prefix_idx.numel())

        # Total number of tokens to keep
        Tprime = base_size + desired_hh_k
        if desired_hh_k == 0:
            # No heavy-hitter tokens; keep only base tokens
            all_keep_tokens = base.unsqueeze(0).expand(H, -1)  # [H, T']
        elif desired_hh_k == prefix_idx.numel():
            # All prefix tokens are selected as heavy-hitters
            # --- no need to sort/select ---
            hh_idx = prefix_idx.unsqueeze(0).expand(H, -1)        # [H, k]
            # Broadcast base to all heads
            base_expanded = base.unsqueeze(0).expand(H, -1)       # [H, base_size]
            # 6) Merge base and HH, sort, and truncate
            merged = torch.cat([base_expanded, hh_idx], dim=1)    # [H, Tprime]
            merged_sorted, _ = torch.sort(merged, dim=1)
            all_keep_tokens = merged_sorted                       # [H, Tprime]
        else:
            # 5) Select top-k tokens in prefix for each head
            # --- vectorized prune segment ---
            scores_prefix = scores_now[:, prefix_idx]                     # [H, N_prefix]
            _, idx_in_prefix = torch.topk(scores_prefix, k=desired_hh_k, dim=-1)
            hh_idx = prefix_idx[idx_in_prefix]                            # [H, desired_hh_k]

            # Broadcast base to all heads
            base_expanded = base.unsqueeze(0).expand(H, -1)               # [H, base_size]

            # 6) Merge base and HH, sort, and truncate
            merged = torch.cat([base_expanded, hh_idx], dim=1)            # [H, Tprime]
            merged_sorted, _ = torch.sort(merged, dim=1)
            all_keep_tokens = merged_sorted 

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
        S = self._scores_hot[slot_idx]             # [H, T]
        SC = self._scores_hot_count[slot_idx]      # [H, T]


        D = K.shape[-1]
        # ---- expand index for batched gather ----
        idx_exp = keep_idx_h.unsqueeze(-1).expand(-1, -1, D)  # [H, T', D]

        # ---- gather in parallel ----
        K_sel  = torch.gather(K, 1, idx_exp)  # [H, T', D]
        V_sel  = torch.gather(V, 1, idx_exp)
        TI_sel = torch.gather(TI, 1, keep_idx_h)   # [H, T']
        S_sel  = torch.gather(S, 1, keep_idx_h)    # [H, T']
        SC_sel = torch.gather(SC, 1, keep_idx_h).contiguous()


        # ---- copy back (in place) ----
        K[:, :Tprime, :].copy_(K_sel)
        V[:, :Tprime, :].copy_(V_sel)
        TI[:, :Tprime].copy_(TI_sel)
        S[:, :Tprime].copy_(S_sel)
        SC[:, :Tprime].copy_(SC_sel)
        # ---- clear out the tail ----
        TI[:, Tprime:].fill_(-1)
        S[:, Tprime:].zero_() #!!!!!!
        SC[:, Tprime:].fill_(0.0)

        self._offset_hot[slot_idx] = Tprime
        self._last_score_offset[slot_idx] = min(self._last_score_offset[slot_idx], Tprime)

    # --- Public API ---

    def get_scores(self) -> torch.Tensor:
        """
        Get the current scores for each layer.
        Returns a tensor of shape [L_eff, H, buffer_T]
        """
        return self._scores_hot
