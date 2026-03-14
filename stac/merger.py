# Copyright (c) 2025 STAC Authors. All rights reserved.

from typing import Optional, Tuple, Union, Dict, Literal
import math
import os
import logging
from collections.abc import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
import enum
from .allocator import (
    BufState,
    BufferInterface, 
    create_buffer,
)

logger = logging.getLogger(__name__)

_MEM_PROFILE = os.environ.get("MERGER_MEM_PROFILE", "0") == "1"

def _gpu_mem_mb():
    """Return (allocated_MB, reserved_MB) for current CUDA device."""
    if not torch.cuda.is_available():
        return 0.0, 0.0
    a = torch.cuda.memory_allocated() / (1024 * 1024)
    r = torch.cuda.memory_reserved() / (1024 * 1024)
    return a, r

def _tensor_mb(t: torch.Tensor) -> float:
    """Return memory footprint of a tensor in MB."""
    return t.nelement() * t.element_size() / (1024 * 1024)

def _mem_checkpoint(label: str, extra: str = ""):
    """Log a memory checkpoint if MERGER_MEM_PROFILE=1."""
    if not _MEM_PROFILE:
        return
    a, r = _gpu_mem_mb()
    msg = f"  [MEM] {label:40s} | alloc={a:10.1f} MB | reserved={r:10.1f} MB"
    if extra:
        msg += f" | {extra}"
    logger.info(msg)

def default_weight_fn(scores: torch.Tensor, sim: torch.Tensor) -> torch.Tensor:
    if scores.shape != sim.shape:
        raise ValueError("scores and sim must have the same shape.")
    return torch.exp(sim)

#! ===================== Main Class: VoxelKVStoreOnlineSlabPool =====================
class VoxelKVMerger:
    """
    insert_and_merge mode:
        element: the operations are performed per-(head, voxel) pair in serial loop
        parrallel: the operations are performed per-head in parallel batch 
    -------------
    Append policy:
        When buffer not full:
        - append to buffer until full.
    Merge policy:
        When buffer full:
        - evaluate similarity between buffer tokens and merge into pivots by weight-FPS clustering.
        - after merging, buffer will be cleaned with count -1, never append token anymore.
    Remerge policy:
        When buffer count set to -1 (no more appends), trigger remerge:
        - For each buffer token, compute similarity to all pivots.
        - If similarity >= sim_thresh, merge into pivots.
        - If similarity < replace_thresh, and sum(scores) >= score_threshold or buffer full,
            find a slot to replace, and insert as new pivot.
            if slot occupied, first merge old pivot into its most similar pivot.
        - Else, ignore the token.
    -------------
    retrieve:
      - return pivots (Optional: with buffers)
    """

    def __init__(self, num_heads: int, head_dim: int,
                 init_voxels: int = 0,
                 pivot_cap: int=4,
                 budget_cap: int = 32,              # per-(h,v) buffer capacity
                 sim_thresh: float = 0.75,  #        trigger merge when >= similarity
                 replace_thresh: float = 0.5,         
                 score_thresh: float = 0.2,
                 dtype: torch.dtype=torch.float16, 
                 device: torch.device=torch.device("cuda"),
                 weight_fn: Optional[Callable] = None,
                 backend: str = "python",
                 **kwargs
                 ):        # or when sum(scores) >= threshold
        self.H = int(num_heads)
        self.D = int(head_dim)
        self.P = int(pivot_cap)
        self.B = int(budget_cap)
        self.dtype = dtype
        self.device = device
        self.debug = kwargs.get("debug", False)
        self.use_seed = kwargs.get("use_seed", False)
        self.backend = backend
        init_voxels = max(1, int(init_voxels))
        self._voxel_alloc = init_voxels
        self._voxel_offset = 0
        self._token_count = 0

        # ---- PIVOT POOL (merged tokens) ----
        self.piv_cap = int(pivot_cap)
        piv_fields_specs = {"K": "vector", "V": "vector", "W": "scalar", "S": "scalar", "C": "scalar"}
        seed_fields_specs = {"K_seed": "vector", "S_seed": "scalar"} if self.use_seed else {}
        # Use a single allocator type for both pivots and buffer
        self._alloc_type = kwargs.get("allocator", "slab")  # also drives CUDA seg_mode

        self_piv_kwargs = dict(
                buf_cap=pivot_cap,         # == self.P
                head_dim=head_dim,         # == self.D
                num_heads=num_heads,
                capacity=init_voxels,
                growth=kwargs.get("slab_growth", 64),
                growth_ratio=kwargs.get("slab_growth_ratio", 0.1),
                free_policy=kwargs.get("slab_free_policy", "immediate"),
                micro_slab_size=kwargs.get("seg_size", 4),
                field_specs={**piv_fields_specs, **seed_fields_specs},
                device=device,
                dtype=dtype,
                debug=self.debug,
            )

        self.pivots: BufferInterface = create_buffer(alloc_type=self._alloc_type, **self_piv_kwargs)
        self.pivots.resize_rows(self.H, self._voxel_alloc)
        # ---- BUFFER (unmerged tokens) ----
        self.buf_cap = int(budget_cap)
        self._buf_kwargs = dict(
                                buf_cap=self.buf_cap,
                                head_dim=head_dim,
                                num_heads=num_heads,
                                capacity=kwargs.get("slab_cap", 1024),
                                growth=kwargs.get("slab_growth", 1024),
                                growth_ratio=kwargs.get("slab_growth_ratio", 0.25),
                                free_policy=kwargs.get("slab_free_policy", "immediate"),
                                micro_slab_size=kwargs.get("seg_size", 4),
                                device=device,
                                dtype=dtype,
                                debug=kwargs.get("alloc_debug", False),
                            )
        self.buffer: BufferInterface = create_buffer(alloc_type=self._alloc_type, **self._buf_kwargs)
        self.buffer.resize_rows(self.H, self._voxel_alloc)
        # ---- Merge control ----
        self.replace_threshold = float(replace_thresh)
        self.sim_threshold = float(sim_thresh)
        self.score_threshold = float(score_thresh)

        if weight_fn is None:
            self.weight_fn = default_weight_fn
        else:
            self.weight_fn = weight_fn

        self.K_over = torch.empty((0, self.D), device=self.device, dtype=self.dtype)
        self.V_over = torch.empty((0, self.D), device=self.device, dtype=self.dtype)
        self.S_over = torch.empty((0,), device=self.device, dtype=torch.float32)
        self.rows_over = torch.empty((0,), device=self.device, dtype=torch.long)
        self._overflow_alloc = self._voxel_alloc
        
        # CUDA merger wrapper (lazily initialized)
        self._stac_merger = None
        self._pending_voxel_zones = None
        self.backend = backend
        if self.backend == "cuda":
            self._init_cuda_merger()

    def _init_cuda_merger(self):
        """Lazily initialize the CUDA MergerWrapper for insert_and_merge_with_rows."""
        if self._stac_merger is not None:
            return
        if _MEM_PROFILE:
            py_buf_mb = sum(_tensor_mb(p) for p in self.buffer.parameters() if hasattr(p, 'nelement')) if hasattr(self.buffer, 'parameters') else 0
            py_piv_mb = sum(_tensor_mb(p) for p in self.pivots.parameters() if hasattr(p, 'nelement')) if hasattr(self.pivots, 'parameters') else 0
        _mem_checkpoint("init_cuda_merger:before",
                        f"V_alloc={self._voxel_alloc}, H={self.H}, P={self.P}, B={self.B}, D={self.D}")
        try:
            from merger_cuda import create_merger_wrapper, has_merger_wrapper
            if not has_merger_wrapper():
                raise RuntimeError("MergerWrapper not available in merger_cuda")
            use_seg = getattr(self, "_alloc_type", "slab") == "segment"
            self._stac_merger = create_merger_wrapper(
                num_heads=self.H,
                head_dim=self.D,
                pivot_cap=self.P,
                budget_cap=self.B,
                init_voxels=self._voxel_alloc,
                dtype=self.dtype,
                device=self.device,
                seg_mode=use_seg,
            )
            _mem_checkpoint("init_cuda_merger:after",
                            f"pool_cap={self.H * self._voxel_alloc}, seg_mode={use_seg}")
            if self.debug:
                logger.debug("[VoxelKVMerger] Initialized CUDA MergerWrapper (seg_mode=%s)", use_seg)
        except Exception as e:
            if self.debug:
                logger.debug("[VoxelKVMerger] Failed to initialize CUDA MergerWrapper: %s", e)
            self._stac_merger = None
            raise

    def reset(self):
        """No-op for compatibility with KV manager lifecycle (state is recreated on ensure_capacity)."""
        pass

    def _flush_cuda_diagnostics(self) -> None:
        """Fetch [SEG-EXPAND]/[SEG-WARN] from CUDA merger and log so they appear above Rich Live."""
        if self._stac_merger is None:
            return
        take = getattr(self._stac_merger, "take_diagnostics", None)
        if take is None:
            return
        for msg in take():
            logger.info("%s", msg)

    def update_voxel_zones(self, voxel_zones: torch.Tensor):
        self._pending_voxel_zones = voxel_zones

    def ensure_capacity(self, V_needed: int):
        self._voxel_offset = max(V_needed, self._voxel_offset)
        if V_needed <= self._voxel_alloc:
            return
        V_old = self._voxel_alloc
        V_new = max(V_needed, int(self._voxel_alloc * 1.5))
        while V_new < V_needed:
            V_new = int(V_new * 1.5)

        _mem_checkpoint("ensure_capacity:grow",
                        f"V_alloc {V_old}->{V_new}, S_tot={self.H * V_new}")

        self._voxel_alloc = V_new

        # Skip Python pool resize when CUDA merger handles all storage
        if self._stac_merger is not None:
            return

        # grow pivot pool
        self.pivots.resize_rows(self.H, self._voxel_alloc)
        # grow token buffer pool
        self.buffer.resize_rows(self.H, self._voxel_alloc)

    def _remap_overflow_rows(self):
        """Remap overflow row indices when _voxel_alloc has grown since they were generated.
        Row encoding: row = head * alloc + voxel. When alloc changes, rows must be re-encoded."""
        if self.rows_over.numel() == 0 or self._overflow_alloc == self._voxel_alloc:
            return
        old_alloc = self._overflow_alloc
        new_alloc = self._voxel_alloc
        h = self.rows_over // old_alloc
        v = self.rows_over % old_alloc
        self.rows_over = h * new_alloc + v
        self._overflow_alloc = new_alloc

    # ===================== Helpers =====================
    @staticmethod
    def _segment_starts(sorted_seg: torch.Tensor) -> torch.Tensor:
        """sorted_seg: [E] int32/64; return start indices for each unique segment.
        (static so it can be called via class or instance)
        """
        E = sorted_seg.numel()
        if E == 0:
            return torch.empty(0, dtype=torch.long, device=sorted_seg.device)
        change = torch.ones(E, dtype=torch.bool, device=sorted_seg.device)
        change[1:] = sorted_seg[1:] != sorted_seg[:-1]
        return torch.nonzero(change, as_tuple=False).squeeze(1)

    def _row_index(self, h: Union[int,torch.Tensor], v: Union[int, torch.Tensor]) -> torch.Tensor:
        """Flattened row index: row = h * self._voxel_alloc + v"""
        # return int(h) * int(self._voxel_alloc) + int(v)
        h = torch.as_tensor(h, device=self.device, dtype=torch.long)
        v = torch.as_tensor(v, device=self.device, dtype=torch.long)
        assert h.shape == v.shape, "h and v must have the same shape"
        return h * int(self._voxel_alloc) + v # torch.long
    
    def _stable_row_index(self, rows: Union[torch.Tensor, int, Tuple[int, int]]) -> torch.Tensor:
        if isinstance(rows, torch.Tensor):
            rows = rows.to(torch.long)
        elif isinstance(rows, int):
            rows = torch.tensor([rows], device=self.device, dtype=torch.long)
        elif isinstance(rows, tuple) and len(rows) == 2:
            rows = torch.tensor([self._row_index(rows[0], rows[1])], device=self.device, dtype=torch.long)
        else:
            raise TypeError("rows must be Tensor / int / (h,v) tuple")
        return rows    
        
    def _is_full_buffer(self, rows: Union[torch.Tensor, int, Tuple[int, int]]) -> torch.Tensor:
        """Check if specified rows have FULL buffer"""
        states = self.buffer.get_state(self._stable_row_index(rows))
        return states == int(BufState.FULL)
    
    def _is_avail_buffer(self, rows: Union[torch.Tensor, int, Tuple[int, int]]) -> torch.Tensor:
        """Check if specified rows have AVAILABLE or RESERVED buffer"""
        states = self.buffer.get_state(self._stable_row_index(rows))
        return (states == int(BufState.AVAILABLE)) | (states == int(BufState.RESERVED))
    
    def _is_retired_buffer(self, rows: Union[torch.Tensor, int, Tuple[int, int]]) -> torch.Tensor:
        """Check if specified rows have RETIRED (HELD or FREE) buffer"""
        states = self.buffer.get_state(self._stable_row_index(rows))
        return ((states == int(BufState.FREE)) | (states == int(BufState.HELD)))


    # ===================== Buffer append =====================
    @torch.no_grad()
    def _append_to_buffer_element(self, h, v, K_new, V_new, S_new):
        rows = self._row_index(h, v)
        return self.buffer.append_element(rows, K_new, V_new, S_new)

    @torch.no_grad()
    def _append_to_buffer_parallel(self, rows_new, K_new, V_new, S_new):
        return self.buffer.append_batch(rows_new, K_new, V_new, S_new)

    # ===================== Merge trigger =====================  
    def _cluster_merge_to_one_flat(self, rows, keys, values, scores, counts=None):
        """
        One element Weighted-FPS clustering merge to single pivot for flattened rows.

        Args:
            rows:   [E] int64, row ids (assumed consecutive/grouped; i.e., same ids are contiguous)
            keys:   [E, D] float / half
            values: [E, D] float / half
            scores: [E]    float
            counts: [E]    float (optional)

        Returns:
            "row_unq": [R] int64,
            {
                "K": [R, D] self.dtype,
                "V": [R, D] self.dtype,
                "W": [R]    float32,   # sum of (coeff * counts) per row
                "S": [R]    float32,   # sum of scores per row
                "C": [R]    float32,   # sum of counts per row
            }
        """
        device = keys.device
        assert keys.dim() == 2, "keys must be 2D tensor."
        E, D = keys.shape

        # Cast to float compute dtype (keep outputs in self.dtype later)
        keys   = keys.float()
        values = values.float()
        scores = scores.clamp_min(1e-6).float()
        counts = counts.float() if counts is not None else torch.ones_like(scores, device=device)

        # R unique rows and inverse map idx: element e -> group r
        row_unq, inv = torch.unique_consecutive(rows, return_inverse=True)
        R = row_unq.numel()

        # Normalize keys for cosine similarity (seed selection step)
        Kn = F.normalize(keys, dim=-1, eps=1e-6)  # [E, D]

        # --- Find the per-row seed as the FIRST element achieving the max score ---
        # Per-group counts and starts (start index of each group in [0..E-1])
        grp_counts = torch.bincount(inv, minlength=R)                              # [R]
        grp_starts = torch.cumsum(grp_counts, dim=0) - grp_counts                  # [R]
        pos_in_grp = torch.arange(E, device=device) - grp_starts[inv]              # [E]

        # Max score per group via scatter-reduce (no loops)
        max_scores = torch.full((R,), float("-inf"), device=device)
        max_scores.scatter_reduce_(0, inv, scores, reduce='amax')

        # Build a mask of elements hitting the group max score
        is_max = scores == max_scores[inv]                                         # [E] bool

        # Among those, pick the FIRST (smallest pos_in_grp): amin over masked positions
        big = torch.full((E,), E, device=device, dtype=pos_in_grp.dtype)
        pos_masked = torch.where(is_max, pos_in_grp, big)                          # [E]
        first_pos = torch.full((R,), E, device=device, dtype=pos_in_grp.dtype)
        first_pos.scatter_reduce_(0, inv, pos_masked, reduce='amin')               # [R]
        is_seed = pos_in_grp == first_pos[inv]                                     # [E] bool

        # Seed vectors and seed scores per group
        # (shape [R, D] / [R]; guaranteed one True per group given consecutive grouping)
        Kn_seed = Kn[is_seed]                                                      # [R, D]
        S_seed  = scores[is_seed]                                                  # [R]

        # Broadcast each element's group's seed to compute similarity
        Kn_seed_g = Kn_seed[inv]                                                   # [E, D]
        sim_to_seed = (Kn * Kn_seed_g).sum(dim=-1).clamp(-1.0, 1.0)                # [E]

        # Weighting coefficients (same weight function as the batched version)
        coeff = self.weight_fn(scores, sim_to_seed)                                # [E]

        # Sum of weighted counts per group: W_piv
        weighted = coeff * counts                                                  # [E]
        W_piv = torch.zeros(R, device=device).scatter_add(0, inv, weighted)        # [R] float32

        # Normalized coefficients
        coeff_norm = coeff / (W_piv[inv] + 1e-8)                                   # [E]

        # Aggregate K,V with index_add along group dimension
        K_piv = torch.zeros(R, D, device=device).index_add_(0, inv, coeff_norm.unsqueeze(-1) * keys)
        V_piv = torch.zeros(R, D, device=device).index_add_(0, inv, coeff_norm.unsqueeze(-1) * values)

        # Other per-row totals (S: sum of scores; C: sum of counts)
        S_piv = torch.zeros(R, device=device).scatter_add(0, inv, scores)          # [R]
        C_piv = torch.zeros(R, device=device).scatter_add(0, inv, counts)          # [R]
        
        pack = {
            "K": K_piv.to(self.dtype),          # [R, D]
            "V": V_piv.to(self.dtype),          # [R, D]
            "W": W_piv,                         # [R] float32
            "S": S_piv,                         # [R] float32
            "C": C_piv,                         # [R] float32
            "K_seed": Kn_seed.to(self.dtype),  # [R, D]
            "S_seed": S_seed,                  # [R]
        }
        return pack,row_unq
    
    def _cluster_merge_to_one_budget(self, keys, values, scores, counts=None):
        """
        Weighted-FPS clustering merge to single pivot.
        --------------
        Args:
            keys: [G, B, D], 
            values: [G, B, D],
            scores: [G, B],
            counts: [G, B], optional
        --------------
        Returns:
            K_piv, V_piv: [G, D] self.dtype
            W_piv, S_piv: [G], float32
            C_piv: [G],       float32,
            K_seed: [G, D] self.dtype
            S_seed: [G] float32
        """
        device = keys.device
        G, B, D = keys.shape

        keys = keys.float(); values = values.float()
        scores = scores.clamp_min(1e-6).float()
        counts = counts.float() if counts is not None else torch.ones_like(scores)

        Kn = F.normalize(keys, dim=-1, eps=1e-6)    # [G,B,D]
        first = scores.argmax(dim=1)

        Kn_seed = Kn.gather(1, first.view(G,1,1).expand(-1,1,D))                       # [G,1,D]
        S_seed= scores.gather(1, first.view(G,1)).squeeze(1)                             # [G]
        sim_to_seed = torch.einsum('gbd,gbd->gb', Kn, Kn_seed).clamp(-1.0, 1.0)         # [G,B]

        coeff = self.weight_fn(scores, sim_to_seed)       # [G,B]
        
        sum_coeff = (coeff*counts).sum(dim=1, keepdim=True)  # [G,1]
        coeff_norm = coeff / (sum_coeff + 1e-8)          # [G,B]

        K_piv = torch.sum(coeff_norm.unsqueeze(-1) * keys, dim=1)       # [G,D]
        V_piv = torch.sum(coeff_norm.unsqueeze(-1) * values, dim=1)       # [G,D]

        W_piv = sum_coeff.squeeze(1)                           # [G]  float32
        S_piv = scores.sum(dim=1)                        # [G]  float32
        # S_piv = (torch.sum(coeff_norm * scores, dim=1))   # [G] float32
        C_piv = counts.sum(dim=1)                        # [G]  float32

        # return K_piv.to(self.dtype), V_piv.to(self.dtype), W_piv, S_piv, C_piv
        return {
                "K": K_piv.to(self.dtype),
                "V": V_piv.to(self.dtype),
                "W": W_piv,
                "S": S_piv,
                "C": C_piv,
                "K_seed": Kn_seed.squeeze(1).to(self.dtype),
                "S_seed": S_seed,
            }
    def _append_to_pivots(self, rows: torch.Tensor,
                          Pack_new: Dict[str, torch.Tensor]
                        #  K_new: torch.Tensor,
                        #  V_new: torch.Tensor,
                        #  W_new: torch.Tensor,
                        #  S_new: torch.Tensor,
                        #  C_new: torch.Tensor,
                        #  Kseed_new: torch.Tensor,
                        #  Sseed_new: torch.Tensor
                         ) -> None:
        """
        Append new pivots into specified rows (element/parallel mode).
        --------------
        Args:
            rows: Tensor [G] long, target rows to append into.
            K_new: Tensor [G,D]
            V_new: Tensor [G,D]
            W_new: Tensor [G]
            S_new: Tensor [G]
            C_new: Tensor [G]
            Kseed_new: Tensor [G,D]
            Sseed_new: Tensor [G]
        --------------
        """
        # 4) Read existing pivots (at most P per row)
        oldPack = self.pivots.read_rows_dict(rows)
        Kp = oldPack["K"]       # [G,P,D]
        Vp = oldPack["V"]         # [G,P,D]
        Wp = oldPack["W"]         # [G,P]
        Sp = oldPack["S"]         # [G,P]
        Cp = oldPack["C"]         # [G,P]
        Mp = oldPack["M"]         # [G,P] bool
        if self.use_seed:
            Kpseed = oldPack["K_seed"]  # [G,P,D] normalized seed keys
            Spseed = oldPack["S_seed"]  # [G,P]
        else:
            Kpseed = F.normalize(Kp, dim=-1, eps=1e-6)
            Spseed = Sp.clone()

        device = Kp.device
        P = Kp.size(1)
        G = rows.size(0)
        arange_g = torch.arange(G, device=device)

        # 5) Count occupied pivots per row and detect full rows
        offset_pivots = Mp.sum(dim=1)                  # [G]
        full_mask = (offset_pivots == P)               # [G] bool

        # 6) When row is full, pick victim: slot with smallest contrib (W here)
        # contrib = 0.7 * torch.log(Sp.clamp_min(1e-6)) + 0.3 * torch.log(Cp.clamp_min(1e-6))  # [G,P]
        contrib = Wp
        j_victim_all = torch.argmin(contrib, dim=1)    # [G]
        idx_full = torch.nonzero(full_mask, as_tuple=False).squeeze(1)  # [Gf]
        Gf = int(idx_full.numel())

        if Gf > 0:
            # 7) For each full row, find nearest neighbour of victim by seed cosine
            Kpn_full = Kpseed.index_select(0, idx_full)              # [Gf,P,D]
            sim_full = torch.matmul(Kpn_full, Kpn_full.transpose(1, 2))  # [Gf,P,P]
            sim_full.diagonal(dim1=1, dim2=2).fill_(-float("inf"))

            j_victim_full = j_victim_all.index_select(0, idx_full)             # [Gf]
            victim_rows = sim_full[torch.arange(Gf, device=device), j_victim_full, :]  # [Gf,P]
            j_nei_full = victim_rows.argmax(dim=1)                             # [Gf]
            sim_j = victim_rows.gather(1, j_nei_full.unsqueeze(1)).squeeze(1)  # [Gf]
            scale_ij = torch.exp(sim_j - 1.0)
            W_jp = Wp[idx_full, j_victim_full] * scale_ij  # [Gf]
            # 8) Merge victim into nearest neighbour by weight, free one slot
            Wj = W_jp
            Wk = Wp[idx_full, j_nei_full]             # [Gf]
            denom = (Wj + Wk).clamp_min(1e-8)         # [Gf]
            frac_j = (Wj / denom).unsqueeze(1)        # [Gf,1]
            frac_k = (Wk / denom).unsqueeze(1)        # [Gf,1]

            Kj = Kp[idx_full, j_victim_full, :]       # [Gf,D]
            Kk = Kp[idx_full, j_nei_full,  :]         # [Gf,D]
            Vk = Vp[idx_full, j_nei_full,  :]         # [Gf,D]
            Vj = Vp[idx_full, j_victim_full, :]       # [Gf,D]

            # Write back into neighbour slot
            Kk_new = frac_j * Kj.float() + frac_k * Kk.float()        # [Gf,D]
            Vk_new = frac_j * Vj.float() + frac_k * Vk.float()        # [Gf,D]
            Wk_new = denom                             # [Gf]
            Sk_new = Sp[idx_full, j_victim_full] + Sp[idx_full, j_nei_full]  # [Gf]
            # Sk_new =
            Ck_new = Cp[idx_full, j_victim_full] + Cp[idx_full, j_nei_full]  # [Gf]

            Kp[idx_full, j_nei_full, :] = Kk_new.to(self.dtype)
            Vp[idx_full, j_nei_full, :] = Vk_new.to(self.dtype)
            Wp[idx_full, j_nei_full]    = Wk_new
            Sp[idx_full, j_nei_full]    = Sk_new
            Cp[idx_full, j_nei_full]    = Ck_new

            # Mark victim slot as invalid (free the slot)
            Mp[idx_full, j_victim_full] = False

        # 9) Compute write slot index per row:
        #    - full row: write to victim slot
        #    - non-full: write at offset (current True count, append at tail)
        write_idx = torch.where(full_mask, j_victim_all, offset_pivots.to(torch.long))  # [G]

        # 10) Write newPack to the corresponding slots in one go (no loop)
        Kp[arange_g, write_idx, :] = Pack_new["K"]
        Vp[arange_g, write_idx, :] = Pack_new["V"]
        Wp[arange_g, write_idx]    = Pack_new["W"]
        Sp[arange_g, write_idx]    = Pack_new["S"]
        Cp[arange_g, write_idx]    = Pack_new["C"]
        Mp[arange_g, write_idx]    = True


        # 11) Write back
        if self.use_seed:
            Kpseed[arange_g, write_idx, :] = Pack_new["K_seed"]
            Spseed[arange_g, write_idx] = Pack_new["S_seed"]
        data = {
            "K": Kp, "V": Vp, "W": Wp, "S": Sp, "C": Cp, "M": Mp,
        }
        if self.use_seed:
            data["K_seed"] = Kpseed.to(self.dtype)
            data["S_seed"] = Spseed
        self.pivots.write_rows_dict(rows, data)


    # ===================== Online Merge: Similarity-based Merge to Pivots =====================
    @torch.no_grad()
    def _one2one_merge_to_pivots(
        self,
        K_tok: torch.Tensor,     # [E', D] token keys
        V_tok: torch.Tensor,     # [E', D] token values
        S_tok: torch.Tensor,     # [E'] token scores
        rows: torch.Tensor,      # [E'] row indices (must have active pivots)
        update_seed: bool = True,
    ) -> torch.Tensor:
        """
        Merge tokens into existing pivots based on similarity threshold.
        
        This method performs online merging of incoming tokens to their corresponding
        pivot clusters. Tokens with similarity >= sim_threshold are merged into the
        nearest pivot using weighted averaging. Tokens with low similarity but high
        scores are marked for buffer insertion.
        
        Args:
            K_tok: [E', D] float, token key vectors
            V_tok: [E', D] float, token value vectors  
            S_tok: [E'] float, token attention scores
            rows: [E'] long, row indices for each token (must have valid pivots)
            update_seed: bool, whether to update seed keys when new tokens have higher scores
            
        Returns:
            low_mask: [E'] bool, True for tokens that should go to buffer
                      (sim < replace_threshold AND score > gate)
        """
        if K_tok.numel() == 0:
            return torch.zeros(0, dtype=torch.bool, device=K_tok.device)
        
        device = K_tok.device
        E = K_tok.size(0)
        
        # Ensure proper dtypes
        rows = rows.to(device=device, dtype=torch.long).view(-1)
        K_tok = K_tok.float()
        V_tok = V_tok.float()
        S_tok = S_tok.clamp_min(1e-6).float()
        
        # Get unique rows and inverse mapping
        uniq_rows, inv_rows = torch.unique(rows, sorted=True, return_inverse=True)  # [G], [E']
        
        # Read pivot data
        pack = self.pivots.read_rows_dict(uniq_rows)
        Kp = pack["K"].float()      # [G, P, D]
        Vp = pack["V"].float()      # [G, P, D]
        Wp = pack["W"].float()      # [G, P]
        Sp = pack["S"].float()      # [G, P]
        Cp = pack["C"].float()      # [G, P]
        Mp = pack["M"].clone()      # [G, P] bool
        
        # Seed keys for similarity: from stored seeds when use_seed, else derive from pivot K/S
        if self.use_seed:
            Kp_seed = pack["K_seed"].float()  # [G, P, D]
            Sp_seed = pack["S_seed"].float()  # [G, P]
        else:
            Kp_seed = F.normalize(Kp, dim=-1, eps=1e-6)
            Sp_seed = Sp.clone()
        
        G, P, D = Kp.shape
        
        # Broadcast seed keys to per-token view
        Ks = Kp_seed.index_select(0, inv_rows)  # [E', P, D]
        Mg = Mp.index_select(0, inv_rows)       # [E', P]
        
        # Compute similarity between tokens and pivot seeds
        Kn = F.normalize(K_tok, dim=-1, eps=1e-6)  # [E', D]
        sim = torch.einsum('ed,epd->ep', Kn, Ks).clamp(-1.0, 1.0)  # [E', P]
        sim = sim.masked_fill(~Mg, float("-inf"))
        
        sim_max, assign = sim.max(dim=-1)  # [E'], [E']
        
        # ---- MERGE: tokens with high similarity ----
        merge_mask = sim_max >= self.sim_threshold
        if merge_mask.any():
            i_merge = merge_mask.nonzero(as_tuple=False).squeeze(1)  # [M]
            g_merge = inv_rows[i_merge]                               # [M] group indices
            j_merge = assign[i_merge].to(torch.long)                  # [M] pivot indices
            
            # Compute merge weights
            alpha = self.weight_fn(S_tok[i_merge], sim_max[i_merge])  # [M]
            
            # Accumulate into (group, pivot) pairs
            key = g_merge * P + j_merge  # [M] flattened index
            K_acc = torch.zeros(G * P, D, device=device)
            V_acc = torch.zeros(G * P, D, device=device)
            A_acc = torch.zeros(G * P, device=device)
            S_acc = torch.zeros(G * P, device=device)
            C_acc = torch.zeros(G * P, device=device)
            
            K_acc.index_add_(0, key, alpha.unsqueeze(1) * K_tok[i_merge])
            V_acc.index_add_(0, key, alpha.unsqueeze(1) * V_tok[i_merge])
            A_acc.index_add_(0, key, alpha)
            S_acc.index_add_(0, key, S_tok[i_merge])
            C_acc.index_add_(0, key, torch.ones_like(alpha))
            
            # Reshape to [G, P, ...]
            K_acc = K_acc.view(G, P, D)
            V_acc = V_acc.view(G, P, D)
            A_acc = A_acc.view(G, P)
            S_acc = S_acc.view(G, P)
            C_acc = C_acc.view(G, P)
            
            # Update pivots with weighted average
            hit = A_acc > 0
            if hit.any():
                w_old = Wp[hit]
                denom = (w_old + A_acc[hit]).clamp_min(1e-8)
                Kp[hit] = (w_old[:, None] / denom[:, None]) * Kp[hit] + (1.0 / denom)[:, None] * K_acc[hit]
                Vp[hit] = (w_old[:, None] / denom[:, None]) * Vp[hit] + (1.0 / denom)[:, None] * V_acc[hit]
                Wp[hit] = denom
                Sp[hit] = Sp[hit] + S_acc[hit]
                Cp[hit] = Cp[hit] + C_acc[hit]
            
            # Build update pack
            pack_piv_update = {
                "K": Kp.to(self.dtype),
                "V": Vp.to(self.dtype),
                "W": Wp,
                "S": Sp,
                "C": Cp,
                "M": Mp,
            }
            
            # Update seeds when new tokens have higher scores (per-pivot); only when use_seed
            if self.use_seed and update_seed:
                ids_merge = merge_mask.nonzero(as_tuple=False).squeeze(1)
                rows_merge = inv_rows[ids_merge]                    # [M] group indices
                pivots_merge = assign[ids_merge].to(torch.long)    # [M] pivot indices
                scores_merge = S_tok.index_select(0, ids_merge)    # [M]

                # Group by (group, pivot) pair to handle P>1
                gp_key = rows_merge * P + pivots_merge             # [M]
                uniq_gp, inv_gp = torch.unique(gp_key, sorted=True, return_inverse=True)
                max_scores = torch.full((uniq_gp.numel(),), float('-inf'), device=device)
                max_scores = max_scores.scatter_reduce(0, inv_gp, scores_merge, reduce='amax', include_self=True)
                is_seed = (scores_merge == max_scores.index_select(0, inv_gp))
                big = torch.full_like(ids_merge, ids_merge.numel())
                ids_masked = torch.where(is_seed, ids_merge, big)
                first_pos = torch.full((uniq_gp.numel(),), ids_merge.numel(), device=device, dtype=torch.long)
                first_pos = first_pos.scatter_reduce(0, inv_gp, ids_masked, reduce='amin', include_self=True)
                seed_pos = first_pos.clamp_max(ids_merge.numel() - 1)
                Kp_seed_update = K_tok.index_select(0, seed_pos)   # [U, D]
                Sp_seed_update = S_tok.index_select(0, seed_pos)   # [U]

                gp_g = uniq_gp // P  # group indices
                gp_p = uniq_gp % P   # pivot indices
                mask = Sp_seed_update > Sp_seed[gp_g, gp_p]       # [U] 1-D comparison
                if mask.any():
                    upd_g = gp_g[mask]
                    upd_p = gp_p[mask]
                    Kp_seed[upd_g, upd_p] = F.normalize(Kp_seed_update[mask], dim=-1, eps=1e-6)
                    Sp_seed[upd_g, upd_p] = Sp_seed_update[mask]
            if self.use_seed:
                pack_piv_update["K_seed"] = Kp_seed.to(self.dtype)
                pack_piv_update["S_seed"] = Sp_seed
            
            # Write updated pivots back
            self.pivots.write_rows_dict(uniq_rows, pack_piv_update)
        
        # ---- Filter: identify low-similarity tokens for buffer ----
        assert self.replace_threshold <= self.sim_threshold
        low_cand = (sim_max < self.replace_threshold)
        
        # Compute per-row average score as threshold gate
        sumS = (Sp * Mp).sum(1)                           # [G]
        sumC = (Cp * Mp).sum(1).clamp_min(1e-6)           # [G]
        S_avg = sumS / sumC                               # [G]
        gate = S_avg.index_select(0, inv_rows) * float(self.score_threshold)  # [E']
        
        # Low mask: low similarity AND score above gate
        low_mask = low_cand & (S_tok > gate)
        
        return low_mask

    # ===================== Re-merge / Replace =====================
    #     
    # ===================== Public API: insert -> buffer + conditional merge =====================
    @torch.no_grad()
    def _insert_and_merge_fusion(self,
                    rows: torch.Tensor,  # [E]
                    keys: torch.Tensor,     # [E,D]
                    values: torch.Tensor,     # [E,D]
                    scores: torch.Tensor,
                    ):    # [E]
        """
        Merge First Strategy: parallel implement to process token by online merge
        -------- Explain ------
        For v in all voxels:
            0) Concate input and past overflow tokens within the same v voxel
            1) if merged buffer is not empty:
                    if sim >= sim_threshold: 
                        merge tokens into merged buffer
                    elif sim < replace_threshold and score > row_avg * score_threshold:
                        overflow1 = append tokens into evited buffer
                    else: discard low-quality tokens
                else:
                    overflow2 = append tokens into evited buffer
            2) if evicted buffer is full:
                a. create a new merge token from evicted buffer
                b. if merged buffer is full:
                    select the lowest weighted merged token to remerge
                    free the slot to insert the new merged token
                c. insert the new merged token into merged buffer

            3) overflow = overflow1 + overflow2
               reserve overflow tokens for next time step
        ----------------------
        Args:
            rows: Tensor, [E] long, row indices for each token
            keys: Tensor, [E,D] float, key vectors for each token
            values: Tensor, [E,D] float, value vectors for each token
            scores: Tensor, [E] float, score for each token              
        """

        if keys.numel() == 0:
            return
        init_time = 0.0
        append_time = 0.0
        merge_one2one_time = 0.0
        merge_all2one_time = 0.0
        remerge_time = 0.0
        if self.debug:
            time_start = torch.cuda.Event(enable_timing=True)
            time_end = torch.cuda.Event(enable_timing=True)
            time_start.record()

        device = keys.device
        self._remap_overflow_rows()
        rows = torch.concat([rows.to(torch.long), self.rows_over], dim=0)
        keys = torch.concat([keys, self.K_over],dim=0).float()
        values = torch.concat([values, self.V_over],dim=0).float()
        scores = torch.concat([scores, self.S_over],dim=0).float()
        
        pivot_states = self.pivots.get_state(rows)
        # reserved mask
        reserved_mask = (pivot_states == int(BufState.RESERVED))
        active_mask = ~reserved_mask
        active_rows = rows[active_mask] # [E']
        low_mask = torch.zeros_like(active_mask)
        if active_rows.numel() > 0:
            if self.debug:
                time_end.record()
                torch.cuda.synchronize()
                init_time += time_start.elapsed_time(time_end)
                time_start.record()

            K_tok_act = keys[active_mask]   # [E']
            V_tok_act = values[active_mask] # [E']
            S_tok_act = scores[active_mask] # [E']
            mapping_act2orig = active_mask.nonzero(as_tuple=False).squeeze(1)

            #& ---- MERGE tokens into exist Pivot (modular call) ----
            low_mask_act = self._one2one_merge_to_pivots(
                K_tok_act, V_tok_act, S_tok_act, active_rows,
                update_seed=True,
            )

            # Map low_mask back to original indices
            low_mask[mapping_act2orig[low_mask_act]] = True

            if self.debug:
                time_end.record()
                torch.cuda.synchronize()
                merge_one2one_time += time_start.elapsed_time(time_end)
                time_start.record()
        else:
            if self.debug:
                time_end.record()
                torch.cuda.synchronize()
                init_time += time_start.elapsed_time(time_end)
                time_start.record()
        sel_mask = reserved_mask | low_mask
        K_tok_sel = keys[sel_mask]
        V_tok_sel = values[sel_mask]
        S_tok_sel = scores[sel_mask]
        row_sel = rows[sel_mask]
        # lifes_sel = lifes[sel_mask]
        
        #& --------- Append to buffer ---------
        K_tok_over, V_tok_over, S_tok_over, rows_tok_over = self._append_to_buffer_parallel(
                                                    row_sel,
                                                    K_tok_sel, V_tok_sel, S_tok_sel,)

        self.K_over = K_tok_over
        self.V_over = V_tok_over
        self.S_over = S_tok_over
        self.rows_over = rows_tok_over
        self._overflow_alloc = self._voxel_alloc

        #--------- collect full buffers ---------
        touched_rows = row_sel.unique()
        rows_buf_full = self.buffer.full_rows(touched_rows)
        # [F,B,D], [F,B,D], [F,B]
        Kb_full, Vb_full, Sb_full, Mb_full = self.buffer.read_rows_fast(rows_buf_full)


        if self.debug:
            time_end.record()
            torch.cuda.synchronize()
            append_time += time_start.elapsed_time(time_end)
            time_start.record() 

        # Merge full buffers to one pivot per row
        pack_piv_new = self._cluster_merge_to_one_budget(
            Kb_full, Vb_full, Sb_full,
        )
        rows_piv_new = rows_buf_full

        if self.debug:
            time_end.record()
            torch.cuda.synchronize()
            merge_all2one_time += time_start.elapsed_time(time_end)
            time_start.record()
        #& --------- Find available slot or remerge
        self._append_to_pivots(rows_piv_new,
                                pack_piv_new)  
        
        self.buffer.clean_rows(rows_buf_full)
        if self.debug:
            time_end.record()
            torch.cuda.synchronize()
            remerge_time += time_start.elapsed_time(time_end)
            time_start.record()
            
        return {
            "init_time": init_time,
            "append_time": append_time,
            "merge_one2one_time": merge_one2one_time,
            "merge_all2one_time": merge_all2one_time,
            "remerge_time": remerge_time,
        }

    @torch.no_grad()
    def _insert_and_merge_fused_cuda(self,
                    rows: torch.Tensor,  # [E]
                    keys: torch.Tensor,     # [E,D]
                    values: torch.Tensor,     # [E,D]
                    scores: torch.Tensor,
                    ):    # [E]
        """
        Merge First Strategy: parallel implement to process token by online merge
        -------- Explain ------
        For v in all voxels:
            0) Concate input and past overflow tokens within the same v voxel
            1) if merged buffer is not empty:
                    if sim >= sim_threshold: 
                        merge tokens into merged buffer
                    elif sim < replace_threshold and score > row_avg * score_threshold:
                        overflow1 = append tokens into evited buffer
                    else: discard low-quality tokens
                else:
                    overflow2 = append tokens into evited buffer
            2) if evicted buffer is full:
                a. create a new merge token from evicted buffer
                b. if merged buffer is full:
                    select the lowest weighted merged token to remerge
                    free the slot to insert the new merged token
                c. insert the new merged token into merged buffer

            3) overflow = overflow1 + overflow2
               reserve overflow tokens for next time step
        ----------------------
        Args:
            rows: Tensor, [E] long, row indices for each token
            keys: Tensor, [E,D] float, key vectors for each token
            values: Tensor, [E,D] float, value vectors for each token
            scores: Tensor, [E] float, score for each token              
        """

        if keys.numel() == 0:
            return
        init_time = 0.0
        append_time = 0.0
        merge_one2one_time = 0.0
        merge_all2one_time = 0.0
        remerge_time = 0.0
        if self.debug:
            time_start = torch.cuda.Event(enable_timing=True)
            time_end = torch.cuda.Event(enable_timing=True)
            time_start.record()

        E_new = keys.shape[0]
        E_overflow_old = self.rows_over.numel()

        _mem_checkpoint("cuda-merge:enter",
                        f"E_new={E_new}, overflow_carry={E_overflow_old}")

        device = keys.device
        self._remap_overflow_rows()

        rows_cuda = torch.concat([rows.to(torch.long), self.rows_over], dim=0)
        keys_cuda = torch.concat([keys.to(self.dtype), self.K_over.to(self.dtype)], dim=0)
        values_cuda = torch.concat([values.to(self.dtype), self.V_over.to(self.dtype)], dim=0)
        scores_cuda = torch.concat([scores.float(), self.S_over.float()], dim=0)

        del keys, values, scores, rows

        E_combined = rows_cuda.shape[0]
        if _MEM_PROFILE:
            concat_kv_mb = _tensor_mb(keys_cuda) + _tensor_mb(values_cuda)
            _mem_checkpoint("cuda-merge:after-concat",
                            f"E_combined={E_combined}, concat_KV={concat_kv_mb:.1f}MB")

        first_init = self._stac_merger is None
        self._init_cuda_merger()

        if first_init:
            _mem_checkpoint("cuda-merge:after-init-pools",
                            f"V_alloc={self._voxel_alloc}, H={self.H}")

        self._stac_merger.ensure_capacity(self._voxel_alloc)
        _mem_checkpoint("cuda-merge:after-ensure-capacity",
                        f"V_alloc={self._voxel_alloc}")
        
        if self.debug:
            time_end.record()
            torch.cuda.synchronize()
            init_time = time_start.elapsed_time(time_end)
            time_start.record()

        K_over, V_over, S_over, rows_over = self._stac_merger.insert_and_merge_with_rows(
            rows_cuda, keys_cuda, values_cuda, scores_cuda,
            sim_thresh=self.sim_threshold,
            replace_thresh=self.replace_threshold,
            score_thresh=self.score_threshold,
        )
        self._flush_cuda_diagnostics()

        E_overflow_new = rows_over.shape[0]
        if _MEM_PROFILE:
            over_mb = _tensor_mb(K_over) + _tensor_mb(V_over) + _tensor_mb(S_over) + _tensor_mb(rows_over)
            _mem_checkpoint("cuda-merge:after-insert_and_merge",
                            f"E_over_new={E_overflow_new}, over_tensors={over_mb:.1f}MB")
        
        # Free CUDA kernel inputs
        del keys_cuda, values_cuda, scores_cuda, rows_cuda
        
        self.K_over = K_over
        self.V_over = V_over
        self.S_over = S_over
        self.rows_over = rows_over
        self._overflow_alloc = self._voxel_alloc

        _mem_checkpoint("cuda-merge:exit",
                        f"overflow={E_overflow_old}->{E_overflow_new}")
        
        if self.debug:
            time_end.record()
            torch.cuda.synchronize()
            total_cuda_time = time_start.elapsed_time(time_end)
            append_time = total_cuda_time * 0.15
            merge_one2one_time = total_cuda_time * 0.40
            merge_all2one_time = total_cuda_time * 0.30
            remerge_time = total_cuda_time * 0.15
        
        return {
            "init_time": init_time,
            "append_time": append_time,
            "merge_one2one_time": merge_one2one_time,
            "merge_all2one_time": merge_all2one_time,
            "remerge_time": remerge_time,
        }

    def _insert_and_merge_fused_cuda_seg(self,
                    rows: torch.Tensor,
                    keys: torch.Tensor,
                    values: torch.Tensor,
                    scores: torch.Tensor):
        """Segmented-pool variant of _insert_and_merge_fused_cuda.
        Uses 1-token-per-segment storage to eliminate internal fragmentation."""
        if keys.numel() == 0:
            return {"init_time": 0, "append_time": 0, "merge_one2one_time": 0,
                    "merge_all2one_time": 0, "remerge_time": 0}
        init_time = 0.0
        append_time = 0.0
        merge_one2one_time = 0.0
        merge_all2one_time = 0.0
        remerge_time = 0.0

        if self.debug:
            time_start = torch.cuda.Event(enable_timing=True)
            time_end = torch.cuda.Event(enable_timing=True)
            time_start.record()

        E_new = keys.shape[0]
        E_overflow_old = self.rows_over.numel()

        device = keys.device
        self._remap_overflow_rows()

        if _MEM_PROFILE:
            torch.cuda.synchronize()
            _a0, _r0 = _gpu_mem_mb()

        # Concat input with overflow. Keys/values stay in self.dtype (bf16)
        # to avoid a wasteful fp32 round-trip that doubles transient memory.
        rows_cuda = torch.concat([rows.to(torch.long), self.rows_over], dim=0)
        keys_cuda = torch.concat([keys.to(self.dtype), self.K_over.to(self.dtype)], dim=0)
        values_cuda = torch.concat([values.to(self.dtype), self.V_over.to(self.dtype)], dim=0)
        scores_cuda = torch.concat([scores.float(), self.S_over.float()], dim=0)

        del keys, values, scores, rows

        if _MEM_PROFILE:
            torch.cuda.synchronize()
            _a1, _r1 = _gpu_mem_mb()
            E_total = keys_cuda.shape[0]
            concat_mb = (_tensor_mb(keys_cuda) + _tensor_mb(values_cuda)
                         + _tensor_mb(scores_cuda) + _tensor_mb(rows_cuda))
            logger.info("  [MEM-SEG] concat | E_new=%d, E_over=%d, E_total=%d, "
                        "concat_tensors=%.1fMB, Δalloc=%+.0fMB, Δres=%+.0fMB",
                        E_new, E_overflow_old, E_total, concat_mb, _a1-_a0, _r1-_r0)

        first_init = self._stac_merger is None
        self._init_cuda_merger()

        if not self._stac_merger.is_seg_mode:
            self._stac_merger.set_seg_mode(True)

        self._stac_merger.ensure_capacity(self._voxel_alloc)
        if self._pending_voxel_zones is not None and self._pending_voxel_zones.numel() > 0:
            self._stac_merger.set_voxel_zones(self._pending_voxel_zones)
            self._pending_voxel_zones = None

        if self.debug:
            time_end.record()
            torch.cuda.synchronize()
            init_time = time_start.elapsed_time(time_end)
            time_start.record()

        K_over, V_over, S_over, rows_over = self._stac_merger.insert_and_merge_with_rows_seg(
            rows_cuda, keys_cuda, values_cuda, scores_cuda,
            sim_thresh=self.sim_threshold,
            replace_thresh=self.replace_threshold,
            score_thresh=self.score_threshold,
        )
        self._flush_cuda_diagnostics()

        del keys_cuda, values_cuda, scores_cuda, rows_cuda

        if _MEM_PROFILE:
            torch.cuda.synchronize()
            _a4, _r4 = _gpu_mem_mb()
            logger.info("  [MEM-SEG] after_cuda_merge | Δalloc=%+.0fMB, alloc=%.0fMB, reserved=%.0fMB",
                        _a4-_a1, _a4, _r4)

        self.K_over = K_over
        self.V_over = V_over
        self.S_over = S_over
        self.rows_over = rows_over
        self._overflow_alloc = self._voxel_alloc

        if self.debug:
            time_end.record()
            torch.cuda.synchronize()
            total_cuda_time = time_start.elapsed_time(time_end)
            append_time = total_cuda_time * 0.15
            merge_one2one_time = total_cuda_time * 0.40
            merge_all2one_time = total_cuda_time * 0.30
            remerge_time = total_cuda_time * 0.15

        return {
            "init_time": init_time,
            "append_time": append_time,
            "merge_one2one_time": merge_one2one_time,
            "merge_all2one_time": merge_all2one_time,
            "remerge_time": remerge_time,
        }

    @torch.no_grad()
    def insert_and_merge(self,
                        K_new: torch.Tensor,  # [H, Tn, D]
                        V_new: torch.Tensor,  # [H, Tn, D]
                        S_new: torch.Tensor,  # [H, Tn]
                        I_new: torch.Tensor,  # [H, Tn]
                        VX_new: torch.Tensor, # [H, Tn]
                        num_voxels: int,
                        mode: str = "element"):  # "element" or "parallel"
        """
        Insert new tokens into the memory structure.
        
        Args:
            K_new: [H, Tn, D]
            V_new: [H, Tn, D]
            S_new: [H, Tn] float32
            I_new: [H, Tn] long
            VX_new: [H, Tn] long, voxel indices in [0, num_voxels)
            num_voxels: total number of voxels
            mode: "element" or "parallel" insertion mode
        """
        init_time = 0.0
        append_time = 0.0
        merge_all2one_time = 0.0
        merge_one2one_time = 0.0
        defualt_res = {
            "init_time": init_time,
            "append_time": append_time,
            "merge_all2one_time": merge_all2one_time,
            "merge_one2one_time": merge_one2one_time,
        }
        H, Tn, D = K_new.shape
        if Tn == 0:
            if self.debug:
                logger.warning("No new tokens to insert")
            return defualt_res
        assert H == self.H and D == self.D
        assert V_new.shape == K_new.shape and S_new.shape == I_new.shape == VX_new.shape == (H, Tn)

        self._token_count += H * Tn

        if self.debug:
            time_start = torch.cuda.Event(enable_timing=True)
            time_end = torch.cuda.Event(enable_timing=True)
            time_start.record()

        self.ensure_capacity(num_voxels)

        valid = (VX_new >= 0) & (VX_new < self._voxel_offset)
        if not valid.any():
            if self.debug:
                logger.warning("No valid tokens to insert")
            return defualt_res
        
        head_ids = torch.arange(H, device=K_new.device, dtype=torch.long)[:, None].expand(H, Tn)
        h_f = head_ids[valid]
        v_f = VX_new[valid].long()
        rows_f = h_f * self._voxel_alloc + v_f

        # sort within row for better memory access
        rows_f, idx = torch.sort(rows_f, stable=True)
        h_f, v_f = h_f[idx], v_f[idx]
        K_new, V_new = K_new[valid][idx], V_new[valid][idx]
        S_new, I_new = S_new[valid][idx], I_new[valid][idx]

        uniq, counts = torch.unique_consecutive(rows_f, return_counts=True)

        if self.debug:
            time_end.record()
            torch.cuda.synchronize()
            init_time = time_start.elapsed_time(time_end)

        if mode in ["fused", "default"] and self.backend == "python":
            timing = self._insert_and_merge_fusion(rows_f, K_new, V_new, S_new)
            timing['init_time'] += init_time
            timing['append_time'] += append_time
            timing['merge_all2one_time'] += merge_all2one_time
            timing['merge_one2one_time'] += merge_one2one_time
            return timing
        elif mode in ["cuda", "stac", "cuda_seg", "stac_seg"] or self.backend == "cuda":
            # Seg vs. contiguous is determined by _alloc_type (set at construction time).
            # _insert_and_merge_fused_cuda_seg keeps a backward-compat guard for set_seg_mode.
            if self._stac_merger is None:
                self._init_cuda_merger()
            if self._stac_merger.is_seg_mode:
                timing = self._insert_and_merge_fused_cuda_seg(rows_f, K_new, V_new, S_new)
            else:
                timing = self._insert_and_merge_fused_cuda(rows_f, K_new, V_new, S_new)
            timing['init_time'] += init_time
            return timing
        else:
            raise ValueError(f"Unknown insert_and_merge mode: {mode}")
        
        return {
            "init_time": init_time,
            "append_time": append_time,
            "merge_all2one_time": merge_all2one_time,
            "merge_one2one_time": merge_one2one_time
        }


    # ===================== Retrieve: optionally pivots/buffer =====================

    @torch.no_grad()
    def _merger_cuda_retrieve(self,
                              voxel_ids: Optional[torch.Tensor] = None,
                              return_buf: bool = False,
                              retrieve_size: Optional[int] = None,
                              ):
        """
        Retrieve pivots (and optionally buffer tokens) from the CUDA MergerWrapper.
        
        This method is called when _stac_merger is active (CUDA mode).
        Delegates to MergerWrapper.retrieve() for pivots and MergerWrapper.retrieve_buf()
        for buffer tokens when return_buf=True.
        
        Args:
            voxel_ids: Optional tensor of voxel indices to retrieve from. 
                       If None, retrieves from all used voxels.
            return_buf: If True, also retrieves unmerged buffer tokens and concatenates
                        them after pivot tokens. Buffer tokens have bias=0.0.
            retrieve_size: Target output size per head. If None or <=0, uses Q*P for
                           pivots (plus Q*B for buffers when return_buf=True).
        """
        H, D, P, B = self.H, self.D, self.P, self.buf_cap
        
        def empty_output(size: int):
            if size <= 0:
                size = 0
            K0 = torch.zeros(H, size, D, dtype=self.dtype, device=self.device)
            V0 = torch.zeros(H, size, D, dtype=self.dtype, device=self.device)
            M0 = torch.zeros(H, size, dtype=torch.bool, device=self.device)
            B0 = torch.full((H, size), float("-inf"), dtype=torch.float32, device=self.device)
            return K0, V0, M0, B0
        
        if self._stac_merger is None:
            return empty_output(retrieve_size if retrieve_size and retrieve_size > 0 else 0)
        
        # Resolve voxel IDs
        if voxel_ids is None:
            if self._voxel_offset == 0:
                return empty_output(retrieve_size if retrieve_size and retrieve_size > 0 else 0)
            voxel_ids = torch.arange(int(self._voxel_offset), device=self.device, dtype=torch.long)
        else:
            voxel_ids = voxel_ids.to(device=self.device, dtype=torch.long)
        
        # Filter valid voxel IDs
        voxel_ids = torch.unique(voxel_ids)
        voxel_ids = voxel_ids[(voxel_ids >= 0) & (voxel_ids < self._voxel_offset)]
        Q = voxel_ids.numel()
        
        if Q == 0:
            return empty_output(retrieve_size if retrieve_size and retrieve_size > 0 else 0)
        
        # Determine pivot retrieve size
        piv_retrieve_size = retrieve_size if (retrieve_size is not None and retrieve_size > 0) else -1
        
        _mem_checkpoint("retrieve:before",
                        f"Q={Q}, V_alloc={self._voxel_alloc}, ret_size={piv_retrieve_size}")

        # Select contiguous or segmented retrieve based on mode
        _use_seg = self._stac_merger.is_seg_mode
        if _use_seg:
            K_piv, V_piv, M_piv_u8, bias_piv = self._stac_merger.retrieve_seg(
                voxel_ids,
                retrieve_size=piv_retrieve_size,
                used_voxel_limit=self._voxel_offset,
            )
        else:
            K_piv, V_piv, M_piv_u8, bias_piv = self._stac_merger.retrieve(
                voxel_ids,
                retrieve_size=piv_retrieve_size,
                used_voxel_limit=self._voxel_offset,
            )
        M_piv = M_piv_u8.to(torch.bool)

        _mem_checkpoint("retrieve:after-pivots",
                        f"piv_shape={list(K_piv.shape)}")
        
        if not return_buf:
            return K_piv, V_piv, M_piv, bias_piv
        
        # return_buf=True: also retrieve buffer tokens
        piv_out_size = K_piv.shape[1]
        
        # Determine buffer retrieve size
        if retrieve_size is not None and retrieve_size > 0:
            buf_retrieve_size = max(0, retrieve_size - piv_out_size)
        else:
            buf_retrieve_size = -1
        
        if buf_retrieve_size == 0:
            return K_piv, V_piv, M_piv, bias_piv
        
        if _use_seg:
            K_buf, V_buf, M_buf_u8, bias_buf = self._stac_merger.retrieve_buf_seg(
                voxel_ids,
                buf_retrieve_size=buf_retrieve_size,
                used_voxel_limit=self._voxel_offset,
            )
        else:
            K_buf, V_buf, M_buf_u8, bias_buf = self._stac_merger.retrieve_buf(
                voxel_ids,
                buf_retrieve_size=buf_retrieve_size,
                used_voxel_limit=self._voxel_offset,
            )
        M_buf = M_buf_u8.to(torch.bool)
        
        # If no buffer tokens, return pivots only
        if K_buf.shape[1] == 0:
            return K_piv, V_piv, M_piv, bias_piv
        
        # Concatenate pivots and buffers along token dimension
        K_out = torch.cat([K_piv, K_buf], dim=1)
        V_out = torch.cat([V_piv, V_buf], dim=1)
        M_out = torch.cat([M_piv, M_buf], dim=1)
        bias_out = torch.cat([bias_piv, bias_buf], dim=1)
        
        return K_out, V_out, M_out, bias_out

    @torch.no_grad()
    def retrieve(self,
                voxel_ids: Optional[torch.Tensor] = None,
                return_buf: bool = False,
                retrieve_size: Optional[int] = None,
                ):
        """
        Returns (pivots only when return_buf=False):
            K_out: [H, Ntok, D]
            V_out: [H, Ntok, D]
            M_out: [H, Ntok]           (valid mask)
            logit_bias_out: [H, Ntok]  (pivot: log(C); invalid: -inf; float32)

        Fixed-size contract for pivots:
        - if retrieve_size is None or <= 0: Ntok = Q * P (Q = queried used voxels)
        - if retrieve_size > 0: Ntok = retrieve_size
        - valid pivots are sorted by W(desc), bias is log(C) at valid positions
        """
        
        # Delegate to CUDA merger if active
        if self._stac_merger is not None:
            return self._merger_cuda_retrieve(voxel_ids, return_buf, retrieve_size)
        
        device = self.device
        H, V_alloc, P, D = self.H, self._voxel_alloc, self.P, self.D
        dtype_k = self.dtype
        dtype_v = self.dtype

        def empty_output(size: int):
            K0 = torch.zeros(H, size, D, dtype=dtype_k, device=device)
            V0 = torch.zeros(H, size, D, dtype=dtype_v, device=device)
            M0 = torch.zeros(H, size, dtype=torch.bool, device=device)
            B0 = torch.full((H, size), float("-inf"), dtype=torch.float32, device=device)
            return K0, V0, M0, B0

        # 0) resolve voxel ids
        if voxel_ids is None:
            if self._voxel_offset == 0:
                if retrieve_size is not None and retrieve_size > 0:
                    return empty_output(int(retrieve_size))
                return empty_output(0)
            voxel_ids = torch.arange(int(self._voxel_offset), device=device, dtype=torch.long)
        else:
            voxel_ids = voxel_ids.to(device=device, dtype=torch.long)

        voxel_ids = torch.unique(voxel_ids)
        voxel_ids = voxel_ids[(voxel_ids >= 0) & (voxel_ids < self._voxel_offset)]
        req_size = int(retrieve_size) if retrieve_size is not None else -1
        out_size = (int(voxel_ids.numel()) * P) if req_size <= 0 else req_size
        if voxel_ids.numel() == 0:
            return empty_output(out_size)

        # 1) gather pivots [H, VN, P, ...]
        heads = torch.arange(H, device=device, dtype=torch.long)[:, None].expand(H, voxel_ids.numel())
        rows = heads * V_alloc + voxel_ids[None, :].expand(H, voxel_ids.numel()) # [H, VN]
        rows_flat = rows.reshape(-1)
        pack = self.pivots.read_rows_dict(rows_flat, fields=["K", "V", "C", "M", "W"])
        K_all = pack["K"] # [H*VN, P, D]
        V_all = pack["V"] # [H*VN, P, D]
        C_all = pack["C"] # [H*VN, P]
        M_all = pack["M"] # [H*VN, P] bool
        W_all = pack["W"] # [H*VN, P]

        if not M_all.any():
            return empty_output(out_size)

        # 2) flatten (VN,P) -> T and pack per-head
        # H_, VN, P_, D_ = K_all.shape
        # T = VN * P

        K_flat = K_all.reshape(H, -1, D) # [H, T, D]
        V_flat = V_all.reshape(H, -1, D)
        M_flat = M_all.reshape(H, -1)
        C_flat = C_all.reshape(H, -1)
        W_all_flat = W_all.reshape(H, -1) # [H, T]
        # 3) gather valid pivots and arrange per-head
        sorted_index = torch.argsort(W_all_flat, dim=1, descending=True)
        K_flat = torch.gather(K_flat, 1, sorted_index.unsqueeze(-1).expand(-1, -1, D))
        V_flat = torch.gather(V_flat, 1, sorted_index.unsqueeze(-1).expand(-1, -1, D))
        M_flat = torch.gather(M_flat, 1, sorted_index)
        C_flat = torch.gather(C_flat, 1, sorted_index)
        # 4) pack output tensors (fixed-size)
        T = K_flat.shape[1]
        if out_size <= 0:
            return empty_output(0)
        pos = (M_flat.to(torch.int32).cumsum(dim=1) - 1).masked_fill_(~M_flat, 0).to(torch.long)

        K_out = torch.zeros((H, out_size, D), dtype=dtype_k, device=device)
        V_out = torch.zeros((H, out_size, D), dtype=dtype_v, device=device)
        M_out = torch.zeros((H, out_size),    dtype=torch.bool, device=device)
        C_out = torch.zeros((H, out_size),    dtype=C_flat.dtype, device=device)

        mask = M_flat & (pos < out_size)
        ii = torch.arange(H, device=device)[:, None].expand(H, T)[mask]
        jj = pos[mask]
        K_out.index_put_((ii, jj), K_flat[mask], accumulate=False)
        V_out.index_put_((ii, jj), V_flat[mask], accumulate=False)
        M_out.index_put_((ii, jj), torch.ones_like(jj, dtype=torch.bool), accumulate=False)
        C_out.index_put_((ii, jj), C_flat[mask], accumulate=False)

        logit_bias_out = torch.full((H, out_size), float("-inf"), dtype=torch.float32, device=device)
        if M_out.any():
            logit_bias_out[M_out] = C_out[M_out].to(torch.float32).clamp_min_(1e-9).log_()

        if not return_buf:
            return K_out, V_out, M_out, logit_bias_out

        if req_size > 0 and req_size <= out_size:
            return K_out, V_out, M_out, logit_bias_out

        # (optional) include live buffers; unchanged from your previous storage-specific path.
        # raise NotImplementedError("return_buf=True: plug your buffer gather and concat here.")
        K_buf, V_buf, S_buf, M_buf = self.buffer.read_rows(rows_flat)
        # K_buf = pack_buf["K"] # [H*VN, B, D]
        # V_buf = pack_buf["V"] # [H*VN, B, D]
        # S_buf = pack_buf["S"] # [H*VN, B]
        # M_buf = pack_buf["M"] # [H*VN, B] bool

        K_buf_flat = K_buf.reshape(H, -1, D) # [H, T_buf, D]
        V_buf_flat = V_buf.reshape(H, -1, D)
        M_buf_flat = M_buf.reshape(H, -1)
        S_buf_flat = S_buf.reshape(H, -1)

        cnt_buf = M_buf_flat.sum(dim=1)         # [H]
        maxN_buf = int(cnt_buf.max().item())
        retr_buf_size = maxN_buf
        if req_size > 0:
            retr_buf_size = min(maxN_buf, req_size - out_size)
        if (retr_buf_size>0) and (retr_buf_size < maxN_buf):
            maxN_buf = retr_buf_size
            sorted_index_buf = torch.argsort(S_buf_flat, dim=1, descending=True)
            K_buf_flat = torch.gather(K_buf_flat, 1, sorted_index_buf.unsqueeze(-1).expand(-1, -1, D))
            V_buf_flat = torch.gather(V_buf_flat, 1, sorted_index_buf.unsqueeze(-1).expand(-1, -1, D))
            M_buf_flat = torch.gather(M_buf_flat, 1, sorted_index_buf)

            K_buf_flat = K_buf_flat[:, :maxN_buf, :]
            V_buf_flat = V_buf_flat[:, :maxN_buf, :]
            M_buf_flat = M_buf_flat[:, :maxN_buf]
            cnt_buf = M_buf_flat.sum(dim=1)         # [H]
            maxN_buf = int(cnt_buf.max().item())

        if maxN_buf <= 0 or retr_buf_size <= 0:
            return K_out, V_out, M_out, logit_bias_out

         # gather valid pivots and arrange per-head
        pos_buf = (M_buf_flat.to(torch.int32).cumsum(dim=1) - 1).masked_fill_(~M_buf_flat, 0).to(torch.long)
        K_buf_out = torch.zeros((H, maxN_buf, D), dtype=dtype_k, device=device)
        V_buf_out = torch.zeros((H, maxN_buf, D), dtype=dtype_v, device=device)
        M_buf_out = torch.zeros((H, maxN_buf),    dtype=torch.bool, device=device)
        mask_buf = M_buf_flat
        Tb = K_buf_flat.shape[1]
        ii_buf = torch.arange(H, device=device)[:, None].expand(H,Tb)[mask_buf] 
        jj_buf = pos_buf[mask_buf]
        K_buf_out.index_put_((ii_buf, jj_buf), K_buf_flat[mask_buf], accumulate=False)
        V_buf_out.index_put_((ii_buf, jj_buf), V_buf_flat[mask_buf], accumulate=False)
        M_buf_out.index_put_((ii_buf, jj_buf), torch.ones_like(jj_buf, dtype=torch.bool), accumulate=False)

        logit_bias_buf = torch.full((H, maxN_buf), float("-inf"), dtype=torch.float32, device=device)
        if M_buf_out.any():
            logit_bias_buf[M_buf_out] = 0.0  # or some default bias for buffer tokens
        
        K_out = torch.cat([K_out, K_buf_out], dim=1)
        V_out = torch.cat([V_out, V_buf_out], dim=1)
        M_out = torch.cat([M_out, M_buf_out], dim=1)
        logit_bias_out = torch.cat([logit_bias_out, logit_bias_buf], dim=1)

        return K_out, V_out, M_out, logit_bias_out

    def _pool_counts(self):
        """Return (buf_data, buf_alloc, piv_data, piv_alloc) token counts.

        Uses CUDA pool_stats() when the CUDA backend is active; otherwise
        falls back to the Python allocator stats.
        """
        if self._stac_merger is not None and hasattr(self._stac_merger, "pool_stats"):
            ps = self._stac_merger.pool_stats()
            return (ps["buf_data_count"], ps["buf_alloc_count"],
                    ps["piv_data_count"], ps["piv_alloc_count"])
        buf_s = self.buffer.stats()
        piv_s = self.pivots.stats()
        return (buf_s.get("data_count", 0), buf_s.get("alloc_count", 1),
                piv_s.get("data_count", 0), piv_s.get("alloc_count", 1))

    def info(self) -> dict:
        buf_data, buf_alloc, piv_data, piv_alloc = self._pool_counts()
        data_count  = piv_data + buf_data
        alloc_count = piv_alloc + buf_alloc

        merge_compress_ratio = piv_data / float(self._token_count + 1e-6)
        best_compress_ratio  = data_count / float(self._token_count + 1e-6)
        real_compress_ratio  = alloc_count / float(self._token_count + 1e-6)

        return dict(
            head_dim=self.D,
            num_heads=self.H,
            pivot_capacity=self.P,
            buffer_capacity=self.buf_cap,
            used_voxels=self._voxel_offset,
            allocated_voxels=self._voxel_alloc,
            token_count=self._token_count,
            merge_compress_ratio=merge_compress_ratio,
            best_compress_ratio=best_compress_ratio,
            real_compress_ratio=real_compress_ratio,
            pivot_pool_used=piv_data,
            pivot_pool_alloc=piv_alloc,
            buffer_pool_used=buf_data,
            buffer_pool_alloc=buf_alloc,
        )
    
    def memory(self) -> dict:
        """Return memory breakdown in MB for buffer and pivot pools."""
        buf_data, buf_alloc, piv_data, piv_alloc = self._pool_counts()
        kv_bytes = torch.finfo(self.dtype).bits // 8

        buf_used_mem  = buf_data  * (self.D * 2 * kv_bytes + 4)
        buf_alloc_mem = buf_alloc * (self.D * 2 * kv_bytes + 4)
        piv_used_mem  = piv_data  * (self.D * 2 * kv_bytes + 4 + 4)
        piv_alloc_mem = piv_alloc * (self.D * 2 * kv_bytes + 4 + 4)

        return dict(
            buffer_used_mem=buf_used_mem / (1024**2),
            buffer_alloc_mem=buf_alloc_mem / (1024**2),
            pivot_used_mem=piv_used_mem / (1024**2),
            pivot_alloc_mem=piv_alloc_mem / (1024**2),
        )

