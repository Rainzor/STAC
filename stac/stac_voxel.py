# Copyright (c) 2025 STAC Authors. All rights reserved.

import logging
import os
from typing import List, Optional, Tuple, Dict, Callable
import torch
import warnings
import numpy as np
from .h2o import HeavyHittersKV
from .voxel import BinaryVoxel
from .flash_attn_triton import fa_forward_colsum_fast, fa_forward_colsum_fast_sub

from .merger import VoxelKVMerger, _MEM_PROFILE, _gpu_mem_mb, _tensor_mb

logger = logging.getLogger(__name__)

try:
    import attn_cuda as _attn_cuda
    _ATTN_CUDA_AVAILABLE = getattr(_attn_cuda, "is_available", lambda: True)()
except Exception:
    _ATTN_CUDA_AVAILABLE = False

class STACVoxelKV(HeavyHittersKV):
    """
    STACVoxelKV: voxel-based, STAC-style KV manager (H2O framework plus voxel pool and merge).

    Decode sequence (per step):
      1) append_kv(layer-wise): write new K/V to hot cache (hot only; voxel pool untouched)
      2) decode_sparse_attn(layer-wise): compute attention over hot (+ merge), update hot scores
    After each step (once for all layers):
      3) append_positions(all layers): voxelize new token 3D coords, write token->voxel id (aligned with hot)
      4) prune_kv(all layers): H2O pruning on hot; on the eviction path, batch-write dropped tokens to VoxelKVStore (buffer) and trigger segment-parallel merge by threshold
      5) retrieve_kv(all layers): retrieve pivots from VoxelKVStore by per-layer voxel importance (per-voxel quota and global cap configurable), write to merge cache for next attention
    """

    def __init__(
        self,
        *args,
        voxel_size: float = 0.05,
        voxelize_layers: Optional[List[int]] = None,
        # --- VoxelKVStore ---
        init_voxels: int = 1024,            # initial voxel count
        voxel_buf_cap: int = 32,            # buffer capacity per (head, voxel)
        voxel_piv_cap: int = 4,             # max pivots per (head, voxel)
        voxel_backend: str = "python",      # backend type
        sim_threshold: float = 0.8,        # pivot similarity threshold
        replace_threshold: float = 0.8,     # pivot replacement threshold
        score_threshold: float = 0,         # or cumulative score threshold
        weight_fn: Optional[Callable]=None, # pivot weight function
        # --- Slab Pool ---
        allocator: str = "slab",            # "static" | "slab" | "segment" | "compact"
        slab_growth = 256,
        slab_free_policy = "immediate",
        slab_cap = 1024,
        seg_size: int = 4,
        alloc_debug: bool = False,
        # --- Retrieve ---
        retrieval_size: int = -1,           # pivots to retrieve per layer (-1 = no limit)
        # --- CPU Offload ---
        enable_alloc_cpu: bool = False,     # enable CPU offload
        gpu_threshold_gb: float = 10.0,    # GPU memory threshold (GB)
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        # ---- config cache ----
        self._voxel_size = torch.tensor(voxel_size, dtype=torch.float32, device=self.device)
        self._vm = BinaryVoxel(voxel_size=voxel_size, device=self.device)
        self._voxel_ids = set()
        self._recent_voxel_ids = None

        if voxelize_layers is None:
            self._voxelize_list = [True for _ in range(self.num_layers)]
        else:
            voxelize_set = set(voxelize_layers)
            self._voxelize_list = [l in voxelize_set for l in range(self.num_layers)]
        assert any(self._voxelize_list), "STACVoxelKV requires at least one voxelized layer."

        # VoxelKVStore hyperparams; reused on reset
        self._vk_conf = dict(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            init_voxels=init_voxels,
            pivot_cap=voxel_piv_cap,
            budget_cap=voxel_buf_cap,
            sim_thresh=sim_threshold,
            replace_thresh=replace_threshold,
            score_thresh=score_threshold,
            weight_fn=weight_fn,
            dtype=self.dtype,
            device=self.device,
            debug=self.debug,
            allocator=allocator,
            backend=voxel_backend,
            slab_growth=slab_growth,
            slab_free_policy=slab_free_policy,
            slab_cap=slab_cap,
            seg_size=seg_size,
            alloc_debug=alloc_debug,
            # CPU Offload parameters
            enable_cpu_offload=enable_alloc_cpu,
            gpu_threshold_gb=gpu_threshold_gb,
        )
        
        # Store CPU offload config for reference
        self._enable_alloc_cpu = enable_alloc_cpu

        # Retrieve default policy
        self.recent_size = retrieval_size
        self.retrieval_token_size = int(retrieval_size*self.token_per_f)

        # token->voxel id (aligned with hot _token_indices_hot)

        # --- VoxelKVStore (per layer) ---
        # reset() will initialize
        self._vstores = None

        cpu_offload_str = ""
        if enable_alloc_cpu:
            cpu_offload_str = f"\ncpu_offload: enabled, threshold={gpu_threshold_gb}GB"
        self._stac_cpu_offload_str = cpu_offload_str

        self._log_registration()

    def _log_registration(self) -> None:
        super()._log_registration()
        logger.info(
            "[STACVoxelKV] retrieval=%dx%d  voxel_size=%.3f  pivot_cap=%d  buf_cap=%d",
            self.recent_size, self.token_per_f, self._voxel_size.item(),
            self._vk_conf["pivot_cap"], self._vk_conf["budget_cap"],
        )
        logger.info(
            "  merge: sim>%.2f  replace<%.2f  score>%.2f",
            self._vk_conf["sim_thresh"], self._vk_conf["replace_thresh"],
            self._vk_conf["score_thresh"],
        )
        logger.info(
            "  pool: backend=%s allocator=%s  growth=%d  cap=%d  seg=%d%s",
            self._vk_conf["backend"],
            self._vk_conf["allocator"], 
            self._vk_conf["slab_growth"],
            self._vk_conf["slab_cap"], 
            self._vk_conf["seg_size"],
            getattr(self, "_stac_cpu_offload_str", ""),
        )
    # ------------------------
    # reset 
    # ------------------------
    def reset(self):
        super().reset()
        # --- Voxel Manager ---
        self._vm.reset()
        self._voxel_ids.clear()
        self._recent_voxel_ids = None
    
        # self._vstores = [
        #     VoxelKVMerger(**self._vk_conf)
        #     if self._voxelize_list[l] else None
        #     for l in range(self._L_eff)
        # ]
        self._vk_conf["num_heads"] = self._L_eff * self.num_heads
        self._vstore = VoxelKVMerger(**self._vk_conf)
        # token->voxel mapping prealloc
        self._token_voxel_indices = torch.full((self._L_eff, self.num_heads, self.reserved_buffer_token_size), -1, dtype=torch.int32, device=self.device)
        self._offset_voxel = [0 for _ in range(self._L_eff)]
        # --- Merge KV（per layer） ---
        self.key_cache_retrieval = [torch.empty(self.num_heads, 0, self.head_dim, device=self.device, dtype=self.dtype)
                                   for _ in range(self._L_eff)]
        self.value_cache_retrieval = [torch.empty(self.num_heads, 0, self.head_dim, device=self.device, dtype=self.dtype)
                                     for _ in range(self._L_eff)]
        self._bias_retrieval = [torch.empty(self.num_heads, 0, device=self.device, dtype=torch.float32) for _ in range(self._L_eff)]
        self._token_indices_retrieval = [torch.empty(self.num_heads, 0, dtype=torch.long, device=self.device)
                                        for _ in range(self._L_eff)]
        self._scores_retrieval = [torch.zeros(self.num_heads, 0, device=self.device, dtype=torch.float32)
                                 for _ in range(self._L_eff)]
        self._offset_retrieval = [0 for _ in range(self._L_eff)]


        torch.cuda.empty_cache()
    
    def free(self):

        self._token_voxel_indices = None
        self._offset_voxel = [0] * self._L_eff
        for l in range(self._L_eff):
            # Retrieval Cache
            self.key_cache_retrieval[l] = None
            self.value_cache_retrieval[l] = None
            self._token_indices_retrieval[l] = None
            self._bias_retrieval[l] = None
            self._offset_retrieval[l] = 0
            self._scores_retrieval[l] = None

        return super().free()
        

    # --------------------------------------------
    # append_kv: same logic as HeavyHittersKV
    # --------------------------------------------
    @torch.no_grad()
    def append_kv(self, key_states, value_states, layer_idx: int):
        super().append_kv(key_states, value_states, layer_idx)

    # --------------------------------------------
    # decode_sparse_attn (layer-wise)
    # attention over hot + retrieve; only hot scores updated (merge scores cached separately)
    # --------------------------------------------
    @torch.no_grad()
    def decode_sparse_attn(self, query_states: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        query_states: [B, Tq, H, D]; B==1
        Returns: attn_output [B, Tq, H, D]
        """
        B, Tq, H, D = query_states.shape
        query_frame_size = Tq // self.token_per_f
        assert B == 1 and H == self.num_heads and D == self.head_dim
        assert query_states.is_cuda, "SasaKV.decode_sparse_attn requires CUDA."
        slot_idx = self._to_slot(layer_idx)
        # === hot ===
        T_main = self._offset_hot[slot_idx]
        K_main = self.key_cache_hot[slot_idx][:, :T_main, :]
        V_main = self.value_cache_hot[slot_idx][:, :T_main, :]

        # === merge ===
        if self._voxelize_list[slot_idx]:
            T_ret = self._offset_retrieval[slot_idx]
        else:
            T_ret = 0

        if T_ret > 0:
            K_ret = self.key_cache_retrieval[slot_idx][:, :T_ret, :]
            V_ret = self.value_cache_retrieval[slot_idx][:, :T_ret, :]
            K_all = torch.cat([K_main, K_ret], dim=1)  # [H, T_total, D]
            V_all = torch.cat([V_main, V_ret], dim=1)
            bias = self._build_bias_for_cache(slot_idx, T_main, T_ret,
                                              dtype=query_states.dtype, device=self.device)
        else:
            K_all, V_all = K_main, V_main
            bias = None

        # [B,T,H,D]
        K_all = K_all.unsqueeze(0).transpose(1, 2).contiguous()
        V_all = V_all.unsqueeze(0).transpose(1, 2).contiguous()
        q = query_states.contiguous()
        use_cuda = bool(self.use_attn_cuda and _ATTN_CUDA_AVAILABLE)
        subsample = self.subsample_ratio

        if use_cuda:
            out, _, col_sum = _attn_cuda.flash_attn_bias_colsum(
                q, K_all, V_all, bias=bias, return_colsum=True, subsample_ratio=subsample)
        else:
            out, _, col_sum = fa_forward_colsum_fast_sub(
                q, K_all, V_all, bias=bias, write_o=True, subsample_ratio=subsample)
        scores_total = col_sum.unsqueeze(2)  # [B, H, 1, T_total]

        # update hot scores (merge scores cached separately for diagnostics)
        self._last_query_offset[slot_idx] = Tq
        if T_ret > 0:
            self._update_scores(scores_total[..., :T_main], slot_idx)
            s_ret = scores_total[..., T_main:T_main + T_ret]  # [B,H,1,T_ret]
            s_ret = s_ret.sum(dim=(0, 2)).to(torch.float32)   # [H,T_ret]
            self._scores_retrieval[slot_idx].resize_(self.num_heads, T_ret)
            self._scores_retrieval[slot_idx][:, :T_ret].copy_(s_ret)
        else:
            self._update_scores(scores_total, slot_idx)

        return out


    def _build_bias_for_cache(self, slot_idx: int, T_main: int, T_ret: int,
                              dtype, device,
                              query_frame_size=1):
        """
        Returns column bias [1,H,1,T_total]; columns with I_ret==-1 in merge segment are set to -inf.
        """
        if T_ret <= 0:
            return None
        bias_ret = self._bias_retrieval[slot_idx][:, :T_ret]  # [H,T_ret]
        bias_ret = bias_ret.unsqueeze(0).unsqueeze(2)  # [1,H,1,T_ret]
        bias_main = torch.zeros(1, self.num_heads, 1, T_main, dtype=dtype, device=device) if T_main > 0 else None

        return bias_ret if bias_main is None else torch.cat([bias_main, bias_ret], dim=-1)

    def _update_scores(self, scores: Optional[torch.Tensor]=None, slot_idx: int=0):
        """
        Efficient GPU-optimized version (no view/copy syncs)
        scores: [B, H, 1, T_live] — already summed over query dim
        """
        super()._update_scores(scores, slot_idx)

    #! -------------------------------
    #! Update Voxel
    #! -------------------------------
    def _search_recent_neighbors(self, slot_idx=0, dist_thres=0.1) -> Dict[int, torch.Tensor]:
        """
        Search for neighbor voxels around voxels that contain the most recently written tokens.
        Returns dict: slot_idx -> neighbor_voxel_ids [N]
        """
        if not self._voxelize_list[slot_idx]:
            self._recent_voxel_ids = torch.tensor(list(self._voxel_ids), dtype=torch.long, device=self.device)
            return

        TV_live = self._offset_voxel[slot_idx]
        R = min(self.recent_token_size, TV_live)
        if R == 0:
            self._recent_voxel_ids = None
            return
        
        if R < self.recent_token_size:
            self._recent_voxel_ids = torch.tensor(list(self._voxel_ids), dtype=torch.long, device=self.device)
            return
        
        recent_idx = torch.arange(TV_live - R, TV_live, device=self.device, dtype=torch.int32)
        recent_voxel_ids = self._token_voxel_indices[slot_idx][0, recent_idx]  # [R]
        recent_voxel_ids = recent_voxel_ids[recent_voxel_ids >= 0].unique()
        neighbor_voxel_ids, dist2, found = self._vm.neighbors(
            voxel_ids=recent_voxel_ids,
            radius_m=dist_thres,
            include_self=True
        )
        neighbor_voxel_ids = neighbor_voxel_ids[neighbor_voxel_ids >= 0]
        voxel_ids = torch.unique(neighbor_voxel_ids)
        self._recent_voxel_ids = voxel_ids


    @torch.no_grad()
    def append_positions(self, new_positions: torch.Tensor, new_pos_mask: torch.Tensor):
        """
        Voxelize new token 3D coordinates and write voxel_id into per-layer token->voxel arrays (aligned with hot).
        new_positions: [T_new,3] (float32, world units)
        new_pos_mask: [T_new] (bool)
        Must be called after append_kv for all layers and before prune_kv.
        """
        T_new = new_positions.shape[0]
        device = self.device
        if T_new == 0:
            return

        vs = self._voxel_size.to(new_positions.device, dtype=new_positions.dtype)
        ijk_new = torch.floor(new_positions / vs).to(torch.int32)  # [T_new,3]
        valid = new_pos_mask.to(torch.bool)

        voxel_id_batch = torch.full((T_new,), -1, dtype=torch.int32, device=device)
        if valid.any():
            voxel_id_valid = self._vm.upsert(ijk_new[valid])  # [Nv]
            voxel_id_batch[valid] = voxel_id_valid
            self._voxel_ids.update(voxel_id_valid.cpu().tolist())

        # write this batch of voxel ids to each layer (broadcast to heads); size must match hot increment
        slot_idx = -1
        old_VT = self._offset_voxel[0]
        total_T_expected = old_VT + T_new
        assert np.all(np.array(self._offset_hot)==total_T_expected), (
            f"[STACVoxelKV] size mismatch: hot_T={self._offset_hot} vs voxel_written={total_T_expected}; "
            f"ensure append_kv -> append_positions"
        )
        # broadcast to [H,T_new]
        self._token_voxel_indices[:, :, old_VT:total_T_expected].copy_(
            voxel_id_batch.view(1, 1, -1).expand(self._L_eff, self.num_heads, -1)
        )
        for l in range(self._L_eff):
            self._offset_voxel[l] = total_T_expected
            slot_idx = l
        
        self._search_recent_neighbors(slot_idx=slot_idx, dist_thres=2*self._voxel_size.item())

    def _clear_retrieval_cache(self, slot_idx: int):
        self._offset_retrieval[slot_idx] = 0
        self._scores_retrieval[slot_idx].resize_(self.num_heads, 0)
        self.key_cache_retrieval[slot_idx].resize_(self.num_heads, 0, self.head_dim)
        self.value_cache_retrieval[slot_idx].resize_(self.num_heads, 0, self.head_dim)

    # --------------------------------------------
    #! Overridden prune & Pool Drops to VoxelKVStore 
    # --------------------------------------------
    @torch.no_grad()
    def prune_kv(self):
        # @call _apply_keep_and_compact
        # @call _pool_drops_vectorized_parallel
        super().prune_kv()
        
        # Step frame counter and check for eviction at end of each prune cycle
        if self._enable_alloc_cpu and self._vstore is not None:
            self._vstore.step_frame()
            n_evicted = self._vstore.check_and_evict()
            if self.debug and n_evicted > 0:
                logger.debug("[STACVoxelKV] Evicted %d buffer rows to CPU", n_evicted)

    def _apply_keep_and_compact(self, slot_idx: int, keep_idx_h: torch.Tensor):
        """
        Before compacting hot by keep indices per head: batch-write dropped tokens into voxel pool (buffer),
        then call parent compact logic; finally compact voxel indices in sync.
        """
        self._pool_drops_vectorized(slot_idx, keep_idx_h, device=self.device)

        # --- delegate hot compaction to parent ---
        super()._apply_keep_and_compact(slot_idx, keep_idx_h)
        if not self._voxelize_list[slot_idx]:
            return
        # --- compact voxel indices in sync ---
        H, Tprime = keep_idx_h.shape
        VX = self._token_voxel_indices[slot_idx]                 # [H,T]
        VX_sel = torch.gather(VX, 1, keep_idx_h).contiguous()     # [H,T']
        VX[:, :Tprime].copy_(VX_sel)
        if Tprime < VX.size(1):
            VX[:, Tprime:].fill_(-1)
        self._offset_voxel[slot_idx] = Tprime

    def _apply_keep_and_compact_parallel(self, keep_idx_lh: torch.Tensor) -> bool:
        self._pool_drops_vectorized_parallel(keep_idx_lh)
        super()._apply_keep_and_compact_parallel(keep_idx_lh)

        L,H,Tprime = keep_idx_lh.shape
        VX = torch.gather(self._token_voxel_indices, 2, keep_idx_lh).contiguous()  # [L,H,T']
        self._token_voxel_indices[:, :, :Tprime].copy_(VX)
        if Tprime < self._token_voxel_indices.size(2):
            self._token_voxel_indices[:, :, Tprime:].fill_(-1)
        for slot_idx in range(self._L_eff):
            self._offset_voxel[slot_idx] = Tprime

    def _pool_drops_vectorized_parallel(self, keep_idx_lh: torch.Tensor):
        L,H,D= self._L_eff, self.num_heads, self.head_dim
        T = self._offset_hot[0]
        TV = self._offset_voxel[0]
        assert T == TV, (
            f"[STACVoxelKV] size mismatch T={T}, TV={TV};",
            f" must call: append_kv → append_positions → prune_kv"
        )
        if T == 0:
            return

        if _MEM_PROFILE:
            torch.cuda.synchronize()
            a0, r0 = _gpu_mem_mb()

        Kh = self.key_cache_hot[:,:, :T, :]  # [L,H,T,D]
        Vh = self.value_cache_hot[:,:, :T, :]  # [L,H,T,D]
        Sh = self._scores_hot[:,:, :T]  # [L,H,T]
        Ih = self._token_indices_hot[:,:, :T].to(self.device)  
        VXh = self._token_voxel_indices[:,:, :T].to(self.device)  # [L,H,T] int32

        # drop mask
        keep_mask = torch.zeros(L, H, T, dtype=torch.bool, device=self.device)
        keep_mask.scatter_(2, keep_idx_lh, True)
        drop_mask_all = ~keep_mask  # [L,H,T] bool
        per_layer_head_counts = drop_mask_all.sum(dim=2)  # [L,H]
        T_d = int(per_layer_head_counts[0,0].item())
        assert torch.all(per_layer_head_counts == T_d), f"[STACVoxelKV] per-head drop count mismatch: {per_layer_head_counts.tolist()}"
        if T_d == 0:
            return
        S_drop = Sh[drop_mask_all].view(L, H, T_d).contiguous().to(torch.float32)
        # extract dropped segments (keep [L,H,Td,*])
        K_drop = Kh[drop_mask_all].view(L, H, T_d, D).contiguous()
        V_drop = Vh[drop_mask_all].view(L, H, T_d, D).contiguous()
        I_drop = Ih[drop_mask_all].view(L, H, T_d).contiguous()
        VX_drop = VXh[drop_mask_all].view(L, H, T_d).contiguous().to(torch.int32)

        if _MEM_PROFILE:
            torch.cuda.synchronize()
            a1, r1 = _gpu_mem_mb()
            drop_mb = (_tensor_mb(K_drop) + _tensor_mb(V_drop) + _tensor_mb(S_drop)
                       + _tensor_mb(I_drop) + _tensor_mb(VX_drop))
            logger.debug("  [MEM-PHASE] pool_drops:extract | T_d=%d, drop_tensors=%.1fMB, "
                         "Δalloc=%+.0fMB, Δres=%+.0fMB", T_d, drop_mb, a1-a0, r1-r0)
        # write to voxel pool (segment-parallel append + trigger merge)

        num_voxels = self._vm.get_voxel_keys().shape[0]  # current valid voxel count
        if self._vk_conf["backend"] == "cuda":
            mode = "cuda"    # seg vs. contiguous resolved inside merger via is_seg_mode
        else:
            mode = "default"
        
        if _MEM_PROFILE:
            torch.cuda.synchronize()
            a_pre, r_pre = _gpu_mem_mb()

        if hasattr(self._vm, '_voxel_zones') and self._vm._voxel_zones.numel() > 0:
            self._vstore.update_voxel_zones(self._vm._voxel_zones)

        insert_merge_time = self._vstore.insert_and_merge(
            K_new=K_drop.view(L*H, T_d, D),
            V_new=V_drop.view(L*H, T_d, D),
            S_new=S_drop.view(L*H, T_d),
            I_new=I_drop.view(L*H, T_d),
            VX_new=VX_drop.view(L*H, T_d),
            num_voxels=num_voxels,
            mode=mode
        )

        if _MEM_PROFILE:
            torch.cuda.synchronize()
            a_post, r_post = _gpu_mem_mb()
            vs = self._vstore
            over_new = vs.rows_over.numel()
            over_mb_new = (_tensor_mb(vs.K_over) + _tensor_mb(vs.V_over)
                          + _tensor_mb(vs.S_over) + _tensor_mb(vs.rows_over))
            logger.debug("  [MEM-PHASE] insert_and_merge | E_in=%d, "
                         "overflow_out=%d (%.1fMB), Δalloc=%+.0fMB, Δres=%+.0fMB, "
                         "alloc=%.0fMB, reserved=%.0fMB",
                         L*H*T_d, over_new, over_mb_new,
                         a_post-a_pre, r_post-r_pre, a_post, r_post)

        # Free drop tensors to reclaim transient memory
        del K_drop, V_drop, S_drop, I_drop, VX_drop
        del keep_mask, drop_mask_all

        if _MEM_PROFILE:
            torch.cuda.synchronize()
            a_after_del, r_after_del = _gpu_mem_mb()
            logger.debug("  [MEM-PHASE] after_del_drops  | Δalloc=%+.0fMB, alloc=%.0fMB, "
                         "reserved=%.0fMB", a_after_del-a_post, a_after_del, r_after_del)

        if self.debug:
            
            timing_str = ", ".join(f"{k}: {v:.3f}ms" for k, v in insert_merge_time.items())
            logger.debug("[STACVoxelKV] Pooled %d tokens, voxels=%d, timing: %s",
                         T_d, num_voxels, timing_str)
        
        if _MEM_PROFILE:
            torch.cuda.synchronize()
            a, r = _gpu_mem_mb()
            vs = self._vstore
            over_mb = (_tensor_mb(vs.K_over) + _tensor_mb(vs.V_over)
                       + _tensor_mb(vs.S_over) + _tensor_mb(vs.rows_over))
            kv_hot_mb = (_tensor_mb(self.key_cache_hot) + _tensor_mb(self.value_cache_hot))

            # Detailed breakdown of "other" memory
            vm = self._vm
            voxel_table_mb = (
                _tensor_mb(vm._voxel_keys) + _tensor_mb(vm._voxel_centers)
                + _tensor_mb(vm._voxel_keys_1d) + _tensor_mb(vm._voxel_keys_1d_sorted)
                + _tensor_mb(vm._voxel_keys_sort_idx)
            ) if hasattr(vm, '_voxel_keys') else 0

            # Python-side pivot/buffer pool sizes (only if not skipped in CUDA mode)
            py_piv_mb = sum(_tensor_mb(getattr(vs.pivots, attr))
                           for attr in ['K', 'V', 'W', 'S', 'C', 'M']
                           if hasattr(vs.pivots, attr)) if hasattr(vs, 'pivots') else 0
            py_buf_mb = sum(_tensor_mb(getattr(vs.buffer, attr))
                           for attr in ['K', 'V', 'S', 'M']
                           if hasattr(vs.buffer, attr)) if hasattr(vs, 'buffer') else 0

            # Retrieval cache sizes
            ret_cache_mb = 0
            for sl in range(self._L_eff):
                ret_cache_mb += _tensor_mb(self.key_cache_retrieval[sl])
                ret_cache_mb += _tensor_mb(self.value_cache_retrieval[sl])

            # Token voxel indices
            tvx_mb = _tensor_mb(self._token_voxel_indices) if hasattr(self, '_token_voxel_indices') else 0

            # CUDA merger workspace estimate
            cuda_ws_mb = 0
            try:
                if vs._stac_merger is not None and hasattr(vs._stac_merger, 'workspace_bytes'):
                    cuda_ws_mb = vs._stac_merger.workspace_bytes() / (1024 * 1024)
            except Exception:
                pass

            # CUDA seg pool sizes (fixed at init)
            cuda_seg_mb = 0
            try:
                if vs._stac_merger is not None:
                    m = vs._stac_merger
                    for attr in ['seg_buf_pool_K', 'seg_buf_pool_V', 'seg_buf_pool_S',
                                 'seg_piv_pool_K', 'seg_piv_pool_V', 'seg_piv_pool_W',
                                 'seg_piv_pool_S', 'seg_piv_pool_C', 'seg_piv_pool_M',
                                 'seg_buf_indirection', 'seg_piv_indirection']:
                        if hasattr(m, attr):
                            t = getattr(m, attr)
                            if isinstance(t, torch.Tensor):
                                cuda_seg_mb += _tensor_mb(t)
            except Exception:
                pass

            logger.debug("  [MEM-SUMMARY] voxels=%d, V_alloc=%d, overflow=%d (%.1fMB), "
                         "KV_hot=%.1fMB, GPU alloc=%.0fMB reserved=%.0fMB",
                         num_voxels, vs._voxel_alloc, vs.rows_over.numel(),
                         over_mb, kv_hot_mb, a, r)
            explained_total = (kv_hot_mb + voxel_table_mb + py_piv_mb + py_buf_mb
                              + ret_cache_mb + tvx_mb + cuda_ws_mb + cuda_seg_mb + over_mb)
            logger.debug("  [MEM-DETAIL] voxel_table=%.2fMB, py_piv=%.1fMB, py_buf=%.1fMB, "
                         "ret_cache=%.1fMB, tvx_idx=%.1fMB, cuda_ws=%.1fMB, "
                         "cuda_seg_pool=%.1fMB, overflow=%.1fMB, explained=%.1fMB, "
                         "total_alloc=%.0fMB, unexplained=%+.0fMB, frag(res-alloc)=%.0fMB",
                         voxel_table_mb, py_piv_mb, py_buf_mb, ret_cache_mb, tvx_mb,
                         cuda_ws_mb, cuda_seg_mb, over_mb, explained_total,
                         a, a - explained_total, r - a)
       

    def _pool_drops_vectorized(self, slot_idx: int, all_keep_tokens: torch.Tensor, device: torch.device):
        """
        Batch-write tokens that will be pruned (per-head) into this layer's VoxelKVStore buffer,
        and trigger segment-parallel merge when threshold is met (no re-merge of existing pivots).
        """
        layer_idx = self._managed_layers[slot_idx]
        if not self._voxelize_list[slot_idx]:
            return

        # --- prepare dropped token data ---
        H, D = self.num_heads, self.head_dim
        T = self._offset_hot[slot_idx]
        TV = self._offset_voxel[slot_idx]
        assert TV == T, (
            f"[STACVoxelKV] size mismatch T={T}, TV={TV}; "
            f"must call: append_kv → append_positions → prune_kv"
        )
        if T == 0:
            return

        # current visible K/V/I/VX in hot
        K = self.key_cache_hot[slot_idx,:, :T, :]
        V = self.value_cache_hot[slot_idx,:, :T, :]
        S = self._scores_hot[slot_idx,:, :T]
        I_mat = self._token_indices_hot[slot_idx,:, :T].to(device)
        VX = self._token_voxel_indices[slot_idx,:, :T].to(device)  # [H,T] int32

        # compute drop mask (per head)
        keep_mask = torch.zeros(H, T, dtype=torch.bool, device=device)
        keep_mask.scatter_(1, all_keep_tokens, True)
        drop_mask_all = ~keep_mask # [H,T] bool

        per_head_counts = drop_mask_all.sum(dim=1) # [H]
        T_d = int(per_head_counts[0].item())
        assert torch.all(per_head_counts == T_d), f"[STACVoxelKV] per-head drop count mismatch: {per_head_counts.tolist()}"
        if T_d == 0:
            return

        S_drop  = S[drop_mask_all].view(H, T_d).contiguous().to(torch.float32)
        # extract dropped segments (keep [H, Td, *])
        K_drop  = K[drop_mask_all].view(H, T_d, D).contiguous()
        V_drop  = V[drop_mask_all].view(H, T_d, D).contiguous()
        I_drop  = I_mat[drop_mask_all].view(H, T_d).contiguous()
        VX_drop = VX[drop_mask_all].view(H, T_d).contiguous().to(torch.int32)


        # write to this layer's voxel pool (segment-parallel append + trigger merge)
        # TODO: consider filter low-score or low-count tokens before insert
        vstore = self._vstores[slot_idx]
        num_voxels = self._vm.get_voxel_keys().shape[0]  # current valid voxel count
        if hasattr(self._vm, '_voxel_zones') and self._vm._voxel_zones.numel() > 0:
            vstore.update_voxel_zones(self._vm._voxel_zones)
        insert_merge_time = vstore.insert_and_merge(K_new=K_drop, V_new=V_drop, 
                                S_new=S_drop, I_new=I_drop, VX_new= VX_drop,
                                num_voxels=num_voxels,
                                mode="default")
                                # mode="test")
                                # mode="parallel")
        
        if self.debug:
            first_layer = torch.nonzero(torch.tensor(self._voxelize_list), as_tuple=False)[0].item()
            
            if layer_idx == first_layer and self._processed_frames % 10 == 0:
                timing_str = ", ".join(f"{k}: {v:.3f}ms" for k, v in insert_merge_time.items())
                logger.debug("[STACVoxelKV:L%d] Pooled %d tokens, voxels=%d, timing: %s",
                             layer_idx, T_d, num_voxels, timing_str)

    def retrieve_kv(self, layer_idx, **kwargs):
        slot_idx = self._to_slot(layer_idx)
        if slot_idx==0:
            return self.retrieve_kv_parallel(**kwargs)
        else:
            return 0,0

    def retrieve_kv_parallel(self, **kwargs)-> Tuple[int, int]:
        """
        Parallel version of retrieve_kv for all voxelized layers.
        Returns:
            max_wrote: int, maximum number of tokens written to any voxelized layer
            total_wrote: int, total number of tokens written across all voxelized layers
        """
        conf = kwargs.copy()
        L,H = self._L_eff, self.num_heads

        # --- select voxel set ---
        voxel_ids: Optional[torch.Tensor] = conf.pop("voxel_ids", None)
        return_buf: bool = conf.pop("return_buf", False)
        if voxel_ids is None:
            # Using all valid voxel ids
            if len(self._voxel_ids) == 0:
                for slot_idx in range(L):
                    self._clear_retrieval_cache(slot_idx)
                return 0, 0
            
            # TODO: score based selection

            #& voxel-position based selection
            voxel_ids = self._recent_voxel_ids
            if voxel_ids is None:
                return 0,0
            # voxel_ids = torch.tensor(list(self._voxel_ids), dtype=torch.long, device=self.device)
        else:
            voxel_ids = voxel_ids.to(self.device, dtype=torch.long)

        max_wrote = 0
        total_wrote = 0

        if _MEM_PROFILE:
            a, r = _gpu_mem_mb()
            logger.debug("  [MEM] retrieve_kv_parallel:before  | Q_voxels=%d, "
                         "ret_size=%d, alloc=%.0fMB reserved=%.0fMB",
                         voxel_ids.numel(), self.retrieval_token_size, a, r)

        K_ret, V_ret, M_ret, B_ret = self._vstore.retrieve(
            voxel_ids=voxel_ids,
            return_buf=return_buf,
            retrieve_size=self.retrieval_token_size
        ) # [L*H,Tm,..]

        if _MEM_PROFILE:
            a, r = _gpu_mem_mb()
            ret_mb = _tensor_mb(K_ret) + _tensor_mb(V_ret) + _tensor_mb(M_ret) + _tensor_mb(B_ret)
            logger.debug("  [MEM] retrieve_kv_parallel:after   | out_shape=%s, "
                         "ret_tensors=%.1fMB, alloc=%.0fMB reserved=%.0fMB",
                         list(K_ret.shape), ret_mb, a, r)
        #TODO:
        for slot_idx in range(L):
            if not self._voxelize_list[slot_idx]:
                continue
            # --- backfill to retrieved cache (overwrite) ---
            self._clear_retrieval_cache(slot_idx)
            Tm = K_ret.shape[1]
            if self.retrieval_token_size>=0:
                Tm = min(Tm, self.retrieval_token_size)
                
            if Tm == 0:
                if self.debug:
                    layer_idx = self._managed_layers[slot_idx]
                    logger.debug("[STACVoxelKV:L%d] No pivots retrieved, K_ret=%s, ret_size=%d",
                                 layer_idx, K_ret.shape, self.retrieval_token_size)
                continue
            
            # resize cache
            # [H,Tm,D]
            self.key_cache_retrieval[slot_idx] = K_ret[slot_idx*H:(slot_idx+1)*H, :Tm, :].contiguous()
            self.value_cache_retrieval[slot_idx] = V_ret[slot_idx*H:(slot_idx+1)*H, :Tm, :].contiguous()
            self._bias_retrieval[slot_idx] = B_ret[slot_idx*H:(slot_idx+1)*H, :Tm].contiguous()
            self._offset_retrieval[slot_idx] = Tm

            def _logits_to_count(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
                """logits: [H,T] float32"""
                valid_mask = (logits > -1e10) & mask
                counts = torch.zeros_like(logits, dtype=torch.int32)
                counts[valid_mask] = torch.exp(logits[valid_mask]).to(torch.int32)
                return counts
            
            # mask_ret = M_ret[slot_idx*H:(slot_idx+1)*H, :Tm]
            # C_ret = _logits_to_count(self._bias_retrieval[slot_idx], mask_ret)  # [H,Tm] int32
            # layer_total_wrote = C_ret.sum().item()
            # layer_max_wrote = C_ret.max().item()
            # total_wrote += layer_total_wrote
            # if layer_max_wrote > max_wrote:
            #     max_wrote = layer_max_wrote

        del K_ret, V_ret, M_ret, B_ret
        if _MEM_PROFILE:
            torch.cuda.synchronize()
            a_end, r_end = _gpu_mem_mb()
            logger.debug("  [MEM] retrieve_kv_parallel:done    | alloc=%.0fMB reserved=%.0fMB "
                         "(after del K/V/M/B_ret)", a_end, r_end)

        return max_wrote, total_wrote

    
    @torch.no_grad()
    def _retrieve_kv(self, layer_idx: int, **kwargs)-> Tuple[int, int]:
        """
        Retrieve required pivots from this layer's VoxelKVStore and write to merge-cache.
        Strategy:
         - 1. If kwargs provides voxel_ids (Long/Int32[M]), fetch those voxels directly;
         - 2. Select all voxels;
         - 3. Select top-M representative voxels by voxel accumulated scores (sum over heads);
         - 4. Select top-M by 3D spatial position (TODO).
        Then call VoxelKVStore.retrieve() to fetch K/V/S/I and backfill merge cache.

        kwargs:
         - voxel_ids: Optional[Tensor[M]]
        """
        if not self._voxelize_list[layer_idx]:
            return 0, 0
        slot_idx = self._to_slot(layer_idx)
        # --- parse config ---
        conf = kwargs.copy()

        vstore = self._vstores[slot_idx]
        H = self.num_heads

        # --- select voxel set ---
        voxel_ids: Optional[torch.Tensor] = conf.pop("voxel_ids", None)
        return_buf: bool = conf.pop("return_buf", False)
        if voxel_ids is None:
            # Using all valid voxel ids
            if len(self._voxel_ids) == 0:
                self._clear_retrieval_cache(slot_idx)
                return 0, 0
            
            # TODO: score based selection

            # TODO: voxel-position based selection
            voxel_ids = self._recent_voxel_ids
            if voxel_ids is None:
                return 0,0
            # voxel_ids = torch.tensor(list(self._voxel_ids), dtype=torch.long, device=self.device)
        else:
            voxel_ids = voxel_ids.to(self.device, dtype=torch.long)

        # --- call VoxelKVStore.retrieve ---
        K_ret, V_ret, M_ret, B_ret = vstore.retrieve(
            voxel_ids=voxel_ids,
            return_buf=return_buf,
        )  # [H,Tm,..]
        # --- backfill to retrieved cache (overwrite) ---
        self._clear_retrieval_cache(slot_idx)
        Tm = K_ret.shape[1]
        if self.retrieval_token_size>=0:
            Tm = min(Tm, self.retrieval_token_size)
            
        if Tm == 0:
            if self.debug:
                logger.debug("[STACVoxelKV:L%d] No pivots retrieved, K_ret=%s, ret_size=%d",
                             layer_idx, K_ret.shape, self.retrieval_token_size)
            return 0, 0
        
        # resize cache
        self.key_cache_retrieval[slot_idx] = K_ret[:, :Tm, :].contiguous() # [H,Tm,D]
        self.value_cache_retrieval[slot_idx] = V_ret[:, :Tm, :].contiguous()
        self._bias_retrieval[slot_idx] = B_ret[:, :Tm].contiguous()
        self._offset_retrieval[slot_idx] = Tm

        max_wrote, total_wrote = 0, 0
        # def _logits_to_count(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        #     """logits: [H,T] float32"""
        #     valid_mask = (logits > -1e10) & mask
        #     counts = torch.zeros_like(logits, dtype=torch.int32)
        #     counts[valid_mask] = torch.exp(logits[valid_mask]).to(torch.int32)
        #     return counts
        
        # C_ret = _logits_to_count(B_ret,M_ret)  # [H,Tm] int32
        # total_wrote = C_ret.sum().item()
        # max_wrote = C_ret.max().item()

        return max_wrote, total_wrote

    # ------------------------ Public API ------------------------
    def get_voxel_info(self) -> Dict[str, torch.Tensor]:
        return self._vm.get_voxel_info()
    
    def get_merger_info(self) -> Dict[str, torch.Tensor]:
        return self._vstore.info()

    def get_retrieval_size(self) -> List[int]:
        return self._offset_retrieval.copy()

    def get_memory_details(self) -> Dict[str, float]:
        """
        Get detailed memory breakdown for all components (in MB).
        
        Returns dict with keys:
            - pinned_memory: memory for pinned tokens (always kept)
            - window_memory: memory for recent/window tokens
            - heavy_hitters_memory: memory for heavy-hitter tokens
            - spatial_cache_usage: memory for retrieved pivots from voxel store
            - voxel_buffer_usage: actual memory used by voxel buffer
            - voxel_buffer_alloc: allocated memory for voxel buffer
            - voxel_pivot_usage: actual memory used by voxel pivots
            - voxel_pivot_alloc: allocated memory for voxel pivots
            - temporal_cache_usage: total actual KV hot cache usage
            - temporal_cache_alloc: total allocated KV hot cache
            - total_usage: total actual memory usage
            - total_alloc: total allocated memory
        """
        # Memory calculation constants
        H, D = self.num_heads, self.head_dim
        bytes_per_elem = torch.finfo(self.dtype).bits // 8
        kv_bytes_per_token = 2 * D * bytes_per_elem  # K and V
        
        # Initialize accumulators
        pinned_memory = 0.0
        window_memory = 0.0
        heavy_hitters_memory = 0.0
        spatial_cache_usage = 0.0
        temporal_cache_usage = 0.0
        temporal_cache_alloc = 0.0
        
        # Calculate per-layer memory
        for l in range(self._L_eff):
            T_live = self._offset_hot[l]
            
            # Pinned tokens (fixed frames that are always kept)
            pinned_count = min(len(self._pinned_token_index), T_live)
            pinned_bytes = H * pinned_count * kv_bytes_per_token
            pinned_memory += pinned_bytes / (1024.0 ** 2)
            
            # Window/recent tokens
            recent_count = min(self.recent_token_size, T_live)
            window_bytes = H * recent_count * kv_bytes_per_token
            window_memory += window_bytes / (1024.0 ** 2)
            
            # Heavy-hitter tokens
            hh_count = min(self.hh_token_size, max(0, T_live - pinned_count - recent_count))
            hh_bytes = H * hh_count * kv_bytes_per_token
            heavy_hitters_memory += hh_bytes / (1024.0 ** 2)
            
            # Retrieval cache (retrieved pivots from voxel store)
            T_ret = self._offset_retrieval[l]
            ret_bytes = H * T_ret * kv_bytes_per_token
            spatial_cache_usage += ret_bytes / (1024.0 ** 2)
            
            # Hot cache (actual usage and allocation)
            hot_usage_bytes = H * T_live * kv_bytes_per_token
            temporal_cache_usage += hot_usage_bytes / (1024.0 ** 2)
            
            hot_alloc_bytes = H * self.reserved_buffer_token_size * kv_bytes_per_token
            temporal_cache_alloc += hot_alloc_bytes / (1024.0 ** 2)
        
        # Voxel store memory (from merger)
        voxel_buffer_usage = 0.0
        voxel_buffer_alloc = 0.0
        voxel_pivot_usage = 0.0
        voxel_pivot_alloc = 0.0
        
        if self._vstore is not None:
            vstore_mem = self._vstore.memory()
            voxel_buffer_usage = vstore_mem.get("buffer_used_mem", 0.0)
            voxel_buffer_alloc = vstore_mem.get("buffer_alloc_mem", 0.0)
            voxel_pivot_usage = vstore_mem.get("pivot_used_mem", 0.0)
            voxel_pivot_alloc = vstore_mem.get("pivot_alloc_mem", 0.0)
        # Total calculations
        total_usage = (temporal_cache_usage + spatial_cache_usage + 
                       voxel_buffer_usage + voxel_pivot_usage)
        total_alloc = (temporal_cache_alloc + spatial_cache_usage + 
                       voxel_buffer_alloc + voxel_pivot_alloc)
        
        return {
            # Component breakdown
            "pinned_memory": pinned_memory,
            "window_memory": window_memory,
            "heavy_hitters_memory": heavy_hitters_memory,
            "spatial_cache_usage": spatial_cache_usage,
            "spatial_cache_alloc": voxel_pivot_usage+voxel_buffer_usage,
            # Voxel store details
            "voxel_buffer_usage": voxel_buffer_usage,
            "voxel_buffer_alloc": voxel_buffer_alloc,
            "voxel_pivot_usage": voxel_pivot_usage,
            "voxel_pivot_alloc": voxel_pivot_alloc,
            # Hot cache totals
            "temporal_cache_usage": temporal_cache_usage,
            "temporal_cache_alloc": temporal_cache_alloc,
            # Grand totals
            "total_usage": total_usage,
            "total_alloc": total_alloc,
        }
    
    def get_memory_usage(self) -> Tuple[float, float]:
        usage_hot, alloc_hot = super().get_memory_usage()
        usage_pool = usage_retrieval = 0.0
        alloc_pool = alloc_retrieval = 0.0
        for l in range(self._L_eff):
            TR = self._offset_retrieval[l]
            H,D = self.num_heads, self.head_dim
            used_mem_bytes_retrieval = (H * TR * D * 2) * \
                                        torch.tensor(torch.finfo(self.dtype).bits // 8)
            usage_retrieval += used_mem_bytes_retrieval.item() / (1024.0 **2)
            alloc_mem_bytes_retrieval = (H * TR * D * 2) * \
                                        torch.tensor(torch.finfo(self.dtype).bits // 8)
            alloc_retrieval += alloc_mem_bytes_retrieval.item() / (1024.0 **2)

            # vstore = self._vstores[l]
            # vstore_mem = vstore.memory()

        alloc = alloc_hot + alloc_pool + alloc_retrieval
        usage = usage_hot + usage_pool + usage_retrieval

        return usage, alloc
    
    def get_memory_voxel(self) -> Tuple[float, float]:
        stats = self._vstore.memory()
        used = stats["buffer_used_mem"] + stats["pivot_used_mem"]
        alloc = stats["buffer_alloc_mem"] + stats["pivot_alloc_mem"]
        return used, alloc
