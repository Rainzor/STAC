# Copyright (c) 2025 STAC Authors. All rights reserved.

from typing import Optional, Tuple, Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import enum
try:
    import open3d as o3d
    import open3d.core as o3c
    _HAS_O3D = True
except Exception:
    _HAS_O3D = False
    print("Warning: Open3D not found; falling back to sorted-array key->id mapping.")


#! ========= GPU-side incremental voxel table manager =========
class BinaryVoxel:
    """
    GPU-side incremental voxel table manager

    Features:
    - Stable voxel IDs (append-only master table)
    - Incremental upsert + binary search lookups
    """

    def __init__(self, voxel_size: float = 0.05, device: str = "cuda"):
        # Convert voxel_size to a torch scalar tensor on the correct device
        self._voxel_size = (
            torch.tensor(voxel_size, dtype=torch.float32, device=device)
            if isinstance(voxel_size, (int, float))
            else voxel_size.to(device, dtype=torch.float32)
        )
        self.device = device

        # Initialize empty tables
        self._voxel_keys = torch.empty(0, 3, dtype=torch.int32, device=device)        # integer voxel grid indices [N,3]
        self._voxel_centers = torch.empty(0, 3, dtype=torch.float32, device=device)   # voxel centers (in world units)
        self._voxel_keys_1d = torch.empty(0, dtype=torch.long, device=device)         # 1D packed keys
        self._voxel_keys_1d_sorted = torch.empty(0, dtype=torch.long, device=device)  # sorted view of 1D keys
        self._voxel_keys_sort_idx = torch.empty(0, dtype=torch.long, device=device)   # sorted→unsorted index map

        self._voxel_zones = torch.empty(0, dtype=torch.int32, device=device)
        self._zone_bbox_min: Optional[torch.Tensor] = None  # [3] int64
        self._zone_bbox_max: Optional[torch.Tensor] = None  # [3] int64

        self._nbr_offsets_1d: Dict[int, torch.Tensor] = {}
        self._nbr_offsets_xyz: Dict[int, torch.Tensor] = {}
        self._nbr_sqdist_lattice: Dict[int, torch.Tensor] = {}
        self._nbr_zero_idx: Dict[int, int] = {}

    def reset(self):
        """Reset voxel table while preserving the same voxel_size and device."""
        self.__init__(voxel_size=self._voxel_size.item(), device=self.device)

    # -------------------------------------------------------------------------
    # Utility functions
    # -------------------------------------------------------------------------
    @staticmethod
    def _pack_keys_1d(ijk: torch.Tensor) -> torch.Tensor:
        """
        Encode (i, j, k) voxel coordinates into a single 1D integer key.
        Each dimension is shifted by +2^20 to ensure non-negative values,
        then bit-packed as: key = i + j*2^21 + k*2^42.
        This allows fast uniqueness and comparison.
        """
        I = ijk.to(torch.int64)
        off = (1 << 20)  # offset to handle negative coordinates
        I += off
        return I[:, 0] + I[:, 1] * (1 << 21) + I[:, 2] * (1 << 42)

    @staticmethod
    def _morton_zone(ijk: torch.Tensor, zone_bits: int = 3,
                     bbox_min: Optional[torch.Tensor] = None,
                     bbox_max: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute Morton (Z-order) zone IDs from integer voxel coordinates.

        Normalizes coordinates to the bounding box ``[bbox_min, bbox_max]``,
        quantizes each axis to ``[0, 2^zone_bits - 1]``, then interleaves in
        Morton order, producing up to ``2^(3*zone_bits)`` distinct zones.

        If no bounding box is given, falls back to a fixed ``2^20`` bias
        (only useful when coordinates span a very wide range).

        Args:
            ijk: [N, 3] int32 voxel grid coordinates.
            zone_bits: Number of bits per axis (default 3 -> 512 zones).
            bbox_min: [3] int64 — per-axis minimum of the coordinate range.
            bbox_max: [3] int64 — per-axis maximum of the coordinate range.

        Returns:
            [N] int32 zone IDs in ``[0, 2^(3*zone_bits))``.
        """
        n_zones = (1 << zone_bits)  # 8 for zone_bits=3
        coords = ijk.to(torch.int64)

        if bbox_min is not None and bbox_max is not None:
            lo = bbox_min.to(coords.device)
            hi = bbox_max.to(coords.device)
            span = (hi - lo).clamp(min=1).float()
            ix = ((coords[:, 0] - lo[0]).float() / span[0] * n_zones).long().clamp(0, n_zones - 1)
            iy = ((coords[:, 1] - lo[1]).float() / span[1] * n_zones).long().clamp(0, n_zones - 1)
            iz = ((coords[:, 2] - lo[2]).float() / span[2] * n_zones).long().clamp(0, n_zones - 1)
        else:
            shift = 20 - zone_bits
            off = 1 << 20
            ix = ((coords[:, 0] + off) >> shift).clamp(0, n_zones - 1)
            iy = ((coords[:, 1] + off) >> shift).clamp(0, n_zones - 1)
            iz = ((coords[:, 2] + off) >> shift).clamp(0, n_zones - 1)
        return (ix | (iy << zone_bits) | (iz << (2 * zone_bits))).to(torch.int32)

    @staticmethod
    def _balanced_morton_zone(ijk: torch.Tensor, num_zones: int = 512,
                              bbox_min: Optional[torch.Tensor] = None,
                              bbox_max: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Balanced Morton zone assignment for non-uniform spatial distributions.

        Instead of quantizing coordinates into a fixed uniform grid (which
        wastes zones on empty space and starves dense regions), this method:
          1. Computes a fine-grained Morton (Z-order) code per voxel
          2. Sorts voxels by Morton code to establish a locality-preserving order
          3. Splits the sorted sequence into ``num_zones`` groups of roughly
             equal size so that each zone receives a similar number of voxels

        Dense regions naturally receive more zones (finer granularity), while
        sparse regions share fewer zones (coarser granularity).

        Args:
            ijk: [N, 3] int32 voxel grid coordinates.
            num_zones: Target number of zones (default 512).
            bbox_min: [3] int64 — per-axis minimum (used for normalization).
            bbox_max: [3] int64 — per-axis maximum (used for normalization).

        Returns:
            [N] int32 zone IDs in ``[0, num_zones)``.
        """
        N = ijk.shape[0]
        if N == 0:
            return torch.empty(0, dtype=torch.int32, device=ijk.device)
        if N <= num_zones:
            return torch.arange(N, dtype=torch.int32, device=ijk.device)

        BITS = 10
        SCALE = 1 << BITS
        coords = ijk.to(torch.int64)

        if bbox_min is not None and bbox_max is not None:
            lo = bbox_min.to(coords.device).float()
            hi = bbox_max.to(coords.device).float()
            span = (hi - lo).clamp(min=1.0)
            x = ((coords[:, 0].float() - lo[0]) / span[0] * SCALE).long().clamp(0, SCALE - 1)
            y = ((coords[:, 1].float() - lo[1]) / span[1] * SCALE).long().clamp(0, SCALE - 1)
            z = ((coords[:, 2].float() - lo[2]) / span[2] * SCALE).long().clamp(0, SCALE - 1)
        else:
            shift = 21 - BITS
            off = 1 << 20
            x = ((coords[:, 0] + off) >> shift).clamp(0, SCALE - 1)
            y = ((coords[:, 1] + off) >> shift).clamp(0, SCALE - 1)
            z = ((coords[:, 2] + off) >> shift).clamp(0, SCALE - 1)

        morton = torch.zeros(N, dtype=torch.int64, device=ijk.device)
        for b in range(BITS):
            morton |= ((x >> b) & 1) << (3 * b)
            morton |= ((y >> b) & 1) << (3 * b + 1)
            morton |= ((z >> b) & 1) << (3 * b + 2)

        sorted_indices = torch.argsort(morton)
        zone_ids = torch.empty(N, dtype=torch.int32, device=ijk.device)
        zone_ids[sorted_indices] = (torch.arange(N, device=ijk.device) * num_zones // N).to(torch.int32)
        return zone_ids

    @staticmethod
    def _first_occurrence_indices(labels: torch.Tensor) -> torch.Tensor:
        """
        Given a 1D integer tensor 'labels' (values 0..L-1),
        return the index of the first occurrence of each unique label
        in the *original* sequence.

        Implementation:
        - Sort labels stably (so equal values keep their order)
        - Detect where value changes
        - Return indices of first elements for each unique segment
        """
        order = torch.argsort(labels, stable=True)
        labels_sorted = labels[order]
        change = torch.ones(labels_sorted.numel(), dtype=torch.bool, device=labels.device)
        if labels_sorted.numel() > 1:
            change[1:] = labels_sorted[1:] != labels_sorted[:-1]
        return order[change]  # indices of first occurrences (aligned with unique ascending order)

    # -------------------------------------------------------------------------
    #! Core operation: incremental insertion (upsert)
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def upsert(self, ijk_valid: torch.Tensor) -> torch.Tensor:
        """
        Incrementally insert voxel coordinates and return their stable voxel IDs.

        Args:
            ijk_valid: [N, 3] int32 — integer voxel grid coordinates

        Returns:
            voxel_id_valid: [N] int32 — stable voxel IDs (row indices in master table)
        """
        if ijk_valid.numel() == 0:
            return torch.empty(0, dtype=torch.int32, device=self.device)

        assert ijk_valid.shape[-1] == 3, "ijk_valid must be [N,3]"
        device = self.device

        # Convert 3D voxel indices to unique 1D encoded keys
        k_new = self._pack_keys_1d(ijk_valid)  # [N], int64

        # -----------------------------------------------------------------
        # 1) Initialization (first frame)
        # -----------------------------------------------------------------
        if self._voxel_keys.numel() == 0:
            # Find unique keys and inverse mapping to group identical voxels
            uniq, inverse = torch.unique(k_new, sorted=True, return_inverse=True)

            # Initialize key tables
            self._voxel_keys_1d = uniq
            self._voxel_keys_1d_sorted = uniq
            self._voxel_keys_sort_idx = torch.arange(uniq.numel(), device=device, dtype=torch.long)

            # Select first occurrence of each voxel from input ijk
            first_idx = self._first_occurrence_indices(inverse)
            ijk_base = ijk_valid.index_select(0, first_idx)
            self._voxel_keys = ijk_base.to(torch.int32)

            # Compute voxel centers (world coordinates)
            self._voxel_centers = (self._voxel_keys.to(torch.float32) + 0.5) * self._voxel_size

            # Establish bounding box and compute Morton zones for the initial voxels
            ijk_i64 = self._voxel_keys.to(torch.int64)
            margin = (ijk_i64.max(0).values - ijk_i64.min(0).values).clamp(min=1) // 5 + 1
            self._zone_bbox_min = ijk_i64.min(0).values - margin
            self._zone_bbox_max = ijk_i64.max(0).values + margin
            self._voxel_zones = self._balanced_morton_zone(
                self._voxel_keys, num_zones=512,
                bbox_min=self._zone_bbox_min, bbox_max=self._zone_bbox_max)

            # Return stable IDs directly (inverse maps to uniq index)
            return inverse.to(torch.int32)

        # -----------------------------------------------------------------
        # 2) Lookup: find which voxels already exist
        # -----------------------------------------------------------------
        if self._voxel_keys_1d_sorted.numel() == 0:
            # Build sorted cache for binary search if missing
            sorted_keys, sort_idx = torch.sort(self._voxel_keys_1d)
            self._voxel_keys_1d_sorted = sorted_keys
            self._voxel_keys_sort_idx = sort_idx  # map sorted -> unsorted (stable ID space)

        # Locate insertion positions via binary search
        pos = torch.searchsorted(self._voxel_keys_1d_sorted, k_new)
        n_sorted = int(self._voxel_keys_1d_sorted.numel())
        if n_sorted > 0:
            pos = torch.clamp(pos, 0, n_sorted - 1)

        # Determine matches (already existing voxels)
        match = (self._voxel_keys_1d_sorted[pos] == k_new)
        old_ids = (
            self._voxel_keys_sort_idx[pos[match]] if match.any()
            else torch.empty(0, dtype=torch.long, device=device)
        )

        # -----------------------------------------------------------------
        # 3) Insert new voxels (append-only update)
        # -----------------------------------------------------------------
        new_mask = ~match
        voxel_id_valid = torch.empty(k_new.shape[0], dtype=torch.int32, device=device)

        if new_mask.any():
            # Find unique new voxel keys and their grouping
            k_new_only, inverse_new_only = torch.unique(k_new[new_mask], sorted=True, return_inverse=True)
            start = self._voxel_keys_1d.numel()  # current number of voxels

            # Append new 1D keys to master table
            self._voxel_keys_1d = torch.cat([self._voxel_keys_1d, k_new_only], dim=0)

            # Pick one representative ijk for each new voxel
            first_idx_new = self._first_occurrence_indices(inverse_new_only)
            ijk_new_only = ijk_valid[new_mask].index_select(0, first_idx_new)

            # Append to geometric table
            self._voxel_keys = torch.cat([self._voxel_keys, ijk_new_only.to(torch.int32)], dim=0)

            # Compute and append corresponding voxel centers
            centers_new = (ijk_new_only.to(torch.float32) + 0.5) * self._voxel_size
            self._voxel_centers = torch.cat([self._voxel_centers, centers_new], dim=0)

            # Expand bounding box if new voxels exceed current range
            ijk_new_i64 = ijk_new_only.to(torch.int64)
            new_min = ijk_new_i64.min(0).values
            new_max = ijk_new_i64.max(0).values
            if self._zone_bbox_min is not None:
                if (new_min < self._zone_bbox_min).any() or (new_max > self._zone_bbox_max).any():
                    self._zone_bbox_min = torch.minimum(self._zone_bbox_min, new_min)
                    self._zone_bbox_max = torch.maximum(self._zone_bbox_max, new_max)

            # Balanced partitioning: always recompute all zones since the
            # partition depends on the global voxel distribution.
            self._voxel_zones = self._balanced_morton_zone(
                self._voxel_keys, num_zones=512,
                bbox_min=self._zone_bbox_min, bbox_max=self._zone_bbox_max)

            # Rebuild sorted cache for next lookup
            sorted_keys, sort_idx = torch.sort(self._voxel_keys_1d)
            self._voxel_keys_1d_sorted = sorted_keys
            self._voxel_keys_sort_idx = sort_idx

            # Compute stable voxel IDs for new entries
            pos_new_only = torch.searchsorted(k_new_only, k_new[new_mask])
            voxel_id_valid[new_mask] = (start + pos_new_only).to(torch.int32)

        # -----------------------------------------------------------------
        # 4) Fill existing voxel IDs
        # -----------------------------------------------------------------
        if match.any():
            voxel_id_valid[match] = old_ids.to(torch.int32)

        return voxel_id_valid

    # =============== neighbor offset cache ===============
    def _ensure_neighbor_offset_cache(self, R: int):
        """
        Precompute and cache all 3D lattice neighbor offsets for Chebyshev radius R:
          Off = {(dx,dy,dz) | dx,dy,dz in [-R,R]}
        Provide:
          - self._nbr_offsets_1d[R]:   [M]  int64 packed offsets (add to 1D key)
          - self._nbr_offsets_xyz[R]:  [M,3] int32 lattice deltas (dx,dy,dz)
          - self._nbr_sqdist_lattice[R]: [M] float32 (dx^2+dy^2+dz^2)
          - self._nbr_zero_idx[R]: index of (dx,dy,dz)=(0,0,0) in the above (for excluding self)
        """
        if not hasattr(self, "_nbr_offsets_1d"):
            self._nbr_offsets_1d: Dict[int, torch.Tensor] = {}
            self._nbr_offsets_xyz: Dict[int, torch.Tensor] = {}
            self._nbr_sqdist_lattice: Dict[int, torch.Tensor] = {}
            self._nbr_zero_idx: Dict[int, int] = {}

        if R in self._nbr_offsets_1d:
            return

        device = self.device
        sY = (1 << 21)
        sZ = (1 << 42)

        # generate integer offsets for [-R, R]^3 (including self 0,0,0)
        rng = torch.arange(-R, R + 1, device=device, dtype=torch.int32)
        dx, dy, dz = torch.meshgrid(rng, rng, rng, indexing='ij')   # [2R+1]^3
        xyz = torch.stack([dx, dy, dz], dim=-1).reshape(-1, 3).contiguous()  # [M,3]
        # 1D additive offsets
        off1d = (xyz[:, 0].to(torch.int64)
                 + xyz[:, 1].to(torch.int64) * sY
                 + xyz[:, 2].to(torch.int64) * sZ)  # [M]
        # lattice squared distance (Euclidean distance up to voxel_size^2 scale)
        sqd_lat = (xyz.to(torch.float32) ** 2).sum(dim=-1)  # [M]

        # index of zero offset
        zero_idx = torch.where((xyz == 0).all(dim=-1))[0]
        zero_idx = int(zero_idx.item()) if zero_idx.numel() > 0 else None

        self._nbr_offsets_1d[R] = off1d
        self._nbr_offsets_xyz[R] = xyz
        self._nbr_sqdist_lattice[R] = sqd_lat
        self._nbr_zero_idx[R] = zero_idx

    # =============== batch key -> id lookup (missing returns -1) ===============
    @torch.no_grad()
    def _lookup_ids_by_keys(self, keys_1d: torch.Tensor) -> torch.Tensor:
        """
        keys_1d: [...], int64
        Returns: same shape int32; stable voxel_id where present, -1 where missing
        """
        if keys_1d.numel() == 0:
            return torch.empty_like(keys_1d, dtype=torch.int32)

        # ensure sorted cache
        if self._voxel_keys_1d_sorted.numel() == 0 and self._voxel_keys_1d.numel() > 0:
            sorted_keys, sort_idx = torch.sort(self._voxel_keys_1d)
            self._voxel_keys_1d_sorted = sorted_keys
            self._voxel_keys_sort_idx = sort_idx

        if self._voxel_keys_1d_sorted.numel() == 0:
            # master table empty; return all -1
            return torch.full_like(keys_1d, -1, dtype=torch.int32)

        # vectorized lookup
        pos = torch.searchsorted(self._voxel_keys_1d_sorted, keys_1d)
        pos = pos.clamp_(0, self._voxel_keys_1d_sorted.numel() - 1)
        match = (self._voxel_keys_1d_sorted.index_select(0, pos) == keys_1d)

        out = torch.full_like(keys_1d, -1, dtype=torch.int32)
        if match.any():
            ids = self._voxel_keys_sort_idx.index_select(0, pos[match]).to(torch.int32)
            out[match] = ids
        return out

    # =============== main API: k nearest neighbors by voxel_id (Euclidean distance) ===============
    @torch.no_grad()
    def knn_by_id(self,
                  voxel_ids: torch.Tensor,     # [Q] int32
                  k: int,
                  include_self: bool = False,
                  max_radius_cells: Optional[int] = None,
                  return_squared_dist: bool = True
                  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Given query voxel_id, return its top-k neighbors (Euclidean distance).
        Uses Chebyshev cube neighborhood [-R,R]^3 as candidates; vectorized lookup + distance filter.
        - If max_radius_cells not set, choose minimal R so (2R+1)^3 >= k + (include_self?1:0) + margin.
        - If a query has fewer than k existing neighbors at that R, fill with -1 and +inf.

        Returns:
            nbr_ids:  [Q, k] int32   (missing = -1)
            nbr_dist: [Q, k] float32 (distance; d^2 if return_squared_dist=True)
            found:    [Q]   int32    (actual neighbor count per query, excluding self)
        """
        assert voxel_ids.dtype in (torch.int32, torch.int64), "voxel_ids must be int32/64"
        device = self.device
        Q = int(voxel_ids.numel())
        if Q == 0 or k <= 0:
            return (torch.empty(Q, 0, dtype=torch.int32, device=device),
                    torch.empty(Q, 0, dtype=torch.float32, device=device),
                    torch.zeros(Q, dtype=torch.int32, device=device))

        # choose R
        if max_radius_cells is None:
            # minimal R so candidate lattice count >= k + (include_self?1:0)
            need = k + (1 if include_self else 0)
            R = int(torch.ceil(((torch.tensor(float(need), device=device) ** (1.0 / 3.0)) - 1.0) / 2.0).item())
            R = max(1, R)  # at least 1
        else:
            R = int(max_radius_cells)
            R = max(1, R)

        self._ensure_neighbor_offset_cache(R)
        off1d = self._nbr_offsets_1d[R]               # [M]
        sqd_lat = self._nbr_sqdist_lattice[R]         # [M]
        zero_idx = self._nbr_zero_idx[R]              # int
        M = int(off1d.numel())

        # query keys
        q_ids = voxel_ids.to(torch.int64)
        # bounds check
        valid_q = (q_ids >= 0) & (q_ids < self._voxel_keys_1d.numel())
        keys_q = torch.empty(Q, dtype=torch.int64, device=device)
        keys_q[valid_q] = self._voxel_keys_1d[q_ids[valid_q]]
        keys_q[~valid_q] = -9223372036854775808  # invalid key (no match)

        # candidate key matrix [Q, M]
        cand_keys = keys_q.view(Q, 1) + off1d.view(1, M)  # broadcast add

        # batch lookup ids -> [Q, M]
        cand_ids = self._lookup_ids_by_keys(cand_keys.view(-1)).view(Q, M)  # int32
        # distance matrix (meters); missing entries +inf
        #   Euclidean^2 = (dx^2+dy^2+dz^2) * voxel_size^2
        vs = float(self._voxel_size.item())
        d2 = sqd_lat.view(1, M) * (vs * vs)           # [1, M] -> [Q, M] broadcast
        d2 = d2.expand(Q, M).clone()

        # missing candidates -> +inf
        missing = (cand_ids < 0)
        d2[missing] = float('inf')

        # exclude self
        if not include_self and (zero_idx is not None):
            d2[:, zero_idx] = float('inf')

        # top-k nearest (smallest distance)
        # if fewer than k valid neighbors, topk returns +inf; we set those ids to -1
        top_vals, top_idx = torch.topk(d2, k=min(k, M), dim=1, largest=False, sorted=True)  # [Q, k]
        nbr_ids = cand_ids.gather(1, top_idx)  # [Q, k]

        # mark +inf positions as -1
        inf_mask = torch.isinf(top_vals)
        if inf_mask.any():
            nbr_ids = nbr_ids.masked_fill(inf_mask, -1)

        # actual neighbor count per query (excluding self)
        found = (~inf_mask).sum(dim=1).to(torch.int32)

        if return_squared_dist:
            nbr_dist = top_vals.to(torch.float32)
        else:
            # sqrt preserves +inf
            nbr_dist = torch.sqrt(top_vals).to(torch.float32)

        return nbr_ids.to(torch.int32), nbr_dist, found

    # =============== Public API: radius in meters (return all ids within radius, up to M) ===============
    @torch.no_grad()
    def neighbors(self,voxel_ids: torch.Tensor,
                    radius_m: float,
                    include_self: bool = False
                    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Given radius in meters, set Chebyshev radius R = ceil(radius_m / voxel_size)
        and return all existing voxels in that cube (cap M=(2R+1)^3).
        Returns:
            nbr_ids:  [Q, M]  (missing = -1; self position -1 if not include_self)
            dist2:    [Q, M]  (meters^2, missing = +inf)
            found:    [Q]     actual count (excluding self)
        """
        vs = float(self._voxel_size.item())
        R = max(1, int(torch.ceil(torch.tensor(radius_m / vs)).item()))
        self._ensure_neighbor_offset_cache(R)

        off1d = self._nbr_offsets_1d[R]
        M = int(off1d.numel())
        # use knn_by_id with k=M to get all valid candidates
        nbr_ids, dist2, found = self.knn_by_id(
            voxel_ids=voxel_ids,
            k=M,
            include_self=include_self,
            max_radius_cells=R,
            return_squared_dist=True
        )
        return nbr_ids, dist2, found

    # -------------------------------------------------------------------------
    # Accessors
    # -------------------------------------------------------------------------
    def get_centers(self, voxel_ids: torch.Tensor) -> torch.Tensor:
        """Return voxel center coordinates for given voxel IDs."""
        return self._voxel_centers[voxel_ids]

    def get_voxel_keys(self) -> torch.Tensor:
        """Return all voxel integer coordinates [N,3]."""
        return self._voxel_keys
    
    def get_voxel_num(self) -> int:
        """Return the total number of voxels stored."""
        return self._voxel_keys.shape[0]
    
    def get_voxel_size(self) -> torch.Tensor:
        """Return the voxel size tensor."""
        return self._voxel_size.item()

    def get_voxel_info(self) -> Dict[str, torch.Tensor]:
        """Return all internal voxel-related tensors as a dictionary."""
        return {
            "voxel_size": self._voxel_size,
            "voxel_keys": self._voxel_keys,
            "voxel_centers": self._voxel_centers,
            "voxel_keys_1d": self._voxel_keys_1d,
            "voxel_keys_1d_sorted": self._voxel_keys_1d_sorted,
            "voxel_keys_sort_idx": self._voxel_keys_sort_idx,
        }

    # -------------------------------------------------------------------------
    # Debug / summary
    # -------------------------------------------------------------------------
    def summary(self):
        """Print basic information about current voxel table."""
        print("=== Voxel Summary ===")
        print(f"Device: {self.device}")
        print(f"Voxel size: {float(self._voxel_size.item()):.4f}")
        print(f"Total voxels: {self._voxel_keys.shape[0]}")
        ok_sorted = (
            (self._voxel_keys_1d.numel() == 0)
            or torch.all(self._voxel_keys_1d.sort().values == self._voxel_keys_1d_sorted)
        )
        print(f"Sorted cache valid: {bool(ok_sorted)}")
