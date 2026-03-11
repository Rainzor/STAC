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

    # =============== 邻域 offset 缓存 ===============
    def _ensure_neighbor_offset_cache(self, R: int):
        """
        Precompute and cache all 3D lattice neighbor offsets for Chebyshev radius R:
          Off = {(dx,dy,dz) | dx,dy,dz in [-R,R]}
        Provide:
          - self._nbr_offsets_1d[R]:   [M]  int64 packed offsets (add to 1D key)
          - self._nbr_offsets_xyz[R]:  [M,3] int32 lattice deltas (dx,dy,dz)
          - self._nbr_sqdist_lattice[R]: [M] float32 (dx^2+dy^2+dz^2)
          - self._nbr_zero_idx[R]: index of (dx,dy,dz)=(0,0,0) in the above (for剔除自身)
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

        # 生成 [-R, R]^3 的整点偏移（包含自身 0,0,0）
        rng = torch.arange(-R, R + 1, device=device, dtype=torch.int32)
        dx, dy, dz = torch.meshgrid(rng, rng, rng, indexing='ij')   # [2R+1]^3
        xyz = torch.stack([dx, dy, dz], dim=-1).reshape(-1, 3).contiguous()  # [M,3]
        # 1D 可加性偏移
        off1d = (xyz[:, 0].to(torch.int64)
                 + xyz[:, 1].to(torch.int64) * sY
                 + xyz[:, 2].to(torch.int64) * sZ)  # [M]
        # 格点平方距离（与实际欧氏距离只差一个 voxel_size^2 的缩放）
        sqd_lat = (xyz.to(torch.float32) ** 2).sum(dim=-1)  # [M]

        # 记录 0 偏移的索引
        zero_idx = torch.where((xyz == 0).all(dim=-1))[0]
        zero_idx = int(zero_idx.item()) if zero_idx.numel() > 0 else None

        self._nbr_offsets_1d[R] = off1d
        self._nbr_offsets_xyz[R] = xyz
        self._nbr_sqdist_lattice[R] = sqd_lat
        self._nbr_zero_idx[R] = zero_idx

    # =============== 新增：批量 key → id 查找（缺失返回 -1） ===============
    @torch.no_grad()
    def _lookup_ids_by_keys(self, keys_1d: torch.Tensor) -> torch.Tensor:
        """
        keys_1d: [...], int64
        返回: 同形状 int32，存在的为稳定 voxel_id，不存在的为 -1
        """
        if keys_1d.numel() == 0:
            return torch.empty_like(keys_1d, dtype=torch.int32)

        # 保证有排序缓存
        if self._voxel_keys_1d_sorted.numel() == 0 and self._voxel_keys_1d.numel() > 0:
            sorted_keys, sort_idx = torch.sort(self._voxel_keys_1d)
            self._voxel_keys_1d_sorted = sorted_keys
            self._voxel_keys_sort_idx = sort_idx

        if self._voxel_keys_1d_sorted.numel() == 0:
            # 主表为空，全部 -1
            return torch.full_like(keys_1d, -1, dtype=torch.int32)

        # 向量化定位
        pos = torch.searchsorted(self._voxel_keys_1d_sorted, keys_1d)
        pos = pos.clamp_(0, self._voxel_keys_1d_sorted.numel() - 1)
        match = (self._voxel_keys_1d_sorted.index_select(0, pos) == keys_1d)

        out = torch.full_like(keys_1d, -1, dtype=torch.int32)
        if match.any():
            ids = self._voxel_keys_sort_idx.index_select(0, pos[match]).to(torch.int32)
            out[match] = ids
        return out

    # =============== 主接口：按 voxel_id 查询 k 邻居（欧氏距离） ===============
    @torch.no_grad()
    def knn_by_id(self,
                  voxel_ids: torch.Tensor,     # [Q] int32
                  k: int,
                  include_self: bool = False,
                  max_radius_cells: Optional[int] = None,
                  return_squared_dist: bool = True
                  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        给定查询 voxel_id，返回其周围 Top-k 邻居（欧氏距离）。
        采用 Chebyshev 立方邻域 [-R,R]^3 做候选，向量化查表 + 距离筛选。
        - 如果 max_radius_cells 未指定，则按 k 自动选择最小 R，使 (2R+1)^3 >= k + (include_self?1:0) + 余裕。
        - 若某些查询在该 R 下不足 k 个存在邻居，则这些位置用 -1 填充，距离用 +inf。

        Returns:
            nbr_ids:  [Q, k] int32   （不存在的为 -1）
            nbr_dist: [Q, k] float32 （对应距离；若 return_squared_dist=True，则为 d^2）
            found:    [Q]   int32    （每个查询实际找到的邻居数量，不含自身）
        """
        assert voxel_ids.dtype in (torch.int32, torch.int64), "voxel_ids must be int32/64"
        device = self.device
        Q = int(voxel_ids.numel())
        if Q == 0 or k <= 0:
            return (torch.empty(Q, 0, dtype=torch.int32, device=device),
                    torch.empty(Q, 0, dtype=torch.float32, device=device),
                    torch.zeros(Q, dtype=torch.int32, device=device))

        # 计算合适的 R
        if max_radius_cells is None:
            # 最小 R 使候选格点数 >= k + (是否包含自身)
            need = k + (1 if include_self else 0)
            # ceil(((need)^(1/3) - 1) / 2)
            R = int(torch.ceil(((torch.tensor(float(need), device=device) ** (1.0 / 3.0)) - 1.0) / 2.0).item())
            R = max(1, R)  # 至少 1
        else:
            R = int(max_radius_cells)
            R = max(1, R)

        self._ensure_neighbor_offset_cache(R)
        off1d = self._nbr_offsets_1d[R]               # [M]
        sqd_lat = self._nbr_sqdist_lattice[R]         # [M]
        zero_idx = self._nbr_zero_idx[R]              # int
        M = int(off1d.numel())

        # 查询键
        q_ids = voxel_ids.to(torch.int64)
        # 越界保护
        valid_q = (q_ids >= 0) & (q_ids < self._voxel_keys_1d.numel())
        keys_q = torch.empty(Q, dtype=torch.int64, device=device)
        keys_q[valid_q] = self._voxel_keys_1d[q_ids[valid_q]]
        keys_q[~valid_q] = -9223372036854775808  # 无效 key（不会匹配）

        # 候选键矩阵 [Q, M]
        cand_keys = keys_q.view(Q, 1) + off1d.view(1, M)  # 广播加法

        # 一次性查 id，得到 [Q, M]
        cand_ids = self._lookup_ids_by_keys(cand_keys.view(-1)).view(Q, M)  # int32
        # 距离矩阵（单位：米），缺失项设为 +inf
        #   欧氏距离^2 = (dx^2+dy^2+dz^2) * voxel_size^2
        vs = float(self._voxel_size.item())
        d2 = sqd_lat.view(1, M) * (vs * vs)                      # [1, M] → [Q, M] 广播
        d2 = d2.expand(Q, M).clone()

        # 不存在的候选 → +inf
        missing = (cand_ids < 0)
        d2[missing] = float('inf')

        # 是否剔除自身
        if not include_self and (zero_idx is not None):
            d2[:, zero_idx] = float('inf')

        # 选取 top-k 最近（距离最小）
        # 注意：如果有效邻居少于 k，topk 会返回若干 +inf；我们随后把 +inf 的 id 置 -1
        top_vals, top_idx = torch.topk(d2, k=min(k, M), dim=1, largest=False, sorted=True)  # [Q, k]
        nbr_ids = cand_ids.gather(1, top_idx)  # [Q, k]

        # 把 +inf 的位置标记为 -1
        inf_mask = torch.isinf(top_vals)
        if inf_mask.any():
            nbr_ids = nbr_ids.masked_fill(inf_mask, -1)

        # 每个查询实际找到的邻居数（不含自身）
        found = (~inf_mask).sum(dim=1).to(torch.int32)

        if return_squared_dist:
            nbr_dist = top_vals.to(torch.float32)
        else:
            # 开根号时保持 +inf 仍为 +inf
            nbr_dist = torch.sqrt(top_vals).to(torch.float32)

        return nbr_ids.to(torch.int32), nbr_dist, found

    # =============== Public API：按「米」设置半径（返回半径内全部 id，最多 M 个） ===============
    @torch.no_grad()
    def neighbors(self,voxel_ids: torch.Tensor,
                    radius_m: float,
                    include_self: bool = False
                    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        给定以米为单位的半径，枚举 Chebyshev 半径 R = ceil(radius_m / voxel_size)，
        返回该立方邻域内的所有存在体素（上限 M=(2R+1)^3）。
        返回：
            nbr_ids:  [Q, M]  （不存在为 -1；若不含自身，自身距离位置会是 -1）
            dist2:    [Q, M]  （米^2，缺失为 +inf）
            found:    [Q]     实际存在数量（不含自身）
        """
        vs = float(self._voxel_size.item())
        R = max(1, int(torch.ceil(torch.tensor(radius_m / vs)).item()))
        self._ensure_neighbor_offset_cache(R)

        off1d = self._nbr_offsets_1d[R]
        M = int(off1d.numel())
        # 借助 knn_by_id 的内部流程：用 k=M 即获取“所有候选中的有效者”
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
        print("=== HashVoxel Summary ===")
        print(f"Device: {self.device}")
        print(f"Voxel size: {float(self._voxel_size.item()):.4f}")
        print(f"Total voxels: {self._voxel_keys.shape[0]}")
        ok_sorted = (
            (self._voxel_keys_1d.numel() == 0)
            or torch.all(self._voxel_keys_1d.sort().values == self._voxel_keys_1d_sorted)
        )
        print(f"Sorted cache valid: {bool(ok_sorted)}")


from typing import Optional, Tuple, Union, Dict
import torch

class MortonVoxel:
    """
    Minimal Morton-ID voxel table (axis_bits=16):
      - Stable voxel ID == Morton(Z-order) code (stored in int64, each axis uses 16 bits)
      - Keep ONLY 'ever-seen' morton codes tensor on device
      - Upsert returns morton codes; Neighbor search by (code or ijk) + radius

    Notes
    -----
    * Why int64 for IDs?
      We interleave 3 axes * 16 bits = 48 bits. Using int64避免碰撞且便于位运算。
      若强制用 uint16 存 ID，最大仅 65536，不满足你给的上限 (~131072)。
    """

    # ---------------- Morton helpers (支持到 axis_bits <= 21，取 16 也可) ----------------
    @staticmethod
    def _part1by2(x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.int64)
        x = (x | (x << 32)) & 0x1f00000000ffff
        x = (x | (x << 16)) & 0x1f0000ff0000ff
        x = (x | (x << 8))  & 0x100f00f00f00f00f
        x = (x | (x << 4))  & 0x10c30c30c30c30c3
        x = (x | (x << 2))  & 0x1249249249249249
        return x

    @staticmethod
    def _compact1by2(x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.int64) & 0x1249249249249249
        x = (x ^ (x >> 2))  & 0x10c30c30c30c30c3
        x = (x ^ (x >> 4))  & 0x100f00f00f00f00f
        x = (x ^ (x >> 8))  & 0x1f0000ff0000ff
        x = (x ^ (x >> 16)) & 0x1f00000000ffff
        x = (x ^ (x >> 32)) & 0x00000000001fffff
        return x.to(torch.int32)

    @classmethod
    def _morton_encode3d_ix(cls, ix: torch.Tensor, iy: torch.Tensor, iz: torch.Tensor) -> torch.Tensor:
        return (cls._part1by2(ix) | (cls._part1by2(iy) << 1) | (cls._part1by2(iz) << 2)).to(torch.int64)

    @classmethod
    def _morton_decode3d_ix(cls, code: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ix = cls._compact1by2(code)
        iy = cls._compact1by2(code >> 1)
        iz = cls._compact1by2(code >> 2)
        return ix, iy, iz

    # ---------------- Core ----------------
    def __init__(self,
                 voxel_size: float = 0.05,
                 device: str = "cuda",
                 axis_bits: int = 16,
                 bias: Union[int, Tuple[int, int, int], None] = None,
                 capacity_limit: int = (2 << 16)):  # ~= 131072
        self.device = device
        self._voxel_size = (
            torch.tensor(voxel_size, dtype=torch.float32, device=device)
            if isinstance(voxel_size, (int, float))
            else voxel_size.to(device, dtype=torch.float32)
        )

        # 编码位数固定为 16（每轴）
        self._bits = int(axis_bits)
        assert self._bits == 16, "按需求固定 axis_bits=16（每轴 16bit Morton 编码）"
        if bias is None:
            bias = 1 << (self._bits - 1)  # 对称默认偏置，避免负 ijk
        if isinstance(bias, int):
            bias = (bias, bias, bias)
        self._bias = torch.tensor(bias, dtype=torch.int32, device=self.device)
        self._axis_max = (1 << self._bits) - 1

        # “曾经存在过”的 Morton code 集合（唯一集）
        self._voxel_morton = torch.empty(0, dtype=torch.int64, device=device)

        # 邻域偏移缓存：仅保留 xyz 与格点距离
        self._nbr_offsets_xyz: Dict[int, torch.Tensor] = {}
        self._nbr_sqdist_lattice: Dict[int, torch.Tensor] = {}
        self._nbr_zero_idx: Dict[int, int] = {}

        self._capacity_limit = int(capacity_limit)

    # -------- encode/decode with clamping & bias --------
    def _encode_ijk_to_code(self, ijk: torch.Tensor) -> torch.Tensor:
        """ijk: [N,3] int32/64 -> int64 code"""
        I = ijk.to(torch.int32) + self._bias.view(1, 3)
        I = torch.clamp(I, 0, self._axis_max)
        return self._morton_encode3d_ix(I[:, 0], I[:, 1], I[:, 2])

    def _decode_code_to_ijk(self, codes: torch.Tensor) -> torch.Tensor:
        ix, iy, iz = self._morton_decode3d_ix(codes.to(torch.int64))
        ijk_biased = torch.stack([ix, iy, iz], dim=-1)  # [N,3] int32
        return (ijk_biased - self._bias.view(1, 3)).to(torch.int32)

    # -------- presence helpers --------
    @torch.no_grad()
    def _present_mask(self, cand_codes: torch.Tensor) -> torch.Tensor:
        if self._voxel_morton.numel() == 0 or cand_codes.numel() == 0:
            return torch.zeros_like(cand_codes, dtype=torch.bool)
        return torch.isin(cand_codes, self._voxel_morton)

    # -------- upsert --------
    @torch.no_grad()
    def upsert(self, ijk: torch.Tensor) -> torch.Tensor:
        """
        Insert ijk (int coords) if new; return their morton codes (int64).
        Only the ever-seen 'code set' is maintained.
        """
        if ijk.numel() == 0:
            return torch.empty(0, dtype=torch.int64, device=self.device)

        assert ijk.shape[-1] == 3, "ijk must be [N,3]"
        codes = self._encode_ijk_to_code(ijk)

        # 以“并集”的方式更新唯一集合（保持旧序优先）
        if self._voxel_morton.numel() == 0:
            uniq = codes.unique(sorted=False)
        else:
            uniq = torch.unique(torch.cat([self._voxel_morton, codes], dim=0), sorted=False)

        # 容量保护（与需求 1 对齐）
        if uniq.numel() > self._capacity_limit:
            raise RuntimeError(
                f"voxel_num ({uniq.numel()}) exceeds limit ({self._capacity_limit}). "
                "Consider increasing capacity_limit or using a larger voxel_size."
            )
        self._voxel_morton = uniq
        return codes  # 直接把 morton code 作为稳定 id

    # -------- neighbor offsets cache --------
    def _ensure_neighbor_offset_cache(self, R: int):
        if R in self._nbr_offsets_xyz:
            return
        device = self.device
        rng = torch.arange(-R, R + 1, device=device, dtype=torch.int32)
        dx, dy, dz = torch.meshgrid(rng, rng, rng, indexing='ij')   # [2R+1]^3
        xyz = torch.stack([dx, dy, dz], dim=-1).reshape(-1, 3).contiguous()  # [M,3]
        sqd = (xyz.to(torch.float32) ** 2).sum(dim=-1)  # [M]
        zero_idx = torch.where((xyz == 0).all(dim=-1))[0]
        zero_idx = int(zero_idx.item()) if zero_idx.numel() > 0 else None
        self._nbr_offsets_xyz[R] = xyz
        self._nbr_sqdist_lattice[R] = sqd
        self._nbr_zero_idx[R] = zero_idx

    # -------- neighbors (by code OR ijk, radius in cells or meters) --------
    @torch.no_grad()
    def neighbors(self,
                  query: torch.Tensor,                 # [Q] morton codes OR [Q,3] ijk
                  radius_cells: Optional[int] = None,
                  radius_m: Optional[float] = None,
                  include_self: bool = False,
                  return_squared_dist: bool = True
                  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return neighbor Morton codes around each query within a cubic radius.

        Output:
          nbr_codes:  [Q, Mq] int64  (padding with -1)
          nbr_dist2:  [Q, Mq] float32 (meters^2 or meters)
          found_cnt:  [Q]     int32   (#valid neighbors per query)
        """
        device = self.device
        if query.numel() == 0:
            return (torch.empty(0, 0, dtype=torch.int64, device=device),
                    torch.empty(0, 0, dtype=torch.float32, device=device),
                    torch.empty(0, dtype=torch.int32, device=device))

        # 规范半径
        if radius_cells is None:
            assert radius_m is not None, "radius_cells 或 radius_m 至少提供一个"
            vs = float(self._voxel_size.item())
            R = max(1, int(torch.ceil(torch.tensor(radius_m / vs)).item()))
        else:
            R = max(1, int(radius_cells))
        self._ensure_neighbor_offset_cache(R)

        # 解析 query -> ijk
        if query.dim() == 2 and query.shape[-1] == 3:
            q_ijk = query.to(torch.int32, device=device)          # [Q,3]
        else:
            q_codes = query.to(torch.int64, device=device).view(-1)
            q_ijk = self._decode_code_to_ijk(q_codes)             # [Q,3]
        Q = int(q_ijk.shape[0])

        xyz = self._nbr_offsets_xyz[R]               # [M,3]
        sqd_lat = self._nbr_sqdist_lattice[R]        # [M]
        zero_idx = self._nbr_zero_idx[R]
        M = int(xyz.shape[0])

        cand_ijk  = q_ijk.view(Q, 1, 3) + xyz.view(1, M, 3)           # [Q,M,3]
        cand_code = self._encode_ijk_to_code(cand_ijk.view(-1, 3)).view(Q, M)

        present = self._present_mask(cand_code)                        # [Q,M] bool

        vs = float(self._voxel_size.item())
        d2 = (sqd_lat.view(1, M) * (vs * vs)).expand(Q, M).clone()
        d2[~present] = float('inf')
        if not include_self and (zero_idx is not None):
            d2[:, zero_idx] = float('inf')

        # 输出为“定长填充”的形式（与 many-kernel 易对齐）
        # 将 inf 置为无效，用 -1 填充 code
        nbr_codes = cand_code.clone()
        inf_mask = torch.isinf(d2)
        nbr_codes = nbr_codes.masked_fill(inf_mask, -1)
        found = (~inf_mask).sum(dim=1).to(torch.int32)
        nbr_dist = d2.to(torch.float32) if return_squared_dist else torch.sqrt(d2).to(torch.float32)

        return nbr_codes.to(torch.int64), nbr_dist, found

    # -------- accessors --------
    def has(self, codes: torch.Tensor) -> torch.Tensor:
        """bool mask: which codes have ever existed"""
        return self._present_mask(codes)

    def num_voxels(self) -> int:
        return int(self._voxel_morton.numel())

    def voxel_size(self) -> float:
        return float(self._voxel_size.item())

    def all_codes(self) -> torch.Tensor:
        return self._voxel_morton


#TODO: check open3d version for HashMap support
class _KeyToIdIndex:
    """
    Backend adapter for key -> id mapping.
    Primary: Open3D GPU HashMap (O(1))
    Fallback: Sorted array + searchsorted (O(logN))
    """
    def __init__(self, device: str, backend: str = "o3d"):
        self.device = device
        self._capacity = 1 << 15  # initial capacity hint
        self._dtype_torch = torch.int64
        self._ids_next = torch.tensor(0, dtype=torch.int64, device=device)  # monotonic id allocator
        self._id2key = torch.empty(0, dtype=torch.int64, device=device)     # id -> packed key
        if _HAS_O3D and backend == "o3d":
            self._o3d_dev = o3c.Device("CUDA:0" if "cuda" in device else "CPU:0")
            # 设备
            dev = "CUDA:0" if (isinstance(self.device, str) and "cuda" in self.device) else "CPU:0"
            self._o3d_dev = o3c.Device(dev)

            # Open3D HashMap 需要 element shape；标量也要 [1]
            key_shape = o3c.SizeVector([1])
            val_shape = o3c.SizeVector([1])

            try:
                # 单值 dtype 签名
                self._map = o3c.HashMap(
                    init_capacity=self._capacity,
                    key_dtype=o3c.Dtype.Int64,
                    key_element_shape=key_shape,
                    value_dtype=o3c.Dtype.Int64,
                    value_element_shape=val_shape,
                    device=self._o3d_dev,
                )
            except TypeError:
                # 多值 dtype 签名（某些版本）
                self._map = o3c.HashMap(
                    init_capacity=self._capacity,
                    key_dtype=o3c.Dtype.Int64,
                    key_element_shape=key_shape,
                    value_dtypes=[o3c.Dtype.Int64],
                    value_element_shapes=[val_shape],
                    device=self._o3d_dev,
                )
            self._backend = "o3d"

        else:
            # Fallback: maintain sorted keys & ids
            self._keys_sorted = torch.empty(0, dtype=torch.int64, device=device)
            self._ids_sorted  = torch.empty(0, dtype=torch.int64, device=device)
            self._backend = "sorted"

    @property
    def id2key(self) -> torch.Tensor:
        return self._id2key

    def _alloc_ids(self, n: int) -> torch.Tensor:
        start = int(self._ids_next.item())
        new_ids = torch.arange(start, start + n, dtype=torch.int64, device=self.device)
        self._ids_next += n
        return new_ids

    # -------- batch find-or-insert: return ids for each key; newly inserted get fresh stable ids --------
    @torch.no_grad()
    def upsert(self, keys: torch.Tensor) -> torch.Tensor:
        assert keys.dtype == torch.int64
        if keys.numel() == 0:
            return torch.empty(0, dtype=torch.int32, device=self.device)

        if self._backend == "o3d":
            k_o3 = o3c.Tensor(keys.contiguous(), device=self._o3d_dev)
            found_mask_o3, iters = self._map.find(k_o3)
            found_mask = torch.from_dlpack(found_mask_o3.to_dlpack()).to(self.device)
            ids_out = torch.empty_like(keys, dtype=torch.int64, device=self.device)

            # 1) 已存在：直接取值
            if found_mask.any():
                vals = self._map.get_value(iters)  # [nnz, 1]
                vals_t = torch.from_dlpack(vals.to_dlpack()).to(self.device).view(-1)
                ids_out[found_mask] = vals_t

            # 2) 不存在：分配新 id 并插入
            miss_mask = ~found_mask
            if miss_mask.any():
                k_miss = o3c.Tensor(keys[miss_mask].contiguous(), device=self._o3d_dev)
                n_new = int(k_miss.shape[0])
                new_ids = self._alloc_ids(n_new)  # torch.int64 on self.device
                # 插入需要 Open3D Tensor
                v_miss = o3c.Tensor(new_ids.contiguous(), device=self._o3d_dev).reshape(n_new, 1)
                ins_mask_o3, _ = self._map.insert(k_miss, v_miss)
                # 记得维护 id->key
                self._id2key = torch.cat([self._id2key, keys[miss_mask]], dim=0)
                ids_out[miss_mask] = new_ids

            return ids_out.to(torch.int32)

        else:
            # Fallback: unique-ify, match in sorted, insert missing, then stitch back
            k = keys
            # probe
            if self._keys_sorted.numel() > 0:
                pos = torch.searchsorted(self._keys_sorted, k)
                pos = pos.clamp_(0, self._keys_sorted.numel()-1)
                match = (self._keys_sorted[pos] == k)
                ids_out = torch.empty_like(k, dtype=torch.int64)
                if match.any():
                    ids_out[match] = self._ids_sorted[pos[match]]
            else:
                match = torch.zeros_like(k, dtype=torch.bool)
                ids_out = torch.empty_like(k, dtype=torch.int64)

            miss = ~match
            if miss.any():
                k_miss = k[miss]
                # unique new keys in ascending order
                k_new, inv = torch.unique(k_miss, sorted=True, return_inverse=True)
                n_new = int(k_new.numel())
                new_ids_unique = self._alloc_ids(n_new)
                # append to sorted arrays (merge like merge-sort)
                self._keys_sorted = torch.cat([self._keys_sorted, k_new], dim=0)
                self._ids_sorted  = torch.cat([self._ids_sorted,  new_ids_unique], dim=0)
                ord_ = torch.argsort(self._keys_sorted)
                self._keys_sorted = self._keys_sorted.index_select(0, ord_)
                self._ids_sorted  = self._ids_sorted.index_select(0, ord_)
                # id->key
                self._id2key = torch.cat([self._id2key, k_new], dim=0)
                # map miss back
                ids_out[miss] = new_ids_unique.index_select(0, inv)

            return ids_out.to(torch.int32)

    # -------- pure lookup: keys -> ids (missing = -1) --------
    @torch.no_grad()
    def lookup_ids(self, keys: torch.Tensor) -> torch.Tensor:
        assert keys.dtype == torch.int64
        if keys.numel() == 0:
            return torch.empty_like(keys, dtype=torch.int32)

        if self._backend == "o3d":
            k_o3 = o3c.Tensor(keys.contiguous(), device=self._o3d_dev)
            found_mask_o3, iters = self._map.find(k_o3)
            found_mask = torch.from_dlpack(found_mask_o3.to_dlpack()).to(self.device)
            out = torch.full_like(keys, -1, dtype=torch.int32, device=self.device)
            if found_mask.any():
                vals = self._map.get_value(iters)  # [nnz, 1]
                vals_t = torch.from_dlpack(vals.to_dlpack()).to(self.device).view(-1).to(torch.int32)
                out[found_mask] = vals_t
            return out
        else:
            if self._keys_sorted.numel() == 0:
                return torch.full_like(keys, -1, dtype=torch.int32)
            pos = torch.searchsorted(self._keys_sorted, keys)
            pos = pos.clamp_(0, self._keys_sorted.numel()-1)
            match = (self._keys_sorted[pos] == keys)
            out = torch.full_like(keys, -1, dtype=torch.int32)
            if match.any():
                out[match] = self._ids_sorted[pos[match]].to(torch.int32)
            return out

class HashVoxel:
    """
    Same public API as before, but key->id uses O(1) GPU hashmap if available.
    """
    def __init__(self, voxel_size: float = 0.05, device: str = "cuda"):
        self._voxel_size = (
            torch.tensor(voxel_size, dtype=torch.float32, device=device)
            if isinstance(voxel_size, (int, float))
            else voxel_size.to(device, dtype=torch.float32)
        )
        self.device = device

        # master tables (append-only, stable ids = row index)
        self._voxel_keys = torch.empty(0, 3, dtype=torch.int32, device=device)        # [N,3] ijk
        self._voxel_centers = torch.empty(0, 3, dtype=torch.float32, device=device)   # [N,3]
        # index: key(int64) -> id(int32)
        self._index = _KeyToIdIndex(device=device)

        # cached neighbor offsets per-R
        self._nbr_offsets_1d: Dict[int, torch.Tensor] = {}
        self._nbr_sqdist_lattice: Dict[int, torch.Tensor] = {}
        self._nbr_zero_idx: Dict[int, int] = {}
    
    def reset(self):
        """Reset voxel table while preserving the same voxel_size and device."""
        self.__init__(float(self._voxel_size.item()), self.device)

    @staticmethod
    def _pack_keys_1d(ijk: torch.Tensor) -> torch.Tensor:
        I = ijk.to(torch.int64)
        off = (1 << 20)
        I += off
        return I[:, 0] + I[:, 1] * (1 << 21) + I[:, 2] * (1 << 42)

    @torch.no_grad()
    def upsert(self, ijk_valid: torch.Tensor) -> torch.Tensor:
        if ijk_valid.numel() == 0:
            return torch.empty(0, dtype=torch.int32, device=self.device)
        assert ijk_valid.shape[-1] == 3
        k_new = self._pack_keys_1d(ijk_valid)
        # 1) find-or-insert to get stable ids
        ids = self._index.upsert(k_new)  # int32

        # 2) 对“新插入”的 id 补齐主表（通过和 id2key 长度对齐检测）
        N_master = self._voxel_keys.shape[0]
        N_index  = int(self._index.id2key.numel())
        if N_index > N_master:
            # 找出本批次新增的 keys 对应的 ijk：按 id 顺序补齐
            ids_new = torch.arange(N_master, N_index, device=self.device, dtype=torch.int32)
            # 需要从 ijk_valid 里捞 representative：这里用“第一次出现”的代表
            # 构建：key -> idx_first_occurrence
            # （注意：大批量时可用散列/unique 优化；此处保持简洁）
            k_all = self._pack_keys_1d(ijk_valid)
            # map: key -> first idx
            order = torch.argsort(k_all, stable=True)
            k_sorted = k_all[order]
            change = torch.ones_like(k_sorted, dtype=torch.bool)
            if k_sorted.numel() > 1:
                change[1:] = (k_sorted[1:] != k_sorted[:-1])
            first_idx = order[change]
            key_rep = k_all[first_idx]
            # 用 id2key 的“新段”去对齐 representative
            k_new_unique = self._index.id2key[ids_new.long()]
            # search key_rep 以定位对应行（简单起见：再来一次 searchsorted；也可以用 hashmap 再查一次）
            pos = torch.searchsorted(key_rep.sort().values, k_new_unique)
            # 重新稳妥一点：直接回查 representative（O(1) 也行）
            # 这里稳妥实现：用匹配掩码
            # 构造 dict 的话会落到 CPU；改用张量匹配
            # 为避免 O(N^2)，我们直接再 hash 查一次：
            # -> 最简单：从 ijk_valid 中筛第一个相等 key。如下实现足够正确：
            rep_map = {}
            # WARNING: Python 循环仅用于新插入的小批次；若担心，可在工程里替换成 kernel
            for ki, idx in zip(key_rep.tolist(), first_idx.tolist()):
                rep_map[ki] = idx
            gather_idx = torch.tensor([rep_map[int(x.item())] for x in k_new_unique],
                                      device=self.device, dtype=torch.long)
            ijk_new_only = ijk_valid.index_select(0, gather_idx).to(torch.int32)

            self._voxel_keys = torch.cat([self._voxel_keys, ijk_new_only], dim=0)
            centers_new = (ijk_new_only.to(torch.float32) + 0.5) * self._voxel_size
            self._voxel_centers = torch.cat([self._voxel_centers, centers_new], dim=0)

        return ids

    # ---- neighbor offset cache ----
    def _ensure_neighbor_offset_cache(self, R: int):
        if R in self._nbr_offsets_1d: return
        device = self.device
        sY, sZ = (1 << 21), (1 << 42)
        rng = torch.arange(-R, R + 1, device=device, dtype=torch.int32)
        dx, dy, dz = torch.meshgrid(rng, rng, rng, indexing='ij')
        xyz = torch.stack([dx, dy, dz], dim=-1).reshape(-1, 3).contiguous()
        off1d = (xyz[:, 0].to(torch.int64)
                 + xyz[:, 1].to(torch.int64) * sY
                 + xyz[:, 2].to(torch.int64) * sZ)
        sqd_lat = (xyz.to(torch.float32) ** 2).sum(-1)
        zero_idx = int(torch.where((xyz == 0).all(-1))[0].item())
        self._nbr_offsets_1d[R] = off1d
        self._nbr_sqdist_lattice[R] = sqd_lat
        self._nbr_zero_idx[R] = zero_idx

    # ---- batch key lookup via backend (O(1) if o3d) ----
    @torch.no_grad()
    def _lookup_ids_by_keys(self, keys_1d: torch.Tensor) -> torch.Tensor:
        return self._index.lookup_ids(keys_1d)

    # ---- KNN / radius API 与原版一致（只是把查找换成 O(1)）----
    @torch.no_grad()
    def knn_by_id(self, voxel_ids: torch.Tensor, k: int,
                  include_self: bool = False,
                  max_radius_cells: Optional[int] = None,
                  return_squared_dist: bool = True):
        device = self.device
        Q = int(voxel_ids.numel())
        if Q == 0 or k <= 0:
            zI = torch.empty(Q, 0, dtype=torch.int32, device=device)
            zF = torch.empty(Q, 0, dtype=torch.float32, device=device)
            zC = torch.zeros(Q, dtype=torch.int32, device=device)
            return zI, zF, zC

        # pick R
        if max_radius_cells is None:
            need = k + (1 if include_self else 0)
            R = int(torch.ceil(((torch.tensor(float(need), device=device) ** (1.0 / 3.0)) - 1.0) / 2.0).item())
            R = max(1, R)
        else:
            R = max(1, int(max_radius_cells))

        self._ensure_neighbor_offset_cache(R)
        off1d = self._nbr_offsets_1d[R]
        sqd_lat = self._nbr_sqdist_lattice[R]
        zero_idx = self._nbr_zero_idx[R]
        M = int(off1d.numel())

        # base keys by id
        valid_q = (voxel_ids >= 0) & (voxel_ids.to(torch.int64) < self._index.id2key.numel())
        base_keys = torch.full((Q,), -2**63, dtype=torch.int64, device=device)
        base_keys[valid_q] = self._index.id2key[voxel_ids[valid_q].to(torch.int64)]

        cand_keys = base_keys.view(Q, 1) + off1d.view(1, M)
        cand_ids = self._lookup_ids_by_keys(cand_keys.view(-1)).view(Q, M)

        vs2 = float(self._voxel_size.item()) ** 2
        d2 = sqd_lat.view(1, M).expand(Q, M).clone() * vs2
        d2[cand_ids.lt(0)] = float('inf')
        if not include_self:
            d2[:, zero_idx] = float('inf')

        k_eff = min(k, M)
        top_vals, top_idx = torch.topk(d2, k=k_eff, dim=1, largest=False, sorted=True)
        nbr_ids = cand_ids.gather(1, top_idx)
        infm = torch.isinf(top_vals)
        if infm.any():
            nbr_ids = nbr_ids.masked_fill(infm, -1)
        found = (~infm).sum(1).to(torch.int32)

        nbr_dist = top_vals.to(torch.float32) if return_squared_dist else torch.sqrt(top_vals).to(torch.float32)
        return nbr_ids.to(torch.int32), nbr_dist, found

    @torch.no_grad()
    def neighbors(self, voxel_ids: torch.Tensor, radius_m: float, include_self: bool = False):
        vs = float(self._voxel_size.item())
        R = max(1, int(torch.ceil(torch.tensor(radius_m / vs)).item()))
        self._ensure_neighbor_offset_cache(R)
        M = int(self._nbr_offsets_1d[R].numel())
        return self.knn_by_id(voxel_ids, k=M, include_self=include_self,
                              max_radius_cells=R, return_squared_dist=True)

    # Accessors (保持原样)
    def get_centers(self, voxel_ids: torch.Tensor) -> torch.Tensor:
        return self._voxel_centers[voxel_ids]
    def get_voxel_keys(self) -> torch.Tensor:
        return self._voxel_keys
    def get_voxel_num(self) -> int:
        return int(self._voxel_keys.shape[0])
    def get_voxel_size(self) -> float:
        return float(self._voxel_size.item())






if __name__ == "__main__":
    pass