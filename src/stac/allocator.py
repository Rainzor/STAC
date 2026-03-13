# Copyright (c) 2025 STAC Authors. All rights reserved.

from typing import List, Optional, Tuple, Union, Dict
from abc import ABC, abstractmethod
import enum

import torch
import torch.nn as nn
import torch.nn.functional as F


class BufState(enum.IntEnum):
    """
    State machine for each buffer row.
    
    States:
        RESERVED (0): No logical buffer bound to this row (initial state)
        AVAILABLE (1): Logical buffer exists; may be unmaterialized or partially filled (0<= count < B)
        FULL (2): Buffer materialized and count == B
        FREE (3): Retired/free; logical buffer unbound, physical memory freed
        HELD (4): Held for lazy release; logical buffer still bound, memory reserved
    """
    RESERVED = 0
    AVAILABLE = 1
    FULL = 2
    FREE = 3
    HELD = 4


class BufferInterface(ABC):
    """Abstract interface for buffer management systems."""
    
    @abstractmethod
    def register_field(self, name: str, kind: str = "vector", dtype: Optional[torch.dtype] = None):
        """Register a new field in the buffer."""
        raise NotImplementedError("register_field is not implemented")

    @abstractmethod
    def resize_rows(self, num_heads: int, alloc_per_head: int):
        """Resize the buffer capacity."""
        raise NotImplementedError("resize_rows is not implemented")

    def close(self):
        """Free all buffer resources."""
        raise NotImplementedError("close is not implemented")

    @abstractmethod
    def append_batch_dict(self, rows_new: torch.Tensor, new_data: Dict[str, torch.Tensor], 
                         by_score: str = "S") -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Append new data to specified rows."""
        raise NotImplementedError("append_batch_dict is not implemented")

    @abstractmethod
    def read_rows_dict(self, rows: Union[List[int], torch.Tensor], 
                      fields: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        """Read data from specified rows."""
        raise NotImplementedError("read_rows_dict is not implemented")

    @abstractmethod
    def read_rows_dict_fast(self, rows: Union[List[int], torch.Tensor], 
                           fields: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        """
        Read data from rows with valid data (AVAILABLE or FULL state).
        
        Contract:
            - Rows must have valid data (not FREE/RESERVED)
            - Does NOT require FULL state
            - Always returns mask "M" indicating valid tokens
            
        Returns:
            Dict with field tensors [G, B, D] or [G, B], plus "M" mask [G, B]
        """
        raise NotImplementedError("read_rows_dict_fast is not implemented")

    @abstractmethod
    def read_rows_dict_compressed(self, rows: Union[List[int], torch.Tensor], 
                              fields: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        """
        Read and pack compressed buffers for given rows.
        
        Args:
            rows: [G] int64/32 row ids. 
                  Rows are not required to be FULL.
                  Rows are not required to be unique.
            fields: Optional list of field names to read. If None, reads all fields.
        
        Returns:
            out: dict of {field_name: tensor}
                - vector fields: [total_valid, D]
                - scalar fields: [total_valid]
                - "M": [G, B] bool mask for reconstruction
        """
        raise NotImplementedError("read_rows_dict_compressed is not implemented")

    @abstractmethod
    def write_rows_dict(self, rows: torch.Tensor, data: Dict[str, torch.Tensor]):
        """
        Write data to specified rows.
        
        Contract:
            - Rows must be unique and not FREE
            - data must contain all registered fields + "M" (mask)
            - "M" is REQUIRED for proper state management
            - State transitions: RESERVED -> AVAILABLE/FULL based on mask count
            
        Args:
            rows: [G] row indices
            data: Dict with field tensors [G, B, D] or [G, B], plus "M" mask [G, B]
        """
        raise NotImplementedError("write_rows_dict is not implemented")

    @abstractmethod
    def get_state(self, rows=None) -> torch.Tensor:
        """Get the state of specified rows."""
        raise NotImplementedError("get_state is not implemented")

    @abstractmethod
    def free_rows(self, rows, free=None):
        """Free specified rows."""
        raise NotImplementedError("free_rows is not implemented")
    @abstractmethod
    def clean_rows(self, rows):
        """Clear data and mark given rows as RESERVED."""
        raise NotImplementedError("clean_rows is not implemented")
    
    @abstractmethod
    def stats(self) -> dict:
        """Get buffer statistics."""
        raise NotImplementedError("stats is not implemented")


class BufferBackend(ABC):
    """
    Abstract backend interface for buffer storage primitives.
    
    This class defines the low-level storage operations that concrete buffer
    implementations (StaticBuffer, SlabPool, SegmentedSlabPool) must provide.
    
    BufferWrapper uses these primitives to implement higher-level operations
    like append_batch_dict with Top-B selection.
    
    Properties that must be set by subclass __init__:
        B: int - buffer capacity per row
        D: int - feature dimension
        device: torch.device - device for storage
        field_specs: Dict[str, str] - field name to type mapping ("vector"/"scalar")
    """
    
    # ---- Properties (must be set by subclass __init__) ----
    B: int
    D: int
    device: torch.device
    field_specs: Dict[str, str]
    
    # ----------------------------------------------------------------------
    # Lightweight dtype query (avoids returning large storage tensor)
    # ----------------------------------------------------------------------
    
    @abstractmethod
    def field_dtype(self, name: str) -> torch.dtype:
        """
        Return the dtype for a field without exposing the storage tensor.
        
        Args:
            name: field name
            
        Returns:
            torch.dtype for the field
        """
        pass
    
    # ----------------------------------------------------------------------
    # State Machine Operations
    # ----------------------------------------------------------------------
    
    @abstractmethod
    def get_state(self, rows=None) -> torch.Tensor:
        """
        Get the state of specified rows.
        
        Args:
            rows: Optional row indices. If None, returns all row states.
            
        Returns:
            Tensor of BufState values
        """
        pass
    
    @abstractmethod
    def set_row_state(self, rows: torch.Tensor, state: BufState):
        """
        Set the state for specified rows.
        
        Args:
            rows: row indices
            state: target BufState
        """
        pass
    
    def set_available(self, rows):
        """Mark rows as AVAILABLE (have logical buffer)."""
        self.set_row_state(rows, BufState.AVAILABLE)
    
    def set_full(self, rows):
        """Mark rows as FULL (buffer completely filled)."""
        self.set_row_state(rows, BufState.FULL)
    
    def set_free(self, rows):
        """Mark rows as FREE (inactive)."""
        self.set_row_state(rows, BufState.FREE)
    
    # ----------------------------------------------------------------------
    # Storage Primitives for Append Operations
    # ----------------------------------------------------------------------
    
    @abstractmethod
    def ensure_materialized(self, rows: torch.Tensor):
        """
        Ensure specified rows have physical storage allocated (lazy allocation).
        Called before reading/writing to rows that are AVAILABLE.
        
        Args:
            rows: [G] unique row indices that need physical storage
        """
        pass
    
    @abstractmethod
    def gather_existing_tokens(self, rows: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Read existing valid tokens from specified rows.
        
        Args:
            rows: [G] unique row indices
            
        Returns:
            existing_data: {field_name: [E_ex, D] or [E_ex]} existing valid tokens
            group_ids: [E_ex] group index (0..G-1) for each existing token
        """
        pass
    
    @abstractmethod
    def clear_row_data(self, rows: torch.Tensor):
        """
        Clear all data and mask for specified rows.
        Does not change row state.
        
        Args:
            rows: [G] unique row indices to clear
        """
        pass
    
    @abstractmethod
    def scatter_tokens(self, rows: torch.Tensor, cols: torch.Tensor,
                       data: Dict[str, torch.Tensor]):
        """
        Write tokens to physical storage at specified positions.
        
        Args:
            rows: [E_kept] row indices for each token
            cols: [E_kept] column indices (0..B-1) for each token
            data: {field_name: [E_kept, D] or [E_kept]} token data to write
        """
        pass
    
    @abstractmethod
    def update_row_indicators(self, rows: torch.Tensor, counts: torch.Tensor,
                              kept_rows: torch.Tensor, kept_cols: torch.Tensor):
        """
        Update row indicators (offset, mask, state) after scatter.
        
        Args:
            rows: [G] unique row indices
            counts: [G] number of tokens kept per row
            kept_rows: [E_kept] row indices of kept tokens (for mask update)
            kept_cols: [E_kept] column indices of kept tokens (for mask update)
        """
        pass
    
    # ----------------------------------------------------------------------
    # Resource Management
    # ----------------------------------------------------------------------
    
    @abstractmethod
    def register_field(self, name: str, kind: str = "vector", dtype: Optional[torch.dtype] = None):
        """Register a new field in the buffer."""
        pass
    
    @abstractmethod
    def resize_rows(self, num_heads: int, alloc_per_head: int):
        """Resize the buffer row capacity."""
        pass
    
    @abstractmethod
    def close(self):
        """Free all buffer resources."""
        pass
    
    # ----------------------------------------------------------------------
    # Read/Write Operations
    # ----------------------------------------------------------------------
    
    @abstractmethod
    def read_rows_dict(self, rows: Union[List[int], torch.Tensor],
                       fields: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        """Read data from specified rows."""
        pass
    
    @abstractmethod
    def read_rows_dict_fast(self, rows: Union[List[int], torch.Tensor],
                            fields: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        """Read data from rows with valid data (AVAILABLE or FULL)."""
        pass
    
    @abstractmethod
    def read_rows_dict_compressed(self, rows: Union[List[int], torch.Tensor],
                                  fields: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        """Read and pack compressed buffers for given rows."""
        pass
    
    @abstractmethod
    def write_rows_dict(self, rows: torch.Tensor, data: Dict[str, torch.Tensor]):
        """Write data to specified rows."""
        pass
    
    # ----------------------------------------------------------------------
    # Row Query Methods
    # ----------------------------------------------------------------------
    
    def full_rows(self, restrict_rows: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return indices of rows that are FULL."""
        return self._query_rows_by_state(BufState.FULL, restrict_rows)
    
    def available_rows(self, restrict_rows: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return indices of rows that are AVAILABLE."""
        return self._query_rows_by_state(BufState.AVAILABLE, restrict_rows)
    
    def reserved_rows(self, restrict_rows: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return indices of rows that are RESERVED."""
        return self._query_rows_by_state(BufState.RESERVED, restrict_rows)
    
    def materialized_rows(self, restrict_rows: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return indices of rows that have any data (AVAILABLE or FULL)."""
        if restrict_rows is not None:
            states = self.get_state(restrict_rows)
            mask = (states == int(BufState.AVAILABLE)) | (states == int(BufState.FULL))
            return restrict_rows[mask]
        else:
            states = self.get_state()
            mask = (states == int(BufState.AVAILABLE)) | (states == int(BufState.FULL))
            return torch.nonzero(mask, as_tuple=False).squeeze(1)
    
    def _query_rows_by_state(self, target_state: BufState,
                             restrict_rows: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Helper method to query rows by state."""
        if restrict_rows is not None:
            states = self.get_state(restrict_rows)
            mask = (states == int(target_state))
            return restrict_rows[mask]
        else:
            states = self.get_state()
            mask = (states == int(target_state))
            return torch.nonzero(mask, as_tuple=False).squeeze(1)
    
    # ----------------------------------------------------------------------
    # Lifecycle Operations
    # ----------------------------------------------------------------------
    
    @abstractmethod
    def free_rows(self, rows, free=None):
        """Free specified rows."""
        pass
    
    @abstractmethod
    def clean_rows(self, rows):
        """Clear data and mark given rows as RESERVED."""
        pass
    
    # ----------------------------------------------------------------------
    # Statistics
    # ----------------------------------------------------------------------
    
    @abstractmethod
    def stats(self) -> dict:
        """Get buffer statistics."""
        pass
    
    # ----------------------------------------------------------------------
    # Common Utility Methods
    # ----------------------------------------------------------------------
    
    @staticmethod
    def _segment_starts(sorted_seg: torch.Tensor) -> torch.Tensor:
        """Return the start indices of each unique segment in sorted order."""
        E = sorted_seg.numel()
        if E == 0:
            return torch.empty(0, dtype=torch.long, device=sorted_seg.device)
        change = torch.ones(E, dtype=torch.bool, device=sorted_seg.device)
        change[1:] = sorted_seg[1:] != sorted_seg[:-1]
        return torch.nonzero(change, as_tuple=False).squeeze(1)
    
    def _stable_rows(self, rows: Union[torch.Tensor, List[int], int]) -> torch.Tensor:
        """Normalize row input to a tensor of shape [N] on device."""
        if isinstance(rows, int):
            return torch.tensor([rows], dtype=torch.long, device=self.device)
        elif isinstance(rows, list):
            return torch.tensor(rows, dtype=torch.long, device=self.device)
        return rows.to(self.device, dtype=torch.long)


class BufferWrapper(BufferInterface):
    """
    Wrapper that implements BufferInterface using BufferBackend primitives.
    
    This class provides the high-level buffer operations (like append_batch_dict
    with Top-B selection) by composing low-level storage primitives from the backend.
    
    Application code should use BufferWrapper (or BufferInterface) rather than
    directly accessing BufferBackend to ensure semantic consistency.
    """
    
    def __init__(self, backend: BufferBackend):
        """
        Initialize wrapper with a backend.
        
        Args:
            backend: BufferBackend instance (StaticBuffer, SlabPool, or SegmentedSlabPool)
        """
        self.backend = backend
    
    # ----------------------------------------------------------------------
    # Properties delegated to backend
    # ----------------------------------------------------------------------
    
    @property
    def B(self) -> int:
        return self.backend.B
    
    @property
    def D(self) -> int:
        return self.backend.D
    
    @property
    def device(self) -> torch.device:
        return self.backend.device
    
    @property
    def field_specs(self) -> Dict[str, str]:
        return self.backend.field_specs
    
    # ----------------------------------------------------------------------
    # Resource Management (delegated)
    # ----------------------------------------------------------------------
    
    def register_field(self, name: str, kind: str = "vector", dtype: Optional[torch.dtype] = None):
        """Register a new field in the buffer."""
        return self.backend.register_field(name, kind, dtype)
    
    def resize_rows(self, num_heads: int, alloc_per_head: int):
        """Resize the buffer capacity."""
        return self.backend.resize_rows(num_heads, alloc_per_head)
    
    def close(self):
        """Free all buffer resources."""
        return self.backend.close()
    
    # ----------------------------------------------------------------------
    # State Management (delegated)
    # ----------------------------------------------------------------------
    
    def get_state(self, rows=None) -> torch.Tensor:
        """Get the state of specified rows."""
        return self.backend.get_state(rows)
    
    def set_available(self, rows):
        """Mark rows as AVAILABLE."""
        return self.backend.set_available(rows)
    
    def set_full(self, rows):
        """Mark rows as FULL."""
        return self.backend.set_full(rows)
    
    def set_free(self, rows):
        """Mark rows as FREE."""
        return self.backend.set_free(rows)
    
    # ----------------------------------------------------------------------
    # Read/Write Operations (delegated)
    # ----------------------------------------------------------------------
    
    def read_rows_dict(self, rows: Union[List[int], torch.Tensor],
                       fields: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        """Read data from specified rows."""
        return self.backend.read_rows_dict(rows, fields)
    
    def read_rows_dict_fast(self, rows: Union[List[int], torch.Tensor],
                            fields: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        """Read data from rows with valid data (AVAILABLE or FULL)."""
        return self.backend.read_rows_dict_fast(rows, fields)
    
    def read_rows_dict_compressed(self, rows: Union[List[int], torch.Tensor],
                                  fields: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        """Read and pack compressed buffers for given rows."""
        return self.backend.read_rows_dict_compressed(rows, fields)
    
    def write_rows_dict(self, rows: torch.Tensor, data: Dict[str, torch.Tensor]):
        """Write data to specified rows."""
        return self.backend.write_rows_dict(rows, data)
    
    # ----------------------------------------------------------------------
    # Row Query Methods (delegated)
    # ----------------------------------------------------------------------
    
    def full_rows(self, restrict_rows: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return indices of rows that are FULL."""
        return self.backend.full_rows(restrict_rows)
    
    def available_rows(self, restrict_rows: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return indices of rows that are AVAILABLE."""
        return self.backend.available_rows(restrict_rows)
    
    def reserved_rows(self, restrict_rows: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return indices of rows that are RESERVED."""
        return self.backend.reserved_rows(restrict_rows)
    
    def materialized_rows(self, restrict_rows: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return indices of rows that have any data."""
        return self.backend.materialized_rows(restrict_rows)
    
    # ----------------------------------------------------------------------
    # Lifecycle Operations (delegated)
    # ----------------------------------------------------------------------
    
    def free_rows(self, rows, free=None):
        """Free specified rows."""
        return self.backend.free_rows(rows, free)
    
    def clean_rows(self, rows):
        """Clear data and mark given rows as RESERVED."""
        return self.backend.clean_rows(rows)
    
    # ----------------------------------------------------------------------
    # Statistics (delegated)
    # ----------------------------------------------------------------------
    
    def stats(self) -> dict:
        """Get buffer statistics."""
        return self.backend.stats()
    
    def detailed_stats(self) -> dict:
        """Get detailed buffer statistics if available."""
        if hasattr(self.backend, 'detailed_stats'):
            return self.backend.detailed_stats()
        return self.stats()
    
    # ----------------------------------------------------------------------
    # Core append_batch_dict Implementation (Top-B selection logic)
    # ----------------------------------------------------------------------

    @torch.no_grad()
    def append_batch_dict(self,
                          rows_new: torch.Tensor,
                          new_data: Dict[str, torch.Tensor],
                          by_score: str = "S"
                          ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Unified Top-B append implementation using backend primitives.
        
        Concatenates existing valid items with new items per row, then keeps
        Top-B by score for all registered fields. Returns aligned overflow.
        
        Args:
            rows_new: [E] flattened logical row indices
            new_data: {field_name: [E, D] or [E]} new token data
            by_score: name of the score field for Top-B selection
            
        Returns:
            overflow: {field_name: [E_over, D] or [E_over]} overflow tokens
            rows_over: [E_over] row indices for overflow tokens
        """
        backend = self.backend
        assert by_score in backend.field_specs, f"Score field '{by_score}' must be registered."
        rows_new = backend._stable_rows(rows_new)
        E = int(rows_new.numel())
        device = backend.device
        B, D = backend.B, backend.D

        # Early exit for empty input
        if E == 0:
            overflow = {}
            for name, kind in backend.field_specs.items():
                dtype = backend.field_dtype(name)
                if kind == "vector":
                    overflow[name] = torch.empty(0, D, dtype=dtype, device=device)
                else:
                    overflow[name] = torch.empty(0, dtype=dtype, device=device)
            rows_over = torch.empty(0, dtype=torch.long, device=device)
            return overflow, rows_over

        # 1) Group rows & check states
        uniq_rows, inv_rows = torch.unique(rows_new, sorted=True, return_inverse=True)
        uniq_st = backend.get_state(uniq_rows)

        ST = BufState
        is_reserved = (uniq_st == int(ST.RESERVED))
        is_available = (uniq_st == int(ST.AVAILABLE))
        ok_mask_rows = is_reserved | is_available
        rows_ok = uniq_rows[ok_mask_rows]

        # Bad elements (FREE/FULL/HELD) → immediate overflow
        state_row = backend.get_state(rows_new)
        is_bad_el = (state_row != int(ST.RESERVED)) & (state_row != int(ST.AVAILABLE))

        overflow: Dict[str, torch.Tensor] = {}
        for name, kind in backend.field_specs.items():
            dtype = backend.field_dtype(name)
            if name in new_data:
                arr = new_data[name].to(dtype, non_blocking=True)
            else:
                if kind == "vector":
                    arr = torch.zeros(E, D, dtype=dtype, device=device)
                else:
                    arr = torch.zeros(E, dtype=dtype, device=device)
            overflow[name] = arr[is_bad_el]
        rows_over = rows_new[is_bad_el]

        # Nothing to append to valid rows
        if rows_ok.numel() == 0:
            return overflow, rows_over

        # 2) RESERVED → AVAILABLE & ensure materialization
        if is_reserved.any():
            backend.set_available(uniq_rows[is_reserved])
        backend.ensure_materialized(rows_ok)

        G_ok = int(rows_ok.numel())

        # 3) Gather existing valid tokens from storage
        existing_data, seg_ex = backend.gather_existing_tokens(rows_ok)

        # 4) Map new elements to ok group indices
        map_u2ok = torch.full((uniq_rows.numel(),), -1, device=device, dtype=torch.long)
        map_u2ok[ok_mask_rows] = torch.arange(G_ok, device=device, dtype=torch.long)
        seg_new = map_u2ok.index_select(0, inv_rows)
        keep_new = seg_new >= 0

        # 5) Concatenate existing + filtered new for each field
        cat_all: Dict[str, torch.Tensor] = {}
        for name, kind in backend.field_specs.items():
            dtype = backend.field_dtype(name)
            Xe = existing_data.get(name, torch.empty(0, D if kind == "vector" else 0, 
                                                      dtype=dtype, device=device).view(-1, D) if kind == "vector" 
                                   else torch.empty(0, dtype=dtype, device=device))
            if kind == "vector":
                if Xe.numel() == 0:
                    Xe = torch.empty(0, D, dtype=dtype, device=device)
                Xn = (new_data.get(name, torch.zeros(E, D, dtype=dtype, device=device))
                      .to(dtype, non_blocking=True))[keep_new]
            else:
                if Xe.numel() == 0:
                    Xe = torch.empty(0, dtype=dtype, device=device)
                Xn = (new_data.get(name, torch.zeros(E, dtype=dtype, device=device))
                      .to(dtype, non_blocking=True))[keep_new]
            cat_all[name] = torch.cat([Xe, Xn], 0)

        grp = torch.cat([seg_ex, seg_new[keep_new]], 0)

        # Handle empty concatenation
        if cat_all[by_score].numel() == 0:
            return overflow, rows_over

        # 6) Stable sort: score ↓ then group asc
        score_all = cat_all[by_score]
        ord1 = torch.argsort(score_all, descending=True, stable=True)
        grp1 = grp[ord1]
        ord2 = torch.argsort(grp1, stable=True)
        idx_lex = ord1.index_select(0, ord2)
        grp2 = grp1[ord2]

        # 7) Per-group Top-B selection
        E_all = grp2.numel()
        start = BufferBackend._segment_starts(grp2)
        pos = torch.arange(E_all, device=device, dtype=torch.long)
        gid = torch.searchsorted(start, pos, right=True) - 1
        rank = pos - start[gid]
        keep = rank < B

        # Per-row kept counts
        offset_vec = torch.zeros(G_ok, dtype=torch.int32, device=device)
        if keep.any():
            offset_vec.index_add_(0, gid[keep], torch.ones_like(rank[keep], dtype=torch.int32))

        # 8) Clear touched rows
        backend.clear_row_data(rows_ok)

        # 9) Scatter kept tokens
        if keep.any():
            rows_keep = rows_ok.index_select(0, gid[keep])
            col_keep = rank[keep].clamp_max(B - 1)
            
            kept_data: Dict[str, torch.Tensor] = {}
            for name, kind in backend.field_specs.items():
                src = cat_all[name].index_select(0, idx_lex)
                kept_data[name] = src[keep]
            
            backend.scatter_tokens(rows_keep, col_keep, kept_data)
            
            # 10) Update row indicators
            backend.update_row_indicators(rows_ok, offset_vec, rows_keep, col_keep)
        else:
            # No tokens kept - still update state to AVAILABLE with 0 count
            backend.update_row_indicators(rows_ok, offset_vec, 
                                          torch.empty(0, dtype=torch.long, device=device),
                                          torch.empty(0, dtype=torch.long, device=device))

        # 11) Collect overflow from ok rows (elements not kept)
        if (~keep).any():
            rows_over_ok = rows_ok.index_select(0, gid[~keep])
            rows_over = torch.cat([rows_over, rows_over_ok], dim=0)
            for name, kind in backend.field_specs.items():
                src = cat_all[name].index_select(0, idx_lex)
                overflow[name] = torch.cat([overflow[name], src[~keep]], dim=0)

        return overflow, rows_over
    
    # ----------------------------------------------------------------------
    # Legacy Interface Support (for backward compatibility)
    # ----------------------------------------------------------------------

    def append_batch(self, rows_new: torch.Tensor, K_new: torch.Tensor, 
                    V_new: torch.Tensor, S_new: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Legacy interface for appending K/V/S data.
        Args:
            rows_new: [E] int64/32 row ids
            K_new: [E,D] float32/16 new key vectors
            V_new: [E,D] float32/16 new value vectors
            S_new: [E] float32 new scores
        Returns:
            K_over: [E_over,D] overflow key vectors
            V_over: [E_over,D] overflow value vectors
            S_over: [E_over] overflow scores
            rows_over: [E_over] row ids corresponding to overflow entries
            E_over <= E
        """
        new_data = {"K": K_new, "V": V_new, "S": S_new}
        overflow, rows_over = self.append_batch_dict(rows_new, new_data, by_score="S")
        return overflow["K"], overflow["V"], overflow["S"], rows_over

    def append_element(self, row: int, K: torch.Tensor, V: torch.Tensor, S: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Append a single element to the buffer.
        Args:
            row: int row id
            K: [N,D] float32/16 key vectors
            V: [N,D] float32/16 value vectors
            S: [N] float32 scores
        Returns:
            K_over, V_over, S_over: overflow key vectors, value vectors, scores (without rows_over)
        """
        K_over, V_over, S_over, _ = self.append_batch([row], K, V, S)
        return K_over, V_over, S_over

    def read_rows(self, rows: Union[List[int], torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Legacy interface for reading K/V/S/M data."""
        out = self.read_rows_dict(rows)
        return out["K"], out["V"], out["S"], out["M"]

    def read_rows_fast(self, rows: Union[List[int], torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Legacy interface for reading FULL rows K/V/S/M data."""
        out = self.read_rows_dict_fast(rows)
        return out["K"], out["V"], out["S"], out["M"]
    
    def read_rows_compressed(self, rows: Union[List[int], torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Legacy interface for reading compressed K/V/S/M data."""
        out = self.read_rows_dict_compressed(rows)
        return out["K"], out["V"], out["S"], out["M"]

    def write_rows(self, rows: torch.Tensor, K_in: torch.Tensor, V_in: torch.Tensor, 
                  S_in: torch.Tensor, M_in: torch.Tensor) -> None:
        """Legacy interface for writing K/V/S/M data.
        Rows are required to be unique and not FREE.

        Args:
            rows: [G] int64/32 row ids
            K_in: [G,B,D] float32/16 key vectors
            V_in: [G,B,D] float32/16 value vectors
            S_in: [G,B] float32 scores
            M_in: [G,B] bool valid mask
        
        """
        data = {"K": K_in, "V": V_in, "S": S_in, "M": M_in}
        self.write_rows_dict(rows, data)


# Backward compatibility: BaseBuffer is now an alias for BufferWrapper
# DEPRECATED: Use BufferWrapper directly with a BufferBackend instance
# or use create_buffer() factory function for new code.
BaseBuffer = BufferWrapper


class StaticBuffer(BufferBackend):
    """
    Multi-field StaticBuffer manager.
    Supports flexible registration of fields (K, V, S, etc.) with identical shape alignment.
    - Each field can be vector-type ([H, A, B, D]) or scalar-type ([H, A, B]).
    - Maintains consistent buffer count and mask per row.

    StaticBuffer keep the physical allocation for all rows, regardless of logical state.
    """

    def __init__(self,
                 buf_cap: int,
                 head_dim: int,
                 num_heads: int=16,
                 capacity: int = 0,
                 field_specs: Optional[Dict[str, str]] = None,  # {"K":"vector", "V":"vector", "S":"scalar"}
                 device: torch.device=torch.device("cuda"),
                 dtype: torch.dtype=torch.float16,
                 **kwargs
                 ):
        """
        Args:
            buf_cap: capacity per buffer row (B)
            D: feature dimension for all vector-type fields
            device: device for storage
            dtype: default dtype for vector-type fields
            capacity: number of rows (per head)
            field_specs: dict mapping field name to type ("vector" or "scalar")
        """
        self.B = int(buf_cap)
        self.D = int(head_dim)
        self.num_heads = int(num_heads)
        self.device = device
        self.dtype = dtype
        self.alloc = int(capacity) if capacity > 0 else 0

        # default fields
        self.field_specs = {"K": "vector", "V": "vector", "S": "scalar"}
        self.field_specs.update(field_specs or {})

        # data buffers
        self.fields: Dict[str, torch.Tensor] = {}
        for name, kind in self.field_specs.items():
            self.fields[name] = self._init_field(name, kind)

        # indicators
        self.buf_M = torch.zeros(self.num_heads, self.alloc, self.B,
                                 dtype=torch.bool, device=self.device)  # valid mask
        self.buf_offset = torch.zeros(self.num_heads, self.alloc,
                                      dtype=torch.int32, device=self.device)  # valid count per row
        self.buf_state = torch.full((self.num_heads, self.alloc),
                                    int(BufState.RESERVED),
                                    dtype=torch.int8,
                                    device=self.device)  # logical state per row

    # ----------------------------------------------------------------------
    # Initialization helpers
    # ----------------------------------------------------------------------

    def _init_field(self, name: str, kind: str) -> torch.Tensor:
        """Initialize a new data field."""
        if kind == "vector":
            return torch.zeros(self.num_heads, self.alloc, self.B, self.D,
                               dtype=self.dtype, device=self.device)
        elif kind == "scalar":
            return torch.zeros(self.num_heads, self.alloc, self.B,
                               dtype=torch.float32, device=self.device)
        else:
            raise ValueError(f"Unknown field kind: {kind}")

    def register_field(self, name: str, kind: str = "vector", dtype: Optional[torch.dtype] = None):
        if name in self.fields:
            raise KeyError(f"Field {name} already exists.")
        dtype = dtype or (self.dtype if kind == "vector" else torch.float32)
        if kind == "vector":
            new_field = torch.zeros(self.num_heads, self.alloc, self.B, self.D,
                                    dtype=dtype, device=self.device)
        elif kind == "scalar":
            new_field = torch.zeros(self.num_heads, self.alloc, self.B,
                                    dtype=dtype, device=self.device)
        else:
            raise ValueError(f"Unsupported kind: {kind}")
        self.fields[name] = new_field
        self.field_specs[name] = kind

    def field_dtype(self, name: str) -> torch.dtype:
        """Return the dtype for a field without exposing the storage tensor."""
        return self.fields[name].dtype

    # ----------------------------------------------------------------------
    # Utility functions
    # ----------------------------------------------------------------------

    def _raw_rows(self, rows):
        return rows.to(self.device, dtype=torch.long)

    def _physical_size(self) -> int:
        return self.num_heads * self.alloc

    def _logical_size(self) -> int:
        return self.num_heads * self.alloc

    # ----------------------------------------------------------------------
    # Row Allocation
    # ----------------------------------------------------------------------

    def resize_rows(self, num_heads: int, alloc_per_head: int):
        assert num_heads > 0 and alloc_per_head > 0
        assert num_heads == self.num_heads, f"Expected head_num={self.num_heads}, got {num_heads}"
        new_A = max(alloc_per_head, self.alloc)
        if new_A == self.alloc:
            return

        def extend_tensor(t: torch.Tensor, new_shape, value):
            new_t = torch.empty(new_shape, dtype=t.dtype, device=self.device)
            if isinstance(value, (bool, int, float)):
                new_t.fill_(value)
            else:
                new_t.copy_(value)
            slices = tuple(slice(0, min(o, n)) for o, n in zip(t.shape, new_shape))
            new_t[slices] = t[slices]
            return new_t

        for name, kind in self.field_specs.items():
            t = self.fields[name]
            if kind == "vector":
                self.fields[name] = extend_tensor(t, (num_heads, new_A, self.B, self.D), 0.0)
            else:
                self.fields[name] = extend_tensor(t, (num_heads, new_A, self.B), 0.0)

        self.buf_M = extend_tensor(self.buf_M, (num_heads, new_A, self.B), False)
        self.buf_offset = extend_tensor(self.buf_offset, (num_heads, new_A), 0)
        self.buf_state = extend_tensor(self.buf_state, (num_heads, new_A), int(BufState.RESERVED))
        self.alloc = new_A
    
    def close(self):
        """Free all buffer resources."""
        self.fields.clear()
        self.fields = {}
        self.buf_M = None
        self.buf_offset = None
        self.buf_state = None

    # ----------------------------------------------------------------------
    # Buffer state ops (StaticBuffer)
    # ----------------------------------------------------------------------

    def get_state(self, rows=None) -> torch.Tensor:
        """
        Return the current logical state for given rows.
        -1 (FREE), 0 (RESERVED), (0,B) AVAILABLE, B FULL
        """
        if rows is None:
            return self.buf_state.view(-1)
        rows = self._stable_rows(rows)
        return self.buf_state.view(-1).index_select(0, rows)

    @torch.no_grad()
    def set_row_state(self, rows: torch.Tensor, state: BufState):
        """Set the state for specified rows."""
        rows = self._stable_rows(rows)
        if rows.numel() == 0:
            return
        if state == BufState.RESERVED:
            self.buf_state.view(-1)[rows] = int(BufState.RESERVED)
            self.buf_offset.view(-1)[rows] = 0
            self.buf_M.view(-1, self.B)[rows] = False
        elif state == BufState.AVAILABLE:
            self.buf_state.view(-1)[rows] = int(BufState.AVAILABLE)
        elif state == BufState.FULL:
            self.buf_state.view(-1)[rows] = int(BufState.FULL)
            self.buf_offset.view(-1)[rows] = self.B
            self.buf_M.view(-1, self.B)[rows] = True
        elif state == BufState.FREE:
            self.buf_state.view(-1)[rows] = int(BufState.FREE)
            self.buf_offset.view(-1)[rows] = -1
            self.buf_M.view(-1, self.B)[rows] = False
        else:
            raise ValueError(f"Unsupported state transition to {state}.")

    #----------------------------------------------------------------------
    # Row Free
    #----------------------------------------------------------------------
    def free_rows(self, rows, free=None):
        """Mark given rows as FREE (inactive)."""
        self.set_free(rows)

    #----------------------------------------------------------------------
    # Row Clean
    #----------------------------------------------------------------------
    def clean_rows(self, rows):
        """Clear data and mark given rows as RESERVED."""
        rows = self._stable_rows(rows)
        if rows.numel() == 0:
            return
        # Clear data
        for name, kind in self.field_specs.items():
            if kind == "vector":
                self.fields[name].view(-1, self.B, self.D)[rows] = 0
            else:
                self.fields[name].view(-1, self.B)[rows] = 0
        self.buf_M.view(-1, self.B)[rows] = False
        self.buf_offset.view(-1)[rows] = 0
        self.buf_state.view(-1)[rows] = int(BufState.RESERVED)

    # ----------------------------------------------------------------------
    # Storage Access Methods (for BufferWrapper.append_batch_dict)
    # ----------------------------------------------------------------------

    def ensure_materialized(self, rows: torch.Tensor):
        """StaticBuffer is always materialized - no-op."""
        pass

    @torch.no_grad()
    def gather_existing_tokens(self, rows: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Read existing valid tokens from specified rows."""
        rows = self._stable_rows(rows)
        G = rows.numel()
        B, D = self.B, self.D
        S_tot = self.num_heads * self.alloc
        device = self.device

        if G == 0:
            existing_data = {}
            for name, kind in self.field_specs.items():
                if kind == "vector":
                    existing_data[name] = torch.empty(0, D, dtype=self.fields[name].dtype, device=device)
                else:
                    existing_data[name] = torch.empty(0, dtype=self.fields[name].dtype, device=device)
            group_ids = torch.empty(0, dtype=torch.long, device=device)
            return existing_data, group_ids

        # Get mask for existing valid elements
        Mb = self.buf_M.view(S_tot, B).index_select(0, rows)  # [G, B]
        m_ex_flat = Mb.view(-1)  # [G*B] bool

        # Group IDs for existing tokens
        seg_ex = torch.repeat_interleave(
            torch.arange(G, device=device, dtype=torch.long), B
        )[m_ex_flat]

        # Gather existing data per field
        existing_data: Dict[str, torch.Tensor] = {}
        for name, kind in self.field_specs.items():
            buf = self.fields[name]
            if kind == "vector":
                Xb = buf.view(S_tot, B, D).index_select(0, rows)  # [G, B, D]
                existing_data[name] = Xb.view(-1, D)[m_ex_flat]   # [E_ex, D]
            else:
                Xb = buf.view(S_tot, B).index_select(0, rows)     # [G, B]
                existing_data[name] = Xb.view(-1)[m_ex_flat]      # [E_ex]

        return existing_data, seg_ex

    @torch.no_grad()
    def clear_row_data(self, rows: torch.Tensor):
        """Clear all data and mask for specified rows."""
        rows = self._stable_rows(rows)
        if rows.numel() == 0:
            return
        S_tot = self.num_heads * self.alloc
        
        # Clear mask
        self.buf_M.view(S_tot, self.B)[rows] = False
        
        # Clear all field data
        for name, kind in self.field_specs.items():
            if kind == "vector":
                self.fields[name].view(S_tot, self.B, self.D)[rows] = 0
            else:
                self.fields[name].view(S_tot, self.B)[rows] = 0

    @torch.no_grad()
    def scatter_tokens(self, rows: torch.Tensor, cols: torch.Tensor,
                       data: Dict[str, torch.Tensor]):
        """Write tokens to physical storage at specified positions."""
        if rows.numel() == 0:
            return
        S_tot = self.num_heads * self.alloc
        
        for name, kind in self.field_specs.items():
            if name not in data:
                continue
            if kind == "vector":
                self.fields[name].view(S_tot, self.B, self.D)[rows, cols] = data[name]
            else:
                self.fields[name].view(S_tot, self.B)[rows, cols] = data[name]

    @torch.no_grad()
    def update_row_indicators(self, rows: torch.Tensor, counts: torch.Tensor,
                              kept_rows: torch.Tensor, kept_cols: torch.Tensor):
        """Update row indicators (offset, mask, state) after scatter."""
        rows = self._stable_rows(rows)
        if rows.numel() == 0:
            return
        S_tot = self.num_heads * self.alloc
        G = rows.numel()
        
        # Update offset
        self.buf_offset.view(S_tot)[rows] = counts
        
        # Update mask for kept positions
        if kept_rows.numel() > 0:
            self.buf_M.view(S_tot, self.B)[kept_rows, kept_cols] = True
        
        # Update state
        state_vec = torch.full((G,), int(BufState.AVAILABLE), dtype=torch.int8, device=self.device)
        state_vec[counts == self.B] = int(BufState.FULL)
        self.buf_state.view(S_tot)[rows] = state_vec

    # ----------------------------------------------------------------------
    # Generic (multi-field) read/write for StaticBuffer
    # ----------------------------------------------------------------------

    #! read
    def read_rows_dict(self, 
                  rows: Union[List[int], torch.Tensor],
                  fields: Optional[List[str]] = None
                  ) -> Dict[str, torch.Tensor]:
        """
        Read and pack all fields for given rows.
        Args: 
            rows: [G] int64/32 row ids. 
            Rows without materialized slabs return zeros / False. 
            Rows are not required to be FULL.
            Rows are not required to be unique.

        Returns:
            out: dict of {field_name: tensor}, shapes: vector=[G,B,D], scalar=[G,B]    
        """
        rows = self._stable_rows(rows)
        B, D = self.B, self.D
        S_tot = self.num_heads * self.alloc
        out: Dict[str, torch.Tensor] = {}
        for name, kind in self.field_specs.items():
            if (fields is not None) and (name not in fields):
                continue
            buf = self.fields[name]
            if kind == "vector":
                out[name] = buf.view(S_tot, B, D).index_select(0, rows)
            else:
                out[name] = buf.view(S_tot, B).index_select(0, rows)
        out["M"] = self.buf_M.view(S_tot, B).index_select(0, rows)
        return out

    def read_rows_dict_fast(self, 
                       rows,
                       fields: Optional[List[str]] = None
                       ) -> Dict[str, torch.Tensor]:
        """ 
            Pack buffers for given rows -> [G, B, ...]. 

            Rows must have valid data (offset > 0, i.e. AVAILABLE or FULL).
            Does NOT require FULL state - returns mask M for partial rows.

            More efficient than read_rows_dict when reading contiguous rows.

            Returns:
                out: dict of {field_name: tensor}, shapes: vector=[G,B,D], scalar=[G,B]
                     "M": [G,B] bool mask indicating valid tokens
        """
        rows = self._stable_rows(rows)
        B, D = self.B, self.D
        states = self.buf_state.view(-1).index_select(0, rows)
        # Only require rows to have valid data (not FREE/RESERVED), not necessarily FULL
        assert torch.all((states == int(BufState.AVAILABLE)) | (states == int(BufState.FULL))), \
            "Some input rows are not AVAILABLE or FULL (no valid data)."

        S_tot = self.num_heads * self.alloc
        out: Dict[str, torch.Tensor] = {}
        for name, kind in self.field_specs.items():
            if (fields is not None) and (name not in fields):
                continue
            buf = self.fields[name]
            if kind == "vector":
                out[name] = buf.view(S_tot, B, D).index_select(0, rows)
            else:
                out[name] = buf.view(S_tot, B).index_select(0, rows)
        out["M"] = self.buf_M.view(S_tot, B).index_select(0, rows)
        return out
    
    def read_rows_dict_compressed(self,
                                  rows: Union[List[int], torch.Tensor],
                                  fields: Optional[List[str]] = None
                                  ) -> Dict[str, torch.Tensor]:
        """
        Read and pack compressed buffers for given rows.
        
        Args:
            rows: [G] int64/32 row ids.
            fields: Optional list of field names to read.
        
        Returns:
            out: dict of {field_name: tensor}
                - vector fields: [total_valid, D]
                - scalar fields: [total_valid]
                - "M": [G, B] bool mask for reconstruction
        """
        rows = self._stable_rows(rows)
        G = rows.numel()
        B, D = self.B, self.D
        S_tot = self.num_heads * self.alloc
        
        out: Dict[str, torch.Tensor] = {}
        
        # Build mask [G, B]
        M_out = torch.zeros(G, B, dtype=torch.bool, device=self.device)
        offsets = self.buf_offset.view(S_tot).index_select(0, rows)  # [G] int32
        valid_off = offsets > 0
        
        if not valid_off.any():
            # No valid rows - return empty tensors
            for name, kind in self.field_specs.items():
                if (fields is not None) and (name not in fields):
                    continue
                if kind == "vector":
                    out[name] = torch.empty(0, D, dtype=self.fields[name].dtype, device=self.device)
                else:
                    out[name] = torch.empty(0, dtype=self.fields[name].dtype, device=self.device)
            out["M"] = M_out
            return out
        
        valid_rows = rows[valid_off]
        M_valid = self.buf_M.view(S_tot, B).index_select(0, valid_rows)  # [G_valid, B]
        M_out[valid_off] = M_valid
        m_flat = M_out.view(-1)  # [G*B] for indexing
        
        for name, kind in self.field_specs.items():
            if (fields is not None) and (name not in fields):
                continue
            buf = self.fields[name]
            # Build full [G, B, ...] tensor then compress
            if kind == "vector":
                Xb_full = torch.zeros(G, B, D, dtype=buf.dtype, device=self.device)
                Xb_valid = buf.view(S_tot, B, D).index_select(0, valid_rows)
                Xb_full[valid_off] = Xb_valid
                out[name] = Xb_full.view(-1, D)[m_flat]
            else:
                Xb_full = torch.zeros(G, B, dtype=buf.dtype, device=self.device)
                Xb_valid = buf.view(S_tot, B).index_select(0, valid_rows)
                Xb_full[valid_off] = Xb_valid
                out[name] = Xb_full.view(-1)[m_flat]
        
        out["M"] = M_out  # [G, B] - consistent with other read methods
        return out
    
    #! write
    @torch.no_grad()
    def write_rows_dict(self, rows: torch.Tensor, data: Dict[str, torch.Tensor]) -> None:
        """ 
        Write given rows' buffers from input tensors.

        Rows are required to be unique. 

        Rows are required to not be FREE 

        If rows are RESERVED or AVAILABLE, they become AVAILABLE/FULL after write.
        """        

        rows = self._stable_rows(rows)
        unique_rows, counts = torch.unique(rows, return_counts=True)
        assert bool((counts == 1).all()), "Input rows must be unique."
        buf_states = self.buf_state.view(-1).index_select(0, rows)
        is_free = (buf_states == int(BufState.FREE))
        assert not bool(is_free.any()), "Cannot write to FREE rows."
        is_reserved = (buf_states == int(BufState.RESERVED))
        if is_reserved.any():
            self.set_available(rows[is_reserved])

        B, D = self.B, self.D
        S_tot = self.num_heads * self.alloc

        for name, kind in self.field_specs.items():
            assert name in data, f"Input data missing field '{name}'."
            buf = self.fields[name]
            if kind == "vector":
                assert data[name].shape[1:] == (B, D)
                buf.view(S_tot, B, D).index_copy_(0, rows, data[name].to(buf.dtype))
            else:
                assert data[name].shape[1:] == (B,)
                buf.view(S_tot, B).index_copy_(0, rows, data[name].to(buf.dtype))

        # M is required for proper state management
        assert "M" in data, "Missing mask 'M' in write_rows_dict"
        M_in = data["M"].to(torch.bool)
        self.buf_M.view(S_tot, B).index_copy_(0, rows, M_in)
        offset = M_in.sum(dim=1).to(torch.int32)
        self.buf_offset.view(-1).index_copy_(0, rows, offset)
        is_full = (offset == B)
        states = torch.where(is_full, int(BufState.FULL), int(BufState.AVAILABLE)).to(torch.int8)
        self.buf_state.view(-1).index_copy_(0, rows, states)

    # ----------------------------------------------------------------------
    # Statistics
    # ----------------------------------------------------------------------

    def stats(self) -> dict:
        """Compute buffer utilization statistics."""
        total_slots = self.num_heads * self.alloc * self.B
        active_rows = int((self.buf_offset > 0).sum().item())
        data_count = int(self.buf_offset.clamp_min(0).sum().item())
        occupancy = data_count / float(total_slots) if total_slots > 0 else 0.0
        return dict(
            fields=list(self.field_specs.keys()),
            buf_capacity=self.B,
            head_dim=self.D,
            head_num=self.num_heads,
            physical_size=self._physical_size(),
            logical_size=self._logical_size(),
            total_slots=total_slots,
            active_rows=active_rows,
            data_count=data_count,
            alloc_count=self._physical_size(),
            occupancy_ratio=occupancy,
        )

    def detailed_stats(self) -> dict:
        stats = self.stats()
        stats.update(self._analyze_usage_patterns())
        return stats
    
    def _analyze_usage_patterns(self) -> dict:
        """Analyze buffer usage patterns for debugging."""
        Mp = self.buf_M            # [H,V,P] bool
        activate_buf = self.buf_offset > 0  # [H,V] bool
        Mp = Mp[activate_buf]
        count_per_row = Mp.sum(dim=-1).to(torch.float32)              # [H,V]
        count_total   = int(count_per_row.sum().item())
        alloc_total   = int(Mp.numel())
        occupancy     = count_total / float(alloc_total + 1e-6)

        frac = count_per_row.flatten() / float(self.B)
        hist = torch.histc(frac.cpu(), bins=self.B+1, min=0.0, max=1.0).tolist()

        low_mask  = (frac < 0.3)
        high_mask = (frac > 0.8)
        low_counts  = float(count_per_row.flatten()[low_mask].sum().item())
        high_counts = float(count_per_row.flatten()[high_mask].sum().item())

        return dict(
            data_count=count_total,
            alloc_count=alloc_total,
            occupancy=occupancy,
            low_pivos=int(low_mask.sum().item()),
            low_counts=low_counts,
            low_ratio=low_counts / float(count_total + 1e-6),
            high_pivos=int(high_mask.sum().item()),
            high_counts=high_counts,
            high_ratio=high_counts / float(count_total + 1e-6),
            occupancy_histogram=hist,
        )


class SlabPool(BufferBackend):
    """
    A row-addressable slab pool for per-(head,voxel) buffers:
      - Row space is [0, S_tot). You can grow S_tot.
      - Lazily materializes GPU slabs on first append.
      - Tracks logical vs physical states per row.
      - Lets the owner (HeadVoxelKVStoreOnlinePool) decide when to remerge or retire;
        the pool just provides primitives and compact batch views.
    Field-flexible SlabPool:
      - Physical slabs in a 'pool' (materialized only when needed).
      - Arbitrary registered fields, each stored as a tensor:
          vector: [pool_cap, B, D]
          scalar: [pool_cap, B]
      - row_ptr maps logical rows → pool slots; row_state tracks state machine.
      - pool_M is the mask per slot; pool_offset is the valid count per slot.
    """

    def __init__(self,
                 buf_cap: int,
                 head_dim: int,
                 num_heads: int=16,
                 growth: int = 256,
                 growth_ratio: float = 0.0,
                 capacity: int = 0,
                 free_policy: str = "immediate",
                 field_specs: Optional[Dict[str, str]] = None,
                 device: torch.device=torch.device("cuda"),
                 dtype: torch.dtype=torch.float16,
                 **kwargs):
        # logical parameters
        self.B = int(buf_cap)
        self.D = int(head_dim)
        self.num_heads = int(num_heads)
        self.device = device
        self.dtype = dtype
        self._growth = int(growth)
        assert self._growth > 0, "Growth must be positive."
        self._growth_ratio = float(growth_ratio)
        self._free_policy = free_policy
        self.debug = kwargs.get("debug", False)

        # field registry (same contract as StaticBuffer)
        self.field_specs: Dict[str, str] = {"K": "vector", "V": "vector", "S": "scalar"}
        if field_specs:
            self.field_specs.update(field_specs)

        # physical pool (start with capacity slots)
        self.pool_cap = int(capacity)
        self.pool_top = 0
        self.free_list: List[int] = []

        # per-field pool tensors
        self.pool_fields: Dict[str, torch.Tensor] = {}
        for name, kind in self.field_specs.items():
            self.pool_fields[name] = self._init_pool_field(kind, self.pool_cap)

        # indicators
        self.pool_M = torch.zeros(self.pool_cap, self.B, dtype=torch.bool, device=device)
        self.pool_offset = torch.zeros(self.pool_cap, dtype=torch.int32, device=device)

        # logical rows (caller grows via resize_rows)
        self.S_tot = 0
        self.row_ptr = torch.full((0,), -1, dtype=torch.int32, device=device)   # -1 = no slot
        self.row_state = torch.full((0,), int(BufState.RESERVED), dtype=torch.int8, device=device)

    # -------------------- field mgmt --------------------

    def _init_pool_field(self, kind: str, cap: int) -> torch.Tensor:
        if kind == "vector":
            return torch.zeros(cap, self.B, self.D, dtype=self.dtype, device=self.device)
        elif kind == "scalar":
            return torch.zeros(cap, self.B, dtype=torch.float32, device=self.device)
        else:
            raise ValueError(f"Unknown kind: {kind}")

    def register_field(self, name: str, kind: str = "vector", dtype: Optional[torch.dtype] = None):
        """
        Dynamically add a field to the materialized pool.
        New storage is allocated for existing slots and will be zero-initialized.
        """
        if name in self.pool_fields:
            raise KeyError(f"Field {name} already exists.")
        # dtype is driven by vector/scalar convention; kept for compatibility
        _ = dtype  # unused in this implementation; tensors follow default dtype rules
        self.field_specs[name] = kind
        self.pool_fields[name] = self._init_pool_field(kind, self.pool_cap)

    def field_dtype(self, name: str) -> torch.dtype:
        """Return the dtype for a field without exposing the storage tensor."""
        return self.pool_fields[name].dtype

    # -------------------- capacity mgmt --------------------

    def _physical_size(self) -> int:
        return self.pool_cap * self.B

    def _logical_size(self) -> int:
        return self.S_tot * self.B

    def _stable_rows(self, rows: Union[torch.Tensor, List[int], int]) -> torch.Tensor:
        """Normalize row input to a tensor of shape [N] on device."""
        if isinstance(rows, int):
            return torch.tensor([rows], dtype=torch.long, device=self.device)
        elif isinstance(rows, list):
            return torch.tensor(rows, dtype=torch.long, device=self.device)
        return rows.to(self.device, dtype=torch.long)
    
    @torch.no_grad()
    def resize_rows(self, num_heads: int, alloc_per_head: int):
        """
        Resize logical row space to (num_heads * alloc_per_head).

        Does NOT materialize slabs.
        """
        assert num_heads == self.num_heads, f"Expected head_num={self.num_heads}, got {num_heads}"
        S_tot_new = num_heads * alloc_per_head
        """Grow row space; does NOT materialize slabs."""
        if S_tot_new <= self.S_tot:
            return
        rp = torch.full((num_heads, alloc_per_head), -1, dtype=torch.int32, device=self.device)
        rm = torch.full((num_heads, alloc_per_head), int(BufState.RESERVED), dtype=torch.int8, device=self.device)
        if self.S_tot > 0:
            old_alloc = self.S_tot // num_heads
            rp[:, :old_alloc].copy_(self.row_ptr.view(num_heads, old_alloc))
            rm[:, :old_alloc].copy_(self.row_state.view(num_heads, old_alloc))
        self.row_ptr, self.row_state = rp.reshape(-1), rm.reshape(-1)
        self.S_tot = S_tot_new
    
    def close(self):
        """Free all pool resources."""
        self.pool_fields.clear()
        self.pool_fields = {}

        self.pool_M = None
        self.pool_offset = None
        self.row_ptr = None
        self.row_state = None
        torch.cuda.empty_cache()

    @torch.no_grad()
    def _grow_pool(self, add: int):
        if add <= 0:
            return
        new_cap = self.pool_cap + add

        # extend per-field tensors
        for name, kind in self.field_specs.items():
            old = self.pool_fields[name]
            if kind == "vector":
                ext = torch.zeros(new_cap, self.B, self.D, dtype=old.dtype, device=self.device)
                if old.numel() > 0:
                    ext[:old.shape[0]].copy_(old)
            else:
                ext = torch.zeros(new_cap, self.B, dtype=old.dtype, device=self.device)
                if old.numel() > 0:
                    ext[:old.shape[0]].copy_(old)
            self.pool_fields[name] = ext

        # extend indicators
        pool_M_new = torch.zeros(new_cap, self.B, dtype=torch.bool, device=self.device)
        pool_offset_new = torch.zeros(new_cap, dtype=torch.int32, device=self.device)
        if self.pool_cap > 0:
            pool_M_new[:self.pool_cap].copy_(self.pool_M)
            pool_offset_new[:self.pool_cap].copy_(self.pool_offset)
        self.pool_M, self.pool_offset = pool_M_new, pool_offset_new
        self.pool_cap = new_cap

    @torch.no_grad()
    def _init_slots(self, ids: torch.Tensor):
        if ids.numel() == 0:
            return
        ids = ids.to(self.device, dtype=torch.long)
        # zero data in all fields
        for name, kind in self.field_specs.items():
            t = self.pool_fields[name]
            if kind == "vector":
                t.index_fill_(0, ids, 0)
            else:
                t.index_fill_(0, ids, 0)
        # reset indicators
        self.pool_M.index_fill_(0, ids, False)
        self.pool_offset.index_fill_(0, ids, 0)

    @torch.no_grad()
    def _alloc_slots(self, n: int) -> torch.Tensor:
        if n <= 0:
            return torch.empty(0, dtype=torch.int32, device=self.device)

        # take from free_list
        k = min(n, len(self.free_list))
        take_from_free = []
        if k > 0:
            take_from_free = self.free_list[-k:]
            del self.free_list[-k:]

        # allocate fresh
        m = n - k
        if self.pool_top + m > self.pool_cap:
            growth_needed = (self.pool_top + m) - self.pool_cap
            growth_exp = int(self._growth_ratio * max(1, self.pool_cap))
            growth_min = min(growth_exp, self._growth)
            growth_try = max(growth_min, growth_needed)
            self._grow_pool(max(growth_min, growth_needed))

        fresh = torch.arange(self.pool_top, self.pool_top + m, device=self.device, dtype=torch.int32) if m > 0 else torch.empty(0, dtype=torch.int32, device=self.device)
        self.pool_top += m

        # init both sets
        ids = fresh
        if k > 0:
            free_ids = torch.tensor(take_from_free, dtype=torch.int32, device=self.device)
            ids = torch.cat([free_ids, ids], dim=0)

        self._init_slots(ids)
        return ids

    @torch.no_grad()
    def _free_slots(self, ids: torch.Tensor):
        if ids.numel() == 0:
            return
        ids = ids.to(self.device, dtype=torch.long)
        # reset indicators & keep data zeroed
        self.pool_M.index_fill_(0, ids, False)
        self.pool_offset.index_fill_(0, ids, 0)
        self.free_list.extend(ids.to("cpu", non_blocking=True).tolist())

    # -------------------- state & materialization --------------------

    @torch.no_grad()
    def get_state(self, rows=None) -> torch.Tensor:
        if rows is None:
            return self.row_state
        rows = self._stable_rows(rows)
        return self.row_state.index_select(0, rows)

    # @torch.no_grad()
    # def set_available(self, rows):
    #     rows = self._stable_rows(rows)
    #     if rows.numel() == 0:
    #         return
    #     self.row_state.index_fill_(0, rows, int(BufState.AVAILABLE))

    # @torch.no_grad()
    # def set_free(self, rows, free: bool = True):
    #     rows = self._stable_rows(rows)
    #     if rows.numel() == 0:
    #         return
    #     ptr = self.row_ptr.index_select(0, rows)
    #     has_slot = (ptr >= 0)
    #     if (free or self._free_policy == "immediate") and has_slot.any():
    #         self._free_slots(ptr[has_slot].to(torch.long))
    #         self.row_ptr.index_fill_(0, rows[has_slot], -1)
    #     self.row_state.index_fill_(0, rows, int(BufState.FREE))

    def set_row_state(self, rows, state):
        """Set the state for specified rows."""
        rows = self._stable_rows(rows)
        if rows.numel() == 0:
            return
        if state == BufState.FREE or state == BufState.RESERVED:
            ptr = self.row_ptr.index_select(0, rows)
            has_slot = (ptr >= 0)
            if has_slot.any():
                self._free_slots(ptr[has_slot].to(torch.long))
                self.row_ptr.index_fill_(0, rows[has_slot], -1)
        self.row_state.index_fill_(0, rows, int(state))
    
    # -------------------- free/clean --------------------
    @torch.no_grad()
    def free_rows(self, rows, free: Optional[bool] = None):
        rows = self._stable_rows(rows)
        if free or self._free_policy == "immediate":
            self.set_row_state(rows, BufState.FREE)
        else:
            self.set_row_state(rows, BufState.HELD)

    def clean_rows(self, rows):
        """Clear data and mark given rows as RESERVED."""
        rows = self._stable_rows(rows)
        if rows.numel() == 0:
            return
        self.set_row_state(rows, BufState.RESERVED)

    # ----------------------------------------------------------------------
    # Storage Access Methods (for BufferWrapper.append_batch_dict)
    # ----------------------------------------------------------------------

    def ensure_materialized(self, rows: torch.Tensor):
        """Ensure specified rows have physical storage allocated (lazy allocation)."""
        rows = self._stable_rows(rows)
        if rows.numel() == 0:
            return
        rows = torch.unique(rows)
        ptr = self.row_ptr.index_select(0, rows)
        st = self.row_state.index_select(0, rows)
        need = (ptr < 0) & (st == int(BufState.AVAILABLE))
        if not need.any():
            return
        rows_need = rows[need]
        new_ids = self._alloc_slots(rows_need.numel())
        self.row_ptr.index_copy_(0, rows_need, new_ids)

    @torch.no_grad()
    def gather_existing_tokens(self, rows: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Read existing valid tokens from specified rows."""
        rows = self._stable_rows(rows)
        device, D, B = self.device, self.D, self.B
        G = rows.numel()

        if G == 0:
            existing_data = {}
            for name, kind in self.field_specs.items():
                if kind == "vector":
                    existing_data[name] = torch.empty(0, D, dtype=self.pool_fields[name].dtype, device=device)
                else:
                    existing_data[name] = torch.empty(0, dtype=self.pool_fields[name].dtype, device=device)
            group_ids = torch.empty(0, dtype=torch.long, device=device)
            return existing_data, group_ids

        slot_ids = self.row_ptr.index_select(0, rows).to(torch.long)
        # Handle unmaterialized rows (slot_ids < 0)
        has_slot = (slot_ids >= 0)
        
        if not has_slot.any():
            existing_data = {}
            for name, kind in self.field_specs.items():
                if kind == "vector":
                    existing_data[name] = torch.empty(0, D, dtype=self.pool_fields[name].dtype, device=device)
                else:
                    existing_data[name] = torch.empty(0, dtype=self.pool_fields[name].dtype, device=device)
            group_ids = torch.empty(0, dtype=torch.long, device=device)
            return existing_data, group_ids

        # Only gather from materialized rows
        valid_slots = slot_ids[has_slot]
        valid_local_idx = torch.nonzero(has_slot, as_tuple=False).squeeze(1)

        Mb = self.pool_M.index_select(0, valid_slots)  # [G_valid, B]
        
        # Build full mask for all G rows
        m_full = torch.zeros(G, B, dtype=torch.bool, device=device)
        m_full[valid_local_idx] = Mb
        m_ex_flat = m_full.view(-1)

        # Group IDs for existing tokens
        seg_ex = torch.repeat_interleave(
            torch.arange(G, device=device, dtype=torch.long), B
        )[m_ex_flat]

        # Gather existing data per field
        existing_data: Dict[str, torch.Tensor] = {}
        for name, kind in self.field_specs.items():
            pool = self.pool_fields[name]
            # Build full data tensor
            if kind == "vector":
                Xb_full = torch.zeros(G, B, D, dtype=pool.dtype, device=device)
                Xb_valid = pool.index_select(0, valid_slots)  # [G_valid, B, D]
                Xb_full[valid_local_idx] = Xb_valid
                existing_data[name] = Xb_full.view(-1, D)[m_ex_flat]
            else:
                Xb_full = torch.zeros(G, B, dtype=pool.dtype, device=device)
                Xb_valid = pool.index_select(0, valid_slots)  # [G_valid, B]
                Xb_full[valid_local_idx] = Xb_valid
                existing_data[name] = Xb_full.view(-1)[m_ex_flat]

        return existing_data, seg_ex

    @torch.no_grad()
    def clear_row_data(self, rows: torch.Tensor):
        """Clear all data and mask for specified rows."""
        rows = self._stable_rows(rows)
        if rows.numel() == 0:
            return
        
        slot_ids = self.row_ptr.index_select(0, rows).to(torch.long)
        has_slot = (slot_ids >= 0)
        
        if has_slot.any():
            valid_slots = slot_ids[has_slot]
            # Clear pool data
            for name, kind in self.field_specs.items():
                self.pool_fields[name].index_fill_(0, valid_slots, 0)
            self.pool_M.index_fill_(0, valid_slots, False)

    @torch.no_grad()
    def scatter_tokens(self, rows: torch.Tensor, cols: torch.Tensor,
                       data: Dict[str, torch.Tensor]):
        """Write tokens to physical storage at specified positions."""
        if rows.numel() == 0:
            return
        
        slots = self.row_ptr.index_select(0, rows).to(torch.long)
        
        for name, kind in self.field_specs.items():
            if name not in data:
                continue
            if kind == "vector":
                self.pool_fields[name][slots, cols] = data[name]
            else:
                self.pool_fields[name][slots, cols] = data[name]
        
        self.pool_M[slots, cols] = True

    @torch.no_grad()
    def update_row_indicators(self, rows: torch.Tensor, counts: torch.Tensor,
                              kept_rows: torch.Tensor, kept_cols: torch.Tensor):
        """Update row indicators (offset, mask, state) after scatter."""
        rows = self._stable_rows(rows)
        if rows.numel() == 0:
            return
        G = rows.numel()
        
        slot_ids = self.row_ptr.index_select(0, rows).to(torch.long)
        
        # Update offset in pool_offset
        self.pool_offset.index_copy_(0, slot_ids, counts)
        
        # State already updated via mask in scatter_tokens
        # Update row state
        new_state = torch.where(counts == self.B,
                                torch.full_like(counts, int(BufState.FULL), dtype=torch.int8),
                                torch.full_like(counts, int(BufState.AVAILABLE), dtype=torch.int8))
        self.row_state.index_copy_(0, rows, new_state)

    # ----------------------------------------------------------------------
    # Generic (multi-field) read/write for SlabPool
    # ----------------------------------------------------------------------

    #! read
    @torch.no_grad()
    def read_rows_dict(self, rows: Union[List[int], torch.Tensor], fields: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        """
        Pack buffers for given rows -> [G, B, ...].
        Non-Materialized rows return zeros.
        """
        rows = self._stable_rows(rows)
        device, D, B = self.device, self.D, self.B
        G = rows.numel()
        slot_ids = self.row_ptr.index_select(0, rows).to(torch.long)
        reserve = (slot_ids < 0)
        have = ~reserve

        out: Dict[str, torch.Tensor] = {}
        # allocate zeros
        for name, kind in self.field_specs.items():
            pool = self.pool_fields[name]
            if fields is not None and name not in fields:
                continue
            if kind == "vector":
                out[name] = torch.zeros(G, B, D, dtype=pool.dtype, device=device)
            else:
                out[name] = torch.zeros(G, B, dtype=pool.dtype, device=device)
            if have.any():
                ids = slot_ids[have]
                out[name][have] = pool.index_select(0, ids)

        out["M"] = torch.zeros(G, B, dtype=torch.bool, device=device)
        if have.any():
            ids = slot_ids[have]
            out["M"][have] = self.pool_M.index_select(0, ids)
        return out

    @torch.no_grad()
    def read_rows_dict_fast(self, rows: Union[List[int], torch.Tensor], fields: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        '''
        Pack buffers for given rows -> [G, B, ...].
        
        Rows must have valid data (materialized with offset > 0, i.e. AVAILABLE or FULL).
        Does NOT require FULL state - returns mask M for partial rows.
        
        More efficient than read_rows_dict when reading contiguous rows.
        
        Returns:
            out: dict of {field_name: tensor}, shapes: vector=[G,B,D], scalar=[G,B]
                 "M": [G,B] bool mask indicating valid tokens
        '''
        rows = self._stable_rows(rows)
        st = self.row_state.index_select(0, rows)
        slot_ids = self.row_ptr.index_select(0, rows).to(torch.long)
        offset = self.pool_offset.index_select(0, slot_ids)
        # Only require rows to have valid data (AVAILABLE or FULL), not necessarily FULL
        assert torch.all((st == int(BufState.AVAILABLE)) | (st == int(BufState.FULL))), \
            "read_rows_dict_fast requires all rows to be AVAILABLE or FULL (have valid data)."
        assert torch.all(slot_ids >= 0) and torch.all(offset > 0), "Some requested rows are not materialized."
        
        out: Dict[str, torch.Tensor] = {}
        for name, kind in self.field_specs.items():
            pool = self.pool_fields[name]
            if fields is not None and name not in fields:
                continue
            ids = slot_ids
            out[name] = pool.index_select(0, ids)
        out["M"] = self.pool_M.index_select(0, slot_ids)
        return out
    
    def read_rows_dict_compressed(self, 
                                  rows: Union[List[int], torch.Tensor],
                                  fields: Optional[List[str]] = None
                                  ) -> Dict[str, torch.Tensor]:
        """
        Read and pack compressed buffers for given rows.
        
        Args:
            rows: [G] int64/32 row ids.
            fields: Optional list of field names to read.
        
        Returns:
            out: dict of {field_name: tensor}
                - vector fields: [total_valid, D]
                - scalar fields: [total_valid]
                - "M": [G, B] bool mask for reconstruction
        """
        rows = self._stable_rows(rows)
        G = rows.numel()
        B, D = self.B, self.D
        device = self.device
        
        # Build mask [G, B]
        M_out = torch.zeros(G, B, dtype=torch.bool, device=device)
        ids = self.row_ptr.index_select(0, rows).to(torch.long)
        ids_valid = ids >= 0
        
        out: Dict[str, torch.Tensor] = {}
        
        if not ids_valid.any():
            # No valid rows - return empty tensors
            for name, kind in self.field_specs.items():
                if (fields is not None) and (name not in fields):
                    continue
                pool = self.pool_fields[name]
                if kind == "vector":
                    out[name] = torch.empty(0, D, dtype=pool.dtype, device=device)
                else:
                    out[name] = torch.empty(0, dtype=pool.dtype, device=device)
            out["M"] = M_out
            return out
        
        valid_ids = ids[ids_valid]
        M_valid = self.pool_M.index_select(0, valid_ids)  # [G_valid, B]
        M_out[ids_valid] = M_valid
        m_flat = M_out.view(-1)  # [G*B] for indexing
        
        for name, kind in self.field_specs.items():
            if (fields is not None) and (name not in fields):
                continue
            pool = self.pool_fields[name]
            # Build full [G, B, ...] tensor then compress
            if kind == "vector":
                Xb_full = torch.zeros(G, B, D, dtype=pool.dtype, device=device)
                Xb_valid = pool.index_select(0, valid_ids)
                Xb_full[ids_valid] = Xb_valid
                out[name] = Xb_full.view(-1, D)[m_flat]
            else:
                Xb_full = torch.zeros(G, B, dtype=pool.dtype, device=device)
                Xb_valid = pool.index_select(0, valid_ids)
                Xb_full[ids_valid] = Xb_valid
                out[name] = Xb_full.view(-1)[m_flat]
        
        out["M"] = M_out  # [G, B] - consistent with other read methods
        return out

    #! write
    @torch.no_grad()
    def write_rows_dict(self, rows,
                        data: Dict[str, torch.Tensor]):  # expects all registered fields + "M"
        rows = self._stable_rows(rows)
        st = self.row_state.index_select(0, rows)
        is_reserved = (st == int(BufState.RESERVED))
        if is_reserved.any():
            self.set_available(rows[is_reserved])
        self.ensure_materialized(rows)
        ids = self.row_ptr.index_select(0, rows).to(torch.long)

        # copy fields
        for name, kind in self.field_specs.items():
            assert name in data, f"Input data missing field '{name}'."
            pool = self.pool_fields[name]
            if kind == "vector":
                assert data[name].shape[1:] == (self.B, self.D)
                pool.index_copy_(0, ids, data[name].to(pool.dtype))
            else:
                assert data[name].shape[1:] == (self.B,)
                pool.index_copy_(0, ids, data[name].to(pool.dtype))

        # mask/state
        assert "M" in data, "Missing mask 'M' in write_rows_dict"
        M_in = data["M"].to(torch.bool)
        self.pool_M.index_copy_(0, ids, M_in)
        sum_M = M_in.sum(dim=1).to(torch.int32)
        self.pool_offset.index_copy_(0, ids, sum_M)
        new_state = torch.where(sum_M == self.B,
                                torch.full_like(sum_M, int(BufState.FULL), dtype=torch.int8),
                                torch.full_like(sum_M, int(BufState.AVAILABLE), dtype=torch.int8))
        self.row_state.index_copy_(0, rows, new_state)

    # -------------------- stats --------------------

    @torch.no_grad()
    def stats(self) -> dict:
        active_slots = int((self.pool_offset > 0).sum().item())
        data_count = int(self.pool_offset.sum().item())
        alloc_count = int(self.pool_cap * self.B)
        occupancy = data_count / float(alloc_count) if self.pool_cap > 0 else 0.0
        compression = float(self.pool_cap) / float(self.S_tot) if self.S_tot > 0 else 0.0
        return dict(
            fields=list(self.field_specs.keys()),
            buf_capacity=self.B,
            head_dim=self.D,
            head_num=self.num_heads,
            voxel_num=self.S_tot // self.num_heads,
            physical_size=self._physical_size(),
            logical_size=self._logical_size(),
            pool_cap=int(self.pool_cap),
            pool_top=int(self.pool_top),
            free_slots=len(self.free_list),
            active_slots=active_slots,
            
            allocated_slots=int(self.S_tot),
            data_count=data_count,
            alloc_count=alloc_count,
            occupancy_ratio=occupancy,
            compression_ratio=compression,
        )

    def detailed_stats(self) -> dict:
        base = self.stats()
        base.update(self._calculate_fragmentation())
        base.update(self._analyze_usage_patterns())
        return base

    @torch.no_grad()
    def _calculate_fragmentation(self) -> dict:
        active_mask = self.pool_offset > 0
        if active_mask.any():
            active_slots = self.pool_offset[active_mask]
            internal_frag = 1.0 - (active_slots.float().sum().item() / (active_mask.sum().item() * self.B))
        else:
            internal_frag = 0.0
        total_free_slots = len(self.free_list) + (self.pool_cap - self.pool_top)
        external_frag = 1.0 - (self.pool_top / self.pool_cap) if self.pool_cap > 0 else 0.0
        return {
            'internal_fragmentation': internal_frag,
            'external_fragmentation': external_frag,
            'total_free_slots': total_free_slots,
        }

    def _analyze_usage_patterns(self) -> dict:
        active_slots = self.pool_offset[self.pool_offset > 0]
        if active_slots.numel() == 0:
            return {
                'avg_occupancy_rate': 0.0,
                'median_occupancy_rate': 0.0,
                'occupancy_std': 0.0,
                'low_slots': 0,
                'high_slots': 0,
                'occupancy_histogram': [0]*10
            }
        occupancy_rates = active_slots.float() / self.B
        low_occupancy_slots = occupancy_rates < 0.3
        high_occupancy_slots = occupancy_rates > 0.8
        low_counts = (active_slots[low_occupancy_slots]).sum().item()
        high_counts = (active_slots[high_occupancy_slots]).sum().item()
        bins = min(10, self.B+1)
        hist = torch.histc(occupancy_rates, bins=bins, min=0, max=1)
        total_counts = active_slots.sum().item()
        return {
            'avg_occupancy_rate': occupancy_rates.mean().item(),
            'median_occupancy_rate': occupancy_rates.median().item(),
            'occupancy_std': occupancy_rates.std().item(),
            'total_counts': total_counts,
            'low_slots': int(low_occupancy_slots.sum().item()),
            "low_counts": low_counts,
            "low_ratio": low_counts / total_counts if total_counts > 0 else 0.0,
            'high_slots': int(high_occupancy_slots.sum().item()),
            "high_counts": high_counts,
            "high_ratio": high_counts / total_counts if total_counts > 0 else 0.0,
            'occupancy_histogram': hist.cpu().tolist(),
        }

# SegmentedSlabPool: micro-slab pooling (e.g., B=32, s=8)
# - Keeps logical per-row buffer size B
# - Stores physically in micro-slabs of size s
# - Reduces internal fragmentation by allocating ceil(n/s) micro-slabs per row
# - Optimized for parallel append_batch and FULL-only read_rows
class SegmentedSlabPool(BufferBackend):
    """
    Row-addressable KV buffer pool with micro-slabs to mitigate fragmentation.

    External semantics (unchanged from big-slab design):
      - Each row represents one logical buffer of size B (e.g., 32).
      - append_element / append_batch: merge existing tokens + new, keep Top-B by score S.
      - FULL rows have exactly B tokens; AVAILABLE rows have < B.
      - read_rows is restricted to FULL rows for performance (vectorized gather).

    Physical storage:
      - Fixed-size micro-slabs of size s (e.g., 8) shared globally across rows.
      - Each row holds up to R = B//s micro-slabs.
      - Tokens in a row are stored densely across the first ceil(offset/s) micro-slabs,
        in order, offsets 0..s-1; FULL rows use exactly R micro-slabs fully.

    Parallel efficiency:
      - append_batch aggregates group-wise Top-B with two stable argsorts, then performs
        a single large scatter into the first need_m micro-slabs per row.
      - read_rows assumes rows are FULL and performs a single index_select + reshape.
    """

    # For consistency with caller; change if needed.

    def __init__(self,
                 buf_cap: int,             # logical per-row cap B (e.g., 32)
                 head_dim: int,                   # feature dim
                 num_heads: int = 16,             # number of heads
                 capacity: int = 0,
                 growth: int = 512,
                 free_policy: str = "immediate",  # or "lazy"
                 micro_slab_size: int = 4, # s (e.g., 4)
                 field_specs: Optional[Dict[str, str]] = None,  # {"K":"vector", "V":"vector", "S":"scalar"}
                 device: torch.device=torch.device("cuda"),
                 dtype: torch.dtype=torch.float16,
                 **kwargs
                 ):

        self.B = int(buf_cap)
        self.s = int(micro_slab_size)
        assert self.B % self.s == 0, "B must be divisible by micro_slab_size"
        self.R = self.B // self.s # micro-slabs per row
        self.D = int(head_dim)
        self.num_heads = int(num_heads)
        self.device = device
        self.dtype = dtype
        self._growth_seg = int(growth*self.R)  # in micro-slabs
        self._free_policy = free_policy
        self.debug = kwargs.get("debug", False)

        # field registry (same contract as SlabPool)
        self.field_specs: Dict[str, str] = {"K": "vector", "V": "vector", "S": "scalar"}
        if field_specs:
            self.field_specs.update(field_specs)

        # micro-slab pool (shared by all rows)
        self.seg_cap = int(capacity*self.R)
        self.seg_top = 0
        self.seg_free_list: List[int] = []

        # per-field segment tensors
        self.seg_fields: Dict[str, torch.Tensor] = {}
        for name, kind in self.field_specs.items():
            self.seg_fields[name] = self._init_seg_field(kind, self.seg_cap)

        # indicators (mask, offset, owner)
        self.seg_M = torch.zeros(self.seg_cap, self.s, dtype=torch.bool, device=device)
        self.seg_offset = torch.zeros(self.seg_cap, dtype=torch.int32, device=device)
        self.seg_owner = torch.full((self.seg_cap,), -1, dtype=torch.int32, device=device)

        # row space
        self.S_tot = 0
        self.row_state = torch.empty(0, dtype=torch.int8, device=device)   # BufState
        self.row_seg = torch.empty(0, self.R, dtype=torch.int32, device=device)  # per-row micro-slab ids (-1 if none)
        self.row_offset  = torch.empty(0, dtype=torch.int32, device=device)   # token count per row

    # ---------------- Utilities ----------------
    def _physical_size(self):
        return self.seg_cap * self.s
    def _logical_size(self):
        return self.S_tot

    def _stable_rows(self, rows: Union[torch.Tensor, List[int], int]) -> torch.Tensor:
        """Normalize row input to a tensor of shape [N] on device."""
        if isinstance(rows, int):
            return torch.tensor([rows], dtype=torch.long, device=self.device)
        elif isinstance(rows, list):
            return torch.tensor(rows, dtype=torch.long, device=self.device)
        return rows.to(self.device, dtype=torch.long)

    # ---------------- Field Management ----------------
    def _init_seg_field(self, kind: str, cap: int) -> torch.Tensor:
        """Initialize a segment field tensor based on kind."""
        if kind == "vector":
            return torch.zeros(cap, self.s, self.D, dtype=self.dtype, device=self.device)
        elif kind == "scalar":
            return torch.zeros(cap, self.s, dtype=torch.float32, device=self.device)
        else:
            raise ValueError(f"Unknown field kind: {kind}")

    def register_field(self, name: str, kind: str = "vector", dtype: Optional[torch.dtype] = None):
        """
        Dynamically add a field to the segment pool.
        New storage is allocated for existing capacity and will be zero-initialized.
        """
        if name in self.seg_fields:
            raise KeyError(f"Field {name} already exists.")
        # dtype is driven by vector/scalar convention; kept for compatibility
        _ = dtype  # unused in this implementation; tensors follow default dtype rules
        self.field_specs[name] = kind
        self.seg_fields[name] = self._init_seg_field(kind, self.seg_cap)

    def field_dtype(self, name: str) -> torch.dtype:
        """Return the dtype for a field without exposing the storage tensor."""
        return self.seg_fields[name].dtype

    @torch.no_grad()
    def resize_rows(self, num_heads: int, alloc_per_head: int):
        assert num_heads == self.num_heads, f"Expected head_num={self.num_heads}, got {num_heads}"
        S_new = num_heads * alloc_per_head
        if S_new <= self.S_tot:
            return
        
        rm = torch.full((num_heads, alloc_per_head), int(BufState.RESERVED), dtype=torch.int8, device=self.device)
        rs = torch.full((num_heads, alloc_per_head, self.R), -1, dtype=torch.int32, device=self.device)
        ro = torch.zeros((num_heads, alloc_per_head), dtype=torch.int32, device=self.device)

        if self.S_tot > 0:
            old_alloc = self.S_tot // num_heads
            rm[:, :old_alloc].copy_(self.row_state.view(num_heads, old_alloc))
            rs[:, :old_alloc, :].copy_(self.row_seg.view(num_heads, old_alloc, self.R))
            ro[:, :old_alloc].copy_(self.row_offset.view(num_heads, old_alloc))

        self.row_state = rm.reshape(-1)
        self.row_seg = rs.reshape(-1, self.R)
        self.row_offset  = ro.reshape(-1)

        self.S_tot = S_new
    
    def close(self):
        """Free all pool resources."""
        self.seg_fields.clear()
        self.seg_fields = {}
        self.seg_M = None
        self.seg_offset = None
        self.seg_owner = None

        self.row_state = None
        self.row_seg = None
        self.row_offset = None

        torch.cuda.empty_cache()

    @torch.no_grad()
    def _grow_seg_pool(self, add: int):
        if add <= 0:
            return
        new_cap = self.seg_cap + add

        def ext(old, shape, dtype=None, value=0):
            out = (old.new_empty if dtype is None else torch.empty)(shape, dtype=(dtype or old.dtype), device=self.device)
            if old.numel() > 0:
                out[:old.shape[0]].copy_(old)
            if out.shape[0] > old.shape[0]:
                if out.dim() == 1:
                    out[old.shape[0]:] = value
                else:
                    out[old.shape[0]:].fill_(value)
            return out

        # extend per-field tensors
        for name, kind in self.field_specs.items():
            old = self.seg_fields[name]
            if kind == "vector":
                self.seg_fields[name] = ext(old, (new_cap, self.s, self.D), dtype=self.dtype)
            else:  # scalar
                self.seg_fields[name] = ext(old, (new_cap, self.s), dtype=torch.float32, value=0.0)

        # extend indicators
        self.seg_M = ext(self.seg_M, (new_cap, self.s), dtype=torch.bool, value=False)
        self.seg_offset = ext(self.seg_offset, (new_cap,), dtype=torch.int32, value=0)
        self.seg_owner = ext(self.seg_owner, (new_cap,), dtype=torch.int32, value=-1)
        self.seg_cap = new_cap
        if self.debug:
            print(f"SegmentedSlabPool: grew micro-slab pool to {new_cap}")

    @torch.no_grad()
    def _init_segments(self, ids: torch.Tensor):
        """Initialize/clear segment data for given segment ids."""
        if ids.numel() == 0:
            return
        ids = ids.to(self.device, dtype=torch.long)
        # clear all fields
        for name, kind in self.field_specs.items():
            t = self.seg_fields[name]
            t.index_fill_(0, ids, 0)
        # reset indicators
        self.seg_M.index_fill_(0, ids, False)
        self.seg_offset.index_fill_(0, ids, 0)
        self.seg_owner.index_fill_(0, ids, -1)

    @torch.no_grad()
    def _alloc_segments(self, n: int) -> torch.Tensor:
        if n <= 0:
            return torch.empty(0, dtype=torch.int32, device=self.device)
        k = min(n, len(self.seg_free_list))
        take = []
        if k > 0:
            take = self.seg_free_list[-k:]
            del self.seg_free_list[-k:]
        m = n - k
        fresh = torch.empty(0, dtype=torch.int32, device=self.device)
        if m > 0:
            if self.seg_top + m > self.seg_cap:
                # over-provision by growth size to reduce future grows
                need = self.seg_top + m - self.seg_cap + self._growth_seg
                self._grow_seg_pool(need)
            fresh = torch.arange(self.seg_top, self.seg_top + m, device=self.device, dtype=torch.int32)
            self.seg_top += m
        ids = torch.empty(0, dtype=torch.int32, device=self.device)
        if k > 0:
            free_ids = torch.tensor(take, dtype=torch.int32, device=self.device)
            ids = free_ids
        if m > 0:
            ids = torch.cat([ids, fresh], 0) if ids.numel() > 0 else fresh
        if ids.numel() > 0:
            self._init_segments(ids)
        return ids

    @torch.no_grad()
    def _free_segments(self, ids: torch.Tensor):
        if ids.numel() == 0:
            return
        ids = ids.to(self.device, dtype=torch.long)
        self._init_segments(ids)
        self.seg_free_list.extend(ids.to("cpu", non_blocking=True).tolist())

    # ---------------- States ----------------
    @torch.no_grad()
    def get_state(self, rows=None) -> torch.Tensor:
        if rows is None:
            return self.row_state
        rows = self._stable_rows(rows)
        return self.row_state.index_select(0, rows)

    @torch.no_grad()
    def set_row_state(self, rows, state):
        """Set the state for specified rows."""
        rows = self._stable_rows(rows)
        if rows.numel() == 0:
            return
        if state == BufState.FREE:
            segs = self.row_seg.index_select(0, rows)  # [G,R]
            seg_ids = segs[segs >= 0]
            if seg_ids.numel() > 0:
                self._free_segments(seg_ids)
                self.row_seg.index_fill_(0, rows, -1)
                self.row_offset.index_fill_(0, rows, 0)
                
        self.row_state.index_fill_(0, rows, int(state))

    # -------------------- free/clean --------------------
    @torch.no_grad()
    def free_rows(self, rows, free: Optional[bool] = None):
        rows = self._stable_rows(rows)
        if free or self._free_policy == "immediate":
            self.set_row_state(rows, BufState.FREE)
        else:
            self.set_row_state(rows, BufState.HELD)

    def clean_rows(self, rows):
        """Clear data and mark given rows as RESERVED."""
        rows = self._stable_rows(rows)
        if rows.numel() == 0:
            return
        # Clear data
        segs = self.row_seg.index_select(0, rows)  # [G,R]
        seg_ids = segs[segs >= 0]
        if seg_ids.numel() > 0:
            self._free_segments(seg_ids)
        self.row_seg.index_fill_(0, rows, -1)
        self.row_offset.index_fill_(0, rows, 0)
        self.row_state.index_fill_(0, rows, int(BufState.RESERVED))
            
    # ---------------- Ensure per-row segment count ----------------
    @torch.no_grad()
    def _ensure_row_segments(self, rows: torch.Tensor, need_m: torch.Tensor):
        rows = self._stable_rows(rows)
        if rows.numel() == 0:
            return
        rs = self.row_seg.index_select(0, rows)            # [G,R]
        cur_m = (rs >= 0).sum(1)                            # [G]
        add = (need_m - cur_m).clamp_min(0)                 # [G]
        total_add = int(add.sum().item())
        if total_add == 0:
            return
        new_ids = self._alloc_segments(total_add)           # [total_add]
        # fill first 'add[g]' negative slots per row, vectorized
        neg = (rs < 0)
        # cumsum over columns to pick first k negatives per row
        neg_cum = neg.cumsum(dim=1)
        select = neg & (neg_cum <= add[:, None])
        idx_rows, idx_cols = torch.nonzero(select, as_tuple=True)
        assert idx_rows.numel() == total_add
        rs[idx_rows, idx_cols] = new_ids.to(torch.int32)
        # owner
        owners = rows.index_select(0, idx_rows).to(torch.int32)
        self.seg_owner.index_copy_(0, new_ids.to(torch.long), owners)
        # write back
        self.row_seg.index_copy_(0, rows, rs)

    # ----------------------------------------------------------------------
    # Storage Access Methods (for BufferWrapper.append_batch_dict)
    # ----------------------------------------------------------------------

    def ensure_materialized(self, rows: torch.Tensor):
        """Ensure specified rows have micro-slabs allocated for at least 1 token."""
        # SegmentedSlabPool allocates segments on-demand in scatter_tokens
        # This is a no-op; segments are allocated in update_row_indicators
        pass

    @torch.no_grad()
    def gather_existing_tokens(self, rows: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Read existing valid tokens from specified rows (from segments)."""
        rows = self._stable_rows(rows)
        device, D, B, s, R = self.device, self.D, self.B, self.s, self.R
        G = rows.numel()

        if G == 0:
            existing_data = {}
            for name, kind in self.field_specs.items():
                if kind == "vector":
                    existing_data[name] = torch.empty(0, D, dtype=self.seg_fields[name].dtype, device=device)
                else:
                    existing_data[name] = torch.empty(0, dtype=self.seg_fields[name].dtype, device=device)
            group_ids = torch.empty(0, dtype=torch.long, device=device)
            return existing_data, group_ids

        rs = self.row_seg.index_select(0, rows)  # [G, R]
        seg_valid = (rs >= 0)

        if not seg_valid.any():
            existing_data = {}
            for name, kind in self.field_specs.items():
                if kind == "vector":
                    existing_data[name] = torch.empty(0, D, dtype=self.seg_fields[name].dtype, device=device)
                else:
                    existing_data[name] = torch.empty(0, dtype=self.seg_fields[name].dtype, device=device)
            group_ids = torch.empty(0, dtype=torch.long, device=device)
            return existing_data, group_ids

        seg_ids = rs[seg_valid].to(torch.long)  # [Eseg]
        Mseg = self.seg_M.index_select(0, seg_ids)  # [Eseg, s]
        seg_rows = torch.nonzero(seg_valid, as_tuple=False)[:, 0]  # [Eseg]

        # Flatten by tokens
        ridx = seg_rows.repeat_interleave(s)  # [Eseg*s]
        Mf = Mseg.view(-1)
        grp_ex = ridx[Mf]

        existing_data: Dict[str, torch.Tensor] = {}
        for name, kind in self.field_specs.items():
            pool = self.seg_fields[name]
            Xseg = pool.index_select(0, seg_ids)  # [Eseg, s, D] or [Eseg, s]
            if kind == "vector":
                Xf = Xseg.view(-1, D)
                existing_data[name] = Xf[Mf]
            else:
                Xf = Xseg.view(-1)
                existing_data[name] = Xf[Mf]

        return existing_data, grp_ex

    @torch.no_grad()
    def clear_row_data(self, rows: torch.Tensor):
        """Clear all data and mask for specified rows (clear segments)."""
        rows = self._stable_rows(rows)
        if rows.numel() == 0:
            return

        rs = self.row_seg.index_select(0, rows)  # [G, R]
        seg_ids = rs[rs >= 0].to(torch.long)

        if seg_ids.numel() > 0:
            for name in self.field_specs.keys():
                self.seg_fields[name].index_fill_(0, seg_ids, 0)
            self.seg_M.index_fill_(0, seg_ids, False)
            self.seg_offset.index_fill_(0, seg_ids, 0)

    @torch.no_grad()
    def scatter_tokens(self, rows: torch.Tensor, cols: torch.Tensor,
                       data: Dict[str, torch.Tensor]):
        """Write tokens to physical storage at specified positions (into segments)."""
        if rows.numel() == 0:
            return

        s, R = self.s, self.R
        E_kept = rows.numel()
        
        # Compute needed segments per unique row
        uniq_rows, inv_rows = torch.unique(rows, sorted=True, return_inverse=True)
        
        # For each unique row, find max column to determine segments needed
        max_cols = torch.zeros(uniq_rows.numel(), dtype=torch.long, device=self.device)
        max_cols.scatter_reduce_(0, inv_rows, cols.to(torch.long), reduce='amax', include_self=False)
        need_m = ((max_cols + s) // s).clamp_max(R)  # need_m[i] = max segments needed for uniq_rows[i]
        
        # Ensure segments are allocated
        self._ensure_row_segments(uniq_rows, need_m.to(torch.int32))
        
        # Now get row_seg for the rows being written
        rs = self.row_seg.index_select(0, rows)  # [E_kept, R]
        
        # Compute segment and offset within segment
        seg_local = (cols // s).clamp_max(R - 1)
        off = cols % s
        
        # Get destination segment ids
        dst_seg = rs[torch.arange(E_kept, device=self.device), seg_local].to(torch.long)

        for name, kind in self.field_specs.items():
            if name not in data:
                continue
            self.seg_fields[name][dst_seg, off] = data[name]

        self.seg_M[dst_seg, off] = True

    @torch.no_grad()
    def update_row_indicators(self, rows: torch.Tensor, counts: torch.Tensor,
                              kept_rows: torch.Tensor, kept_cols: torch.Tensor):
        """Update row indicators (update offset/state, free extra segments)."""
        rows = self._stable_rows(rows)
        if rows.numel() == 0:
            return

        G = rows.numel()
        s, R = self.s, self.R
        ST = BufState

        # Compute needed micro-slabs based on counts
        need_m = ((counts + (s - 1)) // s).clamp_max(R)

        # Update per-segment counts if there are kept tokens
        if kept_rows.numel() > 0:
            rs = self.row_seg.index_select(0, kept_rows)
            seg_local = (kept_cols // s).clamp_max(R - 1)
            E_kept = kept_rows.numel()
            dst_seg = rs[torch.arange(E_kept, device=self.device), seg_local].to(torch.long)
            
            uids, ucnt = torch.unique(dst_seg, return_counts=True)
            self.seg_offset.index_copy_(0, uids, ucnt.to(torch.int32))

        # Update row_offset and row_state
        self.row_offset.index_copy_(0, rows, counts)
        new_state = torch.where(counts == self.B,
                                torch.full_like(counts, int(ST.FULL), dtype=torch.int8),
                                torch.full_like(counts, int(ST.AVAILABLE), dtype=torch.int8))
        self.row_state.index_copy_(0, rows, new_state)

        # Free extra segments beyond need_m
        posR = torch.arange(R, device=self.device)
        rs = self.row_seg.index_select(0, rows)
        cur_m = (rs >= 0).sum(1)
        extra = (cur_m - need_m).clamp_min(0)
        if int(extra.sum().item()) > 0:
            drop_mask = (posR[None, :] >= need_m[:, None]) & (rs >= 0)
            extra_ids = rs[drop_mask]
            if extra_ids.numel() > 0 and self._free_policy == "immediate":
                self._free_segments(extra_ids)
            rs[drop_mask] = -1
            self.row_seg.index_copy_(0, rows, rs)

    # ---------------- FULL-only read ----------------
    @torch.no_grad()
    def full_rows(self, restrict_rows: Optional[torch.Tensor] = None) -> torch.Tensor:
        if restrict_rows is None:
            mask = (self.row_state == int(BufState.FULL))
            return torch.nonzero(mask, as_tuple=False).squeeze(1)
        else:
            rr = self._stable_rows(restrict_rows)
            st = self.row_state.index_select(0, rr)
            return rr[st == int(BufState.FULL)]

    # ---------------- Read (dict interface) ----------------
    @torch.no_grad()
    def read_rows_dict(self, 
                       rows: Union[List[int], torch.Tensor],
                       fields: Optional[List[str]] = None
                       ) -> Dict[str, torch.Tensor]:
        """
        Read and pack all fields for given rows.
        Args: 
            rows: [G] int64/32 row ids. 
            Rows without materialized slabs return zeros / False. 
            Rows are not required to be FULL.
            Rows are not required to be unique.
            fields: Optional list of field names to read. If None, reads all fields.

        Returns:
            out: dict of {field_name: tensor}, shapes: vector=[G,B,D], scalar=[G,B]
            out["M"]: mask tensor [G,B] bool
        """
        rows = self._stable_rows(rows)
        device, D, B, s, R = self.device, self.D, self.B, self.s, self.R
        G = rows.numel()

        if G == 0:
            out: Dict[str, torch.Tensor] = {}
            for name, kind in self.field_specs.items():
                if fields is not None and name not in fields:
                    continue
                if kind == "vector":
                    out[name] = torch.zeros(0, B, D, dtype=self.seg_fields[name].dtype, device=device)
                else:
                    out[name] = torch.zeros(0, B, dtype=self.seg_fields[name].dtype, device=device)
            out["M"] = torch.zeros(0, B, dtype=torch.bool, device=device)
            return out

        # support non-FULL rows
        st = self.row_state.index_select(0, rows)  # [G]
        rs = self.row_seg.index_select(0, rows)    # [G,R]
        have = (st == int(BufState.FULL)) | (st == int(BufState.AVAILABLE))
        have &= (rs >= 0).any(dim=1)  # [G]

        # allocate zeros
        out: Dict[str, torch.Tensor] = {}
        for name, kind in self.field_specs.items():
            if fields is not None and name not in fields:
                continue
            pool = self.seg_fields[name]
            if kind == "vector":
                out[name] = torch.zeros(G, B, D, dtype=pool.dtype, device=device)
            else:
                out[name] = torch.zeros(G, B, dtype=pool.dtype, device=device)
        out["M"] = torch.zeros(G, B, dtype=torch.bool, device=device)

        if have.any():
            have_rows = torch.nonzero(have, as_tuple=False).squeeze(1)
            have_rs = rs.index_select(0, have_rows)  # [Gh,R]
            Gh = have_rows.numel()
            
            # Handle -1 (unallocated segments) by replacing with 0 and masking
            valid_seg_mask = (have_rs >= 0)  # [Gh, R]
            have_seg_ids = have_rs.clone()
            have_seg_ids[~valid_seg_mask] = 0  # Replace -1 with 0 for safe indexing
            have_seg_ids = have_seg_ids.reshape(-1).to(torch.long)  # [Gh*R]

            Mseg = self.seg_M.index_select(0, have_seg_ids)  # [Gh*R, s]
            # Zero out invalid segments' mask
            Mseg_masked = Mseg.view(Gh, R, s)
            Mseg_masked = Mseg_masked * valid_seg_mask.unsqueeze(-1)  # [Gh, R, s]
            Mb_h = Mseg_masked.view(Gh, R * s)
            out["M"].index_copy_(0, have_rows, Mb_h[:, :B])

            for name, kind in self.field_specs.items():
                if fields is not None and name not in fields:
                    continue
                pool = self.seg_fields[name]
                Xseg = pool.index_select(0, have_seg_ids)
                if kind == "vector":
                    # Zero out invalid segments
                    Xseg_masked = Xseg.view(Gh, R, s, D)
                    Xseg_masked = Xseg_masked * valid_seg_mask.unsqueeze(-1).unsqueeze(-1)
                    Xb_h = Xseg_masked.view(Gh, R * s, D)
                    out[name].index_copy_(0, have_rows, Xb_h[:, :B])
                else:
                    # Zero out invalid segments
                    Xseg_masked = Xseg.view(Gh, R, s)
                    Xseg_masked = Xseg_masked * valid_seg_mask.unsqueeze(-1)
                    Xb_h = Xseg_masked.view(Gh, R * s)
                    out[name].index_copy_(0, have_rows, Xb_h[:, :B])

        return out

    @torch.no_grad()
    def read_rows_dict_fast(self, 
                            rows: Union[List[int], torch.Tensor],
                            fields: Optional[List[str]] = None
                            ) -> Dict[str, torch.Tensor]:
        """
        Pack buffers for given rows -> [G, B, ...].
        
        Rows must have valid data (AVAILABLE or FULL state, i.e. at least one segment allocated).
        Does NOT require FULL state - returns mask M for partial rows.
        
        More efficient than read_rows_dict when reading contiguous rows.

        Returns:
            out: dict of {field_name: tensor}, shapes: vector=[G,B,D], scalar=[G,B]
            out["M"]: mask tensor [G,B] bool indicating valid tokens
        """
        rows = self._stable_rows(rows)
        device, D, B, s, R = self.device, self.D, self.B, self.s, self.R
        G = rows.numel()

        if G == 0:
            out: Dict[str, torch.Tensor] = {}
            for name, kind in self.field_specs.items():
                if fields is not None and name not in fields:
                    continue
                if kind == "vector":
                    out[name] = torch.zeros(0, B, D, dtype=self.seg_fields[name].dtype, device=device)
                else:
                    out[name] = torch.zeros(0, B, dtype=self.seg_fields[name].dtype, device=device)
            out["M"] = torch.zeros(0, B, dtype=torch.bool, device=device)
            return out

        st = self.row_state.index_select(0, rows)
        # Only require rows to have valid data (AVAILABLE or FULL), not necessarily FULL
        assert torch.all((st == int(BufState.AVAILABLE)) | (st == int(BufState.FULL))), \
            "read_rows_dict_fast requires all rows to be AVAILABLE or FULL (have valid data)"

        # Gather R segments per row and reshape to [G, B, ...]
        rs = self.row_seg.index_select(0, rows)  # [G,R]
        
        # For AVAILABLE rows, some segments may be unallocated (marked as -1)
        # Replace -1 with 0 for safe indexing, invalid data will be masked out
        valid_mask = (rs >= 0)  # [G,R] - which segments are allocated
        seg_ids_safe = rs.clamp(min=0).reshape(-1).to(torch.long)  # [G*R]

        out: Dict[str, torch.Tensor] = {}
        for name, kind in self.field_specs.items():
            if fields is not None and name not in fields:
                continue
            pool = self.seg_fields[name]
            Xseg = pool.index_select(0, seg_ids_safe)
            if kind == "vector":
                Xb = Xseg.view(G, R * s, D)
                out[name] = Xb[:, :B].contiguous()
            else:
                Xb = Xseg.view(G, R * s)
                out[name] = Xb[:, :B].contiguous()

        # Build mask: read actual segment masks, but zero out unallocated segments
        Mseg = self.seg_M.index_select(0, seg_ids_safe)  # [G*R, s]
        # Mask out segments that were not allocated (-1 in row_seg)
        valid_seg_mask = valid_mask.unsqueeze(-1).expand(G, R, s).reshape(G * R, s)  # [G*R, s]
        Mseg = Mseg & valid_seg_mask
        Mb = Mseg.view(G, R * s)
        out["M"] = Mb[:, :B].contiguous()

        return out

    @torch.no_grad()
    def read_rows_dict_compressed(self, 
                                  rows: Union[List[int], torch.Tensor],
                                  fields: Optional[List[str]] = None
                                  ) -> Dict[str, torch.Tensor]:
        """
        Read and pack compressed buffers for given rows.
        Only valid tokens (where M=True) are returned.

        Args:
            rows: [G] int64/32 row ids. 
                  Rows are not required to be FULL.
                  Rows are not required to be unique.
            fields: Optional list of field names to read. If None, reads all fields.

        Returns:
            out: dict of {field_name: tensor}
                - vector fields: [total_valid, D]
                - scalar fields: [total_valid]
                - "M": [G, B] bool mask for reconstruction
        """
        rows = self._stable_rows(rows)
        device, D, B, s, R = self.device, self.D, self.B, self.s, self.R
        G = rows.numel()

        if G == 0:
            out: Dict[str, torch.Tensor] = {}
            for name, kind in self.field_specs.items():
                if fields is not None and name not in fields:
                    continue
                if kind == "vector":
                    out[name] = torch.empty(0, D, dtype=self.seg_fields[name].dtype, device=device)
                else:
                    out[name] = torch.empty(0, dtype=self.seg_fields[name].dtype, device=device)
            out["M"] = torch.zeros(0, B, dtype=torch.bool, device=device)
            return out

        # Get full data first
        st = self.row_state.index_select(0, rows)
        rs = self.row_seg.index_select(0, rows)
        have = (st == int(BufState.FULL)) | (st == int(BufState.AVAILABLE))
        have &= (rs >= 0).any(dim=1)

        # Build full mask [G, B]
        M_out = torch.zeros(G, B, dtype=torch.bool, device=device)

        out: Dict[str, torch.Tensor] = {}

        if have.any():
            have_rows = torch.nonzero(have, as_tuple=False).squeeze(1)
            have_rs = rs.index_select(0, have_rows)
            Gh = have_rows.numel()
            
            # Handle -1 (unallocated segments) by replacing with 0 and masking
            valid_seg_mask = (have_rs >= 0)  # [Gh, R]
            have_seg_ids = have_rs.clone()
            have_seg_ids[~valid_seg_mask] = 0  # Replace -1 with 0 for safe indexing
            have_seg_ids = have_seg_ids.reshape(-1).to(torch.long)

            Mseg = self.seg_M.index_select(0, have_seg_ids)
            # Zero out invalid segments' mask
            Mseg_masked = Mseg.view(Gh, R, s)
            Mseg_masked = Mseg_masked * valid_seg_mask.unsqueeze(-1)
            Mb_h = Mseg_masked.view(Gh, R * s)[:, :B]
            M_out.index_copy_(0, have_rows, Mb_h)

            m_flat = M_out.view(-1)  # [G*B] for indexing

            for name, kind in self.field_specs.items():
                if fields is not None and name not in fields:
                    continue
                pool = self.seg_fields[name]
                Xseg = pool.index_select(0, have_seg_ids)
                if kind == "vector":
                    Xb = torch.zeros(G, B, D, dtype=pool.dtype, device=device)
                    # Zero out invalid segments
                    Xseg_masked = Xseg.view(Gh, R, s, D)
                    Xseg_masked = Xseg_masked * valid_seg_mask.unsqueeze(-1).unsqueeze(-1)
                    Xb_h = Xseg_masked.view(Gh, R * s, D)[:, :B]
                    Xb.index_copy_(0, have_rows, Xb_h)
                    out[name] = Xb.view(-1, D)[m_flat]
                else:
                    Xb = torch.zeros(G, B, dtype=pool.dtype, device=device)
                    # Zero out invalid segments
                    Xseg_masked = Xseg.view(Gh, R, s)
                    Xseg_masked = Xseg_masked * valid_seg_mask.unsqueeze(-1)
                    Xb_h = Xseg_masked.view(Gh, R * s)[:, :B]
                    Xb.index_copy_(0, have_rows, Xb_h)
                    out[name] = Xb.view(-1)[m_flat]
        else:
            for name, kind in self.field_specs.items():
                if fields is not None and name not in fields:
                    continue
                if kind == "vector":
                    out[name] = torch.empty(0, D, dtype=self.seg_fields[name].dtype, device=device)
                else:
                    out[name] = torch.empty(0, dtype=self.seg_fields[name].dtype, device=device)

        out["M"] = M_out  # [G, B] - consistent with other read methods
        return out

    # ---------------- Write (dict interface) ----------------
    @torch.no_grad()
    def write_rows_dict(self, rows: torch.Tensor, data: Dict[str, torch.Tensor]):
        """
        Write given rows' buffers from input tensors.

        Rows are required to be unique. 
        Rows are required to not be FREE.
        If rows are RESERVED or AVAILABLE, they become AVAILABLE/FULL after write.

        Args:
            rows: [G] int64/32 row ids
            data: dict of {field_name: tensor}
                - vector fields: [G,B,D]
                - scalar fields: [G,B]
                - "M": [G,B] bool mask
        """
        rows = self._stable_rows(rows)
        device, D, B, s, R = self.device, self.D, self.B, self.s, self.R
        G = rows.numel()

        if G == 0:
            return

        unique_rows, counts = torch.unique(rows, return_counts=True)
        assert bool((counts == 1).all()), "Input rows must be unique."

        buf_states = self.row_state.index_select(0, rows)
        is_free = (buf_states == int(BufState.FREE))
        assert not bool(is_free.any()), "Cannot write to FREE rows."

        is_reserved = (buf_states == int(BufState.RESERVED))
        if is_reserved.any():
            self.set_available(rows[is_reserved])

        # Get mask and compute offsets
        assert "M" in data, "Missing mask 'M' in write_rows_dict"
        M_in = data["M"].to(torch.bool)  # [G,B]
        sum_M = M_in.sum(dim=1).to(torch.int32)  # [G]

        # Calculate micro-slabs needed per row
        need_m = ((sum_M + (s - 1)) // s).clamp_max(R)

        # Ensure enough micro-slabs for each row
        self._ensure_row_segments(rows, need_m)

        # Get segment ids for rows
        rs = self.row_seg.index_select(0, rows)  # [G,R]

        # Clear target segs (first need_m per row)
        posR = torch.arange(R, device=device)
        use_mask = posR[None, :] < need_m[:, None]  # [G,R]
        seg_use = rs[use_mask].to(torch.long)
        if seg_use.numel() > 0:
            for name in self.field_specs.keys():
                self.seg_fields[name].index_fill_(0, seg_use, 0)
            self.seg_M.index_fill_(0, seg_use, False)
            self.seg_offset.index_fill_(0, seg_use, 0)

        # Scatter data to segments
        # For each row, we need to scatter its valid tokens to segments
        # Token at position [g, b] goes to segment rs[g, b//s] at offset b%s
        
        # Create indices for scatter
        g_idx = torch.arange(G, device=device, dtype=torch.long).unsqueeze(1).expand(G, B)  # [G,B]
        b_idx = torch.arange(B, device=device, dtype=torch.long).unsqueeze(0).expand(G, B)  # [G,B]
        seg_local = (b_idx // s).clamp_max(R - 1)  # [G,B]
        off = b_idx % s  # [G,B]

        # Only write where mask is True
        m_flat = M_in.reshape(-1)  # [G*B]
        g_flat = g_idx.reshape(-1)[m_flat]
        seg_local_flat = seg_local.reshape(-1)[m_flat]
        off_flat = off.reshape(-1)[m_flat]

        # Get destination segment ids
        dst_seg = rs[g_flat, seg_local_flat].to(torch.long)

        # Write all fields
        for name, kind in self.field_specs.items():
            assert name in data, f"Input data missing field '{name}'."
            pool = self.seg_fields[name]
            if kind == "vector":
                assert data[name].shape == (G, B, D), f"{name}: expected [{G},{B},{D}], got {list(data[name].shape)}"
                src = data[name].view(-1, D)[m_flat]
                pool[dst_seg, off_flat] = src.to(pool.dtype)
            else:
                assert data[name].shape == (G, B), f"{name}: expected [{G},{B}], got {list(data[name].shape)}"
                src = data[name].view(-1)[m_flat]
                pool[dst_seg, off_flat] = src.to(pool.dtype)

        # Write mask
        self.seg_M[dst_seg, off_flat] = True

        # Update per-seg counts
        uids, ucnt = torch.unique(dst_seg, return_counts=True)
        self.seg_offset.index_copy_(0, uids, ucnt.to(torch.int32))

        # Update row offsets and states
        self.row_offset.index_copy_(0, rows, sum_M)
        new_state = torch.where(sum_M == B,
                                torch.full_like(sum_M, int(BufState.FULL), dtype=torch.int8),
                                torch.full_like(sum_M, int(BufState.AVAILABLE), dtype=torch.int8))
        self.row_state.index_copy_(0, rows, new_state)

        # Free extra segs (beyond need_m)
        cur_m = (rs >= 0).sum(1)
        extra = (cur_m - need_m).clamp_min(0)
        if int(extra.sum().item()) > 0:
            drop_mask = (posR[None, :] >= need_m[:, None]) & (rs >= 0)
            extra_ids = rs[drop_mask]
            if extra_ids.numel() > 0 and self._free_policy == "immediate":
                self._free_segments(extra_ids)
            rs[drop_mask] = -1
            self.row_seg.index_copy_(0, rows, rs)

    # ---------------- Stats ----------------
    @torch.no_grad()
    def stats(self) -> dict:
        active_segs = int((self.seg_offset > 0).sum().item())
        data_count = int(self.seg_offset.sum().item())
        alloc_count = self.seg_cap * self.s
        occupancy = data_count / float(max(1, alloc_count))
        # logical max segments = S_tot * R
        compression = float(self.seg_cap) / float(max(1, self.S_tot * self.R))
        return dict(
            fields=list(self.field_specs.keys()),
            buf_capacity=self.B,
            head_num=self.num_heads,
            seg_size=self.s,
            seg_per_row=self.R,
            head_dim=self.D,
            physical_size=self._physical_size(),
            logical_size=self._logical_size(),
            seg_cap=int(self.seg_cap),
            seg_top=int(self.seg_top),
            free_segments=len(self.seg_free_list),
            active_segments=active_segs,
            data_count=data_count,
            alloc_count=alloc_count,
            segment_occupancy_ratio=occupancy,
            segment_compression_ratio=compression,
        )
    
    @torch.no_grad()
    def detailed_stats(self) -> dict:
        stats_basic = self.stats()
        stats_frag = self._calculate_fragmentation()
        stats_usage = self._analyze_usage_patterns()
        stats_basic.update(stats_frag)
        stats_basic.update(stats_usage)
        return stats_basic

    @torch.no_grad()
    def _calculate_fragmentation(self) -> dict:
        """Internal and external fragmentation analysis for micro-slabs."""
        # Internal fragmentation: allocated but unused slots inside active micro-slabs
        active_mask = self.seg_offset > 0
        if active_mask.any():
            active_slots = self.seg_offset[active_mask]
            internal_frag = 1.0 - (active_slots.float().sum().item() / (active_mask.sum().item() * self.s))
        else:
            internal_frag = 0.0

        # External fragmentation: free micro-slab distribution
        total_free_segments = len(self.seg_free_list) + max(0, self.seg_cap - self.seg_top)
        external_frag = 1.0 - (self.seg_top / self.seg_cap) if self.seg_cap > 0 else 0.0

        return {
            'internal_fragmentation': internal_frag,
            'external_fragmentation': external_frag,
            'total_free_segments': total_free_segments,
        }

    def _analyze_usage_patterns(self) -> dict:
        """Analyze occupancy distribution among micro-slabs."""
        active_seg_offset = self.seg_offset[self.seg_offset > 0]
        if active_seg_offset.numel() > 0:
            total_counts = active_seg_offset.sum().item()
            occupancy_rates = active_seg_offset.float() / self.s
            low_segments = occupancy_rates < 0.3
            low_counts = active_seg_offset[low_segments].sum().item()
            high_segments = occupancy_rates > 0.8
            high_counts = active_seg_offset[high_segments].sum().item()
            occupancy_stats = {
                'avg_occupancy_rate': occupancy_rates.mean().item(),
                'median_occupancy_rate': occupancy_rates.median().item(),
                'occupancy_std': occupancy_rates.std().item(),
                'total_counts': total_counts,
                'low_segments': (occupancy_rates < 0.3).sum().item(),
                'low_counts': low_counts,
                'low_ratio': low_counts / total_counts if total_counts > 0 else 0.0,
                'high_segments': (occupancy_rates > 0.8).sum().item(),
                'high_counts': high_counts,
                'high_ratio': high_counts / total_counts if total_counts > 0 else 0
            }

            # Simplified histogram
            hist = torch.histc(occupancy_rates, bins=min(10,self.s+1), min=0, max=1)
            occupancy_stats['occupancy_histogram'] = hist.cpu().tolist()
        else:
            occupancy_stats = {
                'avg_occupancy_rate': 0.0,
                'median_occupancy_rate': 0.0,
                'occupancy_std': 0.0,
                'low_egments': 0,
                'high_segments': 0,
                'occupancy_histogram': [0]
            }

        return occupancy_stats


# Backend registry - maps type names to backend classes
backend_registry = {
    "static": StaticBuffer,
    "slab": SlabPool,
    "segment": SegmentedSlabPool,
}

# Backward compatible alias
allocator_registry = backend_registry


def create_buffer(alloc_type: str, **kwargs) -> BufferInterface:
    """
    Factory function to create a BufferWrapper-wrapped backend.
    
    This is the recommended way to create buffer instances for application code.
    The wrapper provides the append_batch_dict implementation with Top-B selection.
    
    Args:
        alloc_type: One of "static", "slab", or "segment"
        **kwargs: Arguments passed to the backend constructor
        
    Returns:
        BufferWrapper instance wrapping the specified backend
        
    Example:
        buffer = create_buffer("slab", buf_cap=32, head_dim=64, num_heads=16)
        buffer.resize_rows(num_heads=16, alloc_per_head=1000)
        overflow, rows_over = buffer.append_batch_dict(rows, data)
    """
    if alloc_type not in backend_registry:
        raise ValueError(f"Unknown allocator type: {alloc_type}. "
                        f"Available types: {list(backend_registry.keys())}")
    backend_cls = backend_registry[alloc_type]
    backend = backend_cls(**kwargs)
    return BufferWrapper(backend)


if __name__ == "__main__":
    pass