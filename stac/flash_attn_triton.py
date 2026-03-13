# Copyright (c) 2025 STAC Authors. All rights reserved.
# Ref: https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn_interface/src/flash_attn_triton.py

import math
import torch
import triton
import triton.language as tl
import torch.nn.functional as F

#& -----------------------------
#& forward + LSE (+ optional O) kernel  [m-major]
#& -----------------------------
@triton.heuristics({
    "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
    "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
    "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
})
@triton.jit
def _fwd_lse_o_kernel(
    Q, K, V, Bias, Out, LSE, TMP,             # tensors
    softmax_scale,                            # scalar
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_bb, stride_bh, stride_bm,
    stride_ob, stride_oh, stride_om,
    nheads, seqlen_q, seqlen_k, seqlen_q_round, headdim,
    BIAS_TYPE: tl.constexpr,                  # "none" | "vector" | "matrix"
    IS_CAUSAL: tl.constexpr,
    WRITE_O: tl.constexpr,                    # 1: write Out, 0: skip computing O
    BLOCK_HEADDIM: tl.constexpr,              # >= headdim, power of two
    EVEN_M: tl.constexpr, EVEN_N: tl.constexpr, EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # Program ids
    pid_m = tl.program_id(0)                  # block id on sequence M (rows)
    off_hb = tl.program_id(1)                 # fused (batch, head)
    off_b = off_hb // nheads
    off_h = off_hb % nheads

    # Offsets in M / N / D
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    # Build pointers
    q_ptrs = Q + off_b * stride_qb + off_h * stride_qh + (offs_m[:, None] * stride_qm + offs_d[None, :])
    k_ptrs = K + off_b * stride_kb + off_h * stride_kh + (offs_n[:, None] * stride_kn + offs_d[None, :])
    v_ptrs = V + off_b * stride_vb + off_h * stride_vh + (offs_n[:, None] * stride_vn + offs_d[None, :])

    if BIAS_TYPE == "vector":
        b_ptrs = Bias + off_b*stride_bb + off_h*stride_bh + offs_n
    elif BIAS_TYPE == "matrix":
        b_ptrs = Bias + off_b*stride_bb + off_h*stride_bh + (offs_m[:, None]*stride_bm + offs_n[None, :])

    # SRAM accumulators
    acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)
    lse_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)   # running log-sum-exp
    m_i   = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)   # running max per row
    t_ptrs = TMP + off_hb * seqlen_q_round + offs_m               # scratch (compiler quirk)

    # Load Q tile
    if EVEN_M & EVEN_HEADDIM:
        q = tl.load(q_ptrs)
    else:
        q = tl.load(q_ptrs, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0)

    # Determine causal end in N for this row-tile
    end_n = seqlen_k if not IS_CAUSAL else tl.minimum((pid_m + 1) * BLOCK_M, seqlen_k)

    # Sweep N in BLOCK_N
    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        # Load K / V subtile
        if EVEN_N & EVEN_HEADDIM:
            k = tl.load(k_ptrs + start_n * stride_kn)
            if WRITE_O:
                v = tl.load(v_ptrs + start_n * stride_vn)
        else:
            mask_nv = ((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim)
            k = tl.load(k_ptrs + start_n * stride_kn, mask=mask_nv, other=0.0)
            if WRITE_O:
                v = tl.load(v_ptrs + start_n * stride_vn, mask=mask_nv, other=0.0)

        # Scores
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))

        # Tail & causal masks
        if not EVEN_N:
            qk += tl.where((start_n + offs_n)[None, :] < seqlen_k, 0.0, float("-inf"))
        if IS_CAUSAL:
            qk += tl.where(offs_m[:, None] >= (start_n + offs_n)[None, :], 0.0, float("-inf"))

        # Bias
        if BIAS_TYPE != "none":
            if BIAS_TYPE == "vector":
                if EVEN_N:
                    bias = tl.load(b_ptrs + start_n).to(tl.float32)
                else:
                    bias = tl.load(b_ptrs + start_n, mask=(start_n + offs_n) < seqlen_k, other=0.0).to(tl.float32)
                bias = bias[None, :]
                qk = qk * softmax_scale + bias
            else:
                if EVEN_M & EVEN_N:
                    bias = tl.load(b_ptrs + start_n).to(tl.float32)
                else:
                    bias = tl.load(
                        b_ptrs + start_n,
                        mask=(offs_m[:, None] < seqlen_q) & ((start_n + offs_n)[None, :] < seqlen_k),
                        other=0.0,
                    ).to(tl.float32)
                qk = qk * softmax_scale + bias

            # log-sum-exp update
            m_ij = tl.maximum(tl.max(qk, axis=1), lse_i)
            p = tl.exp(qk - m_ij[:, None])
        else:
            # scale inside
            m_ij = tl.maximum(tl.max(qk, axis=1) * softmax_scale, lse_i)
            p = tl.exp(qk * softmax_scale - m_ij[:, None])

        l_ij = tl.sum(p, axis=1)

        # Update accumulators for O if requested
        if WRITE_O:
            # rescale old acc
            alpha = tl.exp(m_i - m_ij)
            tl.store(t_ptrs, alpha); alpha = tl.load(t_ptrs)       # workaround for old compiler
            acc_o *= alpha[:, None]
            # add p @ v
            p = p.to(v.dtype)
            acc_o += tl.dot(p, v)

        # Update running stats
        m_i = m_ij
        lse_i = m_ij + tl.log(tl.exp(lse_i - m_ij) + l_ij)

    # Final rescale & writeback
    # store lse rows
    lse_ptrs = LSE + off_hb * seqlen_q_round + offs_m
    tl.store(lse_ptrs, lse_i, mask=offs_m < seqlen_q)

    if WRITE_O:
        o_scale = tl.exp(m_i - lse_i)
        tl.store(t_ptrs, o_scale); o_scale = tl.load(t_ptrs)
        acc_o *= o_scale[:, None]

        offs_d = tl.arange(0, BLOCK_HEADDIM)
        out_ptrs = Out + off_b*stride_ob + off_h*stride_oh + (offs_m[:, None]*stride_om + offs_d[None, :])
        tl.store(out_ptrs, acc_o, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim))


# ---------------------------------------
def fa_forward_lse(q, k, v, bias=None, causal=False, softmax_scale=None):
    """
    q: [B, M, H, D], k,v: [B, N, H, D] (fp16/bf16, CUDA)
    bias: None | [B,H,1,N] (vector) | [B,H,M,N] (matrix)
    return: o, lse
    """
    B, M, H, D = q.shape
    N = k.shape[1]
    assert q.dtype in (torch.float16, torch.bfloat16) and q.is_cuda
    assert k.shape == (B, N, H, D) and v.shape == (B, N, H, D)
    assert D <= 128, "FlashAttention only support head dimensions up to 128"
    assert k.dtype == v.dtype == q.dtype, "All tensors must have the same type"
    assert bias is None or (bias.dtype == q.dtype and bias.is_cuda)
    softmax_scale = softmax_scale or 1.0 / math.sqrt(D)

    M_rounded = math.ceil(M / 128) * 128

    # bias preproc
    bias_type = "none"
    b_strides = (0, 0, 0)
    if bias is not None:
        if bias.stride(-1) != 1:
            bias = bias.contiguous()
        if bias.shape[2:] == (1, N):
            bias_type = "vector"
        elif bias.shape[2:] == (M, N):
            bias_type = "matrix"
        else:
            raise RuntimeError("bias last two dims must be (1,N) or (M,N)")
        bias = bias.expand(B, H, M, N)
        b_strides = (bias.stride(0), bias.stride(1), bias.stride(2))

    # buffers
    lse = torch.empty((B, H, M_rounded), device=q.device, dtype=torch.float32)
    tmp = torch.empty_like(lse)
    out = torch.empty_like(q)

    # launch
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_HEADDIM = max(triton.next_power_of_2(D), 16)
    num_warps = 4 if D <= 64 else 8
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]), B * H)

    _fwd_lse_o_kernel[grid](
        q, k, v, bias, out, lse, tmp,
        softmax_scale,
        q.stride(0), q.stride(2), q.stride(1),
        k.stride(0), k.stride(2), k.stride(1),
        v.stride(0), v.stride(2), v.stride(1),
        *b_strides,
        out.stride(0),
        out.stride(2),
        out.stride(1),
        H, M, N, M_rounded, D,
        bias_type, causal,
        1,              # WRITE_O
        BLOCK_HEADDIM,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=num_warps, num_stages=1,
    )
    return out, lse, softmax_scale


#& -----------------------------
#& forward + LSE + col_sum kernel
#& -----------------------------
@triton.heuristics({
    "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
    "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
    "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
})
@triton.jit
def _fwd_lse_o_and_colsum_kernel(
    Q, K, V, Bias,
    Out, LSE, TMP,
    ColSum,                               # [B, H, N] fp32
    softmax_scale,
    # strides
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_bb, stride_bh, stride_bm,
    stride_ob, stride_oh, stride_om,
    stride_cb, stride_ch, stride_cn,      # ColSum strides: B,H,N
    nheads, seqlen_q, seqlen_k, seqlen_q_round, headdim,
    BIAS_TYPE: tl.constexpr, IS_CAUSAL: tl.constexpr,
    WRITE_O: tl.constexpr,                # 1 to compute/store O, 0 to skip
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr, EVEN_N: tl.constexpr, EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # ------------------------
    # program ids & offsets
    # ------------------------
    pid_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b  = off_hb // nheads
    off_h  = off_hb % nheads

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    # base pointers for this (b,h) and row tile
    q_ptrs = Q + off_b*stride_qb + off_h*stride_qh + (offs_m[:, None]*stride_qm + offs_d[None, :])
    k_base = K + off_b*stride_kb + off_h*stride_kh
    v_base = V + off_b*stride_vb + off_h*stride_vh

    if BIAS_TYPE == "vector":
        b_vec_base = Bias + off_b*stride_bb + off_h*stride_bh
    elif BIAS_TYPE == "matrix":
        b_mat_base = Bias + off_b*stride_bb + off_h*stride_bh

    # scratch & stat buffers
    t_ptrs = TMP + off_hb*seqlen_q_round + offs_m
    lse_i  = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    m_i    = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    acc_o  = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)

    # load Q tile
    q = tl.load(
        q_ptrs,
        mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
        other=0.0
    )

    # ------------- PASS 1: streaming softmax, optional O -------------
    end_n = seqlen_k if not IS_CAUSAL else tl.minimum((pid_m + 1) * BLOCK_M, seqlen_k)
    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        k_ptrs = k_base + ( (start_n + offs_n)[:, None]*stride_kn + offs_d[None, :] )
        v_ptrs = v_base + ( (start_n + offs_n)[:, None]*stride_vn + offs_d[None, :] )

        k = tl.load(k_ptrs, mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim), other=0.0)
        if WRITE_O:
            v = tl.load(v_ptrs, mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim), other=0.0)

        # scores
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))

        # masks
        if not EVEN_N:
            qk += tl.where((start_n + offs_n)[None, :] < seqlen_k, 0.0, float("-inf"))
        if IS_CAUSAL:
            qk += tl.where(offs_m[:, None] >= (start_n + offs_n)[None, :], 0.0, float("-inf"))

        # bias
        if BIAS_TYPE != "none":
            if BIAS_TYPE == "vector":
                b = tl.load(b_vec_base + (start_n + offs_n), mask=(start_n + offs_n) < seqlen_k, other=0.0).to(tl.float32)
                qk = qk * softmax_scale + b[None, :]
            else:
                b_ptrs = b_mat_base + (offs_m[:, None]*stride_bm + (start_n + offs_n)[None, :])
                b = tl.load(b_ptrs,
                            mask=(offs_m[:, None] < seqlen_q) & ((start_n + offs_n)[None, :] < seqlen_k),
                            other=0.0).to(tl.float32)
                qk = qk * softmax_scale + b
            m_ij = tl.maximum(tl.max(qk, axis=1), lse_i)
            p = tl.exp(qk - m_ij[:, None])
        else:
            m_ij = tl.maximum(tl.max(qk, axis=1) * softmax_scale, lse_i)
            p = tl.exp(qk * softmax_scale - m_ij[:, None])

        l_ij = tl.sum(p, axis=1)

        if WRITE_O:
            alpha = tl.exp(m_i - m_ij)
            tl.store(t_ptrs, alpha); alpha = tl.load(t_ptrs)
            acc_o *= alpha[:, None]
            acc_o += tl.dot(p.to(v.dtype), v)

        # update running stats
        m_i   = m_ij
        lse_i = m_ij + tl.log(tl.exp(lse_i - m_ij) + l_ij)

    # write LSE
    tl.store(LSE + off_hb*seqlen_q_round + offs_m, lse_i, mask=offs_m < seqlen_q)

    # finalize and store O (optional)
    if WRITE_O:
        o_scale = tl.exp(m_i - lse_i)
        tl.store(t_ptrs, o_scale); o_scale = tl.load(t_ptrs)
        acc_o *= o_scale[:, None]
        out_ptrs = Out + off_b*stride_ob + off_h*stride_oh + (offs_m[:, None]*stride_om + offs_d[None, :])
        tl.store(out_ptrs, acc_o, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim))

    # ------------- PASS 2: recompute true p using final LSE -------------
    # Reload row-wise LSE into registers
    lse_final = tl.load(LSE + off_hb*seqlen_q_round + offs_m, mask=offs_m < seqlen_q, other=-float("inf"))

    for start_n in range(0, end_n, BLOCK_N): # Loop over K in blocks of BLOCK_N
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # K tile
        k_ptrs = k_base + ( (start_n + offs_n)[:, None]*stride_kn + offs_d[None, :] )
        k = tl.load(k_ptrs, mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim), other=0.0)
        # scores
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k)) # [BLOCK_M, BLOCK_N]

        # masks
        if not EVEN_N:
            qk += tl.where((start_n + offs_n)[None, :] < seqlen_k, 0.0, float("-inf"))
        if IS_CAUSAL:
            qk += tl.where(offs_m[:, None] >= (start_n + offs_n)[None, :], 0.0, float("-inf"))

        # bias
        if BIAS_TYPE != "none":
            if BIAS_TYPE == "vector":
                b = tl.load(b_vec_base + (start_n + offs_n), mask=(start_n + offs_n) < seqlen_k, other=0.0).to(tl.float32)
                qk = qk * softmax_scale + b[None, :]
            else:
                b_ptrs = b_mat_base + (offs_m[:, None]*stride_bm + (start_n + offs_n)[None, :])
                b = tl.load(b_ptrs,
                            mask=(offs_m[:, None] < seqlen_q) & ((start_n + offs_n)[None, :] < seqlen_k),
                            other=0.0).to(tl.float32)
                qk = qk * softmax_scale + b
            p_true = tl.exp(qk - lse_final[:, None])
        else:
            p_true = tl.exp(qk * softmax_scale - lse_final[:, None])

        # rows beyond M are zero
        p_true = tl.where(offs_m[:, None] < seqlen_q, p_true, 0.0)

        # sum along rows -> [BLOCK_N]
        col_sum_tile = tl.sum(p_true, axis=0)

        # atomic add to [B,H,N]
        c_ptrs = ColSum + off_b*stride_cb + off_h*stride_ch + (start_n + offs_n)*stride_cn
        tl.atomic_add(c_ptrs, col_sum_tile, mask=(start_n + offs_n) < seqlen_k)

def fa_forward_colsum(q, k, v, bias=None, causal=False, softmax_scale=None, write_o=True):
    """
    Args: 
        q,k,v [B,M/H,D], 
        bias: None | [B,H,1,N] | [B,H,M,N]

    Returns: 
        out[B,M,H,D] (or None if write_o=False), 
        lse[B,H,M_rounded], 
        col_sum[B,H,N] (fp32)
    """
    import math, triton
    B, M, H, D = q.shape
    N = k.shape[1]
    assert q.is_cuda and q.dtype in (torch.float16, torch.bfloat16)
    assert k.shape == (B, N, H, D) and v.shape == (B, N, H, D)
    scale = softmax_scale or 1.0 / math.sqrt(D)

    # bias
    bias_type = "none"
    b_strides = (0,0,0)
    if bias is not None:
        if bias.stride(-1) != 1:
            bias = bias.contiguous()
        if bias.shape[2:] == (1, N):
            bias_type = "vector"
        elif bias.shape[2:] == (M, N):
            bias_type = "matrix"
        else:
            raise RuntimeError("bias last two dims must be (1,N) or (M,N)")
        bias = bias.expand(B, H, M, N)
        b_strides = (bias.stride(0), bias.stride(1), bias.stride(2))

    M_rounded = math.ceil(M / 128) * 128
    out = torch.empty_like(q) if write_o else q.new_empty(0)
    lse = torch.empty((B, H, M_rounded), device=q.device, dtype=torch.float32)
    tmp = torch.empty_like(lse)
    col_sum = torch.zeros((B, H, N), device=q.device, dtype=torch.float32)

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_HEADDIM = max(triton.next_power_of_2(D), 16)
    num_warps = 4 if D <= 64 else 8
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]), B * H)

    _fwd_lse_o_and_colsum_kernel[grid](
        q, k, v, bias if bias is not None else torch.empty(1, device=q.device, dtype=q.dtype),
        out, lse, tmp, col_sum,
        scale,
        q.stride(0), q.stride(2), q.stride(1),
        k.stride(0), k.stride(2), k.stride(1),
        v.stride(0), v.stride(2), v.stride(1),
        *b_strides,
        out.stride(0) if write_o else 0,
        (out.stride(2) if write_o else 0),
        (out.stride(1) if write_o else 0),
        col_sum.stride(0), col_sum.stride(1), col_sum.stride(2),
        H, M, N, M_rounded, D,
        bias_type, causal,
        1 if write_o else 0,
        BLOCK_HEADDIM,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=num_warps, num_stages=1,
    )
    return (out if write_o else None), lse, col_sum

#& -----------------------------
#& ColSum kernel (no-atomics)  [n-major]
#& Each program owns (b,h,n-block), scans all m-blocks, accumulates to regs, stores once.
#& -----------------------------
@triton.heuristics({
    "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
    "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
    "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
})
@triton.jit
def _colsum_n_major_kernel(
    Q, K, Bias, LSE, ColSum,
    softmax_scale,
    # strides
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    stride_bb, stride_bh, stride_bm,    # Bias strides if expanded to [B,H,M,N]
    stride_cb, stride_ch, stride_cn,    # ColSum strides: [B,H,N]
    nheads, seqlen_q, seqlen_k, seqlen_q_round, headdim,
    BIAS_TYPE: tl.constexpr, IS_CAUSAL: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr, EVEN_N: tl.constexpr, EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # program ids
    pid_n = tl.program_id(0)            # along N
    off_hb = tl.program_id(1)           # flatten (B,H)
    off_b  = off_hb // nheads
    off_h  = off_hb % nheads

    # index vectors
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    # base ptrs
    k_ptrs = K + off_b*stride_kb + off_h*stride_kh + (offs_n[:, None]*stride_kn + offs_d[None, :])
    k = tl.load(
        k_ptrs,
        mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
        other=0.0
    )

    # bias base
    if BIAS_TYPE == "vector":
        b_vec_base = Bias + off_b*stride_bb + off_h*stride_bh
    elif BIAS_TYPE == "matrix":
        b_mat_base = Bias + off_b*stride_bb + off_h*stride_bh

    # register accumulator for this (b,h,n-block)
    colsum = tl.zeros([BLOCK_N], dtype=tl.float32)

    # scan over m-blocks
    for start_m in range(0, seqlen_q, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        offs_d_m = tl.arange(0, BLOCK_HEADDIM)

        # load Q block and LSE for rows
        q_ptrs = Q + off_b*stride_qb + off_h*stride_qh + (offs_m[:, None]*stride_qm + offs_d_m[None, :])
        q = tl.load(
            q_ptrs,
            mask=(offs_m[:, None] < seqlen_q) & (offs_d_m[None, :] < headdim),
            other=0.0
        )
        lse = tl.load(LSE + off_hb*seqlen_q_round + offs_m,
                      mask=offs_m < seqlen_q, other=-float("inf"))

        # scores for (m-block, n-block)
        qk = tl.dot(q, tl.trans(k)).to(tl.float32)  # [BLOCK_M, BLOCK_N]

        # bias + scale
        if BIAS_TYPE != "none":
            if BIAS_TYPE == "vector":
                b = tl.load(b_vec_base + offs_n,
                            mask=offs_n < seqlen_k, other=0.0).to(tl.float32)
                qk = qk * softmax_scale + b[None, :]
            else:
                b_ptrs = b_mat_base + (offs_m[:, None]*stride_bm + offs_n[None, :])
                b = tl.load(b_ptrs,
                            mask=(offs_m[:, None] < seqlen_q) & (offs_n[None, :] < seqlen_k),
                            other=0.0).to(tl.float32)
                qk = qk * softmax_scale + b
            p = tl.exp(qk - lse[:, None])
        else:
            p = tl.exp(qk * softmax_scale - lse[:, None])

        # causal + bounds mask
        if IS_CAUSAL:
            p = tl.where(offs_m[:, None] >= offs_n[None, :], p, 0.0)
        p = tl.where(
            (offs_m[:, None] < seqlen_q) & (offs_n[None, :] < seqlen_k),
            p, 0.0
        )

        # reduce rows -> [BLOCK_N], accumulate in regs
        colsum += tl.sum(p, axis=0)

    # single store, no atomics
    c_ptrs = ColSum + off_b*stride_cb + off_h*stride_ch + (offs_n*stride_cn)
    tl.store(c_ptrs, colsum, mask=offs_n < seqlen_k)

def fa_forward_colsum_fast(q, k, v, bias=None, causal=False, softmax_scale=None, write_o=True):
    """
    Args:
        q, k, v: Q[K,V] shapes are [B, M/N, H, D] with q: [B,M,H,D], k/v: [B,N,H,D]
        bias: None | [B,H,1,N] | [B,H,M,N]
    Returns:
        out[B,M,H,D] (or None if write_o=False),
        lse[B,H,M_rounded] (fp32),
        col_sum[B,H,N] (fp32)   -- computed without atomics
    """
    assert q.is_cuda and q.dtype in (torch.float16, torch.bfloat16), f"q dtype {q.dtype} not supported"
    B, M, H, D = q.shape
    N = k.shape[1]
    assert k.shape == (B, N, H, D) and v.shape == (B, N, H, D)

    scale = softmax_scale or (1.0 / math.sqrt(D))

    # prepare bias
    bias_type = "none"
    b_strides = (0, 0, 0)
    if bias is not None:
        if bias.stride(-1) != 1:
            bias = bias.contiguous()
        if bias.shape[2:] == (1, N):
            bias_type = "vector"
            # Expand to allow unified stride logic for matrix path when needed
            bias = bias.expand(B, H, M, N)
        elif bias.shape[2:] == (M, N):
            bias_type = "matrix"
        else:
            raise RuntimeError("bias last two dims must be (1,N) or (M,N)")
        b_strides = (bias.stride(0), bias.stride(1), bias.stride(2))

    # buffers
    M_rounded = math.ceil(M / 128) * 128
    out = torch.empty_like(q) if write_o else q.new_empty(0)
    lse = torch.empty((B, H, M_rounded), device=q.device, dtype=torch.float32)
    tmp = torch.empty_like(lse)
    col_sum = torch.empty((B, H, N), device=q.device, dtype=torch.float32)  # will be fully written, no need to zero

    # meta / grids
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_HEADDIM = max(triton.next_power_of_2(D), 16)
    num_warps_a = 4 if D <= 64 else 8
    num_warps_b = 4 if D <= 64 else 8
    grid_a = lambda META: (triton.cdiv(M, META["BLOCK_M"]), B * H)
    grid_b = lambda META: (triton.cdiv(N, META["BLOCK_N"]), B * H)

    # ---- Kernel A: PASS1 (LSE + optional O), m-major ----
    _fwd_lse_o_kernel[grid_a](
        q, k, v, bias if bias is not None else torch.empty(1, device=q.device, dtype=q.dtype),
        out, lse, tmp,
        scale,
        q.stride(0), q.stride(2), q.stride(1),
        k.stride(0), k.stride(2), k.stride(1),
        v.stride(0), v.stride(2), v.stride(1),
        *b_strides,
        out.stride(0) if write_o else 0,
        (out.stride(2) if write_o else 0),
        (out.stride(1) if write_o else 0),
        H, M, N, M_rounded, D,
        bias_type, causal,
        1 if write_o else 0,
        BLOCK_HEADDIM,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=num_warps_a, num_stages=1,
    )

    # ---- Kernel B: PASS2 (ColSum w/o atomics), n-major ----
    _colsum_n_major_kernel[grid_b](
        q, k, bias if bias is not None else torch.empty(1, device=q.device, dtype=q.dtype),
        lse, col_sum,
        scale,
        q.stride(0), q.stride(2), q.stride(1),
        k.stride(0), k.stride(2), k.stride(1),
        *b_strides,
        col_sum.stride(0), col_sum.stride(1), col_sum.stride(2),
        H, M, N, M_rounded, D,
        bias_type, causal,
        BLOCK_HEADDIM,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=num_warps_b, num_stages=2,   # deeper pipeline for K reuse
    )

    return (out if write_o else None), lse, col_sum

