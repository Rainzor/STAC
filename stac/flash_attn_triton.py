# Copyright (c) 2025 STAC Authors. All rights reserved.
# Ref: https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn_interface/src/flash_attn_triton.py

import math
import torch
import triton
import triton.language as tl
import torch.nn.functional as F

#& -----------------------------
#& forward + LSE (+ optional O) kernel  [m-major]
#& FlashAttention v2 style: maintain (m_i, l_i) in loop, compute lse once at epilogue.
#& No TMP buffer — all in registers.
#& -----------------------------
@triton.heuristics({
    "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
    "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
    "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
})
@triton.jit
def _fwd_lse_o_kernel(
    Q, K, V, Bias, Out, LSE,
    softmax_scale,
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_bb, stride_bh, stride_bm,
    stride_ob, stride_oh, stride_om,
    nheads, seqlen_q, seqlen_k, seqlen_q_round, headdim,
    BIAS_TYPE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    WRITE_O: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr, EVEN_N: tl.constexpr, EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    q_ptrs = Q + off_b * stride_qb + off_h * stride_qh + (offs_m[:, None] * stride_qm + offs_d[None, :])
    k_ptrs = K + off_b * stride_kb + off_h * stride_kh + (offs_n[:, None] * stride_kn + offs_d[None, :])
    v_ptrs = V + off_b * stride_vb + off_h * stride_vh + (offs_n[:, None] * stride_vn + offs_d[None, :])

    if BIAS_TYPE == "vector":
        b_ptrs = Bias + off_b * stride_bb + off_h * stride_bh + offs_n
    elif BIAS_TYPE == "matrix":
        b_ptrs = Bias + off_b * stride_bb + off_h * stride_bh + (offs_m[:, None] * stride_bm + offs_n[None, :])

    acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)
    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    if EVEN_M & EVEN_HEADDIM:
        q = tl.load(q_ptrs)
    else:
        q = tl.load(q_ptrs, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0)

    end_n = seqlen_k if not IS_CAUSAL else tl.minimum((pid_m + 1) * BLOCK_M, seqlen_k)

    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        if EVEN_N & EVEN_HEADDIM:
            k = tl.load(k_ptrs + start_n * stride_kn)
            if WRITE_O:
                v = tl.load(v_ptrs + start_n * stride_vn)
        else:
            mask_nv = ((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim)
            k = tl.load(k_ptrs + start_n * stride_kn, mask=mask_nv, other=0.0)
            if WRITE_O:
                v = tl.load(v_ptrs + start_n * stride_vn, mask=mask_nv, other=0.0)

        qk = tl.dot(q, tl.trans(k)).to(tl.float32)

        if not EVEN_N:
            qk += tl.where((start_n + offs_n)[None, :] < seqlen_k, 0.0, float("-inf"))
        if IS_CAUSAL:
            qk += tl.where(offs_m[:, None] >= (start_n + offs_n)[None, :], 0.0, float("-inf"))

        if BIAS_TYPE != "none":
            if BIAS_TYPE == "vector":
                if EVEN_N:
                    bias = tl.load(b_ptrs + start_n).to(tl.float32)
                else:
                    bias = tl.load(b_ptrs + start_n, mask=(start_n + offs_n) < seqlen_k, other=0.0).to(tl.float32)
                qk = qk * softmax_scale + bias[None, :]
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
            m_ij = tl.max(qk, axis=1)
        else:
            qk = qk * softmax_scale
            m_ij = tl.max(qk, axis=1)

        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)

        if WRITE_O:
            acc_o = acc_o * alpha[:, None] + tl.dot(p.to(v.dtype), v)

        m_i = m_new

    lse_i = m_i + tl.log(l_i)
    lse_ptrs = LSE + off_hb * seqlen_q_round + offs_m
    tl.store(lse_ptrs, lse_i, mask=offs_m < seqlen_q)

    if WRITE_O:
        acc_o = acc_o / l_i[:, None]
        out_ptrs = Out + off_b * stride_ob + off_h * stride_oh + (offs_m[:, None] * stride_om + offs_d[None, :])
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

    lse = torch.empty((B, H, M_rounded), device=q.device, dtype=torch.float32)
    out = torch.empty_like(q)

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_HEADDIM = max(triton.next_power_of_2(D), 16)
    num_warps = 4 if D <= 64 else 8
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]), B * H)

    _fwd_lse_o_kernel[grid](
        q, k, v, bias, out, lse,
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
        1,
        BLOCK_HEADDIM,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=num_warps, num_stages=2,
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
    q_m_stride,  # physical row stride for Q/LSE subsampling (1 = no subsampling)
    BIAS_TYPE: tl.constexpr, IS_CAUSAL: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr, EVEN_N: tl.constexpr, EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b  = off_hb // nheads
    off_h  = off_hb % nheads

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    k_ptrs = K + off_b*stride_kb + off_h*stride_kh + (offs_n[:, None]*stride_kn + offs_d[None, :])
    k = tl.load(
        k_ptrs,
        mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
        other=0.0
    )

    if BIAS_TYPE == "vector":
        b_vec_base = Bias + off_b*stride_bb + off_h*stride_bh
    elif BIAS_TYPE == "matrix":
        b_mat_base = Bias + off_b*stride_bb + off_h*stride_bh

    colsum = tl.zeros([BLOCK_N], dtype=tl.float32)

    for start_m in range(0, seqlen_q, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        offs_m_phys = offs_m * q_m_stride
        offs_d_m = tl.arange(0, BLOCK_HEADDIM)

        q_ptrs = Q + off_b*stride_qb + off_h*stride_qh + (offs_m_phys[:, None]*stride_qm + offs_d_m[None, :])
        q = tl.load(
            q_ptrs,
            mask=(offs_m[:, None] < seqlen_q) & (offs_d_m[None, :] < headdim),
            other=0.0
        )
        lse = tl.load(LSE + off_hb*seqlen_q_round + offs_m_phys,
                      mask=offs_m < seqlen_q, other=-float("inf"))

        qk = tl.dot(q, tl.trans(k)).to(tl.float32)

        if BIAS_TYPE != "none":
            if BIAS_TYPE == "vector":
                b = tl.load(b_vec_base + offs_n,
                            mask=offs_n < seqlen_k, other=0.0).to(tl.float32)
                qk = qk * softmax_scale + b[None, :]
            else:
                b_ptrs = b_mat_base + (offs_m_phys[:, None]*stride_bm + offs_n[None, :])
                b = tl.load(b_ptrs,
                            mask=(offs_m[:, None] < seqlen_q) & (offs_n[None, :] < seqlen_k),
                            other=0.0).to(tl.float32)
                qk = qk * softmax_scale + b
            p = tl.exp(qk - lse[:, None])
        else:
            p = tl.exp(qk * softmax_scale - lse[:, None])

        if IS_CAUSAL:
            p = tl.where(offs_m_phys[:, None] >= offs_n[None, :], p, 0.0)
        p = tl.where(
            (offs_m[:, None] < seqlen_q) & (offs_n[None, :] < seqlen_k),
            p, 0.0
        )

        colsum += tl.sum(p, axis=0)

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

    M_rounded = math.ceil(M / 128) * 128
    out = torch.empty_like(q) if write_o else q.new_empty(0)
    lse = torch.empty((B, H, M_rounded), device=q.device, dtype=torch.float32)
    col_sum = torch.empty((B, H, N), device=q.device, dtype=torch.float32)

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_HEADDIM = max(triton.next_power_of_2(D), 16)
    num_warps_a = 4 if D <= 64 else 8
    num_warps_b = 4 if D <= 64 else 8
    grid_a = lambda META: (triton.cdiv(M, META["BLOCK_M"]), B * H)
    grid_b = lambda META: (triton.cdiv(N, META["BLOCK_N"]), B * H)

    _fwd_lse_o_kernel[grid_a](
        q, k, v, bias if bias is not None else torch.empty(1, device=q.device, dtype=q.dtype),
        out, lse,
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
        num_warps=num_warps_a, num_stages=2,
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
        1,  # q_m_stride: no subsampling
        bias_type, causal,
        BLOCK_HEADDIM,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=num_warps_b, num_stages=2,
    )

    return (out if write_o else None), lse, col_sum


#& -----------------------------
#& Fused LSE + O + partial ColSum kernel  [m-major, split-K]
#& Pass 1: streaming softmax → LSE + O  (identical to _fwd_lse_o_kernel)
#& Pass 2: reuse Q (regs) + lse_i (regs), re-sweep K (L2-cached) → partial colsum
#&         stored per (pid_m, off_hb) — no atomics needed.
#& -----------------------------
@triton.heuristics({
    "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
    "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
    "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
})
@triton.jit
def _fwd_lse_o_partial_colsum_kernel(
    Q, K, V, Bias, Out, LSE, TMP,
    PartialColSum,
    softmax_scale,
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_bb, stride_bh, stride_bm,
    stride_ob, stride_oh, stride_om,
    stride_ps, stride_phb,
    nheads, seqlen_q, seqlen_k, seqlen_q_round, headdim,
    BIAS_TYPE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    WRITE_O: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr, EVEN_N: tl.constexpr, EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    q_ptrs = Q + off_b * stride_qb + off_h * stride_qh + (offs_m[:, None] * stride_qm + offs_d[None, :])
    k_ptrs = K + off_b * stride_kb + off_h * stride_kh + (offs_n[:, None] * stride_kn + offs_d[None, :])
    v_ptrs = V + off_b * stride_vb + off_h * stride_vh + (offs_n[:, None] * stride_vn + offs_d[None, :])

    if BIAS_TYPE == "vector":
        b_ptrs = Bias + off_b * stride_bb + off_h * stride_bh + offs_n
    elif BIAS_TYPE == "matrix":
        b_ptrs = Bias + off_b * stride_bb + off_h * stride_bh + (offs_m[:, None] * stride_bm + offs_n[None, :])

    acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)
    lse_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    m_i   = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    t_ptrs = TMP + off_hb * seqlen_q_round + offs_m

    if EVEN_M & EVEN_HEADDIM:
        q = tl.load(q_ptrs)
    else:
        q = tl.load(q_ptrs, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0)

    end_n = seqlen_k if not IS_CAUSAL else tl.minimum((pid_m + 1) * BLOCK_M, seqlen_k)

    # ---- Pass 1: streaming softmax → LSE + O ----
    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        if EVEN_N & EVEN_HEADDIM:
            k = tl.load(k_ptrs + start_n * stride_kn)
            if WRITE_O:
                v = tl.load(v_ptrs + start_n * stride_vn)
        else:
            mask_nv = ((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim)
            k = tl.load(k_ptrs + start_n * stride_kn, mask=mask_nv, other=0.0)
            if WRITE_O:
                v = tl.load(v_ptrs + start_n * stride_vn, mask=mask_nv, other=0.0)

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))

        if not EVEN_N:
            qk += tl.where((start_n + offs_n)[None, :] < seqlen_k, 0.0, float("-inf"))
        if IS_CAUSAL:
            qk += tl.where(offs_m[:, None] >= (start_n + offs_n)[None, :], 0.0, float("-inf"))

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
            p = p.to(v.dtype)
            acc_o += tl.dot(p, v)

        m_i = m_ij
        lse_i = m_ij + tl.log(tl.exp(lse_i - m_ij) + l_ij)

    lse_ptrs = LSE + off_hb * seqlen_q_round + offs_m
    tl.store(lse_ptrs, lse_i, mask=offs_m < seqlen_q)

    if WRITE_O:
        o_scale = tl.exp(m_i - lse_i)
        tl.store(t_ptrs, o_scale); o_scale = tl.load(t_ptrs)
        acc_o *= o_scale[:, None]

        offs_d2 = tl.arange(0, BLOCK_HEADDIM)
        out_ptrs = Out + off_b * stride_ob + off_h * stride_oh + (offs_m[:, None] * stride_om + offs_d2[None, :])
        tl.store(out_ptrs, acc_o, mask=(offs_m[:, None] < seqlen_q) & (offs_d2[None, :] < headdim))

    # ---- Pass 2: partial col_sum using Q (regs) + lse_i (regs), K L2-cached ----
    ps_base = PartialColSum + pid_m * stride_ps + off_hb * stride_phb

    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        if EVEN_N & EVEN_HEADDIM:
            k = tl.load(k_ptrs + start_n * stride_kn)
        else:
            k = tl.load(
                k_ptrs + start_n * stride_kn,
                mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                other=0.0,
            )

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))

        if not EVEN_N:
            qk += tl.where((start_n + offs_n)[None, :] < seqlen_k, 0.0, float("-inf"))
        if IS_CAUSAL:
            qk += tl.where(offs_m[:, None] >= (start_n + offs_n)[None, :], 0.0, float("-inf"))

        if BIAS_TYPE != "none":
            if BIAS_TYPE == "vector":
                if EVEN_N:
                    bias2 = tl.load(b_ptrs + start_n).to(tl.float32)
                else:
                    bias2 = tl.load(b_ptrs + start_n, mask=(start_n + offs_n) < seqlen_k, other=0.0).to(tl.float32)
                bias2 = bias2[None, :]
                qk = qk * softmax_scale + bias2
            else:
                if EVEN_M & EVEN_N:
                    bias2 = tl.load(b_ptrs + start_n).to(tl.float32)
                else:
                    bias2 = tl.load(
                        b_ptrs + start_n,
                        mask=(offs_m[:, None] < seqlen_q) & ((start_n + offs_n)[None, :] < seqlen_k),
                        other=0.0,
                    ).to(tl.float32)
                qk = qk * softmax_scale + bias2

            p_true = tl.exp(qk - lse_i[:, None])
        else:
            p_true = tl.exp(qk * softmax_scale - lse_i[:, None])

        p_true = tl.where(offs_m[:, None] < seqlen_q, p_true, 0.0)
        if not EVEN_N:
            p_true = tl.where((start_n + offs_n)[None, :] < seqlen_k, p_true, 0.0)

        partial_cs = tl.sum(p_true, axis=0)

        tl.store(ps_base + start_n + offs_n, partial_cs, mask=(start_n + offs_n) < seqlen_k)

    if IS_CAUSAL:
        for start_n in range(end_n, seqlen_k, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            tl.store(
                ps_base + start_n + offs_n,
                tl.zeros([BLOCK_N], dtype=tl.float32),
                mask=(start_n + offs_n) < seqlen_k,
            )


#& -----------------------------
#& Reduce partial col_sum along M-block dimension
#& -----------------------------
@triton.jit
def _reduce_partial_colsum_kernel(
    PartialColSum, ColSum,
    stride_ps, stride_phb,
    stride_cb, stride_ch, stride_cn,
    num_m_blocks, nheads, seqlen_k,
    BLOCK_R: tl.constexpr,
):
    pid_n = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads

    offs_n = pid_n * BLOCK_R + tl.arange(0, BLOCK_R)
    mask_n = offs_n < seqlen_k

    acc = tl.zeros([BLOCK_R], dtype=tl.float32)
    base = off_hb * stride_phb
    for i in range(num_m_blocks):
        vals = tl.load(PartialColSum + i * stride_ps + base + offs_n, mask=mask_n, other=0.0)
        acc += vals

    c_ptrs = ColSum + off_b * stride_cb + off_h * stride_ch + offs_n * stride_cn
    tl.store(c_ptrs, acc, mask=mask_n)


def fa_forward_colsum_fast_beta(q, k, v, bias=None, causal=False, softmax_scale=None, write_o=True):
    """Split-K + reduce: fused LSE+O+partial_colsum kernel + lightweight reduce.

    Same signature and return type as fa_forward_colsum_fast.
    """
    assert q.is_cuda and q.dtype in (torch.float16, torch.bfloat16), f"q dtype {q.dtype} not supported"
    B, M, H, D = q.shape
    N = k.shape[1]
    assert k.shape == (B, N, H, D) and v.shape == (B, N, H, D)

    scale = softmax_scale or (1.0 / math.sqrt(D))

    bias_type = "none"
    b_strides = (0, 0, 0)
    if bias is not None:
        if bias.stride(-1) != 1:
            bias = bias.contiguous()
        if bias.shape[2:] == (1, N):
            bias_type = "vector"
            bias = bias.expand(B, H, M, N)
        elif bias.shape[2:] == (M, N):
            bias_type = "matrix"
        else:
            raise RuntimeError("bias last two dims must be (1,N) or (M,N)")
        b_strides = (bias.stride(0), bias.stride(1), bias.stride(2))

    BLOCK_M = 128
    BLOCK_N = 128
    M_rounded = math.ceil(M / BLOCK_M) * BLOCK_M
    num_m_blocks = math.ceil(M / BLOCK_M)
    BLOCK_HEADDIM = max(triton.next_power_of_2(D), 16)
    num_warps = 4 if D <= 64 else 8

    out = torch.empty_like(q) if write_o else q.new_empty(0)
    lse = torch.empty((B, H, M_rounded), device=q.device, dtype=torch.float32)
    tmp = torch.empty_like(lse)
    partial_colsum = torch.empty((num_m_blocks, B * H, N), device=q.device, dtype=torch.float32)
    col_sum = torch.empty((B, H, N), device=q.device, dtype=torch.float32)

    grid_fused = lambda META: (triton.cdiv(M, META["BLOCK_M"]), B * H)

    _fwd_lse_o_partial_colsum_kernel[grid_fused](
        q, k, v, bias if bias is not None else torch.empty(1, device=q.device, dtype=q.dtype),
        out, lse, tmp,
        partial_colsum,
        scale,
        q.stride(0), q.stride(2), q.stride(1),
        k.stride(0), k.stride(2), k.stride(1),
        v.stride(0), v.stride(2), v.stride(1),
        *b_strides,
        out.stride(0) if write_o else 0,
        (out.stride(2) if write_o else 0),
        (out.stride(1) if write_o else 0),
        partial_colsum.stride(0), partial_colsum.stride(1),
        H, M, N, M_rounded, D,
        bias_type, causal,
        1 if write_o else 0,
        BLOCK_HEADDIM,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=num_warps, num_stages=1,
    )

    BLOCK_R = 128
    grid_reduce = (triton.cdiv(N, BLOCK_R), B * H)

    _reduce_partial_colsum_kernel[grid_reduce](
        partial_colsum, col_sum,
        partial_colsum.stride(0), partial_colsum.stride(1),
        col_sum.stride(0), col_sum.stride(1), col_sum.stride(2),
        num_m_blocks, H, N,
        BLOCK_R=BLOCK_R,
    )

    return (out if write_o else None), lse, col_sum


def fa_forward_colsum_fast_sub(q, k, v, bias=None, causal=False,
                               softmax_scale=None, write_o=True,
                               subsample_ratio=0.25):
    """Forward + ColSum with M-subsampled colsum for faster heavy-hitter scoring.

    Pass 1 (fwd): Identical to fa_forward_colsum_fast — full-M O + LSE.
    Pass 2 (colsum): Uses only M' = floor(M * subsample_ratio) uniformly-spaced
                     rows, then scales by M / M'.

    For heavy-hitter ranking (top-K selection) the ordering is nearly identical
    to the exact colsum, while the colsum GEMM cost drops proportionally.
    """
    assert q.is_cuda and q.dtype in (torch.float16, torch.bfloat16)
    B, M, H, D = q.shape
    N = k.shape[1]
    assert k.shape == (B, N, H, D) and v.shape == (B, N, H, D)

    scale = softmax_scale or (1.0 / math.sqrt(D))

    # --- Prepare bias (same as fa_forward_colsum_fast) ---
    bias_type = "none"
    b_strides = (0, 0, 0)
    if bias is not None:
        if bias.stride(-1) != 1:
            bias = bias.contiguous()
        if bias.shape[2:] == (1, N):
            bias_type = "vector"
            bias = bias.expand(B, H, M, N)
        elif bias.shape[2:] == (M, N):
            bias_type = "matrix"
        else:
            raise RuntimeError("bias last two dims must be (1,N) or (M,N)")
        b_strides = (bias.stride(0), bias.stride(1), bias.stride(2))

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_HEADDIM = max(triton.next_power_of_2(D), 16)

    M_rounded = math.ceil(M / 128) * 128
    out = torch.empty_like(q) if write_o else q.new_empty(0)
    lse = torch.empty((B, H, M_rounded), device=q.device, dtype=torch.float32)

    num_warps_a = 4 if D <= 64 else 8
    grid_a = lambda META: (triton.cdiv(M, META["BLOCK_M"]), B * H)

    _fwd_lse_o_kernel[grid_a](
        q, k, v, bias if bias is not None else torch.empty(1, device=q.device, dtype=q.dtype),
        out, lse,
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
        num_warps=num_warps_a, num_stages=2,
    )

    # ---- Subsample parameters (stride-based, zero-copy) ----
    M_sub = max(BLOCK_M, int(M * subsample_ratio))
    M_sub = (M_sub // BLOCK_M) * BLOCK_M

    if M_sub >= M:
        q_m_stride = 1
        M_sub = M
        correction = 1.0
    else:
        q_m_stride = max(1, M // M_sub)
        correction = M / M_sub

    if bias_type == "matrix" and q_m_stride > 1:
        raise NotImplementedError(
            "M-subsampling with matrix bias requires gathering the bias M dim; "
            "use vector bias (the default in STAC).")

    # ---- Pass 2: subsampled colsum (stride-based, no gather/padding) ----
    col_sum = torch.empty((B, H, N), device=q.device, dtype=torch.float32)
    num_warps_b = 4 if D <= 64 else 8
    grid_b = lambda META: (triton.cdiv(N, META["BLOCK_N"]), B * H)

    _colsum_n_major_kernel[grid_b](
        q, k,
        bias if bias is not None else torch.empty(1, device=q.device, dtype=q.dtype),
        lse, col_sum,
        scale,
        q.stride(0), q.stride(2), q.stride(1),
        k.stride(0), k.stride(2), k.stride(1),
        *b_strides,
        col_sum.stride(0), col_sum.stride(1), col_sum.stride(2),
        H, M_sub, N, M_rounded, D,
        q_m_stride,
        bias_type, causal,
        BLOCK_HEADDIM,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=num_warps_b, num_stages=2,
    )

    if correction != 1.0:
        col_sum *= correction

    return (out if write_o else None), lse, col_sum


# =========================================================================
# Self-test: accuracy + efficiency
# =========================================================================
if __name__ == "__main__":
    import time

    torch.manual_seed(42)
    device = "cuda"

    def reference_attention(q, k, v, bias=None, softmax_scale=None):
        B, M, H, D = q.shape
        N = k.shape[1]
        if softmax_scale is None:
            softmax_scale = D ** (-0.5)
        q_t = q.transpose(1, 2).float()
        k_t = k.transpose(1, 2).float()
        v_t = v.transpose(1, 2).float()
        scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * softmax_scale
        if bias is not None:
            b = bias.float()
            if b.dim() == 4:
                b = b.squeeze(2)
            scores = scores + b.unsqueeze(2)
        lse = torch.logsumexp(scores, dim=-1)
        attn = torch.softmax(scores, dim=-1)
        col_sum = attn.sum(dim=2)
        out = torch.matmul(attn, v_t)
        return out.transpose(1, 2), lse, col_sum

    def bench(fn, M, N, warmup=10, repeat=100):
        if M * N >= 8 * 1024 * 1024:
            repeat = min(repeat, 20)
            warmup = min(warmup, 3)
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(repeat):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / repeat

    configs = [
        (1, 1024, 16384, 16, 64, "bias_inf"),
        (1, 1024, 16384, 16, 64, "none"),
        (1, 4096, 16384, 16, 64, "bias_inf"),
        (1, 4096, 16384, 16, 64, "none"),
        (1, 1024, 2048,  16, 64, "bias_inf"),
        (1, 1024, 2048,  16, 64, "none"),
    ]

    print(f"GPU: {torch.cuda.get_device_name()}")
    print()

    sub_ratios = [0.5, 0.25, 0.125]

    # ---- Accuracy ----
    print("=" * 140)
    print("Accuracy (max abs error vs PyTorch fp32 reference)")
    print("=" * 140)
    hdr = (f"{'Config':>36s} {'Bias':>10s} | "
           f"{'fast O':>8s} {'fast CS':>8s} | "
           f"{'sub.50 O':>8s} {'sub.50 CS':>9s} | "
           f"{'sub.25 O':>8s} {'sub.25 CS':>9s} | "
           f"{'sub.12 O':>8s} {'sub.12 CS':>9s}")
    print(hdr)
    print("-" * 140)

    for B, M, N, H, D, bias_mode in configs:
        dtype = torch.float16
        q = torch.randn(B, M, H, D, device=device, dtype=dtype)
        k = torch.randn(B, N, H, D, device=device, dtype=dtype)
        v = torch.randn(B, N, H, D, device=device, dtype=dtype)

        if bias_mode == "bias_inf":
            bias_4d = torch.zeros(1, H, 1, N, device=device, dtype=torch.float32)
            bias_4d[:, :, :, 3 * N // 4:] = float("-inf")
        else:
            bias_4d = None

        scale = D ** (-0.5)
        ref_o, ref_lse, ref_cs = reference_attention(q, k, v, bias=bias_4d, softmax_scale=scale)

        out_f, lse_f, cs_f = fa_forward_colsum_fast(q, k, v, bias=bias_4d, softmax_scale=scale, write_o=True)
        fo = (out_f.float() - ref_o).abs().max().item()
        fc = (cs_f - ref_cs).abs().max().item()

        parts = [f"{fo:8.6f} {fc:8.6f}"]
        for r in sub_ratios:
            out_s, _, cs_s = fa_forward_colsum_fast_sub(
                q, k, v, bias=bias_4d, softmax_scale=scale, write_o=True, subsample_ratio=r)
            so = (out_s.float() - ref_o).abs().max().item()
            sc = (cs_s - ref_cs).abs().max().item()
            parts.append(f"{so:8.6f} {sc:9.6f}")

        tag = f"B={B} M={M} N={N} H={H}"
        print(f"{tag:>36s} {bias_mode:>10s} | " + " | ".join(parts))

    # ---- Ranking quality (Spearman + Top-K overlap) ----
    print()
    print("=" * 140)
    print("Ranking quality (Spearman rho & Top-K overlap vs exact colsum)")
    print("=" * 140)

    def spearman(a, b):
        def _rank(x):
            idx = x.argsort(descending=True)
            r = torch.empty_like(x)
            r[idx] = torch.arange(len(x), device=x.device, dtype=x.dtype)
            return r
        ra, rb = _rank(a), _rank(b)
        d = ra - rb
        n = len(a)
        return 1.0 - 6.0 * (d * d).sum().item() / (n * (n * n - 1))

    def topk_overlap(a, b, k):
        ta = set(a.topk(k).indices.tolist())
        tb = set(b.topk(k).indices.tolist())
        return len(ta & tb) / k

    hdr3 = (f"{'Config':>36s} {'Bias':>10s} {'ratio':>6s} | "
            f"{'Spearman':>9s} {'Top-64':>7s} {'Top-128':>8s} {'Top-256':>8s}")
    print(hdr3)
    print("-" * 140)

    for B, M, N, H, D, bias_mode in configs:
        dtype = torch.float16
        q = torch.randn(B, M, H, D, device=device, dtype=dtype)
        k = torch.randn(B, N, H, D, device=device, dtype=dtype)
        v = torch.randn(B, N, H, D, device=device, dtype=dtype)

        if bias_mode == "bias_inf":
            bias_4d = torch.zeros(1, H, 1, N, device=device, dtype=torch.float32)
            bias_4d[:, :, :, 3 * N // 4:] = float("-inf")
        else:
            bias_4d = None

        scale = D ** (-0.5)
        _, _, cs_ref = fa_forward_colsum_fast(q, k, v, bias=bias_4d, softmax_scale=scale, write_o=False)

        for r in sub_ratios:
            _, _, cs_s = fa_forward_colsum_fast_sub(
                q, k, v, bias=bias_4d, softmax_scale=scale, write_o=False, subsample_ratio=r)
            rho_vals, t64_vals, t128_vals, t256_vals = [], [], [], []
            for h in range(H):
                ref_h = cs_ref[0, h]
                sub_h = cs_s[0, h]
                rho_vals.append(spearman(ref_h, sub_h))
                t64_vals.append(topk_overlap(ref_h, sub_h, 64))
                t128_vals.append(topk_overlap(ref_h, sub_h, 128))
                t256_vals.append(topk_overlap(ref_h, sub_h, 256))
            tag = f"B={B} M={M} N={N} H={H}"
            print(f"{tag:>36s} {bias_mode:>10s} {r:6.3f} | "
                  f"{sum(rho_vals)/H:9.4f} {sum(t64_vals)/H:7.1%} {sum(t128_vals)/H:8.1%} {sum(t256_vals)/H:8.1%}")

    # ---- Efficiency ----
    print()
    print("=" * 140)
    print("Efficiency (ms, lower is better)")
    print("=" * 140)
    hdr4 = (f"{'Config':>36s} {'Bias':>10s} | "
            f"{'fast(ms)':>10s} | "
            + " | ".join(f"sub{r}(ms)" for r in sub_ratios)
            + " | " + " | ".join(f"  spd{r}" for r in sub_ratios))
    print(hdr4)
    print("-" * 140)

    for B, M, N, H, D, bias_mode in configs:
        dtype = torch.float16
        q = torch.randn(B, M, H, D, device=device, dtype=dtype)
        k = torch.randn(B, N, H, D, device=device, dtype=dtype)
        v = torch.randn(B, N, H, D, device=device, dtype=dtype)

        if bias_mode == "bias_inf":
            bias_4d = torch.zeros(1, H, 1, N, device=device, dtype=torch.float32)
            bias_4d[:, :, :, 3 * N // 4:] = float("-inf")
        else:
            bias_4d = None

        t_fast = bench(
            lambda: fa_forward_colsum_fast(q, k, v, bias=bias_4d, write_o=True),
            M, N,
        )
        t_subs = []
        for r in sub_ratios:
            t_s = bench(
                lambda r=r: fa_forward_colsum_fast_sub(q, k, v, bias=bias_4d, write_o=True, subsample_ratio=r),
                M, N,
            )
            t_subs.append(t_s)

        tag = f"B={B} M={M} N={N} H={H}"
        time_parts = f"{t_fast*1000:10.3f} | " + " | ".join(f"{t*1000:10.3f}" for t in t_subs)
        spd_parts = " | ".join(f"{t_fast/t:7.2f}x" for t in t_subs)
        print(f"{tag:>36s} {bias_mode:>10s} | {time_parts} | {spd_parts}")

    print()
    print("speedup > 1 means sub is faster than fast")
