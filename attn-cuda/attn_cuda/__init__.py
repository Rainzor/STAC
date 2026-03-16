import torch  # must be imported before _ext to load libc10.so
from attn_cuda._ext import is_available, get_version, flash_attn_fwd
from typing import Optional, Tuple, Union


def flash_attn_bias_colsum(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    return_colsum: bool = False,
    subsample_ratio: float = 1.0,
) -> Union[Tuple[torch.Tensor, torch.Tensor],
           Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Flash attention forward with optional vector bias and column-sum.

    Args:
        q: [B, M, H, D] query tensor (fp16/bf16)
        k: [B, N, H, D] key tensor (fp16/bf16)
        v: [B, N, H, D] value tensor (fp16/bf16)
        bias: Optional [B, H, N], [B, H, 1, N], or [1, H, 1, N] vector bias
              added to attention scores. Supports any dtype (auto-cast to fp32).
        softmax_scale: scaling factor (default 1/sqrt(D))
        return_colsum: if True, also return column-sum [B, H, N]
        subsample_ratio: fraction of M rows to use for colsum (0, 1].
            E.g. 0.25 scans only 25% of Q rows (uniformly strided), then
            scales by 1/ratio. Speeds up colsum with minimal ranking error.

    Returns:
        out: [B, M, H, D] attention output
        lse: [B, H, M] log-sum-exp
        colsum: [B, H, N] column-sum (only if return_colsum=True)
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    results = flash_attn_fwd(q, k, v, softmax_scale, bias, return_colsum,
                             subsample_ratio)

    if return_colsum:
        return results[0], results[1], results[2]
    return results[0], results[1]
