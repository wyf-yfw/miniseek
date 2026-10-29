"""Single-Pass mHC：残差流保持 hc_mult 份，子层前后由 (pre, post, comb) 混合。

对应官方 Block.hc_mixes / hc_pre / hc_post 与 kernel.py 的 hc_split_sinkhorn。
系数全部在 fp32 里算；comb 经 Sinkhorn 投到双随机矩阵。
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def _fp32(device: torch.device):
    return torch.autocast(device_type=device.type, enabled=False)


def hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    iters: int = 20,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """mixes [..., (2+hc)*hc] → pre [...,hc], post [...,hc], comb [...,hc,hc]。"""
    hc = hc_mult
    pre = torch.sigmoid(mixes[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(mixes[..., hc : 2 * hc] * hc_scale[1] + hc_base[hc : 2 * hc])
    comb = (mixes[..., 2 * hc :] * hc_scale[2] + hc_base[2 * hc :]).unflatten(-1, (hc, hc))
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


def hc_mixes(
    x: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    norm_eps: float,
    hc_mult: int,
    iters: int,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """x [...,hc,d]：整条展平后做一次 RMS（权重已折进 fn），投影出全部系数。"""
    with _fp32(x.device):
        xf = x.flatten(-2).float()
        rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + norm_eps)
        mixes = F.linear(xf, fn.float()) * rsqrt
        return hc_split_sinkhorn(mixes, scale.float(), base.float(), hc_mult, iters, eps)


def hc_pre(x: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
    """按 pre 把 hc 份合成子层输入。x [...,hc,d]，pre [...,hc] → [...,d]。"""
    with _fp32(x.device):
        y = torch.einsum("...k,...kd->...d", pre.float(), x.float())
    return y.to(x.dtype)


def hc_post(y: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor) -> torch.Tensor:
    """X_new[k] = post[k]·y + Σ_j comb[j,k]·X[j]。y [...,d]，residual [...,hc,d]。"""
    with _fp32(y.device):
        out = post.float().unsqueeze(-1) * y.float().unsqueeze(-2)
        out = out + torch.einsum("...jk,...jd->...kd", comb.float(), residual.float())
    return out.to(y.dtype)


def identity_pre_mix(x: torch.Tensor, hc_mult: int) -> torch.Tensor:
    """第一个子层用 one-hot：只取第 0 份。x [...,hc,d] → [...,hc]。"""
    pre = x.new_zeros(*x.shape[:-2], hc_mult, dtype=torch.float32)
    pre[..., 0] = 1.0
    return pre
