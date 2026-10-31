"""RoPE（含 YaRN），与官方 inference/model.py 的 precompute_freqs_cis / apply_rotary_emb 同语义。

官方版本原地修改输入；这里返回新张量，便于反传。相邻两维组成一个复数。
"""

from __future__ import annotations

import math

import torch

from .config import FlashConfig


def precompute_freqs_cis(
    dim: int,
    seqlen: int,
    original_seq_len: int,
    base: float,
    factor: float,
    beta_fast: int,
    beta_slow: int,
) -> torch.Tensor:
    """返回 complex64 [seqlen, dim // 2]。original_seq_len > 0 时启用 YaRN。"""
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:

        def corrected_dim(rotations: float) -> float:
            return dim * math.log(original_seq_len / (rotations * 2 * math.pi)) / (2 * math.log(base))

        low = max(math.floor(corrected_dim(beta_fast)), 0)
        high = min(math.ceil(corrected_dim(beta_slow)), dim - 1)
        ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / max(high - low, 1e-3)).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    freqs = torch.outer(torch.arange(seqlen, dtype=torch.float32), freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """x: [B,S,D] 或 [B,S,H,D]；freqs_cis: [S,D/2] 或 [B,S,D/2]。inverse 用共轭旋转。"""
    dtype = x.dtype
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    if inverse:
        freqs_cis = freqs_cis.conj()
    lead = freqs_cis.size(0) if freqs_cis.dim() == 3 else 1
    if xc.dim() == 3:
        f = freqs_cis.reshape(lead, xc.size(1), xc.size(-1))
    else:
        f = freqs_cis.reshape(lead, xc.size(1), 1, xc.size(-1))
    return torch.view_as_real(xc * f).flatten(-2).to(dtype)


def rotate_tail(x: torch.Tensor, freqs_cis: torch.Tensor, rope_dim: int, inverse: bool = False) -> torch.Tensor:
    """只旋转最后 rope_dim 维。"""
    head, tail = x[..., :-rope_dim], x[..., -rope_dim:]
    return torch.cat([head, apply_rotary_emb(tail, freqs_cis, inverse)], dim=-1)


def layer_freqs(cfg: FlashConfig, layer_id: int, seqlen: int) -> torch.Tensor:
    """压缩层（含其滑窗分支）用 compress_rope_theta + YaRN；纯滑窗层用 rope_theta、不开 YaRN。"""
    if layer_id < cfg.n_layers and cfg.compress_ratios[layer_id]:
        original_seq_len, theta = cfg.original_seq_len, cfg.compress_rope_theta
    else:
        original_seq_len, theta = 0, cfg.rope_theta
    return precompute_freqs_cis(
        cfg.rope_head_dim, seqlen, original_seq_len, theta, cfg.rope_factor, cfg.beta_fast, cfg.beta_slow
    )
