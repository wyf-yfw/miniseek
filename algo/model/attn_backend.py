"""Attention backend: FlashAttention (flash_attn) → torch SDPA flash → math.

Used by CausalSelfAttention for both Dense and MoE. Training defaults ON;
falls back safely when CUDA / dtype / package do not support flash.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F

_FLASH_ATTN_FUNC = None
_FLASH_ATTN_CHECKED = False
_CONFIGURED_NAME: Optional[str] = None


def flash_attn_func_or_none():
    """Lazy-import Dao-lab flash_attn; None if missing / broken."""
    global _FLASH_ATTN_FUNC, _FLASH_ATTN_CHECKED
    if not _FLASH_ATTN_CHECKED:
        _FLASH_ATTN_CHECKED = True
        try:
            from flash_attn import flash_attn_func as _f  # type: ignore

            _FLASH_ATTN_FUNC = _f
        except Exception:  # noqa: BLE001
            _FLASH_ATTN_FUNC = None
    return _FLASH_ATTN_FUNC


def configure_attn_backend(use_flash: bool = True) -> str:
    """Set process-wide SDPA prefs; return human-readable backend name.

    Call once at train / infer startup. Does not require CUDA.
    """
    global _CONFIGURED_NAME
    if not use_flash:
        if torch.cuda.is_available():
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
        _CONFIGURED_NAME = "sdpa (flash off)"
        return _CONFIGURED_NAME

    if flash_attn_func_or_none() is not None:
        if torch.cuda.is_available():
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
        _CONFIGURED_NAME = "flash_attn"
        return _CONFIGURED_NAME

    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        # Probe: can FLASH kernel run on this GPU?
        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel

            q = torch.randn(1, 1, 32, 64, device="cuda", dtype=torch.bfloat16)
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                F.scaled_dot_product_attention(q, q, q, is_causal=True)
            _CONFIGURED_NAME = "torch_sdpa_flash"
        except Exception:  # noqa: BLE001
            _CONFIGURED_NAME = "torch_sdpa (flash probe failed → mem/math)"
        return _CONFIGURED_NAME

    _CONFIGURED_NAME = "sdpa/math (cpu)"
    return _CONFIGURED_NAME


def configured_attn_backend() -> str:
    return _CONFIGURED_NAME or "unconfigured (sdpa default)"


def _flash_attn_cond_ok(q: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> bool:
    if attn_mask is not None:
        return False
    if not q.is_cuda:
        return False
    if q.dtype not in (torch.float16, torch.bfloat16):
        return False
    return flash_attn_func_or_none() is not None


def causal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    head_dim: int,
    attn_mask: Optional[torch.Tensor] = None,
    use_flash: bool = True,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """q/k/v: (B, H, T, D) → (B, H, T, D)."""
    b, h, t, d = q.shape

    # 1) Dao flash_attn (layout B,T,H,D)
    if use_flash and _flash_attn_cond_ok(q, attn_mask):
        fa = flash_attn_func_or_none()
        assert fa is not None
        # (B,H,T,D) → (B,T,H,D)
        q_t = q.transpose(1, 2).contiguous()
        k_t = k.transpose(1, 2).contiguous()
        v_t = v.transpose(1, 2).contiguous()
        out = fa(q_t, k_t, v_t, dropout_p=dropout_p, causal=True)
        return out.transpose(1, 2).contiguous()

    # 2) torch SDPA — prefer FLASH when use_flash and no custom mask
    if hasattr(F, "scaled_dot_product_attention"):
        if (
            use_flash
            and attn_mask is None
            and q.is_cuda
            and q.dtype in (torch.float16, torch.bfloat16)
        ):
            try:
                from torch.nn.attention import SDPBackend, sdpa_kernel

                with sdpa_kernel(
                    [
                        SDPBackend.FLASH_ATTENTION,
                        SDPBackend.EFFICIENT_ATTENTION,
                        SDPBackend.MATH,
                    ],
                    set_priority=True,
                ):
                    return F.scaled_dot_product_attention(
                        q,
                        k,
                        v,
                        attn_mask=None,
                        is_causal=True,
                        dropout_p=dropout_p,
                    )
            except Exception:  # noqa: BLE001
                pass
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=attn_mask is None,
            dropout_p=dropout_p,
        )

    # 3) manual math fallback (ancient torch)
    scale = 1.0 / math.sqrt(head_dim)
    att = (q @ k.transpose(-2, -1)) * scale
    causal = torch.triu(
        torch.ones(t, t, device=q.device, dtype=torch.bool), diagonal=1
    )
    att = att.masked_fill(causal, float("-inf"))
    if attn_mask is not None:
        att = att + attn_mask
    att = F.softmax(att, dim=-1)
    if dropout_p > 0.0 and q.requires_grad:
        att = F.dropout(att, p=dropout_p)
    return att @ v
