"""按层 activation checkpointing。反传重算时模块的副作用（负载计数、KL 记录）要跳过。"""

from __future__ import annotations

import contextlib

import torch
from torch.utils.checkpoint import checkpoint

_RECOMPUTING = False


def recomputing() -> bool:
    return _RECOMPUTING


@contextlib.contextmanager
def _recompute_ctx():
    global _RECOMPUTING
    prev, _RECOMPUTING = _RECOMPUTING, True
    try:
        yield
    finally:
        _RECOMPUTING = prev


def checkpoint_call(fn, *args):
    return checkpoint(
        fn, *args, use_reentrant=False, context_fn=lambda: (contextlib.nullcontext(), _recompute_ctx())
    )


def autocast_dtype(device: torch.device):
    """当前 autocast 的低精度类型；未开 autocast 时返回 None。"""
    try:
        if torch.is_autocast_enabled(device.type):
            return torch.get_autocast_dtype(device.type)
        return None
    except TypeError:  # 老版本 torch 无 device_type 参数
        if device.type == "cuda" and torch.is_autocast_enabled():
            return torch.get_autocast_gpu_dtype()
        return None
