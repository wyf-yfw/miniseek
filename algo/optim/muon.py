"""Muon（DeepSeek-V4 Algorithm 1）：Nesterov 动量 + 混合 Newton–Schulz 正交化 + RMS 重标定。

M ← μM + G；O′ = HybridNS(μM + G)；O = O′·√max(n,m)·γ；W ← W(1 − ηλ) − ηO。
参数可带属性 ``_muon_splits = k``：按行切成 k 个逻辑独立矩阵分别正交化（head-wise Muon 等）。
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
from torch.optim import Optimizer

NS_FAST = (3.4445, -4.7750, 2.0315)
NS_STABLE = (2.0, -1.5, 0.5)


def hybrid_newton_schulz(m: torch.Tensor, steps: int = 10, fast_steps: int = 8, eps: float = 1e-7) -> torch.Tensor:
    """m [..., r, c] → 近似 UVᵀ。前 fast_steps 轮快速收敛，后几轮把奇异值稳在 1。"""
    x = m.float()
    transposed = x.size(-2) > x.size(-1)
    if transposed:
        x = x.mT
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + eps)
    for i in range(steps):
        a, b, c = NS_FAST if i < fast_steps else NS_STABLE
        gram = x @ x.mT
        x = a * x + (b * gram + c * (gram @ gram)) @ x
    return x.mT if transposed else x


class Muon(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.95,
        weight_decay: float = 0.1,
        rms_scale: float = 0.18,
        ns_steps: int = 10,
        ns_fast_steps: int = 8,
    ) -> None:
        defaults = dict(
            lr=lr,
            lr_mult=1.0,
            momentum=momentum,
            weight_decay=weight_decay,
            rms_scale=rms_scale,
            ns_steps=ns_steps,
            ns_fast_steps=ns_fast_steps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        # 同形状（且同 NS 设定）的逻辑矩阵堆起来批量做 Newton–Schulz
        buckets: Dict[Tuple, List[Tuple[torch.Tensor, dict, int]]] = defaultdict(list)
        updates: Dict[int, torch.Tensor] = {}
        for group in self.param_groups:
            mu = group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.dim() != 2:
                    raise ValueError("Muon 只接受二维权重")
                state = self.state[p]
                buf = state.get("momentum_buffer")
                if buf is None:
                    buf = state["momentum_buffer"] = torch.zeros_like(p, dtype=torch.float32)
                g = p.grad.float()
                buf.mul_(mu).add_(g)
                u = g.add(buf, alpha=mu)
                k = int(getattr(p, "_muon_splits", 1))
                mats = u.reshape(k, p.size(0) // k, p.size(1))
                key = (mats.shape[1], mats.shape[2], group["ns_steps"], group["ns_fast_steps"], u.device)
                buckets[key].append((p, group, k))
                updates[id(p)] = mats
        for (rows, cols, steps, fast, _), items in buckets.items():
            stacked = torch.cat([updates[id(p)] for p, _, _ in items], dim=0)
            ortho = hybrid_newton_schulz(stacked, steps=steps, fast_steps=fast)
            offset = 0
            for p, group, k in items:
                o = ortho[offset : offset + k].reshape(p.shape)
                offset += k
                lr = group["lr"] * group["lr_mult"]
                scale = math.sqrt(max(rows, cols)) * group["rms_scale"]
                p.mul_(1 - lr * group["weight_decay"])
                p.add_(o.to(p.dtype), alpha=-lr * scale)
        return loss
