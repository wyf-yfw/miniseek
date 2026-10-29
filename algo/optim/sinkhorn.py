"""Sinkhorn 均衡的动量更新（DeepSeek-V4.1-Flash §2.5 Algorithm 1），用于 embedding、head 与 Engram 表。

M ← βM + (1−β)G；Ĝ = βM + (1−β)G；‖Ĝ_i‖ ≤ τ·mean 的行置零；奇数步行 ℓ2 归一、偶数步列 ℓ2 归一，
共 K 次（K 为奇数）；Δ = √n·U；W ← W − γηΔ。不做权重衰减。
"""

from __future__ import annotations

import math

import torch
from torch.optim import Optimizer


def sinkhorn_direction(g_hat: torch.Tensor, k_iters: int = 11, tau: float = 1e-3, eps: float = 1e-20) -> torch.Tensor:
    """g_hat [m, n] → √n·U：行、列 RMS 都约为 1 的更新方向。"""
    u = g_hat.float().clone()
    rho = u.norm(dim=1)
    u[rho <= tau * rho.mean()] = 0
    for k in range(1, k_iters + 1):
        if k % 2:
            u = u / (u.norm(dim=1, keepdim=True) + eps)
        else:
            u = u / (u.norm(dim=0, keepdim=True) + eps)
    return math.sqrt(u.size(1)) * u


class SinkhornMomentum(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        beta: float = 0.95,
        gamma: float = 0.18,
        k_iters: int = 11,
        tau: float = 1e-3,
        eps: float = 1e-20,
    ) -> None:
        if k_iters % 2 == 0:
            raise ValueError("k_iters 必须为奇数（最后一步是行归一化）")
        defaults = dict(lr=lr, lr_mult=1.0, beta=beta, gamma=gamma, k_iters=k_iters, tau=tau, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            beta = group["beta"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.float()
                state = self.state[p]
                m = state.get("momentum_buffer")
                if m is None:
                    m = state["momentum_buffer"] = torch.zeros_like(p, dtype=torch.float32)
                m.mul_(beta).add_(g, alpha=1 - beta)
                g_hat = m * beta + g * (1 - beta)
                delta = sinkhorn_direction(g_hat, group["k_iters"], group["tau"], group["eps"])
                lr = group["lr"] * group["lr_mult"]
                p.add_(delta.to(p.dtype), alpha=-group["gamma"] * lr)
        return loss
