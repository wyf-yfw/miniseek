"""DeepSeekMoE：1 个共享专家 + 路由专家 top-k，专家为 clamp SwiGLU。

对应官方 inference/model.py 的 Gate / Expert / MoE：打分 sqrt(softplus)，bias 只参与选择，
加权用原始分数（归一化后乘 route_scale）。bias 不进梯度，每个优化器 step 后按负载 ±speed 更新。
另有 V3 公式的序列均衡损失（训练权重 1e-4）。
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import FlashConfig


def _fp32(device: torch.device):
    return torch.autocast(device_type=device.type, enabled=False)


class Gate(nn.Module):
    def __init__(
        self,
        dim: int,
        n_experts: int,
        topk: int,
        score_func: str = "sqrtsoftplus",
        route_scale: float = 1.0,
        norm_topk_prob: bool = True,
    ) -> None:
        super().__init__()
        self.topk = topk
        self.score_func = score_func
        self.route_scale = float(route_scale)
        self.norm_topk_prob = bool(norm_topk_prob)
        self.weight = nn.Parameter(torch.empty(n_experts, dim))
        nn.init.normal_(self.weight, std=0.02)
        # 修正 bias：只影响选哪些专家，不进梯度（官方存为 fp32 参数，这里用持久 buffer）
        self.register_buffer("bias", torch.zeros(n_experts, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """x [N,d] → (weights [N,K] fp32, indices [N,K], scores [N,E] fp32)。"""
        with _fp32(x.device):
            scores = F.linear(x.float(), self.weight.float())
            if self.score_func == "softmax":
                scores = scores.softmax(dim=-1)
            elif self.score_func == "sigmoid":
                scores = scores.sigmoid()
            else:
                scores = F.softplus(scores).sqrt()
            indices = (scores + self.bias).topk(self.topk, dim=-1)[1]
            weights = scores.gather(1, indices)
            if self.norm_topk_prob and self.topk > 1:
                weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
            weights = weights * self.route_scale
        return weights, indices, scores


class Expert(nn.Module):
    """SwiGLU：up 两侧截到 ±limit，gate 只截上界。激活在 fp32 里算，路由权重在 w2 之前乘。"""

    def __init__(self, dim: int, inter_dim: int, swiglu_limit: float = 0.0) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, inter_dim, bias=False)
        self.swiglu_limit = float(swiglu_limit)

    def forward(self, x: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        dtype = x.dtype
        gate = self.w1(x).float()
        up = self.w3(x).float()
        if self.swiglu_limit > 0:
            up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
            gate = torch.clamp(gate, max=self.swiglu_limit)
        h = F.silu(gate) * up
        if weights is not None:
            h = weights * h
        return self.w2(h.to(dtype))


def sequence_balance_loss(scores: torch.Tensor, indices: torch.Tensor, n_experts: int) -> torch.Tensor:
    """V3：每条序列 Σ_i f_i·P_i，f_i = N/(K·T)·count_i，P_i = mean_t s'_{i,t}，s' 按 token 归一化。"""
    B, T, E = scores.shape
    K = indices.size(-1)
    counts = torch.zeros(B, E, device=scores.device, dtype=scores.dtype)
    counts.scatter_add_(1, indices.reshape(B, -1), torch.ones_like(indices.reshape(B, -1), dtype=scores.dtype))
    f = counts * (n_experts / (K * T))
    p = (scores / scores.sum(-1, keepdim=True).clamp_min(1e-20)).mean(dim=1)
    return (f * p).sum(-1).mean()


class MoE(nn.Module):
    def __init__(self, cfg: FlashConfig, n_routed: Optional[int] = None, n_activated: Optional[int] = None) -> None:
        super().__init__()
        self.n_routed = int(n_routed or cfg.n_routed_experts)
        self.n_activated = int(n_activated or cfg.n_activated_experts)
        self.gate = Gate(cfg.dim, self.n_routed, self.n_activated, cfg.score_func, cfg.route_scale, cfg.norm_topk_prob)
        self.experts = nn.ModuleList(
            [Expert(cfg.dim, cfg.moe_inter_dim, cfg.swiglu_limit) for _ in range(self.n_routed)]
        )
        self.shared_experts = Expert(cfg.dim, cfg.moe_inter_dim, cfg.swiglu_limit)
        self._step_counts: Optional[torch.Tensor] = None
        self._record_usage = False
        self._usage_counts: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x [B,T,d] → (y [B,T,d], 序列均衡损失)。按专家排序后每个专家一次 GEMM。"""
        B, T, d = x.shape
        flat = x.reshape(-1, d)
        weights, indices, scores = self.gate(flat)
        k = indices.size(1)
        expert_ids = indices.reshape(-1)
        order = torch.argsort(expert_ids, stable=True)
        token_ids = torch.arange(flat.size(0), device=x.device).repeat_interleave(k)[order]
        w_sorted = weights.reshape(-1)[order].unsqueeze(-1)
        counts = torch.bincount(expert_ids, minlength=self.n_routed)
        self._remember(counts)

        gathered = flat.index_select(0, token_ids)
        bounds = [0] + counts.cumsum(0).tolist()
        outs = []
        for e in range(self.n_routed):
            start, end = bounds[e], bounds[e + 1]
            if start < end:
                outs.append(self.experts[e](gathered[start:end], w_sorted[start:end]))
        y = torch.zeros(flat.shape, device=x.device, dtype=torch.float32)
        if outs:
            y = y.index_add(0, token_ids, torch.cat(outs, dim=0).float())
        y = y + self.shared_experts(flat).float()
        balance = sequence_balance_loss(scores.view(B, T, -1), indices.view(B, T, -1), self.n_routed)
        return y.to(x.dtype).view(B, T, d), balance

    @torch.compiler.disable
    def _remember(self, counts: torch.Tensor) -> None:
        counts = counts.detach()
        if self.training:
            self._step_counts = counts if self._step_counts is None else self._step_counts + counts
        if self._record_usage:
            self._usage_counts = counts if self._usage_counts is None else self._usage_counts + counts

    @torch.no_grad()
    def update_expert_bias(self, speed: float) -> None:
        """负载高于平均就 −speed，低于平均就 +speed；计数为整个 step（含多个 micro-batch）。"""
        counts = self._step_counts
        self._step_counts = None
        if counts is None:
            return
        counts = counts.to(self.gate.bias.device, torch.float32)
        expected = counts.sum() / self.n_routed
        self.gate.bias.add_(float(speed) * torch.sign(expected - counts))
