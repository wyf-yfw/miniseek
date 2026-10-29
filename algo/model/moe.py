"""MiniSeek-M 系列：DeepSeek-V4.1 的 MoE 层（宽度和专家个数仍是 M0.1）。

每层：1 个共享专家始终计算 + N 个路由专家 top-k。
专家是 clamp SwiGLU。选专家用 sigmoid 分数加 bias，加权只用原始分数。
bias 不进梯度，每步按负载加减。序列级均衡损失权重由训练的 aux_coef 控制（V4.1 为 0.0001）。
注意力仍是全因果，残差仍是普通 Pre-Norm 相加。
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .dense import (
    CausalSelfAttention,
    RMSNorm,
    count_parameters,
    precompute_rope_cis,
)


def clamped_swiglu(gate: torch.Tensor, up: torch.Tensor, limit: float) -> torch.Tensor:
    """SwiGLU，pre-activation 按 OpenAI 的方式截断。阈值用 V4.1 的 10。

    gate 只限制上界，up 限制到 [-limit, limit]。激活仍是 SiLU(gate) * up。
    """
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    return F.silu(gate) * up


def route_tokens(
    logits: torch.Tensor, bias: torch.Tensor, top_k: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """sigmoid 分数加 bias 只用来选 top-k；加权用选中项的原始分数，再归一化。

    logits: (T, E)，bias: (E,)。
    返回 scores (T, E)、topk_weight (T, K)、topk_idx (T, K)。
    """
    scores = torch.sigmoid(logits)
    _choice, topk_idx = torch.topk(scores + bias, top_k, dim=-1)
    topk_weight = scores.gather(-1, topk_idx)
    topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    return scores, topk_weight, topk_idx


def sequence_balance_loss(
    scores: torch.Tensor, topk_idx: torch.Tensor, n_experts: int
) -> torch.Tensor:
    """每条序列自己的 N * Σ f_i P_i，再对 batch 取平均。

    scores: (B, T, E) 原始 sigmoid 分数。topk_idx: (B, T, K)。
    f_i 是这条序列里分到专家 i 的槽位比例，P_i 是该专家分数的平均值。
    """
    b, _t, _e = scores.shape
    k = topk_idx.size(-1)
    one_hot = F.one_hot(topk_idx.reshape(b, -1), num_classes=n_experts).to(scores.dtype)
    f = one_hot.mean(dim=1)
    p = scores.mean(dim=1)
    return (n_experts * (f * p).sum(dim=-1)).mean()


class ExpertMLP(nn.Module):
    """单专家：clamp SwiGLU，三套无 bias 线性层。"""

    def __init__(
        self, d_model: int, intermediate: int, dropout: float = 0.0, clamp: float = 10.0
    ) -> None:
        super().__init__()
        self.clamp = float(clamp)
        self.gate_proj = nn.Linear(d_model, intermediate, bias=False)
        self.up_proj = nn.Linear(d_model, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = clamped_swiglu(self.gate_proj(x), self.up_proj(x), self.clamp)
        return self.dropout(self.down_proj(hidden))


class MoELayer(nn.Module):
    """1 shared + N routed, top-k；返回 (y, sequence_balance_loss)。"""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        h = cfg.moe_intermediate_size
        d = cfg.d_model
        drop = cfg.dropout
        clamp = cfg.swiglu_clamp
        self.n_routed = cfg.n_routed_experts
        self.top_k = cfg.num_experts_per_tok
        self.bias_speed = float(cfg.expert_bias_speed)
        self.shared = ExpertMLP(d, h, drop, clamp)
        self.experts = nn.ModuleList(
            [ExpertMLP(d, h, drop, clamp) for _ in range(self.n_routed)]
        )
        self.router = nn.Linear(d, self.n_routed, bias=False)
        # 不进优化器。选专家时加到分数上，过载就减、欠载就加。
        self.register_buffer("expert_bias", torch.zeros(self.n_routed))
        # Probe-only: when True, accumulate top-k assignment counts (CPU long).
        self._record_usage = False
        self._usage_counts: Optional[torch.Tensor] = None
        self._step_counts: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sorted / grouped expert dispatch (torch-only).

        Flatten (token, slot) → sort by expert id → one contiguous GEMM per
        expert with count>0 → scatter-add weighted outputs. Architecture is
        unchanged (1 shared + N routed, top-k). megablocks / grouped_gemm are
        not required; drop them in later if available for fused kernels.
        """
        b, t, d = x.shape
        flat = x.reshape(b * t, d)  # (T, D)
        n_tok = flat.size(0)
        shared_out = self.shared(flat)

        logits = self.router(flat)
        scores, topk_prob, topk_idx = route_tokens(logits, self.expert_bias, self.top_k)

        # (T, K) → (T*K,) assignments
        k = self.top_k
        expert_ids = topk_idx.reshape(-1)  # (T*K,)
        weights = topk_prob.reshape(-1, 1)  # (T*K, 1)
        # token id for each (token, slot) pair
        token_ids = (
            torch.arange(n_tok, device=flat.device)
            .unsqueeze(1)
            .expand(n_tok, k)
            .reshape(-1)
        )

        # Sort by expert so each expert's tokens form a contiguous slice
        order = torch.argsort(expert_ids, stable=True)
        expert_sorted = expert_ids[order]
        token_sorted = token_ids[order]
        weight_sorted = weights[order]

        counts = torch.bincount(expert_sorted, minlength=self.n_routed)
        self._remember_assignment_counts(counts)
        # offsets[e] = start index of expert e in the sorted list
        offsets = torch.zeros(self.n_routed + 1, dtype=torch.long, device=flat.device)
        offsets[1:] = counts.cumsum(0)

        # Gather tokens once in expert order, then slice per expert
        gathered = flat.index_select(0, token_sorted)  # (T*K, D)
        expert_out = torch.empty_like(gathered)

        # One contiguous mm (via ExpertMLP) per expert with tokens.
        # 边界一次性搬到 CPU：原来每个专家两次 .item()，8 层 × 32 专家 = 256 次
        # 强制同步，每次都要抽干 GPU 流水线。
        bounds = offsets.tolist()
        for e in range(self.n_routed):
            start, end = bounds[e], bounds[e + 1]
            if start == end:
                continue
            expert_out[start:end] = self.experts[e](gathered[start:end])

        weighted = expert_out * weight_sorted  # (T*K, D)
        routed = torch.zeros_like(flat)
        routed.index_add_(0, token_sorted, weighted)

        y = (shared_out + routed).view(b, t, d)
        aux = sequence_balance_loss(
            scores.view(b, t, -1), topk_idx.view(b, t, -1), self.n_routed
        )
        return y, aux

    @torch.compiler.disable
    def _remember_assignment_counts(self, counts: torch.Tensor) -> None:
        """训练步留一份计数给 bias 更新；探针另累加。不放进 compile 图。"""
        counts = counts.detach()
        if self.training:
            self._step_counts = counts
        if not self._record_usage:
            return
        if self._usage_counts is None:
            self._usage_counts = counts
        else:
            self._usage_counts = self._usage_counts + counts

    @torch.no_grad()
    def update_expert_bias(self, speed: Optional[float] = None) -> None:
        """这一步谁分到的 token 多于平均数，就减小它的 bias，反之加大。"""
        counts = self._step_counts
        self._step_counts = None
        if counts is None:
            return
        counts = counts.to(device=self.expert_bias.device, dtype=self.expert_bias.dtype)
        expected = counts.sum() / self.n_routed
        step = self.bias_speed if speed is None else float(speed)
        self.expert_bias.add_(step * torch.sign(expected - counts))


class MoETransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, eps=cfg.rms_norm_eps)
        self.attn = CausalSelfAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model, eps=cfg.rms_norm_eps)
        self.moe = MoELayer(cfg)

    def forward(
        self, x: torch.Tensor, rope_cis: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x + self.attn(self.attn_norm(x), rope_cis)
        y, aux = self.moe(self.ffn_norm(x))
        return x + y, aux


class MoELM(nn.Module):
    """MiniSeek-M0.1：每层 DeepSeekMoE，full causal attention。"""

    def __init__(self, cfg: Optional[ModelConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or ModelConfig(arch="moe")
        if self.cfg.arch != "moe" and not self.cfg.use_moe:
            raise ValueError("MoELM requires arch='moe' or use_moe=True")

        c = self.cfg
        self.tok_emb = nn.Embedding(c.vocab_size, c.d_model)
        self.drop = nn.Dropout(c.dropout)
        self.layers = nn.ModuleList(
            [MoETransformerBlock(c) for _ in range(c.n_layers)]
        )
        self.norm = RMSNorm(c.d_model, eps=c.rms_norm_eps)
        self.lm_head = nn.Linear(c.d_model, c.vocab_size, bias=False)
        if c.tie_word_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        rope = precompute_rope_cis(c.head_dim, c.max_seq_len, theta=c.rope_theta)
        self.register_buffer("rope_cis", rope, persistent=False)

        self.apply(self._init_weights)
        for layer in self.layers:
            torch.nn.init.normal_(
                layer.attn.o_proj.weight, mean=0.0, std=0.02 / math.sqrt(2 * c.n_layers)
            )
            torch.nn.init.normal_(
                layer.moe.shared.down_proj.weight,
                mean=0.0,
                std=0.02 / math.sqrt(2 * c.n_layers),
            )
            for exp in layer.moe.experts:
                torch.nn.init.normal_(
                    exp.down_proj.weight, mean=0.0, std=0.02 / math.sqrt(2 * c.n_layers)
                )

        n = count_parameters(self)
        print(
            f"[MoELM] vocab={c.vocab_size} d_model={c.d_model} "
            f"n_layers={c.n_layers} shared={c.n_shared_experts} "
            f"routed={c.n_routed_experts} top_k={c.num_experts_per_tok} "
            f"h={c.moe_intermediate_size} params={n:,} ({n / 1e6:.2f}M)"
        )

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        returns: logits (B,T,V), optional CE loss, aux_loss (scalar)
        """
        b, t = input_ids.shape
        if t > self.cfg.max_seq_len:
            raise ValueError(f"seq_len {t} > max_seq_len {self.cfg.max_seq_len}")

        x = self.drop(self.tok_emb(input_ids))
        cis = self.rope_cis[:t]
        aux_sum = x.new_zeros(())
        for layer in self.layers:
            x, aux = layer(x, cis)
            aux_sum = aux_sum + aux
        aux_loss = aux_sum / max(len(self.layers), 1)

        x = self.norm(x)
        logits = self.lm_head(x)

        ce = None
        if labels is not None:
            ce = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
        return logits, ce, aux_loss

    def begin_expert_usage(self) -> None:
        """Start recording routed top-k assignment counts (probe / eval only)."""
        for block in self.layers:
            block.moe._record_usage = True
            block.moe._usage_counts = None

    def end_expert_usage(self) -> torch.Tensor:
        """Stop recording; return summed counts over layers, shape (n_routed,)."""
        total: Optional[torch.Tensor] = None
        for block in self.layers:
            block.moe._record_usage = False
            counts = block.moe._usage_counts
            block.moe._usage_counts = None
            if counts is None:
                continue
            counts = counts.detach().cpu().long()
            total = counts if total is None else total + counts
        if total is None:
            return torch.zeros(self.cfg.n_routed_experts, dtype=torch.long)
        return total

    @torch.no_grad()
    def update_expert_bias(self, speed: Optional[float] = None) -> None:
        for block in self.layers:
            block.moe.update_expert_bias(speed)

    @torch.no_grad()
    def probe_expert_load(
        self, input_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, int]:
        """One forward with usage recording.

        Returns:
            counts: (n_routed,) long — top-k slot assignments summed over layers
            n_tokens: number of tokens in this batch (B*T)
        """
        was_training = self.training
        self.eval()
        self.begin_expert_usage()
        try:
            _logits, _ce, _aux = self.forward(input_ids)
        finally:
            counts = self.end_expert_usage()
            if was_training:
                self.train()
        n_tokens = int(input_ids.numel())
        return counts, n_tokens

    def _sample_next(
        self,
        ids: torch.Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
    ) -> torch.Tensor:
        ctx = ids[:, -self.cfg.max_seq_len :]
        logits, _, _ = self.forward(ctx)
        logits = logits[:, -1, :] / max(temperature, 1e-6)
        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        self.eval()
        ids = input_ids
        for _ in range(max_new_tokens):
            next_id = self._sample_next(ids, temperature=temperature, top_k=top_k)
            ids = torch.cat([ids, next_id], dim=1)
            if eos_token_id is not None and (next_id == eos_token_id).all():
                break
        return ids

    @torch.no_grad()
    def _sample_next_debug(
        self,
        ids: torch.Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        candidates_k: int = 10,
    ):
        """采样下一 token，并返回温度缩放后的 top candidates（id/logit/prob）。"""
        ctx = ids[:, -self.cfg.max_seq_len :]
        logits, _, _ = self.forward(ctx)
        logits = logits[:, -1, :] / max(temperature, 1e-6)
        cand_k = max(1, min(int(candidates_k), logits.size(-1)))
        cand_logits, cand_ids = torch.topk(logits, cand_k, dim=-1)
        probs_full = F.softmax(logits, dim=-1)
        cand_probs = probs_full.gather(-1, cand_ids)

        sample_logits = logits
        if top_k is not None and top_k > 0:
            v, _ = torch.topk(sample_logits, min(top_k, sample_logits.size(-1)))
            sample_logits = sample_logits.masked_fill(sample_logits < v[:, [-1]], float("-inf"))
        probs = F.softmax(sample_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        info = {
            "ids": [int(x) for x in cand_ids[0].tolist()],
            "logits": [float(x) for x in cand_logits[0].tolist()],
            "probs": [float(x) for x in cand_probs[0].tolist()],
        }
        return next_id, info

    @torch.no_grad()
    def generate_stream(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        eos_token_id: Optional[int] = None,
    ):
        self.eval()
        ids = input_ids
        for _ in range(max_new_tokens):
            next_id = self._sample_next(ids, temperature=temperature, top_k=top_k)
            ids = torch.cat([ids, next_id], dim=1)
            token = int(next_id.item())
            yield token
            if eos_token_id is not None and token == int(eos_token_id):
                break

    @torch.no_grad()
    def generate_stream_debug(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        eos_token_id: Optional[int] = None,
        candidates_k: int = 10,
    ):
        """逐 token 产出 (token_id, candidates_info)；用于逐步调试。"""
        self.eval()
        ids = input_ids
        for _ in range(max_new_tokens):
            next_id, info = self._sample_next_debug(
                ids, temperature=temperature, top_k=top_k, candidates_k=candidates_k
            )
            ids = torch.cat([ids, next_id], dim=1)
            token = int(next_id.item())
            yield token, info
            if eos_token_id is not None and token == int(eos_token_id):
                break


def step_expert_bias(model: nn.Module) -> None:
    """优化器 step 之后调用。compile 包装过的模型走原始模块。"""
    root = getattr(model, "_orig_mod", model)
    fn = getattr(root, "update_expert_bias", None)
    if fn is not None:
        fn()


def build_moe(cfg: ModelConfig) -> MoELM:
    return MoELM(cfg)
