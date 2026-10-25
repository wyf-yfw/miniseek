"""MiniSeek-M 系列：DeepSeekMoE 风格混合专家（M0.1）。

每层：1 shared + N routed，top-k；训练损失 = CE + λ·load-balance aux。
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


def load_balance_loss(
    router_probs: torch.Tensor, topk_idx: torch.Tensor, n_experts: int
) -> torch.Tensor:
    """Switch-style aux: n_experts * sum_i (f_i * P_i).

    router_probs: (T, E) softmax over experts
    topk_idx: (T, K) expert indices
    """
    t = router_probs.size(0)
    k = topk_idx.size(1)
    # f_i = fraction of (token,slot) assignments to expert i
    one_hot = F.one_hot(topk_idx.reshape(-1), num_classes=n_experts).float()  # (T*K, E)
    f = one_hot.mean(dim=0)  # (E,)  each assignment equally weighted → / (T*K)
    p = router_probs.mean(dim=0)  # (E,)
    return n_experts * (f * p).sum()


class ExpertMLP(nn.Module):
    """单专家 MLP：GELU，无 bias。"""

    def __init__(self, d_model: int, intermediate: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, intermediate, bias=False)
        self.fc2 = nn.Linear(intermediate, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


class MoELayer(nn.Module):
    """1 shared + N routed, top-k；返回 (y, aux_loss)。"""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        h = cfg.moe_intermediate_size
        d = cfg.d_model
        drop = cfg.dropout
        self.n_routed = cfg.n_routed_experts
        self.top_k = cfg.num_experts_per_tok
        self.shared = ExpertMLP(d, h, drop)
        self.experts = nn.ModuleList(
            [ExpertMLP(d, h, drop) for _ in range(self.n_routed)]
        )
        self.router = nn.Linear(d, self.n_routed, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, t, d = x.shape
        flat = x.reshape(b * t, d)
        shared_out = self.shared(flat)

        logits = self.router(flat)
        probs = F.softmax(logits, dim=-1)
        topk_prob, topk_idx = torch.topk(probs, self.top_k, dim=-1)
        topk_prob = topk_prob / topk_prob.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        routed = torch.zeros_like(flat)
        for slot in range(self.top_k):
            idx = topk_idx[:, slot]
            w = topk_prob[:, slot].unsqueeze(-1)
            for e in range(self.n_routed):
                mask = idx == e
                if not mask.any():
                    continue
                routed[mask] = routed[mask] + w[mask] * self.experts[e](flat[mask])

        y = (shared_out + routed).view(b, t, d)
        aux = load_balance_loss(probs, topk_idx, self.n_routed)
        return y, aux


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
                layer.moe.shared.fc2.weight,
                mean=0.0,
                std=0.02 / math.sqrt(2 * c.n_layers),
            )
            for exp in layer.moe.experts:
                torch.nn.init.normal_(
                    exp.fc2.weight, mean=0.0, std=0.02 / math.sqrt(2 * c.n_layers)
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


def build_moe(cfg: ModelConfig) -> MoELM:
    return MoELM(cfg)
