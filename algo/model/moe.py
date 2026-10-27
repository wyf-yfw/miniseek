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
from .sampling import sample_next
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
        # Probe-only: when True, accumulate top-k assignment counts (CPU long).
        self._record_usage = False
        self._usage_counts: Optional[torch.Tensor] = None

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
        probs = F.softmax(logits, dim=-1)
        topk_prob, topk_idx = torch.topk(probs, self.top_k, dim=-1)
        topk_prob = topk_prob / topk_prob.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        if self._record_usage:
            batch_counts = torch.bincount(
                topk_idx.reshape(-1), minlength=self.n_routed
            ).detach().cpu()
            if self._usage_counts is None:
                self._usage_counts = batch_counts
            else:
                self._usage_counts += batch_counts

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
        # offsets[e] = start index of expert e in the sorted list
        offsets = torch.zeros(self.n_routed + 1, dtype=torch.long, device=flat.device)
        offsets[1:] = counts.cumsum(0)

        # Gather tokens once in expert order, then slice per expert
        gathered = flat.index_select(0, token_sorted)  # (T*K, D)
        expert_out = torch.empty_like(gathered)

        # One contiguous mm (via ExpertMLP) per expert with tokens
        # counts/offsets are tiny (n_routed); .item() is fine here
        for e in range(self.n_routed):
            start = int(offsets[e].item())
            end = int(offsets[e + 1].item())
            if start == end:
                continue
            expert_out[start:end] = self.experts[e](gathered[start:end])

        weighted = expert_out * weight_sorted  # (T*K, D)
        routed = torch.zeros_like(flat)
        routed.index_add_(0, token_sorted, weighted)

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
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
    ) -> torch.Tensor:
        ctx = ids[:, -self.cfg.max_seq_len :]
        out = self.forward(ctx)
        logits = out[0] if isinstance(out, tuple) else out
        return sample_next(
            logits[:, -1, :].squeeze(0),
            ids[0].tolist(),
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        eos_token_id: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
    ) -> torch.Tensor:
        self.eval()
        ids = input_ids
        for _ in range(max_new_tokens):
            next_id = self._sample_next(
                ids,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )
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
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
    ):
        self.eval()
        ids = input_ids
        for _ in range(max_new_tokens):
            next_id = self._sample_next(
                ids,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )
            ids = torch.cat([ids, next_id], dim=1)
            token = int(next_id.item())
            yield token
            if eos_token_id is not None and token == int(eos_token_id):
                break


def build_moe(cfg: ModelConfig) -> MoELM:
    return MoELM(cfg)
