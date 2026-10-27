"""MiniSeek-D：稠密 Decoder-only Transformer LM（RoPE + RMSNorm + FFN）.

产品线 MiniSeek-D*（当前 D0.1）。MoE 见 model/moe.py（MiniSeek-M*）。
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .sampling import sample_next


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        norm = x.float().pow(2).mean(-1, keepdim=True)
        x_normed = x.float() * torch.rsqrt(norm + self.eps)
        return (x_normed * self.weight.float()).type_as(x)


def precompute_rope_cis(
    head_dim: int, max_seq_len: int, theta: float = 1e6
) -> torch.Tensor:
    """预计算 RoPE 复数频率，形状 (max_seq_len, head_dim//2) complex64."""
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
    freqs = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2).float() / head_dim)
    )
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)  # (T, head_dim/2)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex


def apply_rope(x: torch.Tensor, cis: torch.Tensor) -> torch.Tensor:
    """x: (B, H, T, D), cis: (T, D/2) complex → 同形状实数张量."""
    b, h, t, d = x.shape
    x_ = torch.view_as_complex(x.float().reshape(b, h, t, d // 2, 2))
    cis = cis[:t].unsqueeze(0).unsqueeze(0)  # (1,1,T,D/2)
    out = torch.view_as_real(x_ * cis).reshape(b, h, t, d)
    return out.type_as(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.d_model = cfg.d_model
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self, x: torch.Tensor, rope_cis: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        b, t, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.d_model, dim=-1)
        # (B, T, H, D) -> (B, H, T, D)
        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        q = apply_rope(q, rope_cis)
        k = apply_rope(k, rope_cis)

        # scaled dot-product；CPU smoke 用 math 路径，CUDA 可走 SDPA
        if hasattr(F, "scaled_dot_product_attention"):
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, is_causal=attn_mask is None, dropout_p=0.0
            )
        else:
            scale = 1.0 / math.sqrt(self.head_dim)
            att = (q @ k.transpose(-2, -1)) * scale
            causal = torch.triu(
                torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1
            )
            att = att.masked_fill(causal, float("-inf"))
            att = F.softmax(att, dim=-1)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(b, t, self.d_model)
        return self.dropout(self.o_proj(y))


class FeedForward(nn.Module):
    """Dense FFN：GELU，intermediate=2048（冻结）."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(cfg.d_model, cfg.intermediate_size, bias=False)
        self.fc2 = nn.Linear(cfg.intermediate_size, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, eps=cfg.rms_norm_eps)
        self.attn = CausalSelfAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model, eps=cfg.rms_norm_eps)
        self.ffn = FeedForward(cfg)
        # TODO Stage1+: 按 cfg.use_moe 替换 self.ffn
        # TODO Stage2+: 按 cfg.use_swa 替换注意力窗口
        # TODO Stage3+: CSA2
        # TODO Stage4+: CED 分 Encoder/Decoder 层

    def forward(self, x: torch.Tensor, rope_cis: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), rope_cis)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class DenseLM(nn.Module):
    """Dense Decoder-only LM — Stage 0 唯一实现路径."""

    def __init__(self, cfg: Optional[ModelConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or ModelConfig()
        if self.cfg.use_moe or self.cfg.use_swa or self.cfg.use_csa2 or self.cfg.use_ced:
            raise NotImplementedError(
                "Stage 0 仅 Dense；MoE/SWA/CSA2/CED 为后续阶段 TODO"
            )

        c = self.cfg
        self.tok_emb = nn.Embedding(c.vocab_size, c.d_model)
        self.drop = nn.Dropout(c.dropout)
        self.layers = nn.ModuleList([TransformerBlock(c) for _ in range(c.n_layers)])
        self.norm = RMSNorm(c.d_model, eps=c.rms_norm_eps)
        self.lm_head = nn.Linear(c.d_model, c.vocab_size, bias=False)
        if c.tie_word_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        rope = precompute_rope_cis(c.head_dim, c.max_seq_len, theta=c.rope_theta)
        self.register_buffer("rope_cis", rope, persistent=False)

        self.apply(self._init_weights)
        # 残差出口：加深网络时缩小输出投影，避免初始 logits 爆炸
        for layer in self.layers:
            torch.nn.init.normal_(
                layer.attn.o_proj.weight, mean=0.0, std=0.02 / math.sqrt(2 * c.n_layers)
            )
            torch.nn.init.normal_(
                layer.ffn.fc2.weight, mean=0.0, std=0.02 / math.sqrt(2 * c.n_layers)
            )

        n = count_parameters(self)
        print(
            f"[DenseLM] vocab={c.vocab_size} d_model={c.d_model} "
            f"n_layers={c.n_layers} params={n:,} ({n / 1e6:.2f}M)"
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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        input_ids: (B, T)
        returns: logits (B, T, V), optional loss
        """
        b, t = input_ids.shape
        if t > self.cfg.max_seq_len:
            raise ValueError(f"seq_len {t} > max_seq_len {self.cfg.max_seq_len}")

        x = self.drop(self.tok_emb(input_ids))
        cis = self.rope_cis[:t]
        for layer in self.layers:
            x = layer(x, cis)
        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
        return logits, loss

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
        """逐 token 产出 next_id（CPU int）；遇到 eos 后停止。"""
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
