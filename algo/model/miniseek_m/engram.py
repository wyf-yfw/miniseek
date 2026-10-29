"""Engram：n-gram 哈希查表，按残差流分支做上下文门控后写回残差流。

对应官方 inference/engram.py（tokenizer 压缩、质数桶布局、乘子、XOR 滚动哈希）与 model.py 的 Engram
（每条流一个 key、共享 value、带符号平方根后 sigmoid 门控）。DeepSeek-M 去掉了原版的短因果卷积。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from .config import FlashConfig


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def find_next_prime(start: int, seen: set) -> int:
    """大于 start、且没分配过的最小质数。"""
    candidate = start + 1
    while not is_prime(candidate) or candidate in seen:
        candidate += 1
    return candidate


def build_compressed_token_map(tokenizer) -> Tuple[List[int], int]:
    """把规范化后相同的 token 合并（大小写、重音、空白），返回 (lookup, 压缩词表大小)。

    没有 HF fast tokenizer 后端时返回恒等映射。
    """
    backend = getattr(tokenizer, "backend_tokenizer", None)
    n = len(tokenizer) if hasattr(tokenizer, "__len__") else int(tokenizer.vocab_size)
    if backend is None:
        return list(range(n)), n

    from tokenizers import Regex, normalizers

    sentinel = "\ue000"
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )
    key_to_new: dict = {}
    lookup = [0] * n
    for token_id in range(n):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:
            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text
        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[token_id] = new_id
    return lookup, len(key_to_new)


def compute_hash_multipliers(layer_ids: Sequence[int], max_ngram_size: int, vocab_size: int) -> torch.Tensor:
    """每个 (层, 回看位置) 一个奇数乘子；上界保证 token_id × 乘子不溢出 int64。"""
    max_long = np.iinfo(np.int64).max
    bound = max(1, (max_long // vocab_size) // 2)
    rows = []
    for layer_id in layer_ids:
        rng = np.random.default_rng(10007 * layer_id)
        values = rng.integers(low=0, high=bound, size=(max_ngram_size,), dtype=np.int64)
        rows.append(torch.tensor(values * 2 + 1))
    return torch.stack(rows)


@dataclass(frozen=True)
class EngramLayout:
    """每层：(n-gram 阶 2..N) × 头，各占一个互不重复的质数大小的桶区间。"""

    max_ngram_size: int
    layer_ids: Tuple[int, ...]
    primes: Tuple[Tuple[Tuple[int, ...], ...], ...]  # [layer][order][head]
    n_heads: int
    head_dim: int

    @classmethod
    def from_config(cls, cfg: FlashConfig) -> Optional["EngramLayout"]:
        if not cfg.engram_layer_ids:
            return None
        primes, seen = [], set()
        for _ in cfg.engram_layer_ids:
            per_order = []
            for _ in range(cfg.engram_max_ngram_size - 1):
                sizes, current = [], cfg.engram_vocab_size - 1
                for _ in range(cfg.engram_n_heads):
                    current = find_next_prime(current, seen)
                    seen.add(current)
                    sizes.append(current)
                per_order.append(tuple(sizes))
            primes.append(tuple(per_order))
        return cls(
            cfg.engram_max_ngram_size, tuple(cfg.engram_layer_ids), tuple(primes), cfg.engram_n_heads, cfg.engram_head_dim
        )

    @property
    def n_cols(self) -> int:
        return (self.max_ngram_size - 1) * self.n_heads

    @property
    def num_embeddings(self) -> Tuple[int, ...]:
        return tuple(sum(sum(order) for order in layer) for layer in self.primes)


class NgramHasher(nn.Module):
    """每个位置以它结尾的 2..N-gram 的哈希 id：[B,T,n_engram_layers,n_cols]。

    回看在序列开头与文档边界处停止，空位填 pad token 的压缩 id。
    """

    def __init__(self, cfg: FlashConfig, layout: EngramLayout, token_map: Optional[Sequence[int]] = None) -> None:
        super().__init__()
        self.layout = layout
        self.pad_token_id = int(cfg.engram_pad_id)
        compressed = int(cfg.engram_compressed_vocab_size or 0)
        if token_map is None:
            if compressed and compressed != cfg.vocab_size:
                # 从 checkpoint 加载：压缩表由 load_state_dict 填入，乘子按配置里的压缩词表大小算
                token_map, vocab = [0] * cfg.vocab_size, compressed
            else:
                token_map, vocab = list(range(cfg.vocab_size)), cfg.vocab_size
        else:
            vocab = int(max(token_map)) + 1
            if compressed and compressed != vocab:
                raise ValueError(f"压缩词表大小 {vocab} 与配置 {compressed} 不一致，哈希会整体变掉")
        flat = [[p for order in layer for p in order] for layer in layout.primes]
        offsets = [np.cumsum([0, *sizes[:-1]]) for sizes in flat]
        self.register_buffer("token_map", torch.tensor(list(token_map), dtype=torch.long))
        self.register_buffer("primes", torch.tensor(layout.primes, dtype=torch.long), persistent=False)
        self.register_buffer("offsets", torch.tensor(np.array(offsets), dtype=torch.long), persistent=False)
        self.register_buffer(
            "multipliers",
            compute_hash_multipliers(layout.layer_ids, layout.max_ngram_size, vocab),
            persistent=False,
        )

    def hash_compressed(self, compressed: torch.Tensor, blocked_fn) -> torch.Tensor:
        B, T = compressed.shape
        pad = self.token_map[self.pad_token_id]
        positions = torch.arange(T, device=compressed.device)
        tokens, blocked = [], torch.zeros(B, T, dtype=torch.bool, device=compressed.device)
        for shift in range(self.layout.max_ngram_size):
            src = (positions - shift).clamp_min(0)
            source = compressed[:, src]
            blocked = blocked | (positions < shift).unsqueeze(0) | blocked_fn(src)
            tokens.append(torch.where(blocked, pad, source))
        tokens = torch.stack(tokens, dim=-1)  # [B,T,N]
        products = tokens.unsqueeze(2) * self.multipliers  # [B,T,L,N]
        rolling, hashes = products[..., 0], []
        for i in range(1, self.layout.max_ngram_size):
            rolling = torch.bitwise_xor(rolling, products[..., i])
            hashes.append(rolling.unsqueeze(-1) % self.primes[:, i - 1])
        return torch.cat(hashes, dim=-1) + self.offsets

    def forward(self, input_ids: torch.Tensor, doc_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        compressed = self.token_map[input_ids]
        if doc_ids is None:
            return self.hash_compressed(compressed, lambda src: torch.zeros_like(compressed, dtype=torch.bool))
        return self.hash_compressed(compressed, lambda src: doc_ids[:, src] != doc_ids)


class Engram(nn.Module):
    def __init__(self, cfg: FlashConfig, layer_id: int, layout: EngramLayout) -> None:
        super().__init__()
        self.layer_hash_index = layout.layer_ids.index(layer_id)
        self.dim = cfg.dim
        self.hc_mult = cfg.hc_mult
        self.eps = cfg.norm_eps
        self.embed = nn.Embedding(layout.num_embeddings[self.layer_hash_index], layout.head_dim)
        self.wkv = nn.Linear(layout.n_cols * layout.head_dim, cfg.dim * (cfg.hc_mult + 1), bias=False)
        self.q_weight = nn.Parameter(torch.ones(cfg.hc_mult, cfg.dim))
        self.k_weight = nn.Parameter(torch.ones(cfg.hc_mult, cfg.dim))
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        # 原版只把短卷积置 0（卷积分支恒等，门控值照常写入）；DeepSeek-M 去掉了卷积，因此不置零
        nn.init.normal_(self.embed.weight, std=0.02)
        nn.init.normal_(self.wkv.weight, std=0.02)
        self.q_weight.fill_(1.0)
        self.k_weight.fill_(1.0)

    def forward(
        self, x: torch.Tensor, hash_ids: torch.Tensor, token_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """x [B,T,hc,d]，hash_ids [B,T,n_cols]。"""
        kv = self.wkv(self.embed(hash_ids).flatten(-2))
        key, value = kv.split([self.hc_mult * self.dim, self.dim], dim=-1)
        with torch.autocast(device_type=x.device.type, enabled=False):
            key = key.float().unflatten(-1, (self.hc_mult, self.dim))
            weight = self.q_weight.float() * self.k_weight.float()
            h = x.float()
            # 按 (token, 流) 在 dim 上各自归一化，不跨流
            rstd = torch.rsqrt(h.square().mean(-1) + self.eps) * torch.rsqrt(key.square().mean(-1) + self.eps)
            dot = (h * weight * key).sum(-1) * rstd * self.dim**-0.5
            gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
            if token_mask is not None:
                gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0)
            out = h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)
        return out.to(x.dtype)
