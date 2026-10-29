"""DeepSeek-M 官方 inference 代码的纯 torch / numpy 语义移植，仅供对拍测试。

来源：https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/main/inference
（model.py / kernel.py / engram.py；MIT License, Copyright (c) 2023 DeepSeek）。
去掉了分布式、FP8/FP4 量化与 TileLang kernel；其余计算顺序保持原样。
函数只覆盖 prefill（start_pos == 0）路径，单测按需取用。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------- 基础 ----


def rms_norm(x, weight, eps):
    dtype = x.dtype
    x = x.float()
    var = x.square().mean(-1, keepdim=True)
    x = x * torch.rsqrt(var + eps)
    return (weight * x).to(dtype)


def precompute_freqs_cis(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow):
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:

        def corrected_dim(rotations):
            return dim * math.log(original_seq_len / (rotations * 2 * math.pi)) / (2 * math.log(base))

        low = max(math.floor(corrected_dim(beta_fast)), 0)
        high = min(math.ceil(corrected_dim(beta_slow)), dim - 1)
        ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / max(high - low, 1e-3)).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    freqs = torch.outer(torch.arange(seqlen), freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb_(x, freqs_cis, inverse=False):
    """官方原地版本。"""
    y = x
    x = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if x.ndim == 3:
        freqs_cis = freqs_cis.view(1, x.size(1), x.size(-1))
    else:
        freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    x = torch.view_as_real(x * freqs_cis).flatten(-2)
    y.copy_(x)
    return y


# ------------------------------------------------------------ 注意力 ----


def get_window_topk_idxs(window_size, bsz, seqlen, start_pos):
    if start_pos == 0:
        end = torch.arange(seqlen).unsqueeze(1)
        idxs = (end - window_size + 1).clamp(0) + torch.arange(min(seqlen, window_size))
        idxs = torch.where(idxs > end, -1, idxs)
    else:
        oldest = start_pos % window_size + 1
        idxs = torch.cat([torch.arange(oldest, window_size), torch.arange(oldest)])
        idxs = torch.where(idxs > start_pos, -1, idxs)
    return idxs.int().unsqueeze(0).expand(bsz, -1, -1).contiguous()


def compressor_prefill(x, wkv, wgate, norm_w, eps, ratio):
    """官方 Compressor.forward(start_pos=0)，返回完整组的 latent（RoPE 之前），不满一组返回 None。"""
    seqlen = x.size(1)
    dtype = x.dtype
    if ratio == 1:
        return rms_norm(F.linear(x, wkv), norm_w, eps)
    x = x.float()
    kv, score = F.linear(x, wkv.float()), F.linear(x, wgate.float())
    if seqlen < ratio:
        return None
    cutoff = seqlen - seqlen % ratio
    kv, score = kv[:, :cutoff], score[:, :cutoff]
    kv = kv.unflatten(1, (-1, ratio))
    score = score.unflatten(1, (-1, ratio))
    kv = (kv * score.softmax(dim=2)).sum(dim=2)
    return rms_norm(kv.to(dtype), norm_w, eps)


def select_candidate_blocks(logits, compress_lens, topk_blocks, block_size):
    width = logits.size(-1)
    scores = F.pad(logits, (0, -width % block_size), value=-torch.inf)
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = scores.size(-1)
    last = (compress_lens - 1) // block_size
    scores = scores.masked_fill(torch.arange(num_blocks, device=logits.device) == last, torch.inf)
    top = scores.topk(min(topk_blocks, num_blocks), dim=-1)
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, top.indices, top.values > -torch.inf)
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


def indexer_prefill(
    x,
    qr,
    index_k,
    *,
    wq_b,
    weights_proj,
    freqs_cis,
    ratio,
    rope_dim,
    n_heads,
    head_dim,
    index_topk,
    offset,
    candidates=None,
    candidate_source=None,
):
    """官方 Indexer.forward(start_pos=0) 的打分与 top-k。

    index_k: 已 RoPE 的 [B, Tc, head_dim]。candidate_source=(topk_blocks, block_size) 时返回候选池；
    candidates 非空时按官方 uses_candidates 屏蔽。返回 (idxs, candidates_or_None, index_score)。
    """
    bsz, seqlen, _ = x.size()
    end_pos = seqlen
    q = F.linear(qr, wq_b).unflatten(-1, (n_heads, head_dim))
    q = q.clone()
    apply_rotary_emb_(q[..., -rope_dim:], freqs_cis[:end_pos])
    index_k = index_k[:bsz, : end_pos // ratio]
    softmax_scale = head_dim**-0.5
    weights = F.linear(x, weights_proj) * (softmax_scale * n_heads**-0.5)
    index_score = torch.einsum("bshd,btd->bsht", q, index_k)
    index_score = (index_score.relu() * weights.unsqueeze(-1)).sum(dim=2)
    compress_lens = (torch.arange(1, seqlen + 1, device=x.device) // ratio).unsqueeze(-1)
    index_score = index_score.masked_fill(torch.arange(seqlen // ratio, device=x.device) >= compress_lens, -torch.inf)
    new_candidates = None
    if candidate_source is not None:
        new_candidates = select_candidate_blocks(index_score, compress_lens, *candidate_source)
    elif candidates is not None:
        index_score = index_score.masked_fill(~candidates, -torch.inf)
    topk = min(index_topk, end_pos // ratio)
    idxs = index_score.topk(topk, dim=-1, sorted=False).indices.sort(dim=-1).values
    return torch.where(idxs < compress_lens, idxs + offset, -1).int(), new_candidates, index_score


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    """kernel.py sparse_attn_kernel 的数学：按索引取条目，在线 softmax，分母加 exp(sink)。"""
    b, m, h, d = q.shape
    idx = topk_idxs.long()
    valid = idx >= 0
    g = kv.gather(1, idx.clamp(min=0).reshape(b, -1, 1).expand(-1, -1, d)).reshape(b, m, -1, d)
    g = torch.where(valid.unsqueeze(-1), g, torch.zeros_like(g)).float()
    s = torch.einsum("bmhd,bmkd->bmhk", q.float(), g) * softmax_scale
    s = s.masked_fill(~valid.unsqueeze(2), -torch.inf)
    mx = s.amax(dim=-1).clamp(min=-1e30)
    e = torch.exp(s - mx.unsqueeze(-1))
    denom = e.sum(-1) + torch.exp(attn_sink.float().view(1, 1, h) - mx)
    o = torch.einsum("bmhk,bmkd->bmhd", e, g) / denom.unsqueeze(-1)
    return o.to(q.dtype)


# ----------------------------------------------------------------- mHC ----


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    """kernel.py hc_split_sinkhorn_kernel 的数学。"""
    hc = hc_mult
    pre = torch.sigmoid(mixes[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(mixes[..., hc : 2 * hc] * hc_scale[1] + hc_base[hc : 2 * hc])
    comb = (mixes[..., 2 * hc :] * hc_scale[2] + hc_base[2 * hc :]).unflatten(-1, (hc, hc))
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


def hc_mixes(x, hc_fn, hc_scale, hc_base, norm_eps, hc_mult, iters, eps):
    x = x.flatten(2).float()
    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + norm_eps)
    mixes = F.linear(x, hc_fn) * rsqrt
    return hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult, iters, eps)


def hc_pre(x, pre_mix):
    y = torch.sum(pre_mix.unsqueeze(-1) * x.float(), dim=2)
    return y.to(x.dtype)


def hc_post(x, residual, post, comb):
    y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
    return y.type_as(x)


# ----------------------------------------------------------------- MoE ----


def gate_forward(x, weight, bias, topk, score_func="sqrtsoftplus", route_scale=1.0, norm_topk_prob=True):
    scores = F.linear(x.float(), weight.float())
    if score_func == "softmax":
        scores = scores.softmax(dim=-1)
    elif score_func == "sigmoid":
        scores = scores.sigmoid()
    else:
        scores = F.softplus(scores).sqrt()
    indices = (scores + bias).topk(topk, dim=-1)[1]
    weights = scores.gather(1, indices)
    if norm_topk_prob and topk > 1:
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    weights = weights * route_scale
    return weights, indices


def expert_forward(x, w1, w2, w3, swiglu_limit, weights=None):
    dtype = x.dtype
    gate = F.linear(x, w1).float()
    up = F.linear(x, w3).float()
    if swiglu_limit > 0:
        up = torch.clamp(up, min=-swiglu_limit, max=swiglu_limit)
        gate = torch.clamp(gate, max=swiglu_limit)
    x = F.silu(gate) * up
    if weights is not None:
        x = weights * x
    return F.linear(x.to(dtype), w2)


def moe_forward(x, gate_weight, gate_bias, experts, shared, topk, swiglu_limit, route_scale, score_func="sqrtsoftplus"):
    """experts: list[(w1, w2, w3)]；shared: (w1, w2, w3)。"""
    shape = x.size()
    x = x.reshape(-1, shape[-1])
    weights, indices = gate_forward(x, gate_weight, gate_bias, topk, score_func, route_scale)
    y = torch.zeros_like(x, dtype=torch.float32)
    for i, (w1, w2, w3) in enumerate(experts):
        idx, top = torch.where(indices == i)
        if idx.numel() == 0:
            continue
        y[idx] += expert_forward(x[idx], w1, w2, w3, swiglu_limit, weights[idx, top, None])
    y += expert_forward(x, *shared, swiglu_limit)
    return y.type_as(x).view(shape)


# -------------------------------------------------------------- Engram ----


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


def find_next_prime(start, seen_primes):
    candidate = start + 1
    while not is_prime(candidate) or candidate in seen_primes:
        candidate += 1
    return candidate


def compute_hash_multipliers(layer_ids, max_ngram_size, tokenizer_vocab_size):
    max_long = np.iinfo(np.int64).max
    multiplier_bound = max(1, (max_long // tokenizer_vocab_size) // 2)
    rows = []
    for layer_id in layer_ids:
        generator = np.random.default_rng(10007 * layer_id)
        values = generator.integers(low=0, high=multiplier_bound, size=(max_ngram_size,), dtype=np.int64)
        rows.append(torch.tensor(values * 2 + 1))
    return torch.stack(rows)


@dataclass(frozen=True)
class EngramLayout:
    max_ngram_size: int
    layer_ids: tuple
    primes: tuple
    n_heads: int
    head_dim: int

    @classmethod
    def build(cls, layer_ids, max_ngram_size, n_heads, head_dim, engram_vocab_size):
        primes, seen = [], set()
        for _ in layer_ids:
            per_ngram = []
            for _ in range(max_ngram_size - 1):
                sizes, current = [], engram_vocab_size - 1
                for _ in range(n_heads):
                    current = find_next_prime(current, seen)
                    seen.add(current)
                    sizes.append(current)
                per_ngram.append(tuple(sizes))
            primes.append(tuple(per_ngram))
        return cls(max_ngram_size, tuple(layer_ids), tuple(primes), n_heads, head_dim)


def ngram_hash(compressed, layout: EngramLayout, compressed_vocab_size, pad_id_compressed, dead_mask=None):
    """官方 NgramHashState.forward(start_pos=0)。compressed: [B,L] 压缩 id；dead_mask 为 True 的位置视为 DEAD。"""
    DEAD = -1
    batch, seqlen = compressed.shape
    cache = compressed.clone()
    if dead_mask is not None:
        cache = torch.where(dead_mask, torch.full_like(cache, DEAD), cache)
    flat = [[p for per_ngram in layer for p in per_ngram] for layer in layout.primes]
    offsets = torch.tensor(np.array([np.cumsum([0, *sizes[:-1]]) for sizes in flat]))
    primes = torch.tensor(layout.primes)
    multipliers = compute_hash_multipliers(layout.layer_ids, layout.max_ngram_size, compressed_vocab_size)
    positions = torch.arange(seqlen).expand(batch, seqlen)
    tokens, blocked = [], torch.zeros_like(positions, dtype=torch.bool)
    for shift in range(layout.max_ngram_size):
        source = cache.gather(1, (positions - shift).clamp_min(0))
        blocked = blocked | (positions < shift) | (source == DEAD)
        tokens.append(torch.where(blocked, pad_id_compressed, source))
    tokens = torch.stack(tokens, dim=-1)
    products = tokens.unsqueeze(2) * multipliers
    rolling, hashes = products[..., 0], []
    for i in range(1, layout.max_ngram_size):
        rolling = torch.bitwise_xor(rolling, products[..., i])
        hashes.append(rolling.unsqueeze(-1) % primes[:, i - 1])
    return torch.cat(hashes, dim=-1) + offsets


def engram_forward(x, hash_ids, table, wkv, q_weight, k_weight, eps, dim, hc_mult, token_mask=None):
    """官方 Engram.forward（表为普通浮点矩阵）。x: [B,L,hc,dim]；hash_ids: [B,L,n_cols]。"""
    clamp_value = 1e-6
    kv = F.linear(F.embedding(hash_ids, table).flatten(-2), wkv)
    key, value = kv.split([hc_mult * dim, dim], dim=-1)
    key = key.float().unflatten(-1, (hc_mult, dim))
    weight = q_weight.float() * k_weight.float()
    h = x.float()
    rstd = torch.rsqrt(h.square().mean(-1) + eps) * torch.rsqrt(key.square().mean(-1) + eps)
    dot = (h * weight * key).sum(-1) * rstd * dim**-0.5
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(clamp_value).sqrt(), dot))
    if token_mask is not None:
        gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0)
    return (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(x.dtype)


def build_compressed_token_map(tokenizer):
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
    backend = tokenizer.backend_tokenizer
    key_to_new, lookup = {}, [0] * len(tokenizer)
    for token_id in range(len(tokenizer)):
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


# -------------------------------------------------------------- DSpark ----


def get_dspark_topk_idxs(window_size, bsz, block_size, start_pos):
    assert start_pos > 0
    matrix = torch.cat([torch.arange(min(window_size, start_pos + 1)), window_size + torch.arange(block_size)])
    return matrix.int().view(1, 1, -1).expand(bsz, block_size, -1).contiguous()
