"""带缓存的推理：分块 forward、CED 预填充（Decoder SWA Bounded Replay）、逐 token 解码。

缓存与官方 inference/model.py 的 buffer 对应：每层一个 window 槽的滑窗环形缓存；kv source 层的
压缩主 KV、indexer K 与未满组的中间状态（kv_state / score_state）；Engram 的压缩 id 历史。
环形缓存另记每个槽存的绝对位置，用来判断某个滑窗位置是否真的在缓存里（Bounded Replay 之后，
更早的位置不可见）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .attention import Attention, select_candidate_blocks, sparse_attention
from .ckpt import autocast_dtype
from .mhc import hc_pre, identity_pre_mix
from .rope import rotate_tail


@dataclass
class LayerCache:
    win_kv: torch.Tensor  # [B,W,hd]
    win_pos: torch.Tensor  # [B,W]，槽里存的绝对位置，−1 为空
    ckv: Optional[torch.Tensor] = None  # [B,cap,hd] 已 RoPE 的压缩主 KV（仅 kv source 层）
    ik: Optional[torch.Tensor] = None  # [B,cap,Ihd]
    kv_state: Optional[torch.Tensor] = None  # [B,m,hd] fp32
    score_state: Optional[torch.Tensor] = None


@dataclass
class InferenceCache:
    layers: List[LayerCache]
    hist: torch.Tensor  # [B,max_len] Engram 压缩 id 历史
    max_len: int
    pos: int = 0

    @classmethod
    def create(cls, model, batch: int, max_len: Optional[int] = None, device=None, dtype=None) -> "InferenceCache":
        cfg = model.cfg
        max_len = int(max_len or cfg.max_seq_len)
        device = device or model.embed.weight.device
        dtype = dtype or model.embed.weight.dtype
        layers = []
        for i, block in enumerate(model.layers):
            attn = block.attn
            lc = LayerCache(
                win_kv=torch.zeros(batch, cfg.window_size, cfg.head_dim, device=device, dtype=dtype),
                win_pos=torch.full((batch, cfg.window_size), -1, device=device, dtype=torch.long),
            )
            if attn.compressor is not None:
                m = attn.compress_ratio
                cap = max_len // m + 1
                lc.ckv = torch.zeros(batch, cap, cfg.head_dim, device=device, dtype=dtype)
                lc.ik = torch.zeros(batch, cap, cfg.index_head_dim, device=device, dtype=dtype)
                lc.kv_state = torch.zeros(batch, m, cfg.head_dim, device=device, dtype=torch.float32)
                lc.score_state = torch.full((batch, m, cfg.head_dim), float("-inf"), device=device, dtype=torch.float32)
            layers.append(lc)
        hist = torch.zeros(batch, max_len, device=device, dtype=torch.long)
        return cls(layers=layers, hist=hist, max_len=max_len)

    def drop_window(self) -> None:
        """丢掉所有滑窗缓存（全局 KV 保留）：对应部署时 SWA KV 不进持久缓存。"""
        for lc in self.layers:
            lc.win_pos.fill_(-1)


@dataclass
class StepState:
    """一次分块 forward 内的跨层共享（来源层写、后续层读）。"""

    src: Optional[LayerCache] = None
    topk: Optional[torch.Tensor] = None
    candidates: Optional[torch.Tensor] = None
    use_candidates: bool = False


def compressor_append(attn: Attention, x: torch.Tensor, p0: int, lc: LayerCache) -> None:
    """把从绝对位置 p0 开始的 token 并入压缩缓存；未满一组的部分留在中间状态里。"""
    comp = attn.compressor
    m = comp.ratio
    rd = attn.rope_dim
    if m == 1:
        latent = comp.norm(comp.wkv(x))
        j0, n_new = p0, x.size(1)
    else:
        r = p0 % m
        with torch.autocast(device_type=x.device.type, enabled=False):
            kv = F.linear(x.float(), comp.wkv.weight.float())
            score = F.linear(x.float(), comp.wgate.weight.float())
            kv = torch.cat([lc.kv_state[:, :r], kv], dim=1)
            score = torch.cat([lc.score_state[:, :r], score], dim=1)
            total = kv.size(1)
            n_new = total // m
            rem = total - n_new * m
            pooled = (kv[:, : n_new * m].unflatten(1, (n_new, m)) * score[:, : n_new * m].unflatten(1, (n_new, m)).softmax(2)).sum(2)
            lc.kv_state[:, :rem] = kv[:, n_new * m :]
            lc.score_state[:, :rem] = score[:, n_new * m :]
        latent = comp.norm(pooled.to(x.dtype))
        j0 = (p0 - r) // m
    if n_new <= 0:
        return
    freqs_c = attn.freqs_cis[j0 * m : (j0 + n_new) * m : m]
    if attn.indexer is not None and attn.indexer.owns_k:
        lc.ik[:, j0 : j0 + n_new] = attn.indexer.make_keys(latent, freqs_c).to(lc.ik.dtype)
    lc.ckv[:, j0 : j0 + n_new] = rotate_tail(latent, freqs_c, rd).to(lc.ckv.dtype)


def attention_infer(
    attn: Attention, x: torch.Tensor, start: int, lc: LayerCache, st: StepState, *, global_from: int
) -> torch.Tensor:
    """x 是绝对位置 [start, start+S) 的注意力输入。只有位置 ≥ global_from 的 token 写全局缓存。"""
    B, S, _ = x.shape
    W = attn.window_size
    dev = x.device
    freqs = attn.freqs_cis[start : start + S]
    qr, q = attn.project_q(x, freqs)
    kv = attn.project_window_kv(x, freqs)

    # 滑窗：位置 i 看 [i−W+1, i]；早于本块的从环形缓存取，且槽里必须恰好存着那个位置
    pos = torch.arange(start, start + S, device=dev)
    kpos = pos.unsqueeze(1) - (W - 1) + torch.arange(W, device=dev)
    in_chunk = kpos >= start
    slot = torch.remainder(kpos, W)
    idx = torch.where(in_chunk, W + (kpos - start), slot).unsqueeze(0).expand(B, -1, -1)
    ring_ok = lc.win_pos.gather(1, slot.flatten().unsqueeze(0).expand(B, -1)).view(B, S, W) == kpos
    valid = (kpos >= 0).unsqueeze(0) & (in_chunk.unsqueeze(0) | ring_ok)
    win_idx = torch.where(valid, idx, -1)
    kv_all = torch.cat([lc.win_kv.to(kv.dtype), kv], dim=1)
    idx_all = win_idx

    if attn.compress_ratio:
        m = attn.compress_ratio
        if attn.compressor is not None:
            new_from = max(start, global_from)
            if new_from < start + S:
                compressor_append(attn, x[:, new_from - start :], new_from, lc)
            st.src = lc
        src = st.src
        n_total = (start + S) // m
        ckv = src.ckv[:, :n_total].to(kv.dtype)
        vis_n = (pos + 1) // m
        if attn.indexer is not None:
            ind = attn.indexer
            score = ind.scores(x, qr, src.ik[:, :n_total].to(kv.dtype), freqs)
            visible = torch.arange(n_total, device=dev).unsqueeze(0) < vis_n.unsqueeze(1)
            score = score.masked_fill(~visible, float("-inf"))
            if st.use_candidates and ind.is_candidate_source:
                st.candidates = select_candidate_blocks(
                    score, vis_n.unsqueeze(-1), ind.candidate_topk_blocks, ind.candidate_block_size
                )
            elif st.use_candidates and ind.uses_candidates and st.candidates is not None:
                score = score.masked_fill(~st.candidates, float("-inf"))
            k = min(ind.topk, n_total)
            top = score.topk(k, dim=-1).indices.sort(dim=-1).values
            st.topk = torch.where(top < vis_n.view(1, S, 1), top, -1)
        topk = st.topk
        kv_all = torch.cat([kv_all, ckv], dim=1)
        idx_all = torch.cat([win_idx, torch.where(topk >= 0, topk + W + S, -1)], dim=-1)

    o, _ = sparse_attention(q, kv_all, idx_all, attn.attn_sink, attn.softmax_scale)

    # 写回环形缓存（只需本块最后 W 个位置）
    keep = min(S, W)
    tail_pos = pos[S - keep :]
    tail_slot = torch.remainder(tail_pos, W)
    lc.win_kv[:, tail_slot] = kv[:, S - keep :].to(lc.win_kv.dtype)
    lc.win_pos[:, tail_slot] = tail_pos.unsqueeze(0).expand(B, -1)
    return attn.output(o, freqs)


def _embed(model, ids: torch.Tensor, start: int, cache: InferenceCache):
    """embedding + Engram 哈希（回看用缓存里的压缩 id 历史）。"""
    cfg = model.cfg
    S = ids.size(1)
    h = model.embed(ids)
    dtype = autocast_dtype(h.device)
    if dtype is not None:
        h = h.to(dtype)
    x = h.unsqueeze(2).repeat(1, 1, cfg.hc_mult, 1)
    hashes = None
    if model.hasher is not None:
        cache.hist[:, start : start + S] = model.hasher.token_map[ids]
        lo = max(0, start - (model.engram_layout.max_ngram_size - 1))
        window = cache.hist[:, lo : start + S]
        full = model.hasher.hash_compressed(window, lambda src: torch.zeros_like(window, dtype=torch.bool))
        hashes = full[:, start - lo :]
    return x, hashes


def run_layers(
    model,
    x: torch.Tensor,
    pre_mix: torch.Tensor,
    hashes: Optional[torch.Tensor],
    start: int,
    layer_ids,
    cache: InferenceCache,
    st: StepState,
    *,
    global_from: int,
    targets: Optional[Dict[int, torch.Tensor]] = None,
):
    for i in layer_ids:
        block = model.layers[i]
        if block.engram is not None and hashes is not None:
            x = block.engram(x, hashes[:, :, block.engram.layer_hash_index])
        if targets is not None and i in model.cfg.dspark_target_layer_ids:
            targets[i] = x.float().mean(dim=2)
        lc = cache.layers[i]
        x, pre_mix, _ = block.run(
            x, pre_mix, lambda h, b=block, c=lc: attention_infer(b.attn, h, start, c, st, global_from=global_from)
        )
    return x, pre_mix


def _logits(model, x: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
    h = model.norm(hc_pre(x, pre_mix))
    with torch.autocast(device_type=h.device.type, enabled=False):
        return F.linear(h.float(), model.head.weight.float())


@torch.no_grad()
def ced_prefill(
    model,
    ids: torch.Tensor,
    cache: InferenceCache,
    *,
    prefix_len: int = 0,
    use_candidates: bool = False,
    targets: Optional[Dict[int, torch.Tensor]] = None,
) -> torch.Tensor:
    """CED 预填充，返回末位 logits [B,V]。

    ids 是完整 prompt（位置 0..P−1）。prefix_len > 0 表示前缀的全局 KV 已在缓存里：编码器只对
    [prefix_len − W, P) 运行，其中前缀部分是 Encoder SWA Bounded Replay（只重建滑窗、不写全局）。
    之后用第一个解码器层的压缩器补全解码器全局 KV，解码器只跑最后 W 个 token
    （Decoder SWA Bounded Replay）。
    """
    cfg = model.cfg
    P = ids.size(1)
    W = cfg.window_size
    E = cfg.n_encoder_layers
    enc_start = max(0, prefix_len - W) if prefix_len > 0 else 0
    x, hashes = _embed(model, ids[:, enc_start:], enc_start, cache)
    pre_mix = identity_pre_mix(x, cfg.hc_mult)
    st = StepState(use_candidates=use_candidates)
    x, pre_mix = run_layers(model, x, pre_mix, hashes, enc_start, range(E), cache, st, global_from=prefix_len)

    # 解码器全局 KV：第一个解码器层（Full）把编码器输出投影成压缩主 KV 与 indexer K
    first = model.layers[E]
    if first.attn.compressor is not None and prefix_len < P:
        h = first.attn_norm(hc_pre(x, pre_mix))
        compressor_append(first.attn, h[:, prefix_len - enc_start :], prefix_len, cache.layers[E])

    # 解码器只重放最后 W 个 token；更早的解码器滑窗位置不在缓存里，自然不可见
    r0 = max(enc_start, P - W)
    for lc in cache.layers[E:]:
        lc.win_pos.fill_(-1)
    xd, pd = x[:, r0 - enc_start :], pre_mix[:, r0 - enc_start :]
    hd = None if hashes is None else hashes[:, r0 - enc_start :]
    st = StepState(use_candidates=use_candidates)
    xd, pd = run_layers(model, xd, pd, hd, r0, range(E, cfg.n_layers), cache, st, global_from=P, targets=targets)
    cache.pos = P
    return _logits(model, xd[:, -1:], pd[:, -1:])[:, -1]


@torch.no_grad()
def decode_chunk(
    model,
    ids: torch.Tensor,
    cache: InferenceCache,
    *,
    use_candidates: bool = False,
    targets: Optional[Dict[int, torch.Tensor]] = None,
) -> torch.Tensor:
    """在缓存末尾续 S 个 token，完整走全部层，返回这 S 个位置的 logits [B,S,V]。"""
    start = cache.pos
    S = ids.size(1)
    if start + S > cache.max_len:
        raise ValueError("超出缓存容量")
    x, hashes = _embed(model, ids, start, cache)
    pre_mix = identity_pre_mix(x, model.cfg.hc_mult)
    st = StepState(use_candidates=use_candidates)
    x, pre_mix = run_layers(
        model, x, pre_mix, hashes, start, range(model.cfg.n_layers), cache, st, global_from=start, targets=targets
    )
    cache.pos = start + S
    return _logits(model, x, pre_mix)


def sample(logits: torch.Tensor, temperature: float = 1.0, top_k: Optional[int] = 50) -> torch.Tensor:
    """logits [B,V] → [B,1]。temperature ≤ 0 时取 argmax。"""
    if temperature is None or temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits / max(temperature, 1e-6)
    if top_k is not None and top_k > 0:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
    return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)


def candidates_info(logits: torch.Tensor, temperature: float, k: int) -> dict:
    scaled = logits[0] / max(temperature, 1e-6) if temperature and temperature > 0 else logits[0]
    k = max(1, min(int(k), scaled.size(-1)))
    top_logits, top_ids = torch.topk(scaled, k)
    probs = F.softmax(scaled, dim=-1).gather(-1, top_ids)
    return {
        "ids": [int(x) for x in top_ids.tolist()],
        "logits": [float(x) for x in top_logits.tolist()],
        "probs": [float(x) for x in probs.tolist()],
    }


@torch.no_grad()
def stream_tokens(
    model,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: Optional[int] = 50,
    eos_token_id: Optional[int] = None,
    use_candidates: bool = False,
    debug_k: int = 0,
) -> Iterator[Tuple[torch.Tensor, Optional[dict]]]:
    """逐 token 产出 (next_id [B,1], 调试信息)。缓存写满时用后半段上下文重新预填充。"""
    model.eval()
    max_len = model.cfg.max_seq_len
    ids = input_ids[:, -max_len:]
    cache = InferenceCache.create(model, ids.size(0), max_len)
    logits = ced_prefill(model, ids, cache, use_candidates=use_candidates)
    for step in range(max_new_tokens):
        info = candidates_info(logits, temperature, debug_k) if debug_k else None
        nxt = sample(logits, temperature, top_k)
        yield nxt, info
        if eos_token_id is not None and bool((nxt == int(eos_token_id)).all()):
            return
        if step == max_new_tokens - 1:
            return
        ids = torch.cat([ids, nxt], dim=1)
        if cache.pos >= max_len:
            ids = ids[:, -(max_len // 2) :]
            cache = InferenceCache.create(model, ids.size(0), max_len)
            logits = ced_prefill(model, ids, cache, use_candidates=use_candidates)
        else:
            logits = decode_chunk(model, nxt, cache, use_candidates=use_candidates)[:, -1]
