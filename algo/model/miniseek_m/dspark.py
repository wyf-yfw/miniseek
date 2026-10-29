"""DSpark 草稿头：半自回归起草 + 置信度调度验证（arXiv:2607.05147；DeepSeek-M §2.4.3）。

结构对应官方 inference/model.py 的 DSparkBlock / DSparkAttention / DSparkMarkovHead /
DSparkConfidenceHead：
- 3 个 mHC Block，注意力只有滑窗：query 来自草稿块（anchor + noise×(γ−1)），key/value 是主干目标层
  特征投影出的上下文 KV（最近 window 个位置）加草稿块自身（块内双向可见）；MoE 比主干小。
- 并行阶段一次 forward 得到 γ 个位置的基础 logits（共享且冻结主干的 embedding 与 head）；
- 顺序阶段由低秩 Markov 头按上一个 token 加转移偏置；置信度头预测每个位置被接受的条件概率。
训练时主干冻结，损失 = 0.1·CE + 0.9·TV + 1.0·置信度 BCE，位置权重 exp(−k/γ)。
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import Attention, RMSNorm, sparse_attention
from .config import FlashConfig
from .mhc import hc_pre, identity_pre_mix
from .model import Block


class DSparkAttention(Attention):
    def forward_draft(
        self,
        x: torch.Tensor,
        main_x: torch.Tensor,
        ctx_pos: torch.Tensor,
        ctx_valid: torch.Tensor,
        q_pos: torch.Tensor,
    ) -> torch.Tensor:
        """x [N,γ,d] 草稿块；main_x [N,W,d] 上下文特征；ctx_pos/ctx_valid [N,W]；q_pos [N,γ]。"""
        N, G, _ = x.shape
        W = main_x.size(1)
        freqs_q = self.freqs_cis[q_pos]
        _, q = self.project_q(x, freqs_q)
        kv_blk = self.project_window_kv(x, freqs_q)
        kv_ctx = self.project_window_kv(main_x, self.freqs_cis[ctx_pos.clamp(min=0)])
        kv_all = torch.cat([kv_ctx, kv_blk], dim=1)
        ctx_idx = torch.where(ctx_valid, torch.arange(W, device=x.device), -1)
        blk_idx = (W + torch.arange(G, device=x.device)).expand(N, G)
        idx = torch.cat([ctx_idx, blk_idx], dim=-1).unsqueeze(1).expand(N, G, W + G)
        o, _ = sparse_attention(q, kv_all, idx, self.attn_sink, self.softmax_scale)
        return self.output(o, freqs_q)


class MarkovHead(nn.Module):
    """低秩一阶转移 B(x_{k−1}, ·) = W1[x_{k−1}]·W2。"""

    def __init__(self, vocab_size: int, rank: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, rank)
        self.head = nn.Linear(rank, vocab_size, bias=False)

    def forward(self, token_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        e = self.embed(token_ids)
        with torch.autocast(device_type=e.device.type, enabled=False):
            bias = F.linear(e.float(), self.head.weight.float())
        return bias, e


class ConfidenceHead(nn.Module):
    """c_k = σ(w·[h_k; W1[x_{k−1}]])，输出 logit（fp32）。"""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, 1, bias=False)

    def forward(self, hidden: torch.Tensor, markov_embed: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            z = torch.cat([hidden.float(), markov_embed.float()], dim=-1)
            return F.linear(z, self.proj.weight.float()).squeeze(-1)


def position_weights(gamma: int, device=None) -> torch.Tensor:
    return torch.exp(-torch.arange(gamma, device=device, dtype=torch.float32) / gamma)


def dspark_loss_terms(
    draft_logits: torch.Tensor,
    target_logits: torch.Tensor,
    targets: torch.Tensor,
    conf_logits: torch.Tensor,
    weights: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """草稿 [N,γ,V]、目标分布 logits [N,γ,V]、真值 [N,γ]、置信度 logit [N,γ]。各项对 N 取平均。"""
    logp_d = torch.log_softmax(draft_logits.float(), dim=-1)
    p_d = logp_d.exp()
    p_t = torch.softmax(target_logits.float(), dim=-1)
    ce = -logp_d.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    l1 = (p_d - p_t).abs().sum(-1)
    accept = (1 - 0.5 * l1).detach().clamp(0, 1)
    conf = F.binary_cross_entropy_with_logits(conf_logits.float(), accept, reduction="none")
    return {
        "ce": (ce * weights).sum(-1).mean(),
        "tv": (l1 * weights).sum(-1).mean(),
        "conf": (conf * weights).sum(-1).mean(),
        "accept": accept.mean(),
    }


class DSparkDrafter(nn.Module):
    def __init__(self, cfg: FlashConfig) -> None:
        super().__init__()
        self.cfg = cfg
        L = cfg.n_layers
        self.blocks = nn.ModuleList(
            [
                Block(
                    cfg,
                    L + i,
                    None,
                    n_routed=cfg.dspark_n_routed_experts,
                    n_activated=cfg.dspark_n_activated_experts,
                    attention_cls=DSparkAttention,
                )
                for i in range(cfg.n_mtp_layers)
            ]
        )
        self.main_proj = nn.Linear(cfg.dim * len(cfg.dspark_target_layer_ids), cfg.dim, bias=False)
        self.main_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.markov_head = MarkovHead(cfg.vocab_size, cfg.dspark_markov_rank)
        self.confidence_head = ConfidenceHead(cfg.dim + cfg.dspark_markov_rank)
        # STS 校准得到的逐位置温度（作用在置信度 logit 上）；未校准时为 1
        self.register_buffer("sts_temps", torch.ones(cfg.dspark_block_size))
        self._init_weights()

    @torch.no_grad()
    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, std=0.02)
        out_std = 0.02 / math.sqrt(2 * len(self.blocks))
        for blk in self.blocks:
            nn.init.normal_(blk.attn.wo_b.weight, std=out_std)
            for expert in list(blk.ffn.experts) + [blk.ffn.shared_experts]:
                nn.init.normal_(expert.w2.weight, std=out_std)
        nn.init.zeros_(self.confidence_head.proj.weight)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def update_expert_bias(self) -> None:
        for blk in self.blocks:
            blk.ffn.update_expert_bias(self.cfg.expert_bias_speed)

    def context_features(self, target_hidden: torch.Tensor) -> torch.Tensor:
        """主干目标层特征 [.., n_targets·d] → 上下文特征 main_x [.., d]。"""
        return self.main_norm(self.main_proj(target_hidden))

    def run_blocks(
        self,
        x_tokens: torch.Tensor,
        main_x: torch.Tensor,
        ctx_pos: torch.Tensor,
        ctx_valid: torch.Tensor,
        q_pos: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """x_tokens [N,γ,d] 草稿块 embedding → (合成后的隐状态 h [N,γ,d]，MoE 均衡损失)。"""
        x = x_tokens.unsqueeze(2).repeat(1, 1, self.cfg.hc_mult, 1)
        pre_mix = identity_pre_mix(x, self.cfg.hc_mult)
        balance = torch.zeros((), device=x.device)
        for blk in self.blocks:
            x, pre_mix, bal = blk.run(
                x, pre_mix, lambda h, b=blk: b.attn.forward_draft(h, main_x, ctx_pos, ctx_valid, q_pos)
            )
            balance = balance + bal
        return hc_pre(x, pre_mix), balance

    def base_logits(self, backbone, h: torch.Tensor) -> torch.Tensor:
        hn = self.norm(h)
        with torch.autocast(device_type=hn.device.type, enabled=False):
            return F.linear(hn.float(), backbone.head.weight.float())


# ------------------------------------------------------------------ 推理 ----


class DraftCache:
    """草稿头每个 block 一份上下文环形缓存：最近 window 个主干位置的上下文 KV（已 RoPE）。"""

    def __init__(self, drafter: DSparkDrafter, batch: int, device, dtype) -> None:
        cfg = drafter.cfg
        W = cfg.window_size
        self.kv = [torch.zeros(batch, W, cfg.head_dim, device=device, dtype=dtype) for _ in drafter.blocks]
        self.pos = torch.full((batch, W), -1, device=device, dtype=torch.long)

    @torch.no_grad()
    def push(self, drafter: DSparkDrafter, target_hidden: torch.Tensor, start: int) -> None:
        """写入绝对位置 [start, start+S) 的主干目标层特征 [B,S,n_targets·d]。"""
        W = drafter.cfg.window_size
        S = target_hidden.size(1)
        keep = min(S, W)
        pos = torch.arange(start + S - keep, start + S, device=target_hidden.device)
        slot = torch.remainder(pos, W)
        main_x = drafter.context_features(target_hidden[:, S - keep :].to(drafter.main_proj.weight.dtype))
        for blk, buf in zip(drafter.blocks, self.kv):
            kv = blk.attn.project_window_kv(main_x, blk.attn.freqs_cis[pos])
            buf[:, slot] = kv.to(buf.dtype)
        self.pos[:, slot] = pos.unsqueeze(0).expand(self.pos.size(0), -1)


def _draft_attention_cached(attn: DSparkAttention, x: torch.Tensor, ring_kv, ring_pos, n: int) -> torch.Tensor:
    B, G, _ = x.shape
    W = ring_kv.size(1)
    freqs_q = attn.freqs_cis[n : n + G]
    _, q = attn.project_q(x, freqs_q)
    kv_blk = attn.project_window_kv(x, freqs_q)
    kv_all = torch.cat([ring_kv.to(kv_blk.dtype), kv_blk], dim=1)
    ok = (ring_pos >= max(0, n - W)) & (ring_pos <= n - 1)
    ctx_idx = torch.where(ok, torch.arange(W, device=x.device), -1)
    blk_idx = (W + torch.arange(G, device=x.device)).expand(B, G)
    idx = torch.cat([ctx_idx, blk_idx], dim=-1).unsqueeze(1).expand(B, G, W + G)
    o, _ = sparse_attention(q, kv_all, idx, attn.attn_sink, attn.softmax_scale)
    return attn.output(o, freqs_q)


def _scaled_probs(logits: torch.Tensor, temperature: float, top_k: Optional[int]) -> torch.Tensor:
    logits = logits.float()
    if temperature is None or temperature <= 0:
        return F.one_hot(logits.argmax(-1), logits.size(-1)).float()
    logits = logits / max(temperature, 1e-6)
    if top_k is not None and top_k > 0:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits = logits.masked_fill(logits < v[..., [-1]], float("-inf"))
    return torch.softmax(logits, dim=-1)


@torch.no_grad()
def draft_block(
    drafter: DSparkDrafter,
    backbone,
    anchor: torch.Tensor,
    n: int,
    dcache: DraftCache,
    temperature: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """anchor [B,1] 位于位置 n。返回 (草稿 token [B,γ]、草稿分布 [B,γ,V]、置信度原始 logit [B,γ])。"""
    cfg = drafter.cfg
    G = cfg.dspark_block_size
    ids = torch.full((anchor.size(0), G), int(cfg.dspark_noise_token_id), device=anchor.device, dtype=torch.long)
    ids[:, 0] = anchor[:, 0]
    x = backbone.embed(ids).unsqueeze(2).repeat(1, 1, cfg.hc_mult, 1)
    pre_mix = identity_pre_mix(x, cfg.hc_mult)
    for blk, ring in zip(drafter.blocks, dcache.kv):
        x, pre_mix, _ = blk.run(x, pre_mix, lambda h, b=blk, r=ring: _draft_attention_cached(b.attn, h, r, dcache.pos, n))
    h = hc_pre(x, pre_mix)
    base = drafter.base_logits(backbone, h)
    tokens, probs, embeds = [], [], []
    prev = anchor[:, 0]
    for k in range(G):
        bias, emb = drafter.markov_head(prev)
        p = _scaled_probs(base[:, k] + bias, temperature, None)
        nxt = p.argmax(-1) if temperature is None or temperature <= 0 else torch.multinomial(p, 1).squeeze(-1)
        tokens.append(nxt)
        probs.append(p)
        embeds.append(emb)
        prev = nxt
    conf_logits = drafter.confidence_head(h, torch.stack(embeds, dim=1))
    return torch.stack(tokens, dim=1), torch.stack(probs, dim=1), conf_logits


def schedule_prefix(conf: torch.Tensor, sps) -> int:
    """DSpark Algorithm 1（单请求）：按存活概率 a_j = Π c_i 依次加入，吞吐 τ·SPS(B) 不再上升即停。"""
    a = torch.cumprod(conf.float(), dim=0).tolist()
    batch, tau = 1, 1.0
    best, chosen = tau * sps(batch), 0
    for j, aj in enumerate(a, start=1):
        if aj <= 0:
            break
        batch += 1
        tau += aj
        theta = tau * sps(batch)
        if theta > best:
            best, chosen = theta, j
        else:
            break
    return chosen


def accept_or_resample(x: int, p_d: torch.Tensor, p_t: torch.Tensor, generator=None) -> Tuple[bool, int]:
    """无损验证：以 min(1, p_t/p_d) 接受；拒绝时从 norm(max(0, p_t − p_d)) 重采样。"""
    pd, pt = float(p_d[x]), float(p_t[x])
    u = float(torch.rand((), generator=generator))
    if pd <= 0 or u < min(1.0, pt / pd):
        return True, x
    residual = (p_t - p_d).clamp(min=0)
    if float(residual.sum()) <= 0:
        residual = p_t
    return False, int(torch.multinomial(residual / residual.sum(), 1, generator=generator))


def _snapshot(cache) -> list:
    snap = []
    for lc in cache.layers:
        snap.append(
            (
                lc.win_kv.clone(),
                lc.win_pos.clone(),
                None if lc.kv_state is None else lc.kv_state.clone(),
                None if lc.score_state is None else lc.score_state.clone(),
            )
        )
    return [snap, cache.pos]


def _restore(cache, snap) -> None:
    layers, pos = snap
    for lc, (wkv, wpos, kst, sst) in zip(cache.layers, layers):
        lc.win_kv.copy_(wkv)
        lc.win_pos.copy_(wpos)
        if kst is not None:
            lc.kv_state.copy_(kst)
            lc.score_state.copy_(sst)
    cache.pos = pos


def measure_sps(backbone, max_tokens: int, device, repeats: int = 3) -> Dict[int, float]:
    """实测主干一次处理 B 个 token 的步频（steps/s），B = 1..max_tokens。"""
    import time

    from .cache import InferenceCache, ced_prefill, decode_chunk

    table = {}
    ids = torch.randint(1, backbone.cfg.vocab_size, (1, 16), device=device)
    for b in range(1, max_tokens + 1):
        cost = []
        for _ in range(repeats):
            cache = InferenceCache.create(backbone, 1)
            ced_prefill(backbone, ids, cache)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            decode_chunk(backbone, ids[:, :b], cache)
            if device.type == "cuda":
                torch.cuda.synchronize()
            cost.append(time.perf_counter() - t0)
        table[b] = 1.0 / max(min(cost), 1e-9)
    return table


@torch.no_grad()
def speculative_stream(
    backbone,
    drafter: DSparkDrafter,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: Optional[int] = 50,
    eos_token_id: Optional[int] = None,
    sps: Optional[Dict[int, float]] = None,
    stats: Optional[dict] = None,
    generator=None,
    force_len: Optional[int] = None,
    recorder=None,
):
    """DSpark 投机解码（单请求）。逐个产出 token（int）；输出分布与普通解码一致。

    force_len 固定每轮验证长度（STS 校准收数据用）；recorder(conf_logits [γ], n_accepted, ell) 记录每轮。
    """
    from .cache import InferenceCache, ced_prefill, decode_chunk, sample

    if input_ids.size(0) != 1:
        raise ValueError("投机解码只支持 batch=1")
    backbone.eval()
    drafter.eval()
    cfg = backbone.cfg
    G = cfg.dspark_block_size
    max_len = cfg.max_seq_len
    sps_table = sps or {b: 1.0 for b in range(1, G + 2)}
    sps_fn = lambda b: sps_table.get(b, sps_table[max(sps_table)])  # noqa: E731
    use_cand = bool(getattr(backbone, "infer_candidate_pool", False))
    ids = input_ids[:, -max_len:]
    cache = InferenceCache.create(backbone, 1, max_len)
    targets: Dict[int, torch.Tensor] = {}
    logits = ced_prefill(backbone, ids, cache, use_candidates=use_cand, targets=targets)
    dcache = DraftCache(drafter, 1, ids.device, drafter.main_proj.weight.dtype)
    W = cfg.window_size
    r0 = max(0, cache.pos - W)
    dcache.push(drafter, torch.cat([targets[i] for i in cfg.dspark_target_layer_ids], dim=-1), r0)
    st = stats if stats is not None else {}
    st.setdefault("cycles", 0)
    st.setdefault("drafted", 0)
    st.setdefault("accepted", 0)

    produced = 0
    anchor = sample(logits, temperature, top_k)
    while True:
        tok = int(anchor[0, 0])
        yield tok
        produced += 1
        if produced >= max_new_tokens or (eos_token_id is not None and tok == int(eos_token_id)):
            return
        n = cache.pos
        if n + G + 1 > max_len:
            # 余量不够一整块：退回普通解码
            if n + 1 > max_len:
                return
            targets = {}
            logits = decode_chunk(backbone, anchor, cache, use_candidates=use_cand, targets=targets)[:, -1]
            dcache.push(drafter, torch.cat([targets[i] for i in cfg.dspark_target_layer_ids], dim=-1), n)
            anchor = sample(logits, temperature, top_k)
            continue
        draft, p_d, conf_logits = draft_block(drafter, backbone, anchor, n, dcache, temperature)
        conf = torch.sigmoid(conf_logits / drafter.sts_temps.to(conf_logits.device))
        ell = int(force_len) if force_len is not None else schedule_prefix(conf[0], sps_fn)
        st["cycles"] += 1
        st["drafted"] += ell
        snap = _snapshot(cache)
        targets = {}
        chunk = torch.cat([anchor, draft[:, :ell]], dim=1)
        vlogits = decode_chunk(backbone, chunk, cache, use_candidates=use_cand, targets=targets)
        out, rejected = [], False
        for k in range(ell):
            p_t = _scaled_probs(vlogits[0, k], temperature, top_k)
            x = int(draft[0, k])
            if temperature is None or temperature <= 0:
                ok, y = (x == int(p_t.argmax())), int(p_t.argmax())
            else:
                ok, y = accept_or_resample(x, p_d[0, k], p_t, generator)
            if ok:
                out.append(x)
            else:
                out.append(y)
                rejected = True
                break
        n_acc = len(out) - (1 if rejected else 0)
        st["accepted"] += n_acc
        if recorder is not None:
            recorder(conf_logits[0].float().cpu(), n_acc, ell)
        if rejected:
            # 回滚：恢复验证前的缓存，只重放 anchor 与已接受的草稿
            _restore(cache, snap)
            targets = {}
            decode_chunk(backbone, chunk[:, : 1 + n_acc], cache, use_candidates=use_cand, targets=targets)
            final = torch.tensor([[out[-1]]], device=ids.device)
        else:
            final = sample(vlogits[:, ell], temperature, top_k)
        dcache.push(drafter, torch.cat([targets[i] for i in cfg.dspark_target_layer_ids], dim=-1), n)
        for x in out[:n_acc]:
            yield x
            produced += 1
            if produced >= max_new_tokens or (eos_token_id is not None and x == int(eos_token_id)):
                return
        anchor = final


def sample_anchors(labels: torch.Tensor, doc_ids: Optional[torch.Tensor], gamma: int, n: int, generator=None):
    """每行随机挑最多 n 个 anchor：其后 γ 个预测目标都有效且同文档。返回 (row, pos) 两个 1-D 张量。"""
    B, T = labels.shape
    ok = labels != -100
    valid = torch.zeros_like(ok)
    if T >= gamma:
        win = ok.unfold(1, gamma, 1).all(-1)  # [B, T−γ+1]：labels[a..a+γ−1] 全有效
        valid[:, : win.size(1)] = win
        if doc_ids is not None:
            same = doc_ids.unfold(1, gamma, 1)
            valid[:, : win.size(1)] &= (same == same[..., :1]).all(-1)
    rows, poss = [], []
    for b in range(B):
        cand = valid[b].nonzero().flatten()
        if cand.numel() == 0:
            continue
        pick = cand[torch.randperm(cand.numel(), generator=generator, device="cpu")[:n].to(cand.device)]
        rows.append(torch.full_like(pick, b))
        poss.append(pick)
    if not rows:
        return labels.new_zeros(0), labels.new_zeros(0)
    return torch.cat(rows), torch.cat(poss)


def dspark_training_losses(
    drafter: DSparkDrafter,
    backbone,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    doc_ids: Optional[torch.Tensor] = None,
    *,
    n_anchors: int = 32,
    generator=None,
    loss_weights=(0.1, 0.9, 1.0),
) -> Dict[str, torch.Tensor]:
    """主干冻结：no-grad 得到目标层特征与目标分布；在随机 anchor 上组草稿块，按 DSpark 损失训练草稿头。"""
    cfg = drafter.cfg
    G, W = cfg.dspark_block_size, cfg.window_size
    with torch.no_grad():
        tgt_logits, _, extras = backbone(input_ids, doc_ids=doc_ids, return_target_hidden=True)
        target_hidden = extras["target_hidden"]
    rows, anchors = sample_anchors(labels, doc_ids, G, n_anchors, generator)
    if rows.numel() == 0:
        zero = drafter.main_proj.weight.sum() * 0.0
        return {"total": zero, "ce": zero, "tv": zero, "conf": zero, "accept": zero, "balance": zero}
    dev = input_ids.device
    main_all = drafter.context_features(target_hidden.to(drafter.main_proj.weight.dtype))
    ctx_pos = anchors.unsqueeze(1) - W + torch.arange(W, device=dev)
    ctx_valid = ctx_pos >= 0
    if doc_ids is not None:
        ctx_doc = doc_ids[rows.unsqueeze(1), ctx_pos.clamp(min=0)]
        ctx_valid = ctx_valid & (ctx_doc == doc_ids[rows, anchors].unsqueeze(1))
    main_x = main_all[rows.unsqueeze(1), ctx_pos.clamp(min=0)]
    q_pos = anchors.unsqueeze(1) + torch.arange(G, device=dev)
    block_ids = torch.full((rows.numel(), G), int(cfg.dspark_noise_token_id), device=dev, dtype=torch.long)
    block_ids[:, 0] = input_ids[rows, anchors]
    with torch.no_grad():
        x_tok = backbone.embed(block_ids)
    h, balance = drafter.run_blocks(x_tok, main_x, ctx_pos, ctx_valid, q_pos)
    base = drafter.base_logits(backbone, h)
    targets = labels[rows.unsqueeze(1), q_pos]  # 块内位置 k 预测 x_{a+k+1}
    prev = torch.cat([input_ids[rows, anchors].unsqueeze(1), targets[:, :-1]], dim=1)  # teacher forcing
    bias, m_embed = drafter.markov_head(prev)
    draft = base + bias
    conf_logits = drafter.confidence_head(h, m_embed)
    terms = dspark_loss_terms(draft, tgt_logits[rows.unsqueeze(1), q_pos], targets, conf_logits, position_weights(G, dev))
    a, b, c = loss_weights
    terms["balance"] = balance
    terms["total"] = a * terms["ce"] + b * terms["tv"] + c * terms["conf"]
    return terms
