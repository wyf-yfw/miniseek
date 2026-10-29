"""注意力：每层一个 128 滑窗分支；压缩层再叠加 CSA2（Full / Reindex / Reuse）。

逐算子对应官方 inference/model.py 的 Attention / Compressor / Indexer / select_candidate_blocks，
核心注意力对应 kernel.py 的 sparse_attn：按索引取条目（−1 为空），softmax 分母额外加 exp(sink)。
训练路径是全量 forward（官方的 prefill 分支），另加打包训练用的文档掩码与 Bounded Replay 截断。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import FlashConfig
from .rope import layer_freqs, rotate_tail


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        return (self.weight.float() * x).to(dtype)


@dataclass
class AttnState:
    """一次 forward 内跨层共享的东西（官方 SharedAttentionRuntime）。来源层写，后续层读。"""

    compress_kv: Optional[torch.Tensor] = None  # [B,Tc,head_dim]，已 RoPE
    compress_doc: Optional[torch.Tensor] = None  # [B,Tc]，条目首 token 所属文档
    index_k: Optional[torch.Tensor] = None  # [B,Tc,index_head_dim]，已 RoPE
    topk: Optional[torch.Tensor] = None  # [B,T,K]，压缩条目下标，−1 为空
    candidates: Optional[torch.Tensor] = None  # [B,T,Tc] bool
    use_candidates: bool = False
    kl: List[torch.Tensor] = field(default_factory=list)


def window_indices(seqlen: int, window: int, device=None) -> torch.Tensor:
    """官方 get_window_topk_idxs(start_pos=0)：[T, min(T, window)]，−1 表示序列开始之前。"""
    end = torch.arange(seqlen, device=device).unsqueeze(1)
    idxs = (end - window + 1).clamp(min=0) + torch.arange(min(seqlen, window), device=device)
    return torch.where(idxs > end, -1, idxs)


def sparse_attention(
    q: torch.Tensor, kv: torch.Tensor, idx: torch.Tensor, sink: torch.Tensor, scale: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """q [B,T,H,D]，kv [B,N,D]（同时作 K 与 V），idx [B,T,K]（−1 为空），sink [H]。

    返回 (o [B,T,H,D], probs [B,T,H,K] fp32)。全为空的行输出 0，与官方 kernel 一致。
    """
    B, T, H, D = q.shape
    K = idx.size(-1)
    valid = idx >= 0
    gathered = kv.gather(1, idx.clamp(min=0).reshape(B, T * K, 1).expand(-1, -1, D)).reshape(B, T, K, D)
    logits = torch.einsum("bthd,btkd->bthk", q, gathered).float() * scale
    logits = logits.masked_fill(~valid.unsqueeze(2), float("-inf"))
    logits = torch.cat([logits, sink.float().view(1, 1, H, 1).expand(B, T, H, 1)], dim=-1)
    probs = torch.softmax(logits, dim=-1)[..., :K]
    o = torch.einsum("bthk,btkd->bthd", probs.to(gathered.dtype), gathered)
    return o, probs


def select_candidate_blocks(
    logits: torch.Tensor, compress_lens: torch.Tensor, topk_blocks: int, block_size: int
) -> torch.Tensor:
    """官方 select_candidate_blocks：按块取最大分，钉住含最新条目的块，留前 topk_blocks 块。"""
    width = logits.size(-1)
    scores = F.pad(logits, (0, -width % block_size), value=float("-inf"))
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = scores.size(-1)
    last = (compress_lens - 1) // block_size
    scores = scores.masked_fill(torch.arange(num_blocks, device=logits.device) == last, float("inf"))
    top = scores.topk(min(topk_blocks, num_blocks), dim=-1)
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, top.indices, top.values > float("-inf"))
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


class Compressor(nn.Module):
    """每 ratio 个 token 不重叠地压成一个 KV latent（RoPE 之前）。ratio=1 是直接投影。"""

    def __init__(self, cfg: FlashConfig, layer_id: int) -> None:
        super().__init__()
        self.ratio = cfg.compress_ratios[layer_id]
        self.wkv = nn.Linear(cfg.dim, cfg.head_dim, bias=False)
        self.wgate = nn.Linear(cfg.dim, cfg.head_dim, bias=False) if self.ratio > 1 else None
        self.norm = RMSNorm(cfg.head_dim, cfg.norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.ratio == 1:
            return self.norm(self.wkv(x))
        dtype = x.dtype
        n = x.size(1) // self.ratio
        # 池化在 fp32 里做（官方把这两层权重提成 fp32）
        with torch.autocast(device_type=x.device.type, enabled=False):
            xf = x[:, : n * self.ratio].float()
            kv = F.linear(xf, self.wkv.weight.float()).unflatten(1, (n, self.ratio))
            score = F.linear(xf, self.wgate.weight.float()).unflatten(1, (n, self.ratio))
            kv = (kv * score.softmax(dim=2)).sum(dim=2)
        return self.norm(kv.to(dtype))


class Indexer(nn.Module):
    """用 indexer Q 给压缩条目打分：Σ_h w_h · ReLU(q_h · k)。只有 kv source 层拥有 wk。"""

    def __init__(self, cfg: FlashConfig, layer_id: int) -> None:
        super().__init__()
        self.owns_k = layer_id in cfg.kv_source_layers
        self.n_heads = cfg.index_n_heads
        self.head_dim = cfg.index_head_dim
        self.rope_dim = cfg.rope_head_dim
        self.topk = cfg.index_topk
        self.is_candidate_source = layer_id == cfg.candidate_source_layer
        self.uses_candidates = 0 <= cfg.candidate_source_layer < layer_id
        self.candidate_topk_blocks = cfg.candidate_topk_blocks
        self.candidate_block_size = cfg.candidate_block_size
        self.wq_b = nn.Linear(cfg.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.weights_proj = nn.Linear(cfg.dim, self.n_heads, bias=False)
        if self.owns_k:
            self.wk = nn.Linear(cfg.head_dim, self.head_dim, bias=False)
            self.k_norm = RMSNorm(self.head_dim, cfg.norm_eps)

    def make_keys(self, latent: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        """由 RoPE 之前的压缩 latent 投影出 indexer K。"""
        return rotate_tail(self.k_norm(self.wk(latent)), freqs, self.rope_dim)

    def scores(self, x: torch.Tensor, qr: torch.Tensor, index_k: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        q = self.wq_b(qr).unflatten(-1, (self.n_heads, self.head_dim))
        q = rotate_tail(q, freqs, self.rope_dim)
        w = self.weights_proj(x) * (self.head_dim**-0.5 * self.n_heads**-0.5)
        s = torch.einsum("bthd,bsd->bths", q, index_k).relu()
        return (s * w.unsqueeze(-1)).sum(dim=2)


def mask_window(
    win: torch.Tensor, doc_ids: Optional[torch.Tensor], swa_floor: Optional[torch.Tensor]
) -> torch.Tensor:
    """win [B,T,W]：跨文档或低于 swa_floor 的滑窗位置置为 −1。"""
    valid = win >= 0
    if doc_ids is not None:
        key_doc = doc_ids.gather(1, win.clamp(min=0).flatten(1)).view_as(win)
        valid = valid & (key_doc == doc_ids.unsqueeze(-1))
    if swa_floor is not None:
        valid = valid & (win >= swa_floor.unsqueeze(-1))
    return torch.where(valid, win, -1)


def indexer_kl(compress_probs: torch.Tensor, sel_scores: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """V3.2 DSA：KL(主注意力在选中条目上的分布 ‖ softmax(indexer 选中分数))，对有效 token 取平均。

    compress_probs [B,T,H,K]（主注意力对选中压缩条目的概率），sel_scores [B,T,K]，valid [B,T,K]。
    """
    ok = valid & torch.isfinite(sel_scores)
    with torch.no_grad():
        target = compress_probs.sum(dim=2).masked_fill(~ok, 0.0)
        denom = target.sum(-1, keepdim=True)
        rows = denom.squeeze(-1) > 0
        target = target / denom.clamp_min(1e-30)
        log_target = torch.log(target.clamp_min(1e-30))
    logp = torch.log_softmax(sel_scores.float().masked_fill(~ok, -1e30), dim=-1)
    kl = (target * (log_target - logp)).masked_fill(~ok, 0.0).sum(-1)
    if not bool(rows.any()):
        return sel_scores.float().masked_fill(~ok, 0.0).sum() * 0.0
    return kl[rows].mean()


class Attention(nn.Module):
    def __init__(self, cfg: FlashConfig, layer_id: int) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.rope_dim = cfg.rope_head_dim
        self.n_groups = cfg.o_groups
        self.o_lora_rank = cfg.o_lora_rank
        self.window_size = cfg.window_size
        self.compress_ratio = cfg.compress_ratios[layer_id]
        self.softmax_scale = self.head_dim**-0.5

        self.attn_sink = nn.Parameter(torch.zeros(self.n_heads))
        self.wq_a = nn.Linear(cfg.dim, cfg.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(cfg.q_lora_rank, cfg.norm_eps)
        self.wq_b = nn.Linear(cfg.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.wkv = nn.Linear(cfg.dim, self.head_dim, bias=False)
        self.kv_norm = RMSNorm(self.head_dim, cfg.norm_eps)
        # 块对角：每组只投影本组的头；权重布局与官方 wo_a 相同 [g*o_lora, H*hd/g]
        self.wo_a = nn.Linear(self.n_heads * self.head_dim // self.n_groups, self.n_groups * self.o_lora_rank, bias=False)
        self.wo_b = nn.Linear(self.n_groups * self.o_lora_rank, cfg.dim, bias=False)

        self.compressor = Compressor(cfg, layer_id) if layer_id in cfg.kv_source_layers else None
        self.indexer = Indexer(cfg, layer_id) if layer_id in cfg.index_source_layers else None
        self.register_buffer("freqs_cis", layer_freqs(cfg, layer_id, cfg.max_seq_len), persistent=False)

    @property
    def mode(self) -> str:
        if self.compress_ratio == 0:
            return "SWA"
        if self.compressor is not None:
            return "Full"
        return "Reindex" if self.indexer is not None else "Reuse"

    def project_q(self, x: torch.Tensor, freqs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).unflatten(-1, (self.n_heads, self.head_dim))
        return qr, rotate_tail(q, freqs, self.rope_dim)

    def project_window_kv(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        return rotate_tail(self.kv_norm(self.wkv(x)), freqs, self.rope_dim)

    def output(self, o: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        B, T = o.shape[:2]
        o = rotate_tail(o, freqs, self.rope_dim, inverse=True)
        o = o.reshape(B, T, self.n_groups, -1)
        wo_a = self.wo_a.weight.view(self.n_groups, self.o_lora_rank, -1)
        o = torch.einsum("btgd,grd->btgr", o, wo_a)
        return self.wo_b(o.flatten(2))

    def forward(
        self,
        x: torch.Tensor,
        state: AttnState,
        doc_ids: Optional[torch.Tensor] = None,
        swa_floor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        freqs = self.freqs_cis[:T]
        qr, q = self.project_q(x, freqs)
        kv = self.project_window_kv(x, freqs)
        win = window_indices(T, self.window_size, x.device).unsqueeze(0).expand(B, -1, -1)
        win = mask_window(win, doc_ids, swa_floor)

        kl_ctx = None
        if self.compress_ratio == 0:
            kv_all, idx_all = kv, win
        else:
            ckv, topk, kl_ctx = self._compressed(x, qr, state, doc_ids, freqs)
            kv_all = torch.cat([kv, ckv.to(kv.dtype)], dim=1)
            idx_all = torch.cat([win, torch.where(topk >= 0, topk + T, -1)], dim=-1)

        o, probs = sparse_attention(q, kv_all, idx_all, self.attn_sink, self.softmax_scale)
        if kl_ctx is not None:
            sel, valid = kl_ctx
            state.kl.append(indexer_kl(probs[..., win.size(-1):], sel, valid))
        return self.output(o, freqs)

    def _compressed(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        state: AttnState,
        doc_ids: Optional[torch.Tensor],
        freqs: torch.Tensor,
    ):
        B, T, _ = x.shape
        ratio = self.compress_ratio
        n = T // ratio
        if self.compressor is not None:
            freqs_c = self.freqs_cis[: n * ratio : ratio]  # 第 j 个条目代表位置 j*ratio
            latent = self.compressor(x)
            if self.indexer is not None and self.indexer.owns_k:
                state.index_k = self.indexer.make_keys(latent.detach(), freqs_c)
            state.compress_kv = rotate_tail(latent, freqs_c, self.rope_dim)
            state.compress_doc = None if doc_ids is None else doc_ids[:, : n * ratio : ratio]
        ckv = state.compress_kv
        if self.indexer is None:
            return ckv, state.topk, None

        # 可见：query 已走过该组最后一个 token，且条目与 query 同文档
        lens = torch.arange(1, T + 1, device=x.device) // ratio
        visible = (torch.arange(n, device=x.device).unsqueeze(0) < lens.unsqueeze(1)).unsqueeze(0)
        if doc_ids is not None:
            visible = visible & (doc_ids.unsqueeze(-1) == state.compress_doc.unsqueeze(1))
        else:
            visible = visible.expand(B, -1, -1)

        # indexer 的输入全部 detach：它只吃 KL 的梯度
        score = self.indexer.scores(x.detach(), qr.detach(), state.index_k, freqs)
        score = score.masked_fill(~visible, float("-inf"))
        if state.use_candidates and self.indexer.is_candidate_source:
            state.candidates = select_candidate_blocks(
                score.detach(), lens.unsqueeze(-1), self.indexer.candidate_topk_blocks, self.indexer.candidate_block_size
            )
        elif state.use_candidates and self.indexer.uses_candidates and state.candidates is not None:
            score = score.masked_fill(~state.candidates, float("-inf"))

        k = min(self.indexer.topk, n)
        idx = score.detach().topk(k, dim=-1).indices.sort(dim=-1).values
        valid = visible.gather(-1, idx)
        topk = torch.where(valid, idx, -1)
        state.topk = topk
        return ckv, topk, (score.gather(-1, idx), valid)
