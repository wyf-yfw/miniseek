"""DSpark 草稿头：半自回归起草 + 置信度调度验证（arXiv:2607.05147；V4.1 §2.4.3）。

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
