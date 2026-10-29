"""MiniSeek-F 主干：embedding → 复制成 hc 份 → 8 个 Block（编码 0–3 / 解码 4–7）→ 合成 → head。

对应官方 inference/model.py 的 Block / Transformer。CED 体现在 KV 来源层的安排上：解码器只有
第一个解码器层（Full）有压缩器，它的输入就是编码器输出 H_{L/2}，其余解码器层复用这份全局 KV；
每层的滑窗 KV 仍由本层隐状态生成。训练 forward 返回 (logits, ce, extras)。
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import Attention, AttnState, RMSNorm
from .ckpt import autocast_dtype, checkpoint_call
from .config import FlashConfig
from .engram import Engram, EngramLayout, NgramHasher
from .mhc import hc_mixes, hc_post, hc_pre, identity_pre_mix
from .moe import MoE


class Block(nn.Module):
    """注意力与 MoE 两个子层都夹在 hc_pre / hc_post 之间。

    每个子层由自己的 hc_mixes 算出 (pre, post, comb)：post、comb 当场用，pre 交给下一个子层
    （Single-Pass mHC：注意力用上一层 MoE 给的 pre，MoE 用本层注意力给的 pre）。
    """

    def __init__(
        self,
        cfg: FlashConfig,
        layer_id: int,
        layout: Optional[EngramLayout] = None,
        *,
        n_routed: Optional[int] = None,
        n_activated: Optional[int] = None,
        attention_cls=Attention,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.norm_eps = cfg.norm_eps
        self.hc_mult = cfg.hc_mult
        self.hc_sinkhorn_iters = cfg.hc_sinkhorn_iters
        self.hc_eps = cfg.hc_eps
        self.attn = attention_cls(cfg, layer_id)
        self.ffn = MoE(cfg, n_routed, n_activated)
        self.engram = Engram(cfg, layer_id, layout) if layout is not None and layer_id in layout.layer_ids else None
        self.attn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.ffn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        mix = (2 + cfg.hc_mult) * cfg.hc_mult
        self.hc_attn_fn = nn.Parameter(torch.empty(mix, cfg.hc_mult * cfg.dim))
        self.hc_ffn_fn = nn.Parameter(torch.empty(mix, cfg.hc_mult * cfg.dim))
        self.hc_attn_base = nn.Parameter(torch.zeros(mix))
        self.hc_ffn_base = nn.Parameter(torch.zeros(mix))
        self.hc_attn_scale = nn.Parameter(torch.full((3,), 0.01))
        self.hc_ffn_scale = nn.Parameter(torch.full((3,), 0.01))
        nn.init.normal_(self.hc_attn_fn, std=0.02)
        nn.init.normal_(self.hc_ffn_fn, std=0.02)

    def mixes(self, x: torch.Tensor, which: str):
        fn, scale, base = (
            (self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
            if which == "attn"
            else (self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        )
        return hc_mixes(x, fn, scale, base, self.norm_eps, self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps)

    def forward(
        self,
        x: torch.Tensor,
        pre_mix: torch.Tensor,
        state: AttnState,
        doc_ids: Optional[torch.Tensor] = None,
        swa_floor: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = x
        attn_pre, attn_post, attn_comb = self.mixes(x, "attn")
        h = self.attn(self.attn_norm(hc_pre(x, pre_mix)), state, doc_ids=doc_ids, swa_floor=swa_floor)
        x = hc_post(h, residual, attn_post, attn_comb)

        residual = x
        ffn_pre, ffn_post, ffn_comb = self.mixes(x, "ffn")
        h, balance = self.ffn(self.ffn_norm(hc_pre(x, attn_pre)))
        x = hc_post(h, residual, ffn_post, ffn_comb)
        return x, ffn_pre, balance


class Transformer(nn.Module):
    def __init__(self, cfg: Optional[FlashConfig] = None, token_map: Optional[Sequence[int]] = None) -> None:
        super().__init__()
        self.cfg = cfg = cfg or FlashConfig()
        self.act_ckpt = False
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.engram_layout = EngramLayout.from_config(cfg)
        self.hasher = NgramHasher(cfg, self.engram_layout, token_map) if self.engram_layout is not None else None
        self.layers = nn.ModuleList([Block(cfg, i, self.engram_layout) for i in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self._init_weights()

    @torch.no_grad()
    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, std=0.02)
        out_std = 0.02 / math.sqrt(2 * self.cfg.n_layers)
        for layer in self.layers:
            nn.init.normal_(layer.attn.wo_b.weight, std=out_std)
            for expert in list(layer.ffn.experts) + [layer.ffn.shared_experts]:
                nn.init.normal_(expert.w2.weight, std=out_std)
            if layer.engram is not None:
                layer.engram.reset_parameters()

    # ------------------------------------------------------------ forward ----
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        *,
        doc_ids: Optional[torch.Tensor] = None,
        swa_floor: Optional[torch.Tensor] = None,
        use_candidate_pool: bool = False,
        return_target_hidden: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        cfg = self.cfg
        _, T = input_ids.shape
        if T > cfg.max_seq_len:
            raise ValueError(f"seq_len {T} > max_seq_len {cfg.max_seq_len}")
        h = self.embed(input_ids)
        dtype = autocast_dtype(h.device)
        if dtype is not None:
            h = h.to(dtype)
        hashes = self.hasher(input_ids, doc_ids) if self.hasher is not None else None
        x = h.unsqueeze(2).repeat(1, 1, cfg.hc_mult, 1)
        pre_mix = identity_pre_mix(x, cfg.hc_mult)
        state = AttnState(use_candidates=use_candidate_pool)
        balance = torch.zeros((), device=h.device)
        targets: List[torch.Tensor] = []
        for i, layer in enumerate(self.layers):
            if layer.engram is not None:
                x = layer.engram(x, hashes[:, :, layer.engram.layer_hash_index])
            if return_target_hidden and i in cfg.dspark_target_layer_ids:
                targets.append(x.float().mean(dim=2))
            floor = swa_floor if cfg.is_decoder(i) else None
            state = state.fork()
            if self.act_ckpt and self.training:
                x, pre_mix, bal = checkpoint_call(layer, x, pre_mix, state, doc_ids, floor)
            else:
                x, pre_mix, bal = layer(x, pre_mix, state, doc_ids, floor)
            balance = balance + bal
        h = self.norm(hc_pre(x, pre_mix))
        with torch.autocast(device_type=h.device.type, enabled=False):
            logits = F.linear(h.float(), self.head.weight.float())

        ce = None
        if labels is not None:
            ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100)
        kl = torch.stack(state.kl).mean() if state.kl else torch.zeros((), device=h.device)
        extras: Dict[str, torch.Tensor] = {"balance": balance, "indexer_kl": kl}
        if return_target_hidden:
            extras["target_hidden"] = torch.cat(targets, dim=-1)
        return logits, ce, extras

    # -------------------------------------------------------- MoE 负载 ----
    @torch.no_grad()
    def update_expert_bias(self) -> None:
        for layer in self.layers:
            layer.ffn.update_expert_bias(self.cfg.expert_bias_speed)

    def begin_expert_usage(self) -> None:
        for layer in self.layers:
            layer.ffn._record_usage = True
            layer.ffn._usage_counts = None

    def end_expert_usage(self) -> torch.Tensor:
        total = torch.zeros(self.cfg.n_routed_experts, dtype=torch.long)
        for layer in self.layers:
            layer.ffn._record_usage = False
            counts, layer.ffn._usage_counts = layer.ffn._usage_counts, None
            if counts is not None:
                total += counts.detach().cpu().long()
        return total

    @torch.no_grad()
    def probe_expert_load(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, int]:
        was_training = self.training
        self.eval()
        self.begin_expert_usage()
        try:
            self.forward(input_ids)
        finally:
            counts = self.end_expert_usage()
            if was_training:
                self.train()
        return counts, int(input_ids.numel())

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def step_expert_bias(model: nn.Module) -> None:
    """优化器 step 之后调用；torch.compile 包装过的模型走原始模块。"""
    root = getattr(model, "_orig_mod", model)
    fn = getattr(root, "update_expert_bias", None)
    if fn is not None:
        fn()


def bounded_replay_floor(labels: torch.Tensor, doc_ids: Optional[torch.Tensor], window: int) -> torch.Tensor:
    """SFT 里模拟 Decoder SWA Bounded Replay：每个回答起点 P 之前的 window 个位置看不到更早的滑窗 key。

    labels 已右移（labels[t] 是位置 t 要预测的 token），所以首个有效 label 的位置是 prompt 最后一个 token，
    P = 该位置 + 1。返回 [B,T]：位置 t 的滑窗下界（默认 0）。
    """
    B, T = labels.shape
    valid = labels != -100
    prev = torch.zeros_like(valid)
    prev[:, 1:] = valid[:, :-1]
    if doc_ids is not None:
        same = torch.ones_like(valid)
        same[:, 1:] = doc_ids[:, 1:] == doc_ids[:, :-1]
        prev = prev & same
    starts = (valid & ~prev).nonzero().tolist()
    floor = torch.zeros(B, T, dtype=torch.long, device=labels.device)
    for b, t in starts:
        end = t + 1
        s = max(end - window, 0)
        floor[b, s:end] = torch.clamp(floor[b, s:end], min=s)
    return floor
