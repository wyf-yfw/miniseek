"""v41（MiniSeek-M）训练接入：train_pretrain / train_sft 共用。

- 构建：FlashConfig + tokenizer 压缩表（Engram）+ Transformer
- 组批：打包时文档对齐到压缩比的公倍数，并带逐 token 文档编号
- 损失：ce + aux_coef·balance + indexer_kl；SFT 可模拟 Decoder SWA Bounded Replay
- 优化器：Muon / AdamW / Sinkhorn 动量（algo.optim）
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, Optional, Tuple

import torch

from algo.model import build_model
from algo.model.miniseek_m import FlashConfig, bounded_replay_floor
from algo.model.miniseek_m.engram import build_compressed_token_map
from algo.optim import build_miniseek_m_optimizer

MM_DEFAULT_LR = 3e-4
MM_DEFAULT_SEQ_LEN = 1024


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, "_orig_mod", model)


def clean_state_dict(sd: Dict[str, Any]) -> Dict[str, Any]:
    """去掉 torch.compile 包装留下的 `_orig_mod.` 前缀。"""
    prefix = "_orig_mod."
    return {(k[len(prefix):] if k.startswith(prefix) else k): v for k, v in sd.items()}


def state_for_ckpt(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """浮点张量存 fp32，整型 buffer（如 Engram 压缩表）原样保存。"""
    out = {}
    for k, v in unwrap(model).state_dict().items():
        v = v.detach().cpu()
        out[k] = v.float() if v.is_floating_point() else v
    return out


def build_miniseek_m(tok, vocab: int, pad_id: int, max_seq_len: int, device: torch.device):
    """M0.1；环境变量 MINISEEK_M_PRESET=tiny 时换成同层模式的极小宽度（仅冒烟测试用）。"""
    token_map, compressed = build_compressed_token_map(tok)
    preset = (os.environ.get("MINISEEK_M_PRESET") or "").strip().lower()
    base = (FlashConfig.tiny() if preset == "tiny" else FlashConfig()).to_dict()
    base.update(
        vocab_size=int(vocab),
        max_seq_len=int(max_seq_len),
        engram_pad_id=int(pad_id),
        engram_compressed_vocab_size=int(compressed),
        dspark_noise_token_id=int(pad_id),
    )
    cfg = FlashConfig.from_dict(base)
    model = build_model(cfg, token_map=token_map).to(device)
    return cfg, model


def doc_align(cfg: FlashConfig) -> int:
    align = 1
    for r in cfg.compress_ratios:
        if r:
            align = align * r // math.gcd(align, r)
    return align


def describe(cfg: FlashConfig, model: torch.nn.Module) -> str:
    modes = cfg.layer_modes()
    parts = []
    for i, mode in enumerate(modes):
        tag = mode if mode == "SWA" else f"{mode}(m={cfg.compress_ratios[i]})"
        if i in cfg.engram_layer_ids:
            tag += "+Engram"
        parts.append(f"L{i} {tag}")
    half = cfg.n_encoder_layers
    n = sum(p.numel() for p in unwrap(model).parameters())
    return (
        f"[v41] {cfg.release_name} encoder[{' | '.join(parts[:half])}] "
        f"decoder[{' | '.join(parts[half:])}] kv_sources={list(cfg.kv_source_layers)} "
        f"index_sources={list(cfg.index_source_layers)} experts=1+{cfg.n_routed_experts} "
        f"top{cfg.n_activated_experts} mhc={cfg.hc_mult} params={n:,}"
    )


def unpack_batch(batch, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    input_ids, labels = batch[0].to(device), batch[1].to(device)
    doc_ids = batch[2].to(device) if len(batch) > 2 else None
    return input_ids, labels, doc_ids


def forward_loss(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    doc_ids: Optional[torch.Tensor],
    *,
    aux_coef: float,
    use_candidate_pool: bool = False,
    bounded_replay: bool = False,
    window: int = 128,
):
    swa_floor = bounded_replay_floor(labels, doc_ids, window) if bounded_replay else None
    _logits, ce, extras = model(
        input_ids, labels=labels, doc_ids=doc_ids, swa_floor=swa_floor, use_candidate_pool=use_candidate_pool
    )
    total = ce + float(aux_coef) * extras["balance"] + extras["indexer_kl"]
    return total, ce, extras


def build_optimizer(model: torch.nn.Module, lr: float, *, fused: bool):
    return build_miniseek_m_optimizer(model, lr=lr, fused_adamw=fused)


def log_line(epoch: int, step: int, loss: float, ce: torch.Tensor, extras: Dict[str, torch.Tensor], lr: float) -> str:
    ce_v = float(ce.item())
    return (
        f"epoch={epoch} step={step} loss={loss:.4f} ce={ce_v:.4f} "
        f"aux={float(extras['balance'].item()):.4f} kl={float(extras['indexer_kl'].item()):.4f} "
        f"ppl={math.exp(min(ce_v, 20)):.2f} lr={lr:.2e}"
    )
