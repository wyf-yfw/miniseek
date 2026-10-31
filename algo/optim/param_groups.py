"""M 参数分组（DeepSeek-V4.1-Flash §2.5、§4.2.2）与组合优化器。

- Muon：所有线性层权重矩阵（注意力、压缩器、indexer、路由器、专家、mHC fn、Engram wkv）
- AdamW：RMSNorm 权重（wd 0.1）；attn sink、mHC base/scale、Engram q/k 权重（wd 0）
- Sinkhorn 动量：embedding、head、Engram 表（Engram 表 lr ×5）、DSpark Markov 表
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW

from .muon import Muon
from .sinkhorn import SinkhornMomentum

SINKHORN_SUFFIXES = ("engram.embed.weight", "markov_head.embed.weight", "markov_head.head.weight")
NOWD_SUFFIXES = ("attn_sink", "_base", "_scale", "q_weight", "k_weight")


def classify_v41_param(name: str, p: torch.Tensor) -> str:
    """返回 'sinkhorn' | 'adamw_wd' | 'adamw_nowd' | 'muon'。"""
    if name in ("embed.weight", "head.weight") or name.endswith(SINKHORN_SUFFIXES):
        return "sinkhorn"
    if name.endswith("norm.weight"):
        return "adamw_wd"
    if name.endswith(NOWD_SUFFIXES) or p.dim() != 2 or min(p.shape) == 1:
        return "adamw_nowd"
    return "muon"


def muon_row_splits(name: str, p: torch.Tensor, cfg) -> int:
    """逻辑独立矩阵个数：Q 按头（head-wise Muon）、wo_a 按组、Engram wkv 按 4 key + 1 value。"""
    if name.endswith("indexer.wq_b.weight"):
        return cfg.index_n_heads
    if name.endswith("attn.wq_b.weight"):
        return cfg.n_heads
    if name.endswith("attn.wo_a.weight"):
        return cfg.o_groups
    if name.endswith("engram.wkv.weight"):
        return cfg.hc_mult + 1
    return 1


class CompositeOptimizer:
    """把多个优化器当一个用：param_groups 合并（便于逐步设学习率），step / zero_grad / 存取一起做。"""

    def __init__(self, optimizers: Dict[str, torch.optim.Optimizer]) -> None:
        self.optimizers = {k: v for k, v in optimizers.items() if v is not None}

    @property
    def param_groups(self) -> List[dict]:
        groups: List[dict] = []
        for opt in self.optimizers.values():
            groups.extend(opt.param_groups)
        return groups

    def step(self, closure=None):
        for opt in self.optimizers.values():
            opt.step()

    def zero_grad(self, set_to_none: bool = True) -> None:
        for opt in self.optimizers.values():
            opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict:
        return {k: opt.state_dict() for k, opt in self.optimizers.items()}

    def load_state_dict(self, state: dict) -> None:
        for k, opt in self.optimizers.items():
            if k in state:
                opt.load_state_dict(state[k])


def named_trainable(model: nn.Module, skip_prefixes: Tuple[str, ...] = ()) -> Iterable[Tuple[str, nn.Parameter]]:
    for name, p in model.named_parameters():
        if p.requires_grad and not name.startswith(skip_prefixes):
            yield name, p


def build_miniseek_v_optimizer(
    model: nn.Module,
    lr: float,
    cfg=None,
    *,
    weight_decay: float = 0.1,
    engram_lr_mult: float = 5.0,
    momentum: float = 0.95,
    fused_adamw: bool = False,
    skip_prefixes: Tuple[str, ...] = (),
) -> CompositeOptimizer:
    root = getattr(model, "_orig_mod", model)
    cfg = cfg or root.cfg
    muon: List[nn.Parameter] = []
    adam_wd: List[nn.Parameter] = []
    adam_nowd: List[nn.Parameter] = []
    sink: List[nn.Parameter] = []
    sink_engram: List[nn.Parameter] = []
    for name, p in named_trainable(root, skip_prefixes):
        kind = classify_v41_param(name, p)
        if kind == "muon":
            p._muon_splits = muon_row_splits(name, p, cfg)
            muon.append(p)
        elif kind == "adamw_wd":
            adam_wd.append(p)
        elif kind == "adamw_nowd":
            adam_nowd.append(p)
        elif name.endswith("engram.embed.weight"):
            sink_engram.append(p)
        else:
            sink.append(p)

    optimizers: Dict[str, torch.optim.Optimizer] = {}
    if muon:
        optimizers["muon"] = Muon(muon, lr=lr, momentum=momentum, weight_decay=weight_decay)
    adam_groups = [g for g in ({"params": adam_wd, "weight_decay": weight_decay}, {"params": adam_nowd, "weight_decay": 0.0}) if g["params"]]
    if adam_groups:
        kwargs = dict(lr=lr, betas=(0.9, 0.95), eps=1e-20)
        if fused_adamw:
            kwargs["fused"] = True
        optimizers["adamw"] = AdamW(adam_groups, **kwargs)
    sink_groups = [g for g in ({"params": sink, "lr_mult": 1.0}, {"params": sink_engram, "lr_mult": engram_lr_mult}) if g["params"]]
    if sink_groups:
        optimizers["sinkhorn"] = SinkhornMomentum(sink_groups, lr=lr, beta=momentum)
    return CompositeOptimizer(optimizers)
