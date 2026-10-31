"""MiniSeek 模型包：仅 MiniSeek-V（DeepSeek-V4.1-Flash 全架构）。

Dense / Tiny 完整快照见分支 `archive/dense-tiny`。
简单 MoE 对照线 MiniSeek-M0.1 仅 release / HF，不在 main。
"""

from typing import Any, Dict, Optional, Sequence

import torch.nn as nn

from .miniseek_v.config import FlashConfig, is_flash_arch

__all__ = [
    "FlashConfig",
    "count_parameters",
    "build_model",
    "config_from_dict",
    "load_model_from_blob",
]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def config_from_dict(d: Dict[str, Any]) -> FlashConfig:
    """按 checkpoint / config.json 里的 arch 还原配置（仅 miniseek-v）。"""
    arch = (d or {}).get("arch")
    if d and not is_flash_arch(arch):
        raise ValueError(
            f"不支持的 arch={arch!r}；main 只保留 MiniSeek-V（arch=miniseek-v）。"
            " Dense / Tiny 见分支 archive/dense-tiny。"
        )
    return FlashConfig.from_dict(d or {})


def build_model(cfg: FlashConfig, token_map: Optional[Sequence[int]] = None):
    """构建 MiniSeek-V Transformer；需要 tokenizer 压缩表（缺省为恒等映射）。"""
    if not isinstance(cfg, FlashConfig):
        raise TypeError(f"期望 FlashConfig，得到 {type(cfg).__name__}")
    from .miniseek_v.model import Transformer

    return Transformer(cfg, token_map=token_map)


def load_model_from_blob(blob: Dict[str, Any]):
    """训练 checkpoint（{"model", "config", ...}）→ (model, cfg)，留在 CPU、未切 eval。

    兼容 torch.compile 留下的 `_orig_mod.` 前缀；SFT 阶段的权重推理时启用候选池。
    """
    cfg = config_from_dict(blob.get("config") or {})
    model = build_model(cfg)
    prefix = "_orig_mod."
    state = {(k[len(prefix):] if k.startswith(prefix) else k): v for k, v in blob["model"].items()}
    model.load_state_dict(state, strict=True)
    model.infer_candidate_pool = blob.get("stage") == "sft"
    return model, cfg
