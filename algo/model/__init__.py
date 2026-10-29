"""MiniSeek 模型包。

- Dense（MiniSeek-D*）：`from algo.model.dense import DenseLM`
- DeepSeek-M 架构（MiniSeek-M*，arch="miniseek-m"）：`from algo.model.miniseek_m import Transformer`
"""

from typing import Any, Dict, Optional, Sequence, Union

from .config import ModelConfig
from .dense import DenseLM, count_parameters
from .miniseek_m.config import FlashConfig, is_flash_arch

__all__ = [
    "ModelConfig",
    "FlashConfig",
    "DenseLM",
    "count_parameters",
    "build_model",
    "config_from_dict",
    "load_model_from_blob",
]

AnyConfig = Union[ModelConfig, FlashConfig]


def config_from_dict(d: Dict[str, Any]) -> AnyConfig:
    """按 checkpoint / config.json 里的 arch 还原配置。"""
    if is_flash_arch((d or {}).get("arch")):
        return FlashConfig.from_dict(d)
    return ModelConfig.from_dict(d or {})


def build_model(cfg: AnyConfig, token_map: Optional[Sequence[int]] = None):
    """按 cfg.arch 构建模型。M 线需要 tokenizer 压缩表（缺省为恒等映射）。"""
    if isinstance(cfg, FlashConfig):
        from .miniseek_m.model import Transformer

        return Transformer(cfg, token_map=token_map)
    if getattr(cfg, "arch", "dense") == "moe" or getattr(cfg, "use_moe", False):
        raise ValueError("旧 MoELM 已移除，请改用 arch=m（MiniSeek-M，DeepSeek-M 全架构）")
    return DenseLM(cfg)


def load_model_from_blob(blob: Dict[str, Any]):
    """训练 checkpoint（{"model", "config", ...}）→ (model, cfg)，留在 CPU、未切 eval。

    兼容 torch.compile 留下的 `_orig_mod.` 前缀；SFT 阶段的 M 权重推理时启用候选池
    （论文：分层稀疏索引在后训练引入，训练与推理用同样的候选限制）。
    """
    cfg = config_from_dict(blob.get("config") or {})
    model = build_model(cfg)
    prefix = "_orig_mod."
    state = {(k[len(prefix):] if k.startswith(prefix) else k): v for k, v in blob["model"].items()}
    model.load_state_dict(state, strict=True)
    if isinstance(cfg, FlashConfig):
        model.infer_candidate_pool = blob.get("stage") == "sft"
    return model, cfg
