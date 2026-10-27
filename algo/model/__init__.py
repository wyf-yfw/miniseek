"""MiniSeek 模型包。

- Dense（MiniSeek-D*）：`from algo.model.dense import DenseLM`
- MoE（MiniSeek-M*）：`from algo.model.moe import MoELM`
"""

from .config import ModelConfig
from .dense import DenseLM, count_parameters

__all__ = ["ModelConfig", "DenseLM", "count_parameters", "build_model"]


def build_model(cfg: ModelConfig):
    """按 cfg.arch 构建模型。"""
    if getattr(cfg, "arch", "dense") == "moe" or getattr(cfg, "use_moe", False):
        from .moe import MoELM

        return MoELM(cfg)
    return DenseLM(cfg)
