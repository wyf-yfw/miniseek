"""MiniSeek 模型包。

- Dense（MiniSeek-D*）：`from algo.model.dense import DenseLM`
- V4.1-Flash 架构（MiniSeek-F*，arch="v41"）：`from algo.model.v41 import Transformer`
"""

from typing import Any, Dict, Optional, Sequence, Union

from .config import ModelConfig
from .dense import DenseLM, count_parameters
from .v41.config import FlashConfig

__all__ = ["ModelConfig", "FlashConfig", "DenseLM", "count_parameters", "build_model", "config_from_dict"]

AnyConfig = Union[ModelConfig, FlashConfig]


def config_from_dict(d: Dict[str, Any]) -> AnyConfig:
    """按 checkpoint / config.json 里的 arch 还原配置。"""
    if (d or {}).get("arch") == "v41":
        return FlashConfig.from_dict(d)
    return ModelConfig.from_dict(d or {})


def build_model(cfg: AnyConfig, token_map: Optional[Sequence[int]] = None):
    """按 cfg.arch 构建模型。v41 需要 tokenizer 压缩表（缺省为恒等映射）。"""
    if isinstance(cfg, FlashConfig):
        from .v41.model import Transformer

        return Transformer(cfg, token_map=token_map)
    if getattr(cfg, "arch", "dense") == "moe" or getattr(cfg, "use_moe", False):
        from .moe import MoELM

        return MoELM(cfg)
    return DenseLM(cfg)
