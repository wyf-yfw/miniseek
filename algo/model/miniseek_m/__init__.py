"""MiniSeek-M：DeepSeek-V4.1-Flash 架构的小尺寸复现（arch="miniseek-m"；旧名 v41 兼容）。"""

from .config import FlashConfig
from .model import Transformer, bounded_replay_floor, step_expert_bias

__all__ = ["FlashConfig", "Transformer", "bounded_replay_floor", "step_expert_bias"]
