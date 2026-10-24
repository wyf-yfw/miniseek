"""MiniSeek-M 系列：DeepSeekMoE 风格混合专家（尚未实现）。

与 `model/dense.py`（MiniSeek-D）并列。实现前请保持 `ModelConfig.arch="dense"`。
目标结构（见 docs/STAGE_PLAN.md 阶段 1）：
  - 1 shared + 8 routed experts, top-2
  - expert intermediate = 1024 (=2d)
  - auxiliary load-balancing loss
"""

from __future__ import annotations

import torch.nn as nn

from .config import ModelConfig


class MoELM(nn.Module):
    """占位：调用即报错，避免误训。"""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        raise NotImplementedError(
            "MiniSeek-M (MoE) 尚未实现。当前请使用 model.dense.DenseLM / MiniSeek-D。"
        )


def build_moe(cfg: ModelConfig) -> MoELM:
    return MoELM(cfg)
