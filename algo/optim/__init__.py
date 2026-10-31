"""MiniSeek-V 训练用优化器：Muon、Sinkhorn 动量与参数分组。"""

from .muon import Muon, hybrid_newton_schulz
from .param_groups import CompositeOptimizer, build_miniseek_v_optimizer, classify_v41_param, muon_row_splits
from .sinkhorn import SinkhornMomentum, sinkhorn_direction

# 旧名兼容
build_miniseek_m_optimizer = build_miniseek_v_optimizer

__all__ = [
    "Muon",
    "hybrid_newton_schulz",
    "SinkhornMomentum",
    "sinkhorn_direction",
    "CompositeOptimizer",
    "build_miniseek_v_optimizer",
    "build_miniseek_m_optimizer",
    "classify_v41_param",
    "muon_row_splits",
]
