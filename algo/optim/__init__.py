"""MiniSeek-M 训练用优化器：Muon、Sinkhorn 动量与 v41 参数分组。"""

from .muon import Muon, hybrid_newton_schulz
from .param_groups import CompositeOptimizer, build_miniseek_m_optimizer, classify_v41_param, muon_row_splits
from .sinkhorn import SinkhornMomentum, sinkhorn_direction

__all__ = [
    "Muon",
    "hybrid_newton_schulz",
    "SinkhornMomentum",
    "sinkhorn_direction",
    "CompositeOptimizer",
    "build_miniseek_m_optimizer",
    "classify_v41_param",
    "muon_row_splits",
]
