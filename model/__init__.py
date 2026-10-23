"""ds41-mini model package — Stage 0 Dense Decoder-only."""

from .config import ModelConfig
from .dense import DenseLM, count_parameters

__all__ = ["ModelConfig", "DenseLM", "count_parameters"]
