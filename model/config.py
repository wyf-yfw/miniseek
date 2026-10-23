"""Stage-0 ModelConfig（Dense Decoder-only）.

后续阶段（MoE / SWA / CSA2 / CED）仅保留扩展钩子字段，Stage 0 不实现。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ModelConfig:
    """冻结规格：d_model=512, n_layers=8, n_heads=8, head_dim=64, FFN=2048."""

    # --- Stage 0 Dense（冻结）---
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    head_dim: int = 64  # d_model // n_heads
    intermediate_size: int = 2048  # Dense FFN
    vocab_size: int = 6400  # MiniMind tokenizer 实测；见 tokenizer.py / README
    max_seq_len: int = 2048
    rope_theta: float = 1e6
    rms_norm_eps: float = 1e-5
    dropout: float = 0.0
    tie_word_embeddings: bool = True

    # --- 扩展钩子（Stage 1+ TODO，Stage 0 忽略）---
    use_moe: bool = False  # MoE（混合专家）
    n_shared_experts: int = 1
    n_routed_experts: int = 8
    num_experts_per_tok: int = 2
    moe_intermediate_size: int = 1024

    use_swa: bool = False  # SWA（滑动窗口注意力）
    swa_window: int = 128

    use_csa2: bool = False  # CSA2（压缩稀疏注意力）
    csa2_topk: int = 128
    csa2_m: int = 1

    use_ced: bool = False  # CED（因果编码器-解码器）
    ced_encoder_layers: int = 4  # L0–L3
    ced_decoder_layers: int = 4  # L4–L7

    def __post_init__(self) -> None:
        if self.d_model != self.n_heads * self.head_dim:
            raise ValueError(
                f"d_model={self.d_model} != n_heads*head_dim="
                f"{self.n_heads * self.head_dim}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})

    def with_vocab_size(self, vocab_size: int) -> "ModelConfig":
        """返回覆盖 vocab_size 的新配置（通常由 tokenizer 检测写入）."""
        d = self.to_dict()
        d["vocab_size"] = int(vocab_size)
        return ModelConfig.from_dict(d)
