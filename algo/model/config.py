"""MiniSeek 模型配置。

架构线：
- dense → MiniSeek-D*（稠密 Decoder-only）
- tiny  → MiniSeek-D 冒烟档（同 DenseLM，极小超参）
- v41   → MiniSeek-F*（DeepSeek-V4.1-Flash 全架构），配置见 model/v41/config.py 的 FlashConfig
版本号写在 model_family / model_version（如 D0.1）。旧的 M0.1 MoE 已移除。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Literal


Arch = Literal["dense", "moe", "tiny"]


@dataclass
class ModelConfig:
    """冻结规格：d_model=512, n_layers=8, n_heads=8, head_dim=64, FFN=2048."""

    # --- 产品命名 ---
    model_family: str = "MiniSeek-D"  # MiniSeek-D / MiniSeek-M
    model_version: str = "0.1"  # D0.1 → family=MiniSeek-D, version=0.1
    arch: Arch = "dense"

    # --- Dense 主干 ---
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    head_dim: int = 64  # d_model // n_heads
    intermediate_size: int = 2048  # Dense FFN
    vocab_size: int = 6400  # MiniMind tokenizer 实测
    max_seq_len: int = 2048
    rope_theta: float = 1e6
    rms_norm_eps: float = 1e-5
    dropout: float = 0.0
    tie_word_embeddings: bool = True

    # --- MoE（arch=moe 时使用；dense 忽略）---
    use_moe: bool = False
    n_shared_experts: int = 1
    n_routed_experts: int = 64
    num_experts_per_tok: int = 3
    # 1+64 专家把中间维收到 256。总参约 216.4M，比 h=512 的 1+32 略小。
    moe_intermediate_size: int = 256
    # V4.1 MoE：SwiGLU clamp 阈值；bias 每步按负载更新的步长。不计入可训练参数。
    swiglu_clamp: float = 10.0
    expert_bias_speed: float = 0.001

    # --- Attention 实现（训练/推理默认开 Flash；装不上 flash_attn 时走 torch SDPA flash）---
    use_flash_attn: bool = True

    # --- 后续注意力/结构钩子（未实现前勿打开）---
    use_swa: bool = False
    swa_window: int = 128
    use_csa2: bool = False
    csa2_topk: int = 128
    csa2_m: int = 1
    use_ced: bool = False
    ced_encoder_layers: int = 4
    ced_decoder_layers: int = 4

    def __post_init__(self) -> None:
        if self.d_model != self.n_heads * self.head_dim:
            raise ValueError(
                f"d_model={self.d_model} != n_heads*head_dim="
                f"{self.n_heads * self.head_dim}"
            )
        if self.arch == "moe":
            self.use_moe = True
            if self.model_family == "MiniSeek-D":
                self.model_family = "MiniSeek-M"
        elif self.arch in ("dense", "tiny"):
            self.use_moe = False

    @property
    def release_name(self) -> str:
        """例如 MiniSeek-D0.1"""
        letter = "M" if self.arch == "moe" or self.model_family.endswith("-M") else "D"
        ver = self.model_version.lstrip("DMdm")
        return f"MiniSeek-{letter}{ver}"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def tiny(cls) -> "ModelConfig":
        return cls(
            model_family="MiniSeek-D",
            model_version="0.0-tiny",
            arch="tiny",
            d_model=64,
            n_layers=2,
            n_heads=1,
            head_dim=64,
            intermediate_size=256,
            vocab_size=6400,
            max_seq_len=256,
        )

    def with_vocab_size(self, vocab_size: int) -> "ModelConfig":
        d = self.to_dict()
        d["vocab_size"] = int(vocab_size)
        return ModelConfig.from_dict(d)
