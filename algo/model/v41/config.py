"""MiniSeek-F 配置。字段名与 DeepSeek-V4.1-Flash 官方 inference/model.py 的 ModelArgs 一致。

默认值即 MiniSeek-F0.1：8 层（编码 0–3 / 解码 4–7），d=512，1+64 专家 top-6。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Tuple

_TUPLE_FIELDS = (
    "compress_ratios",
    "kv_source_layers",
    "index_source_layers",
    "engram_layer_ids",
    "dspark_target_layer_ids",
)


@dataclass
class FlashConfig:
    arch: str = "v41"
    model_family: str = "MiniSeek-F"
    model_version: str = "0.1"

    vocab_size: int = 6400
    dim: int = 512
    n_layers: int = 8
    max_seq_len: int = 1024

    # attention：低秩 Q，单头共享 KV（K=V），分组低秩输出投影
    n_heads: int = 16
    head_dim: int = 128
    rope_head_dim: int = 32
    q_lora_rank: int = 128
    o_groups: int = 2
    o_lora_rank: int = 256
    norm_eps: float = 1e-20
    window_size: int = 128
    # 每层：0 = 只有滑窗；m = 主 KV 每 m 个 token 压成一个条目
    compress_ratios: Tuple[int, ...] = (0, 0, 2, 2, 1, 1, 1, 1)
    # 压缩器所在层（Full）；其后同 ratio 的层复用它的主 KV 与 indexer K
    kv_source_layers: Tuple[int, ...] = (2, 4)
    # 跑 indexer 的层（Full 或 Reindex）；其后的层复用最新 top-k
    index_source_layers: Tuple[int, ...] = (2, 4, 6)
    index_n_heads: int = 8
    index_head_dim: int = 64
    index_topk: int = 64
    # 分层稀疏索引：候选池由该层建立，之后的索引层只在池内选（训练时由开关启用）
    candidate_source_layer: int = 4
    candidate_topk_blocks: int = 32
    candidate_block_size: int = 8

    # rope：纯滑窗层用 rope_theta 且不开 YaRN；压缩层用 compress_rope_theta（+YaRN）
    rope_theta: float = 10000.0
    compress_rope_theta: float = 160000.0
    original_seq_len: int = 0
    rope_factor: float = 16.0
    beta_fast: int = 32
    beta_slow: int = 1

    # moe
    n_routed_experts: int = 64
    n_shared_experts: int = 1
    n_activated_experts: int = 6
    moe_inter_dim: int = 176
    score_func: str = "sqrtsoftplus"
    norm_topk_prob: bool = True
    route_scale: float = 1.5
    swiglu_limit: float = 10.0
    expert_bias_speed: float = 0.001

    # single-pass mHC
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6

    # engram
    engram_layer_ids: Tuple[int, ...] = (1, 3)
    engram_max_ngram_size: int = 4
    engram_n_heads: int = 8
    engram_head_dim: int = 32
    engram_vocab_size: int = 16000
    engram_pad_id: int = 0
    # tokenizer 压缩后的词表大小；0 表示不压缩（压缩 id = token id）
    engram_compressed_vocab_size: int = 0

    # dspark（草稿头，参数不计入主干预算）
    n_mtp_layers: int = 3
    dspark_block_size: int = 5
    dspark_noise_token_id: int = 0
    dspark_target_layer_ids: Tuple[int, ...] = (5, 6, 7)
    dspark_markov_rank: int = 64
    dspark_n_routed_experts: int = 16
    dspark_n_activated_experts: int = 3

    tie_word_embeddings: bool = False

    def __post_init__(self) -> None:
        for name in _TUPLE_FIELDS:
            setattr(self, name, tuple(int(v) for v in getattr(self, name)))
        self._validate()

    def _validate(self) -> None:
        L = self.n_layers
        if len(self.compress_ratios) != L:
            raise ValueError(f"compress_ratios 长度 {len(self.compress_ratios)} != n_layers {L}")
        if L % 2:
            raise ValueError("CED 需要偶数层（编码器与解码器各半）")
        if self.n_heads % self.o_groups:
            raise ValueError("n_heads 必须能被 o_groups 整除")
        if self.rope_head_dim % 2 or self.rope_head_dim > min(self.head_dim, self.index_head_dim):
            raise ValueError("rope_head_dim 必须为偶数且不超过 head_dim / index_head_dim")
        for i in self.kv_source_layers:
            if not 0 <= i < L or self.compress_ratios[i] == 0:
                raise ValueError(f"kv source 层 {i} 必须是压缩层")
            if i not in self.index_source_layers:
                raise ValueError(f"kv source 层 {i} 必须同时是 index source（Full 模式）")
        for i in self.index_source_layers:
            if not 0 <= i < L or self.compress_ratios[i] == 0:
                raise ValueError(f"index source 层 {i} 必须是压缩层")
        for i, r in enumerate(self.compress_ratios):
            if r and self.kv_source_for(i) is None:
                raise ValueError(f"压缩层 {i} 之前没有同压缩比的 kv source")
        if self.candidate_source_layer >= 0 and self.candidate_source_layer not in self.kv_source_layers:
            raise ValueError("candidate_source_layer 必须是 Full 层")
        for i in tuple(self.engram_layer_ids) + tuple(self.dspark_target_layer_ids):
            if not 0 <= i < L:
                raise ValueError(f"层号 {i} 越界")
        if self.n_shared_experts != 1:
            raise ValueError("官方结构固定 1 个共享专家")

    # --- 层模式 ---
    @property
    def n_encoder_layers(self) -> int:
        return self.n_layers // 2

    def is_decoder(self, layer_id: int) -> bool:
        return layer_id >= self.n_encoder_layers

    def kv_source_for(self, layer_id: int):
        """该层读取的主 KV 来源层（最近的、压缩比相同的 Full 层）。"""
        r = self.compress_ratios[layer_id]
        if r == 0:
            return None
        src = None
        for i in self.kv_source_layers:
            if i <= layer_id and self.compress_ratios[i] == r:
                src = i
        return src

    def layer_mode(self, layer_id: int) -> str:
        if self.compress_ratios[layer_id] == 0:
            return "SWA"
        if layer_id in self.kv_source_layers:
            return "Full"
        if layer_id in self.index_source_layers:
            return "Reindex"
        return "Reuse"

    def layer_modes(self):
        return [self.layer_mode(i) for i in range(self.n_layers)]

    @property
    def release_name(self) -> str:
        return f"MiniSeek-F{self.model_version}"

    # --- 序列化 ---
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        for name in _TUPLE_FIELDS:
            d[name] = list(d[name])
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FlashConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def tiny(cls) -> "FlashConfig":
        """与 F0.1 同样的 8 层模式，只把宽度缩到 CPU 单测可跑。"""
        return cls(
            model_version="0.0-tiny",
            vocab_size=97,
            dim=32,
            max_seq_len=64,
            n_heads=4,
            head_dim=16,
            rope_head_dim=8,
            q_lora_rank=16,
            o_groups=2,
            o_lora_rank=16,
            window_size=8,
            index_n_heads=2,
            index_head_dim=16,
            index_topk=4,
            candidate_topk_blocks=2,
            candidate_block_size=4,
            n_routed_experts=8,
            n_activated_experts=2,
            moe_inter_dim=16,
            engram_n_heads=2,
            engram_head_dim=8,
            engram_vocab_size=50,
            dspark_block_size=3,
            dspark_markov_rank=8,
            dspark_n_routed_experts=4,
            dspark_n_activated_experts=2,
        )
