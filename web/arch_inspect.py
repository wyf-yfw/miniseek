"""Architecture diagram data from MiniSeek design configs (and optional ckpt inspect)."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from model.config import ModelConfig

_LAYER_RE = re.compile(r"^layers\.(\d+)\.(.+)$")

PRESET_ARCHS = ("tiny", "dense", "moe")


def preset_config(arch: str) -> ModelConfig:
    a = (arch or "").strip().lower()
    if a == "tiny":
        return ModelConfig.tiny()
    if a == "dense":
        return ModelConfig()  # D0.1 defaults
    if a == "moe":
        return ModelConfig(arch="moe")
    raise ValueError(f"未知架构: {arch}（可选: tiny / dense / moe）")


def _attn_params(d: int) -> int:
    # qkv + o_proj (no bias)
    return 3 * d * d + d * d


def _dense_ffn_params(d: int, intermediate: int) -> int:
    return d * intermediate + intermediate * d


def _expert_ffn_params(d: int, moe_intermediate: int) -> int:
    return d * moe_intermediate + moe_intermediate * d


def inspect_from_config(cfg: ModelConfig, *, source: str = "design") -> Dict[str, Any]:
    """Build viz JSON from ModelConfig (no checkpoint). Param counts are design formulas."""
    c = cfg.to_dict()
    d = int(cfg.d_model)
    vocab = int(cfg.vocab_size)
    n_layers = int(cfg.n_layers)
    tie = bool(cfg.tie_word_embeddings)
    is_moe = bool(cfg.arch == "moe" or cfg.use_moe)

    embed_params = vocab * d
    blocks: List[Dict[str, Any]] = [
        {
            "id": "embed",
            "kind": "embedding",
            "label": "Token Embedding",
            "params": embed_params,
        }
    ]

    attn_p = _attn_params(d)
    attn_norm_p = d
    ffn_norm_p = d

    for i in range(n_layers):
        parts: List[Dict[str, Any]] = [
            {
                "id": "attn_norm",
                "label": "RMSNorm (pre-Attn)",
                "params": attn_norm_p,
                "detail": {"eps": cfg.rms_norm_eps},
            },
            {
                "id": "attn",
                "label": "Attention",
                "params": attn_p,
                "detail": {
                    "n_heads": cfg.n_heads,
                    "head_dim": cfg.head_dim,
                    "qkv": f"Linear({d}→{3*d})",
                    "o_proj": f"Linear({d}→{d})",
                },
            },
            {
                "id": "residual_attn",
                "label": "Residual + Attn",
                "params": 0,
                "detail": {"formula": "x = x + Attn(RMSNorm(x))"},
            },
            {
                "id": "ffn_norm",
                "label": "RMSNorm (pre-FFN)",
                "params": ffn_norm_p,
                "detail": {"eps": cfg.rms_norm_eps},
            },
        ]
        if is_moe:
            n_shared = int(cfg.n_shared_experts)
            n_routed = int(cfg.n_routed_experts)
            topk = int(cfg.num_experts_per_tok)
            mid = int(cfg.moe_intermediate_size)
            gate_p = n_routed * d
            one_exp = _expert_ffn_params(d, mid)
            shared_p = n_shared * one_exp
            routed_p = n_routed * one_exp
            parts.extend(
                [
                    {
                        "id": "router",
                        "label": "Router / Gate",
                        "params": gate_p,
                        "detail": {
                            "n_routed_experts": n_routed,
                            "num_experts_per_tok": topk,
                        },
                    },
                    {
                        "id": "shared_experts",
                        "label": f"Shared Experts ×{n_shared}",
                        "params": shared_p,
                        "detail": {
                            "n_shared_experts": n_shared,
                            "moe_intermediate_size": mid,
                            "params_per_expert": one_exp,
                        },
                    },
                    {
                        "id": "routed_experts",
                        "label": f"Routed Experts ×{n_routed}",
                        "params": routed_p,
                        "detail": {
                            "n_routed_experts": n_routed,
                            "num_experts_per_tok": topk,
                            "moe_intermediate_size": mid,
                            "params_per_expert": one_exp,
                        },
                    },
                ]
            )
            ffn_block_p = gate_p + shared_p + routed_p
        else:
            mid = int(cfg.intermediate_size)
            ffn_p = _dense_ffn_params(d, mid)
            parts.append(
                {
                    "id": "ffn",
                    "label": "FFN (GELU)",
                    "params": ffn_p,
                    "detail": {
                        "intermediate_size": mid,
                        "fc1": f"Linear({d}→{mid})",
                        "fc2": f"Linear({mid}→{d})",
                    },
                }
            )
            ffn_block_p = ffn_p

        parts.append(
            {
                "id": "residual_ffn",
                "label": "Residual + FFN",
                "params": 0,
                "detail": {"formula": "x = x + FFN(RMSNorm(x))"},
            }
        )
        layer_params = attn_norm_p + attn_p + ffn_norm_p + ffn_block_p
        blocks.append(
            {
                "id": f"layer-{i}",
                "kind": "layer",
                "index": i,
                "label": f"Layer {i}",
                "params": layer_params,
                "parts": parts,
            }
        )

    norm_p = d
    lm_p = vocab * d
    tail = [
        {
            "id": "norm",
            "kind": "norm",
            "label": "Final RMSNorm",
            "params": norm_p,
        },
        {
            "id": "lm_head",
            "kind": "lm_head",
            "label": "LM Head",
            "params": lm_p,
            "tied": tie,
        },
    ]

    total = embed_params + sum(b["params"] for b in blocks if b["kind"] == "layer") + norm_p
    if not tie:
        total += lm_p

    return {
        "path": f"design:{cfg.arch}",
        "config": c,
        "meta": {
            "step": None,
            "loss": None,
            "source": source,
            "release_name": cfg.release_name,
        },
        "total_params": total,
        "tie_word_embeddings": tie,
        "blocks": blocks,
        "tail": tail,
    }


def inspect_design(arch: str) -> Dict[str, Any]:
    return inspect_from_config(preset_config(arch), source="design")


def load_ckpt_blob(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))

    weight_path: Optional[Path] = None
    cfg_path: Optional[Path] = None
    if path.is_dir() and (path / "pytorch_model.bin").exists():
        weight_path = path / "pytorch_model.bin"
        cfg_path = path / "config.json"
    elif path.name == "pytorch_model.bin":
        weight_path = path
        cfg_path = path.parent / "config.json"

    if weight_path is not None:
        sd = torch.load(weight_path, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "model" in sd:
            blob = sd
            blob.setdefault("source", "hf-blob")
        else:
            cfg_raw: Dict[str, Any] = {}
            if cfg_path and cfg_path.exists():
                cfg_raw = json.loads(cfg_path.read_text(encoding="utf-8"))
            allowed = set(ModelConfig.__dataclass_fields__)  # type: ignore[attr-defined]
            cfg_dict = {k: v for k, v in cfg_raw.items() if k in allowed}
            blob = {
                "model": sd,
                "config": cfg_dict,
                "step": cfg_raw.get("train_step", cfg_raw.get("step")),
                "loss": cfg_raw.get("train_loss", cfg_raw.get("loss")),
                "source": "hf",
            }
        return blob

    blob = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(blob, dict) or "model" not in blob:
        raise ValueError(f"无法识别的 checkpoint 格式: {path}")
    blob.setdefault("source", "training")
    return blob


def _numel(t: torch.Tensor) -> int:
    return int(t.numel())


def _sum_prefix(sd: Dict[str, torch.Tensor], prefix: str) -> int:
    return sum(_numel(v) for k, v in sd.items() if k.startswith(prefix))


def inspect_blob(blob: Dict[str, Any], *, path: str = "") -> Dict[str, Any]:
    if "model" not in blob:
        raise ValueError("checkpoint 缺少 model 权重")
    sd: Dict[str, torch.Tensor] = blob["model"]
    if not isinstance(sd, dict):
        raise ValueError("model 不是 state_dict")

    cfg_raw = blob.get("config") or {}
    if not isinstance(cfg_raw, dict):
        cfg_raw = {}
    try:
        cfg = ModelConfig.from_dict(cfg_raw)
        cfg_out = cfg.to_dict()
    except Exception:
        cfg_out = dict(cfg_raw)

    tie = bool(cfg_out.get("tie_word_embeddings", True))
    if "tok_emb.weight" in sd and "lm_head.weight" in sd:
        try:
            if sd["tok_emb.weight"].data_ptr() == sd["lm_head.weight"].data_ptr():
                tie = True
        except Exception:
            pass

    n_layers = int(cfg_out.get("n_layers") or 0)
    if not n_layers:
        idxs = []
        for k in sd:
            m = _LAYER_RE.match(k)
            if m:
                idxs.append(int(m.group(1)))
        n_layers = (max(idxs) + 1) if idxs else 0

    blocks: List[Dict[str, Any]] = []
    embed_params = _sum_prefix(sd, "tok_emb.")
    blocks.append(
        {
            "id": "embed",
            "kind": "embedding",
            "label": "Token Embedding",
            "params": embed_params,
        }
    )

    is_moe = bool(
        (cfg_out.get("arch") == "moe")
        or cfg_out.get("use_moe")
        or any(".ffn.experts." in k or ".ffn.shared_experts." in k for k in sd)
    )

    for i in range(n_layers):
        pref = f"layers.{i}."
        layer_keys = {k: v for k, v in sd.items() if k.startswith(pref)}
        attn_params = sum(
            _numel(v)
            for k, v in layer_keys.items()
            if k.startswith(pref + "attn")
        )
        parts: List[Dict[str, Any]] = [
            {
                "id": "attn",
                "label": "Attention",
                "params": attn_params,
                "detail": {
                    "n_heads": cfg_out.get("n_heads"),
                    "head_dim": cfg_out.get("head_dim"),
                },
            }
        ]
        if is_moe:
            gate_params = _sum_prefix(sd, pref + "ffn.gate.")
            shared_params = _sum_prefix(sd, pref + "ffn.shared_experts.")
            routed_params = _sum_prefix(sd, pref + "ffn.experts.")
            ffn_norm = _sum_prefix(sd, pref + "ffn_norm.")
            parts.append(
                {
                    "id": "router",
                    "label": "Router / Gate",
                    "params": gate_params + ffn_norm,
                    "detail": {
                        "num_experts_per_tok": cfg_out.get("num_experts_per_tok"),
                    },
                }
            )
            parts.append(
                {
                    "id": "shared_experts",
                    "label": "Shared Experts",
                    "params": shared_params,
                    "detail": {
                        "n_shared_experts": cfg_out.get("n_shared_experts"),
                        "moe_intermediate_size": cfg_out.get("moe_intermediate_size"),
                    },
                }
            )
            n_routed = int(cfg_out.get("n_routed_experts") or 0)
            per = (routed_params // n_routed) if n_routed else routed_params
            parts.append(
                {
                    "id": "routed_experts",
                    "label": "Routed Experts",
                    "params": routed_params,
                    "detail": {
                        "n_routed_experts": n_routed,
                        "num_experts_per_tok": cfg_out.get("num_experts_per_tok"),
                        "params_per_expert": per,
                        "moe_intermediate_size": cfg_out.get("moe_intermediate_size"),
                    },
                }
            )
        else:
            ffn_params = sum(
                _numel(v)
                for k, v in layer_keys.items()
                if k.startswith(pref + "ffn")
            )
            parts.append(
                {
                    "id": "ffn",
                    "label": "FFN",
                    "params": ffn_params,
                    "detail": {
                        "intermediate_size": cfg_out.get("intermediate_size"),
                    },
                }
            )
        layer_params = sum(_numel(v) for v in layer_keys.values())
        blocks.append(
            {
                "id": f"layer-{i}",
                "kind": "layer",
                "index": i,
                "label": f"Layer {i}",
                "params": layer_params,
                "parts": parts,
            }
        )

    norm_params = _sum_prefix(sd, "norm.")
    lm_params = _sum_prefix(sd, "lm_head.")
    tail = [
        {
            "id": "norm",
            "kind": "norm",
            "label": "Final RMSNorm",
            "params": norm_params,
        },
        {
            "id": "lm_head",
            "kind": "lm_head",
            "label": "LM Head",
            "params": lm_params,
            "tied": tie,
        },
    ]

    total = 0
    seen_ptrs: set[int] = set()
    for k, v in sd.items():
        if tie and k.startswith("lm_head."):
            continue
        try:
            ptr = v.data_ptr()
            if ptr in seen_ptrs:
                continue
            seen_ptrs.add(ptr)
        except Exception:
            pass
        total += _numel(v)

    meta = {
        "step": blob.get("step"),
        "loss": blob.get("loss"),
        "source": blob.get("source"),
    }
    return {
        "path": path,
        "config": cfg_out,
        "meta": meta,
        "total_params": total,
        "tie_word_embeddings": tie,
        "blocks": blocks,
        "tail": tail,
    }


def inspect_path(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    blob = load_ckpt_blob(path)
    return inspect_blob(blob, path=str(path))
