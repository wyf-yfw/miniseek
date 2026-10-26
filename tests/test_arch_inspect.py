# tests/test_arch_inspect.py
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from model.config import ModelConfig
from model.dense import DenseLM, count_parameters
from web.arch_inspect import inspect_blob, inspect_path, load_ckpt_blob


def _tiny_blob():
    cfg = ModelConfig.tiny()
    model = DenseLM(cfg)
    return {
        "config": cfg.to_dict(),
        "model": model.state_dict(),
        "step": 12,
        "loss": 2.5,
    }


def test_inspect_tiny_dense_layers_and_params():
    blob = _tiny_blob()
    out = inspect_blob(blob, path="tiny.pt")
    assert out["config"]["arch"] == "tiny"
    assert out["config"]["n_layers"] == 2
    assert out["config"]["d_model"] == 64
    assert out["tie_word_embeddings"] is True
    layers = [b for b in out["blocks"] if b["kind"] == "layer"]
    assert len(layers) == 2
    assert layers[0]["id"] == "layer-0"
    part_ids = {p["id"] for p in layers[0]["parts"]}
    assert "attn" in part_ids and "ffn" in part_ids
    expected = count_parameters(DenseLM(ModelConfig.tiny()))
    assert out["total_params"] == expected
    lm = next(x for x in out["tail"] if x["id"] == "lm_head")
    assert lm["tied"] is True


def test_inspect_path_missing():
    with pytest.raises(FileNotFoundError):
        inspect_path("/no/such/ckpt.pt")


def test_load_and_inspect_roundtrip(tmp_path: Path):
    blob = _tiny_blob()
    p = tmp_path / "t.pt"
    torch.save(blob, p)
    loaded = load_ckpt_blob(p)
    assert "model" in loaded and "config" in loaded
    out = inspect_path(p)
    assert out["meta"]["step"] == 12
    assert out["blocks"][0]["kind"] == "embedding"


def test_synthetic_moe_parts():
    """MoE implementation may be stub; inspect must still classify expert keys."""
    cfg = ModelConfig(
        arch="moe",
        d_model=64,
        n_layers=1,
        n_heads=1,
        head_dim=64,
        intermediate_size=256,
        vocab_size=100,
        max_seq_len=32,
        n_shared_experts=1,
        n_routed_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=128,
        tie_word_embeddings=True,
    )
    sd = {
        "tok_emb.weight": torch.zeros(100, 64),
        "layers.0.attn_norm.weight": torch.ones(64),
        "layers.0.attn.qkv.weight": torch.zeros(192, 64),
        "layers.0.attn.o_proj.weight": torch.zeros(64, 64),
        "layers.0.ffn_norm.weight": torch.ones(64),
        "layers.0.ffn.gate.weight": torch.zeros(4, 64),
        "layers.0.ffn.shared_experts.0.fc1.weight": torch.zeros(128, 64),
        "layers.0.ffn.shared_experts.0.fc2.weight": torch.zeros(64, 128),
        "layers.0.ffn.experts.0.fc1.weight": torch.zeros(128, 64),
        "layers.0.ffn.experts.0.fc2.weight": torch.zeros(64, 128),
        "layers.0.ffn.experts.1.fc1.weight": torch.zeros(128, 64),
        "layers.0.ffn.experts.1.fc2.weight": torch.zeros(64, 128),
        "layers.0.ffn.experts.2.fc1.weight": torch.zeros(128, 64),
        "layers.0.ffn.experts.2.fc2.weight": torch.zeros(64, 128),
        "layers.0.ffn.experts.3.fc1.weight": torch.zeros(128, 64),
        "layers.0.ffn.experts.3.fc2.weight": torch.zeros(64, 128),
        "norm.weight": torch.ones(64),
        "lm_head.weight": torch.zeros(100, 64),
    }
    blob = {"config": cfg.to_dict(), "model": sd, "step": 1, "loss": 1.0}
    out = inspect_blob(blob)
    layer = next(b for b in out["blocks"] if b["kind"] == "layer")
    ids = {p["id"] for p in layer["parts"]}
    assert "attn" in ids
    assert "router" in ids or "gate" in ids
    assert "shared_experts" in ids
    assert "routed_experts" in ids
    routed = next(p for p in layer["parts"] if p["id"] == "routed_experts")
    assert routed["detail"]["n_routed_experts"] == 4
    assert routed["detail"]["num_experts_per_tok"] == 2
