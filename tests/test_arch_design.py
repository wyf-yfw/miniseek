from __future__ import annotations

from model.config import ModelConfig
from model.dense import DenseLM, count_parameters
from web.arch_inspect import inspect_design, preset_config


def test_design_tiny_matches_denselm():
    out = inspect_design("tiny")
    assert out["config"]["arch"] == "tiny"
    assert out["config"]["n_layers"] == 2
    assert out["total_params"] == count_parameters(DenseLM(ModelConfig.tiny()))
    layer = next(b for b in out["blocks"] if b["kind"] == "layer")
    ids = {p["id"] for p in layer["parts"]}
    assert "attn" in ids and "ffn" in ids
    assert "residual_attn" not in ids and "residual_ffn" not in ids
    assert "add_attn" in ids and "add_ffn" in ids
    assert layer["residuals"] == [
        {"id": "attn", "from": "layer_in", "to": "add_attn", "label": "x"},
        {"id": "ffn", "from": "post_attn", "to": "add_ffn", "label": "x"},
    ]


def test_design_dense_matches_denselm():
    out = inspect_design("dense")
    assert out["config"]["n_layers"] == 8
    assert out["total_params"] == count_parameters(DenseLM(ModelConfig()))


def test_design_moe_has_experts():
    out = inspect_design("moe")
    assert out["config"]["arch"] == "moe"
    layer = next(b for b in out["blocks"] if b["kind"] == "layer")
    ids = {p["id"] for p in layer["parts"]}
    assert "router" in ids and "shared_experts" in ids and "routed_experts" in ids
    assert "residual_attn" not in ids
    assert "add_attn" in ids and "add_ffn" in ids
    assert out["total_params"] > inspect_design("dense")["total_params"]


def test_preset_unknown():
    import pytest
    with pytest.raises(ValueError):
        preset_config("xl")
