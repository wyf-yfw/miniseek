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
    by_id = {p["id"]: p for p in layer["parts"]}
    # tiny: max_seq_len=256, d_model=64, vocab_size=6400, intermediate=256
    assert by_id["attn"]["shape_in"] == "[b,256,64]"
    assert by_id["attn"]["shape_out"] == "[b,256,64]"
    assert by_id["ffn"]["shape_in"] == "[b,256,64]"
    assert by_id["attn_norm"]["shape_in"] == "[b,256,64]"
    assert by_id["attn"]["weight_shapes"] == [
        {"name": "qkv.weight", "shape": "[192,64]"},
        {"name": "o_proj.weight", "shape": "[64,64]"},
    ]
    emb = next(b for b in out["blocks"] if b["kind"] == "embedding")
    assert emb["shape_in"] == "[b,256]" and emb["shape_out"] == "[b,256,64]"
    lm = next(b for b in out["tail"] if b["id"] == "lm_head")
    assert lm["shape_in"] == "[b,256,64]" and lm["shape_out"] == "[b,256,6400]"


def test_design_dense_matches_denselm():
    out = inspect_design("dense")
    assert out["config"]["n_layers"] == 8
    assert out["total_params"] == count_parameters(DenseLM(ModelConfig()))
    layer = next(b for b in out["blocks"] if b["kind"] == "layer")
    by_id = {p["id"]: p for p in layer["parts"]}
    # dense defaults: max_seq_len=2048, d_model=512, vocab_size=6400, intermediate=2048
    assert by_id["attn"]["shape_in"] == "[b,2048,512]"
    assert by_id["attn"]["shape_out"] == "[b,2048,512]"
    assert by_id["ffn"]["shape_in"] == "[b,2048,512]"
    assert by_id["ffn"]["shape_out"] == "[b,2048,512]"
    assert by_id["attn"]["weight_shapes"] == [
        {"name": "qkv.weight", "shape": "[1536,512]"},
        {"name": "o_proj.weight", "shape": "[512,512]"},
    ]
    assert by_id["ffn"]["weight_shapes"] == [
        {"name": "fc1.weight", "shape": "[2048,512]"},
        {"name": "fc2.weight", "shape": "[512,2048]"},
    ]
    assert by_id["attn_norm"]["weight_shapes"] == [{"name": "weight", "shape": "[512]"}]
    emb = next(b for b in out["blocks"] if b["kind"] == "embedding")
    assert emb["shape_in"] == "[b,2048]" and emb["shape_out"] == "[b,2048,512]"
    assert emb["weight_shapes"] == [{"name": "tok_emb.weight", "shape": "[6400,512]"}]
    lm = next(b for b in out["tail"] if b["id"] == "lm_head")
    assert lm["shape_in"] == "[b,2048,512]" and lm["shape_out"] == "[b,2048,6400]"
    assert lm["weight_shapes"] == [{"name": "lm_head.weight", "shape": "[6400,512] (tied)"}]


def test_design_moe_has_experts():
    out = inspect_design("moe")
    assert out["config"]["arch"] == "moe"
    layer = next(b for b in out["blocks"] if b["kind"] == "layer")
    ids = {p["id"] for p in layer["parts"]}
    assert "router" in ids and "shared_experts" in ids and "routed_experts" in ids
    assert "residual_attn" not in ids
    assert "add_attn" in ids and "add_ffn" in ids
    assert out["total_params"] > inspect_design("dense")["total_params"]
    by_id = {p["id"]: p for p in layer["parts"]}
    # moe: d=512, moe_intermediate=1024, n_routed=8, n_shared=1
    assert by_id["router"]["weight_shapes"] == [{"name": "gate.weight", "shape": "[8,512]"}]
    assert by_id["shared_experts"]["weight_shapes"] == [
        {"name": "fc1.weight", "shape": "[1024,512] ×1"},
        {"name": "fc2.weight", "shape": "[512,1024] ×1"},
    ]
    assert by_id["routed_experts"]["weight_shapes"] == [
        {"name": "fc1.weight", "shape": "[1024,512] ×8"},
        {"name": "fc2.weight", "shape": "[512,1024] ×8"},
    ]


def test_preset_unknown():
    import pytest
    with pytest.raises(ValueError):
        preset_config("xl")
