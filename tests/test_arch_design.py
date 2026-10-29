from __future__ import annotations

from algo.model.config import ModelConfig
from algo.model.dense import DenseLM, count_parameters
from api.arch_inspect import inspect_design, preset_config


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
    assert by_id["attn"]["shape_in"] == "[b,t,64]"
    assert by_id["attn"]["shape_out"] == "[b,t,64]"
    assert by_id["ffn"]["shape_in"] == "[b,t,64]"
    assert by_id["attn_norm"]["shape_in"] == "[b,t,64]"
    assert by_id["attn"]["weight_shapes"] == [
        {"name": "qkv.weight", "shape": "[192,64]"},
        {"name": "o_proj.weight", "shape": "[64,64]"},
    ]
    emb = next(b for b in out["blocks"] if b["kind"] == "embedding")
    assert emb["shape_in"] == "[b,t]" and emb["shape_out"] == "[b,t,64]"
    lm = next(b for b in out["tail"] if b["id"] == "lm_head")
    assert lm["shape_in"] == "[b,t,64]" and lm["shape_out"] == "[b,t,6400]"


def test_design_dense_matches_denselm():
    out = inspect_design("dense")
    assert out["config"]["n_layers"] == 8
    assert out["total_params"] == count_parameters(DenseLM(ModelConfig()))
    layer = next(b for b in out["blocks"] if b["kind"] == "layer")
    by_id = {p["id"]: p for p in layer["parts"]}
    # dense defaults: max_seq_len=2048, d_model=512, vocab_size=6400, intermediate=2048
    assert by_id["attn"]["shape_in"] == "[b,t,512]"
    assert by_id["attn"]["shape_out"] == "[b,t,512]"
    assert by_id["ffn"]["shape_in"] == "[b,t,512]"
    assert by_id["ffn"]["shape_out"] == "[b,t,512]"
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
    assert emb["shape_in"] == "[b,t]" and emb["shape_out"] == "[b,t,512]"
    assert emb["weight_shapes"] == [{"name": "tok_emb.weight", "shape": "[6400,512]"}]
    lm = next(b for b in out["tail"] if b["id"] == "lm_head")
    assert lm["shape_in"] == "[b,t,512]" and lm["shape_out"] == "[b,t,6400]"
    assert lm["weight_shapes"] == [{"name": "lm_head.weight", "shape": "[6400,512] (tied)"}]


def test_design_v41_fig3():
    out = inspect_design("miniseek-m")
    assert out["config"]["arch"] == "miniseek-m"
    assert out["meta"]["diagram"] == "fig3" and out["meta"]["trainable"] is True
    assert out["total_params"] <= 200_000_000
    fig = out["fig3"]

    def sig(groups):
        return [(g["rows"][0]["mode"], g["repeat"], (g.get("nested") or {}).get("repeat")) for g in groups]

    assert sig(fig["encoder"]["groups"]) == [("SWA", 2, None), ("Full", 1, 1)]
    assert sig(fig["decoder"]["groups"]) == [("Full", 1, None), ("Reuse", 1, None), ("Reindex", 1, 1)]
    rows = [
        r["id"]
        for g in fig["encoder"]["groups"] + fig["decoder"]["groups"]
        for r in g["rows"] + (g.get("nested") or {}).get("rows", [])
    ]
    assert {"enc_swa_attn", "enc_full", "enc_reuse", "dec_full", "dec_reindex"} <= set(rows)
    attn_layers = sorted(
        i
        for g in fig["encoder"]["groups"] + fig["decoder"]["groups"]
        for r in g["rows"] + (g.get("nested") or {}).get("rows", [])
        if r["type"] != "moe"
        for i in r["layers"]
    )
    assert attn_layers == list(range(out["config"]["n_layers"]))
    assert fig["head"]["id"] == "lm_head"
    assert fig["engram_targets"] == ["enc_swa_attn", "enc_reuse"] and fig["show_vision"] is False
    assert {f["id"] for f in fig["kv_flows"]} >= {"ced", "indexer", "reindex"}
    layer4 = next(b for b in out["blocks"] if b["id"] == "layer-4")
    parts = {p["id"] for p in layer4["parts"]}
    assert {"hc_attn", "attn", "compressor", "indexer", "router", "shared_experts", "routed_experts"} <= parts
    assert next(b for b in out["tail"] if b["id"] == "dspark")["params"] > 0
    assert next(b for b in out["tail"] if b["id"] == "lm_head")["tied"] is False


def test_preset_unknown():
    import pytest
    with pytest.raises(ValueError):
        preset_config("xl")
