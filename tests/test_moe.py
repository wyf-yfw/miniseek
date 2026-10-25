"""MiniSeek-M0.1 MoE unit tests."""

from __future__ import annotations

from model.config import ModelConfig


def test_moe_config_defaults():
    cfg = ModelConfig(arch="moe")
    assert cfg.use_moe is True
    assert cfg.model_family == "MiniSeek-M"
    assert cfg.n_shared_experts == 1
    assert cfg.n_routed_experts == 32
    assert cfg.num_experts_per_tok == 2
    assert cfg.moe_intermediate_size == 512
    assert cfg.release_name.startswith("MiniSeek-M")


def test_dense_config_unchanged():
    cfg = ModelConfig(arch="dense")
    assert cfg.use_moe is False
    assert cfg.model_family == "MiniSeek-D"
