"""MiniSeek-M0.1 MoE unit tests."""

from __future__ import annotations

import torch

from model.config import ModelConfig
from model.dense import count_parameters


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


def test_expert_mlp_shape():
    from model.moe import ExpertMLP

    cfg = ModelConfig(arch="moe", vocab_size=6400)
    m = ExpertMLP(cfg.d_model, cfg.moe_intermediate_size, cfg.dropout)
    x = torch.randn(2, 8, cfg.d_model)
    y = m(x)
    assert y.shape == x.shape


def test_moe_layer_forward_and_aux():
    from model.moe import MoELayer

    cfg = ModelConfig(arch="moe", vocab_size=6400)
    layer = MoELayer(cfg)
    x = torch.randn(2, 16, cfg.d_model)
    y, aux = layer(x)
    assert y.shape == x.shape
    assert aux.ndim == 0 and torch.isfinite(aux)
    y.sum().backward()
    assert any(p.grad is not None for p in layer.parameters())


def test_load_balance_uniform_low():
    from model.moe import load_balance_loss

    n, t, e = 4, 32, 8
    probs = torch.full((n * t, e), 1.0 / e)
    idx = torch.arange(n * t) % e
    topk = idx.view(-1, 1)
    aux = load_balance_loss(probs, topk, e)
    assert torch.isfinite(aux)


def test_moelm_param_count_and_forward():
    from model.moe import MoELM

    cfg = ModelConfig(arch="moe", vocab_size=6400, max_seq_len=128)
    model = MoELM(cfg)
    n = count_parameters(model)
    assert abs(n - 150_200_000) / 150_200_000 < 0.05, n
    ids = torch.randint(0, 6400, (2, 16))
    labels = torch.randint(0, 6400, (2, 16))
    logits, ce, aux = model(ids, labels=labels)
    assert logits.shape == (2, 16, 6400)
    assert ce is not None and torch.isfinite(ce)
    assert torch.isfinite(aux)
    (ce + 0.01 * aux).backward()


def test_moelm_generate():
    from model.moe import MoELM

    cfg = ModelConfig(arch="moe", vocab_size=6400, max_seq_len=64)
    model = MoELM(cfg).eval()
    ids = torch.randint(0, 6400, (1, 4))
    out = model.generate(ids, max_new_tokens=3, temperature=1.0, top_k=10)
    assert out.shape[1] == 7
