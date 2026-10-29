"""MiniSeek-M0.1 MoE unit tests."""

from __future__ import annotations

import torch

from algo.model.config import ModelConfig
from algo.model.dense import count_parameters


def test_moe_config_defaults():
    cfg = ModelConfig(arch="moe")
    assert cfg.use_moe is True
    assert cfg.model_family == "MiniSeek-M"
    assert cfg.n_shared_experts == 1
    assert cfg.n_routed_experts == 64
    assert cfg.num_experts_per_tok == 3
    assert cfg.moe_intermediate_size == 256
    assert cfg.release_name.startswith("MiniSeek-M")


def test_dense_config_unchanged():
    cfg = ModelConfig(arch="dense")
    assert cfg.use_moe is False
    assert cfg.model_family == "MiniSeek-D"


def test_expert_mlp_shape():
    from algo.model.moe import ExpertMLP

    cfg = ModelConfig(arch="moe", vocab_size=6400)
    m = ExpertMLP(cfg.d_model, cfg.moe_intermediate_size, cfg.dropout)
    x = torch.randn(2, 8, cfg.d_model)
    y = m(x)
    assert y.shape == x.shape


def test_moe_layer_forward_and_aux():
    from algo.model.moe import MoELayer

    cfg = ModelConfig(arch="moe", vocab_size=6400)
    layer = MoELayer(cfg)
    x = torch.randn(2, 16, cfg.d_model)
    y, aux = layer(x)
    assert y.shape == x.shape
    assert aux.ndim == 0 and torch.isfinite(aux)
    y.sum().backward()
    assert any(p.grad is not None for p in layer.parameters())


def test_sequence_balance_uniform_finite():
    from algo.model.moe import sequence_balance_loss

    b, t, e = 4, 32, 8
    scores = torch.full((b, t, e), 1.0 / e)
    idx = torch.arange(b * t) % e
    topk = idx.view(b, t, 1)
    aux = sequence_balance_loss(scores, topk, e)
    assert aux.ndim == 0 and torch.isfinite(aux)


def test_moelm_param_count_and_forward():
    from algo.model.moe import MoELM

    cfg = ModelConfig(arch="moe", vocab_size=6400, max_seq_len=128)
    model = MoELM(cfg)
    n = count_parameters(model)
    # SwiGLU 三矩阵：8 层 × (1+64) 专家 × 3 × 512 × 256，加上注意力和词表
    assert n == 216_408_576, n
    ids = torch.randint(0, 6400, (2, 16))
    labels = torch.randint(0, 6400, (2, 16))
    logits, ce, aux = model(ids, labels=labels)
    assert logits.shape == (2, 16, 6400)
    assert ce is not None and torch.isfinite(ce)
    assert torch.isfinite(aux)
    (ce + 0.01 * aux).backward()


def test_moelm_generate():
    from algo.model.moe import MoELM

    cfg = ModelConfig(arch="moe", vocab_size=6400, max_seq_len=64)
    model = MoELM(cfg).eval()
    ids = torch.randint(0, 6400, (1, 4))
    out = model.generate(ids, max_new_tokens=3, temperature=1.0, top_k=10)
    assert out.shape[1] == 7

def _naive_moe_forward(layer, x: torch.Tensor):
    """Reference: nested slot × expert Python loops (pre-sorted dispatch)."""
    from algo.model.moe import route_tokens, sequence_balance_loss

    b, t, d = x.shape
    flat = x.reshape(b * t, d)
    shared_out = layer.shared(flat)
    scores, topk_prob, topk_idx = route_tokens(layer.router(flat), layer.expert_bias, layer.top_k)
    routed = torch.zeros_like(flat)
    for slot in range(layer.top_k):
        idx = topk_idx[:, slot]
        w = topk_prob[:, slot].unsqueeze(-1)
        for e in range(layer.n_routed):
            mask = idx == e
            if not mask.any():
                continue
            routed[mask] = routed[mask] + w[mask] * layer.experts[e](flat[mask])
    y = (shared_out + routed).view(b, t, d)
    aux = sequence_balance_loss(scores.view(b, t, -1), topk_idx.view(b, t, -1), layer.n_routed)
    return y, aux


def test_moe_grouped_dispatch_matches_naive():
    """Sorted/grouped GEMM dispatch must match nested-loop reference (atol 1e-4)."""
    from algo.model.moe import MoELayer

    cfg = ModelConfig(
        arch="moe",
        d_model=32,
        n_heads=1,
        head_dim=32,
        n_routed_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
        vocab_size=100,
        max_seq_len=64,
        dropout=0.0,
    )
    layer = MoELayer(cfg)
    layer.eval()
    torch.manual_seed(0)
    x = torch.randn(2, 8, cfg.d_model)

    with torch.no_grad():
        y_grouped, aux_grouped = layer(x)
        y_naive, aux_naive = _naive_moe_forward(layer, x)

    assert torch.allclose(y_grouped, y_naive, atol=1e-4, rtol=1e-4), (
        (y_grouped - y_naive).abs().max().item()
    )
    assert torch.allclose(aux_grouped, aux_naive, atol=1e-4, rtol=1e-4)


def test_moe_sorted_dispatch_32_experts():
    """Same check at M0.1-ish expert count (1+32 top-2), float32 atol 1e-4."""
    from algo.model.moe import MoELayer

    cfg = ModelConfig(
        arch="moe",
        d_model=64,
        n_heads=2,
        head_dim=32,
        n_routed_experts=32,
        num_experts_per_tok=2,
        moe_intermediate_size=128,
        vocab_size=100,
        max_seq_len=64,
        dropout=0.0,
    )
    layer = MoELayer(cfg)
    layer.eval()
    torch.manual_seed(1)
    x = torch.randn(2, 16, cfg.d_model)

    with torch.no_grad():
        y_grouped, aux_grouped = layer(x)
        y_naive, aux_naive = _naive_moe_forward(layer, x)

    assert torch.allclose(y_grouped, y_naive, atol=1e-4, rtol=1e-4), (
        (y_grouped - y_naive).abs().max().item()
    )
    assert torch.allclose(aux_grouped, aux_naive, atol=1e-4, rtol=1e-4)



def test_probe_expert_load_counts_sum():
    from algo.model.moe import MoELM

    cfg = ModelConfig(
        arch="moe",
        vocab_size=6400,
        max_seq_len=64,
        n_layers=2,
        n_routed_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
        d_model=64,
        n_heads=2,
        head_dim=32,
        dropout=0.0,
    )
    model = MoELM(cfg).eval()
    ids = torch.randint(0, 6400, (2, 16))
    counts, n_tokens = model.probe_expert_load(ids)
    assert n_tokens == 32
    assert counts.shape == (8,)
    # each token contributes top_k assignments per layer
    assert int(counts.sum().item()) == 2 * 16 * 2 * 2


def test_train_window_expert_usage_and_log_line(capsys):
    """训练 log 窗口：begin → forward×N → end/emit expert_load= JSON。"""
    import json
    from algo.model.moe import MoELM
    from algo.train_pretrain import _emit_expert_load_line

    cfg = ModelConfig(
        arch="moe",
        vocab_size=6400,
        max_seq_len=64,
        n_layers=2,
        n_routed_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
        d_model=64,
        n_heads=2,
        head_dim=32,
        dropout=0.0,
    )
    model = MoELM(cfg)
    model.train()
    model.begin_expert_usage()
    b, t = 2, 8
    for _ in range(3):
        ids = torch.randint(0, 6400, (b, t))
        labels = ids.clone()
        _logits, _ce, _aux = model(ids, labels=labels)
    _emit_expert_load_line(model=model, step=3, window_steps=3, cfg=cfg)
    out = capsys.readouterr().out.strip().splitlines()
    assert out and out[-1].startswith("expert_load=")
    payload = json.loads(out[-1].split("=", 1)[1])
    assert payload["step"] == 3
    assert payload["window_steps"] == 3
    assert payload["n_routed"] == 8
    assert len(payload["fractions"]) == 8
    assert len(payload["counts"]) == 8
    # 3 steps × layers × B × T × top_k
    assert payload["total_assignments"] == 3 * 2 * b * t * 2
    assert abs(sum(payload["fractions"]) - 1.0) < 1e-6


def test_swiglu_clamps_preactivation():
    from algo.model.moe import clamped_swiglu

    gate = torch.tensor([100.0, -3.0])
    up = torch.tensor([100.0, -100.0])
    y = clamped_swiglu(gate, up, 10.0)
    silu10 = 10.0 * torch.sigmoid(torch.tensor(10.0))
    assert torch.allclose(y[0], silu10 * 10.0)
    silu_neg = (-3.0) * torch.sigmoid(torch.tensor(-3.0))
    assert torch.allclose(y[1], silu_neg * -10.0)


def test_bias_changes_selection_not_weights():
    from algo.model.moe import MoELayer, route_tokens

    cfg = ModelConfig(
        arch="moe",
        d_model=32,
        n_heads=1,
        head_dim=32,
        n_routed_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
        vocab_size=100,
        max_seq_len=32,
        dropout=0.0,
    )
    layer = MoELayer(cfg).eval()
    torch.manual_seed(0)
    x = torch.randn(1, 4, cfg.d_model)
    flat = x.reshape(-1, cfg.d_model)
    logits = layer.router(flat)
    scores, weight, idx = route_tokens(logits, layer.expert_bias, layer.top_k)
    layer.expert_bias.zero_()
    layer.expert_bias[0] = 50.0
    scores2, weight2, idx2 = route_tokens(logits, layer.expert_bias, layer.top_k)
    assert (idx2[:, 0] == 0).all()
    assert not torch.equal(idx, idx2)
    gathered = scores2.gather(-1, idx2)
    expect = gathered / gathered.sum(dim=-1, keepdim=True)
    assert torch.allclose(weight2, expect)


def test_bias_update_moves_overloaded_expert_down():
    from algo.model.moe import MoELayer

    cfg = ModelConfig(
        arch="moe",
        d_model=32,
        n_heads=1,
        head_dim=32,
        n_routed_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        vocab_size=100,
        max_seq_len=32,
        dropout=0.0,
        expert_bias_speed=0.001,
    )
    layer = MoELayer(cfg)
    layer.train()
    counts = torch.zeros(4)
    counts[0] = 100
    layer._step_counts = counts
    layer.update_expert_bias()
    assert layer.expert_bias[0].item() < 0
    assert layer.expert_bias[1].item() > 0
    assert layer.expert_bias.requires_grad is False
