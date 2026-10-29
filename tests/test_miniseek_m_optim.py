import math

import torch

from algo.model.miniseek_m import FlashConfig, Transformer
from algo.optim import (
    Muon,
    SinkhornMomentum,
    build_miniseek_m_optimizer,
    classify_v41_param,
    hybrid_newton_schulz,
    sinkhorn_direction,
)


def _ns_reference(m):
    x = m.float()
    t = x.size(-2) > x.size(-1)
    if t:
        x = x.mT
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for i in range(10):
        a, b, c = (3.4445, -4.7750, 2.0315) if i < 8 else (2.0, -1.5, 0.5)
        x = a * x + b * (x @ x.mT) @ x + c * (x @ x.mT) @ (x @ x.mT) @ x
    return x.mT if t else x


def test_hybrid_ns_coeffs_and_orthogonality():
    torch.manual_seed(0)
    m = torch.randn(3, 8, 5)
    out = hybrid_newton_schulz(m)
    torch.testing.assert_close(out, _ns_reference(m), atol=1e-5, rtol=1e-5)
    s = torch.linalg.svdvals(out)
    assert torch.all((s - 1).abs() < 0.1), s


def test_muon_update_rule():
    torch.manual_seed(1)
    w = torch.nn.Parameter(torch.randn(6, 4))
    w0 = w.detach().clone()
    g = torch.randn(6, 4)
    opt = Muon([w], lr=0.1, momentum=0.9, weight_decay=0.05)
    w.grad = g.clone()
    opt.step()
    expect = w0 * (1 - 0.1 * 0.05) - 0.1 * math.sqrt(6) * 0.18 * _ns_reference((1 + 0.9) * g)
    torch.testing.assert_close(w.detach(), expect, atol=1e-5, rtol=1e-5)


def test_headwise_split_equivalence():
    torch.manual_seed(2)
    w = torch.nn.Parameter(torch.randn(12, 5))
    w._muon_splits = 4
    w0 = w.detach().clone()
    g = torch.randn(12, 5)
    w.grad = g.clone()
    Muon([w], lr=0.1, weight_decay=0.0).step()
    blocks = [
        w0[i * 3 : (i + 1) * 3] - 0.1 * math.sqrt(5) * 0.18 * _ns_reference(1.95 * g[i * 3 : (i + 1) * 3])
        for i in range(4)
    ]
    torch.testing.assert_close(w.detach(), torch.cat(blocks), atol=1e-5, rtol=1e-5)


def _sinkhorn_reference(g_hat, k=11, tau=1e-3, eps=1e-20):
    rho = torch.stack([row.norm() for row in g_hat])
    u = g_hat.clone()
    for i in range(u.size(0)):
        if rho[i] <= tau * rho.mean():
            u[i] = 0
    for step in range(1, k + 1):
        if step % 2:
            for i in range(u.size(0)):
                u[i] = u[i] / (u[i].norm() + eps)
        else:
            for j in range(u.size(1)):
                u[:, j] = u[:, j] / (u[:, j].norm() + eps)
    return math.sqrt(u.size(1)) * u


def test_sinkhorn_matches_algorithm_and_masks_rows():
    torch.manual_seed(3)
    w = torch.nn.Parameter(torch.randn(7, 4))
    w0 = w.detach().clone()
    g = torch.randn(7, 4)
    g[2] = 0
    w.grad = g.clone()
    SinkhornMomentum([w], lr=0.1, beta=0.9, gamma=0.2).step()
    m = 0.1 * g
    g_hat = 0.9 * m + 0.1 * g
    torch.testing.assert_close(w.detach(), w0 - 0.2 * 0.1 * _sinkhorn_reference(g_hat), atol=1e-6, rtol=1e-5)
    assert torch.equal(w.detach()[2], w0[2])
    d = sinkhorn_direction(torch.randn(50, 8))
    torch.testing.assert_close(d.square().mean(1), torch.ones(50), atol=1e-4, rtol=0)


def _opt():
    torch.manual_seed(4)
    model = Transformer(FlashConfig.tiny())
    return model, build_miniseek_m_optimizer(model, lr=1e-3)


def test_every_param_in_one_group():
    model, opt = _opt()
    seen = [id(p) for g in opt.param_groups for p in g["params"]]
    assert len(seen) == len(set(seen))
    assert set(seen) == {id(p) for p in model.parameters()}


def test_group_assignment_examples():
    model, opt = _opt()
    cfg = model.cfg
    named = dict(model.named_parameters())
    kinds = {n: classify_v41_param(n, p) for n, p in named.items()}
    for n in ("embed.weight", "head.weight", "layers.1.engram.embed.weight"):
        assert kinds[n] == "sinkhorn", n
    for n in ("layers.0.attn_norm.weight", "layers.2.attn.q_norm.weight", "norm.weight", "layers.4.attn.indexer.k_norm.weight"):
        assert kinds[n] == "adamw_wd", n
    for n in ("layers.0.attn.attn_sink", "layers.0.hc_attn_base", "layers.0.hc_ffn_scale", "layers.3.engram.q_weight"):
        assert kinds[n] == "adamw_nowd", n
    for n in ("layers.0.hc_attn_fn", "layers.0.ffn.gate.weight", "layers.0.ffn.experts.0.w1.weight", "layers.3.engram.wkv.weight"):
        assert kinds[n] == "muon", n
    assert named["layers.0.attn.wq_b.weight"]._muon_splits == cfg.n_heads
    assert named["layers.6.attn.indexer.wq_b.weight"]._muon_splits == cfg.index_n_heads
    assert named["layers.0.attn.wo_a.weight"]._muon_splits == cfg.o_groups
    assert named["layers.3.engram.wkv.weight"]._muon_splits == cfg.hc_mult + 1
    assert named["layers.0.attn.wkv.weight"]._muon_splits == 1
    adam = opt.optimizers["adamw"]
    assert {g["weight_decay"] for g in adam.param_groups} == {0.1, 0.0}
    assert adam.defaults["eps"] == 1e-20 and adam.defaults["betas"] == (0.9, 0.95)


def test_lr_mult_and_state_roundtrip():
    model, opt = _opt()
    engram_groups = [g for g in opt.optimizers["sinkhorn"].param_groups if g["lr_mult"] == 5.0]
    assert engram_groups and all(p.dim() == 2 for p in engram_groups[0]["params"])
    for g in opt.param_groups:
        g["lr"] = 3e-4
    ids = torch.randint(0, 97, (2, 16))
    _, ce, ex = model(ids, labels=ids)
    (ce + ex["indexer_kl"]).backward()
    before = model.embed.weight.detach().clone()
    opt.step()
    opt.zero_grad()
    assert not torch.equal(before, model.embed.weight)
    sd = opt.state_dict()
    _, opt2 = _opt()
    opt2.load_state_dict(sd)
    assert set(opt2.state_dict()) == {"muon", "adamw", "sinkhorn"}
