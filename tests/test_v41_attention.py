import pytest
import torch
import torch.nn.functional as F

from algo.model.v41.attention import (
    Attention,
    AttnState,
    Compressor,
    mask_window,
    select_candidate_blocks,
    sparse_attention,
    window_indices,
)
from algo.model.v41.config import FlashConfig
from tests.v41_ref import official as ref


def _cfg():
    return FlashConfig.tiny()


def _layers(cfg, seed=0):
    torch.manual_seed(seed)
    layers = [Attention(cfg, i) for i in range(cfg.n_layers)]
    with torch.no_grad():
        for layer in layers:
            for name, p in layer.named_parameters():
                if name.endswith("norm.weight"):
                    p.uniform_(0.5, 1.5)
                elif name == "attn_sink":
                    p.normal_(0, 1.0)
    return layers


def _run(layers, x, *, use_candidates=False, doc_ids=None, swa_floor=None):
    state = AttnState(use_candidates=use_candidates)
    outs = [layer(x, state, doc_ids=doc_ids, swa_floor=swa_floor) for layer in layers]
    return outs, state


def _ref_stack(cfg, layers, x, use_candidates):
    """把官方函数按层串起来：共享主 KV / indexer K / top-k / 候选池。"""
    B, T, _ = x.shape
    rd, eps = cfg.rope_head_dim, cfg.norm_eps
    shared = {}
    outs = []
    for layer in layers:
        lid, ratio = layer.layer_id, cfg.compress_ratios[layer.layer_id]
        freqs = layer.freqs_cis[:T]
        qr = ref.rms_norm(F.linear(x, layer.wq_a.weight), layer.q_norm.weight, eps)
        q = F.linear(qr, layer.wq_b.weight).unflatten(-1, (cfg.n_heads, cfg.head_dim)).clone()
        ref.apply_rotary_emb_(q[..., -rd:], freqs)
        kv = ref.rms_norm(F.linear(x, layer.wkv.weight), layer.kv_norm.weight, eps).clone()
        ref.apply_rotary_emb_(kv[..., -rd:], freqs)
        idxs = ref.get_window_topk_idxs(cfg.window_size, B, T, 0)
        if ratio:
            if layer.compressor is not None:
                c = layer.compressor
                latent = ref.compressor_prefill(
                    x, c.wkv.weight, None if c.wgate is None else c.wgate.weight, c.norm.weight, eps, ratio
                )
                freqs_c = layer.freqs_cis[: T - T % ratio : ratio]
                ik = ref.rms_norm(F.linear(latent, layer.indexer.wk.weight), layer.indexer.k_norm.weight, eps).clone()
                ref.apply_rotary_emb_(ik[..., -rd:], freqs_c)
                lat = latent.clone()
                ref.apply_rotary_emb_(lat[..., -rd:], freqs_c)
                shared["index_k"], shared["kv"] = ik, lat
            if layer.indexer is not None:
                ind = layer.indexer
                cand_src = (
                    (cfg.candidate_topk_blocks, cfg.candidate_block_size)
                    if use_candidates and ind.is_candidate_source
                    else None
                )
                cands = shared.get("cand") if use_candidates and ind.uses_candidates else None
                ci, new_c, _ = ref.indexer_prefill(
                    x,
                    qr,
                    shared["index_k"],
                    wq_b=ind.wq_b.weight,
                    weights_proj=ind.weights_proj.weight,
                    freqs_cis=freqs,
                    ratio=ratio,
                    rope_dim=rd,
                    n_heads=cfg.index_n_heads,
                    head_dim=cfg.index_head_dim,
                    index_topk=cfg.index_topk,
                    offset=T,
                    candidates=cands,
                    candidate_source=cand_src,
                )
                if new_c is not None:
                    shared["cand"] = new_c
                shared["idx"] = ci
            kv = torch.cat([kv, shared["kv"]], dim=1)
            idxs = torch.cat([idxs, shared["idx"]], dim=-1)
        o = ref.sparse_attn(q, kv, layer.attn_sink, idxs, cfg.head_dim**-0.5).clone()
        ref.apply_rotary_emb_(o[..., -rd:], freqs, True)
        o = o.view(B, T, cfg.o_groups, -1)
        wo_a = layer.wo_a.weight.view(cfg.o_groups, cfg.o_lora_rank, -1)
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
        outs.append(F.linear(o.flatten(2), layer.wo_b.weight))
    return outs


@pytest.mark.parametrize("seqlen", [3, 8, 21])
def test_window_indices_match_official(seqlen):
    ours = window_indices(seqlen, 8)
    expect = ref.get_window_topk_idxs(8, 1, seqlen, 0)[0].long()
    assert torch.equal(ours, expect)


@pytest.mark.parametrize("layer_id", [2, 4])
@pytest.mark.parametrize("seqlen", [13, 16])
def test_compressor_matches_official(layer_id, seqlen):
    cfg = _cfg()
    torch.manual_seed(1)
    comp = Compressor(cfg, layer_id)
    x = torch.randn(2, seqlen, cfg.dim)
    expect = ref.compressor_prefill(
        x, comp.wkv.weight, None if comp.wgate is None else comp.wgate.weight, comp.norm.weight, cfg.norm_eps, comp.ratio
    )
    torch.testing.assert_close(comp(x), expect, atol=1e-5, rtol=1e-5)


def test_sparse_attention_matches_official():
    torch.manual_seed(2)
    q = torch.randn(2, 5, 3, 8)
    kv = torch.randn(2, 9, 8)
    idx = torch.randint(-1, 9, (2, 5, 6))
    idx[0, 2] = -1  # 整行为空：输出 0
    sink = torch.randn(3)
    o, probs = sparse_attention(q, kv, idx, sink, 8**-0.5)
    expect = ref.sparse_attn(q, kv, sink, idx.int(), 8**-0.5)
    torch.testing.assert_close(o, expect, atol=1e-5, rtol=1e-5)
    assert torch.all(o[0, 2] == 0)
    assert torch.all(probs.sum(-1) <= 1 + 1e-6)


def test_candidate_blocks_match_official():
    torch.manual_seed(3)
    scores = torch.randn(2, 10, 37)
    lens = (torch.arange(1, 11) * 3).unsqueeze(-1)
    scores = scores.masked_fill(torch.arange(37) >= lens, float("-inf"))
    ours = select_candidate_blocks(scores, lens, 3, 4)
    assert torch.equal(ours, ref.select_candidate_blocks(scores, lens, 3, 4))


@pytest.mark.parametrize("use_candidates", [False, True])
def test_attention_modes_match_reference(use_candidates):
    cfg = _cfg()
    layers = _layers(cfg)
    torch.manual_seed(4)
    x = torch.randn(2, 37, cfg.dim)
    ours, _ = _run(layers, x, use_candidates=use_candidates)
    expect = _ref_stack(cfg, layers, x, use_candidates)
    for i, (a, b) in enumerate(zip(ours, expect)):
        torch.testing.assert_close(a, b, atol=2e-5, rtol=1e-4, msg=f"layer {i}")


def test_layer_modes_and_params():
    cfg = _cfg()
    layers = _layers(cfg)
    assert [layer.mode for layer in layers] == cfg.layer_modes()
    for i in (0, 1, 3, 5, 7):
        assert layers[i].compressor is None and layers[i].indexer is None
    assert layers[6].indexer is not None and not hasattr(layers[6].indexer, "wk")
    assert layers[2].indexer.owns_k and layers[4].indexer.owns_k


def test_attention_causal():
    cfg = _cfg()
    layers = _layers(cfg)
    torch.manual_seed(5)
    x = torch.randn(1, 40, cfg.dim)
    y = x.clone()
    y[:, 25:] = torch.randn_like(y[:, 25:])
    a, _ = _run(layers, x)
    b, _ = _run(layers, y)
    for oa, ob in zip(a, b):
        torch.testing.assert_close(oa[:, :25], ob[:, :25])


@pytest.mark.parametrize("use_candidates", [False, True])
def test_doc_mask_matches_separate_run(use_candidates):
    # top-k 与候选池覆盖全部条目，排除 ReLU 打分并列时任意取舍带来的差异
    cfg = FlashConfig.from_dict({**_cfg().to_dict(), "index_topk": 64, "candidate_topk_blocks": 16})
    layers = _layers(cfg)
    torch.manual_seed(6)
    x = torch.randn(1, 40, cfg.dim)
    doc_ids = torch.tensor([[0] * 16 + [1] * 24])
    packed, _ = _run(layers, x, use_candidates=use_candidates, doc_ids=doc_ids)
    alone_a, _ = _run(layers, x[:, :16], use_candidates=use_candidates)
    alone_b, _ = _run(layers, x[:, 16:], use_candidates=use_candidates)
    for i, (p, a, b) in enumerate(zip(packed, alone_a, alone_b)):
        torch.testing.assert_close(p[:, :16], a, atol=2e-5, rtol=1e-4, msg=f"doc A layer {i}")
        torch.testing.assert_close(p[:, 16:], b, atol=2e-5, rtol=1e-4, msg=f"doc B layer {i}")


@pytest.mark.parametrize("use_candidates", [False, True])
def test_doc_mask_no_leakage(use_candidates):
    cfg = _cfg()
    layers = _layers(cfg)
    torch.manual_seed(8)
    x = torch.randn(1, 40, cfg.dim)
    y = x.clone()
    y[:, :16] = torch.randn_like(y[:, :16])
    doc_ids = torch.tensor([[0] * 16 + [1] * 24])
    a, _ = _run(layers, x, use_candidates=use_candidates, doc_ids=doc_ids)
    b, _ = _run(layers, y, use_candidates=use_candidates, doc_ids=doc_ids)
    for oa, ob in zip(a, b):
        assert torch.equal(oa[:, 16:], ob[:, 16:])


def test_indexer_grad_isolation():
    cfg = _cfg()
    layers = _layers(cfg)
    params = {f"{i}.{n}": p for i, layer in enumerate(layers) for n, p in layer.named_parameters()}
    torch.manual_seed(7)
    x = torch.randn(2, 30, cfg.dim, requires_grad=True)

    outs, state = _run(layers, x)
    sum(o.square().mean() for o in outs).backward()
    for name, p in params.items():
        if ".indexer." in name:
            assert p.grad is None or torch.all(p.grad == 0), name

    for p in params.values():
        p.grad = None
    x.grad = None
    outs, state = _run(layers, x)
    assert len(state.kl) == 3
    torch.stack(state.kl).mean().backward()
    assert x.grad is None or torch.all(x.grad == 0)
    for name, p in params.items():
        if ".indexer." in name:
            continue
        assert p.grad is None or torch.all(p.grad == 0), name
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in params.items() if ".indexer." in n)


def test_swa_floor_truncates_window():
    win = window_indices(10, 4).unsqueeze(0)
    floor = torch.tensor([[0] * 6 + [7] * 4])
    out = mask_window(win, None, floor)
    assert torch.all(out[0, 7:][(win[0, 7:] < 7)] == -1)
    assert torch.equal(out[0, :6], win[0, :6])
