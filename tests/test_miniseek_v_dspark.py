import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from algo import miniseek_v_train as mvt
from algo.model.miniseek_v import FlashConfig, Transformer
from algo.model.miniseek_v.dspark import (
    DSparkAttention,
    DSparkDrafter,
    dspark_loss_terms,
    dspark_training_losses,
    position_weights,
    sample_anchors,
)
from algo.optim import build_miniseek_v_optimizer
from tests.miniseek_v_ref import official as ref

ROOT = Path(__file__).resolve().parents[1]


def _cfg(**over):
    return FlashConfig.from_dict({**FlashConfig.tiny().to_dict(), **over})


def test_draft_attention_matches_official_window_pattern():
    cfg = _cfg()
    torch.manual_seed(0)
    attn = DSparkAttention(cfg, cfg.n_layers)
    with torch.no_grad():
        attn.attn_sink.normal_(0, 1)
    W, G, a = cfg.window_size, cfg.dspark_block_size, 20
    x = torch.randn(1, G, cfg.dim)
    main_x = torch.randn(1, W, cfg.dim)
    ctx_pos = (a - W + torch.arange(W)).unsqueeze(0)
    q_pos = (a + torch.arange(G)).unsqueeze(0)
    ours = attn.forward_draft(x, main_x, ctx_pos, torch.ones(1, W, dtype=torch.bool), q_pos)

    # 官方：上下文 KV 存在环形缓存（槽 = 位置 % W），草稿块接在后面，索引来自 get_dspark_topk_idxs
    start_pos = a - 1
    freqs = attn.freqs_cis
    ring = torch.zeros(1, W, cfg.head_dim)
    kv_ctx = attn.project_window_kv(main_x, freqs[ctx_pos[0]])
    ring[0, ctx_pos[0] % W] = kv_ctx[0]
    _, q = attn.project_q(x, freqs[a : a + G])
    kv_blk = attn.project_window_kv(x, freqs[a : a + G])
    idx = ref.get_dspark_topk_idxs(W, 1, G, start_pos)
    o = ref.sparse_attn(q, torch.cat([ring, kv_blk], dim=1), attn.attn_sink, idx, cfg.head_dim**-0.5)
    expect = attn.output(o, freqs[a : a + G])
    torch.testing.assert_close(ours, expect, atol=1e-5, rtol=1e-4)


def test_loss_terms_math():
    torch.manual_seed(1)
    d, t = torch.randn(2, 3, 7), torch.randn(2, 3, 7)
    y = torch.randint(0, 7, (2, 3))
    c = torch.randn(2, 3)
    w = position_weights(3)
    terms = dspark_loss_terms(d, t, y, c, w)
    pd, pt = d.softmax(-1), t.softmax(-1)
    l1 = (pd - pt).abs().sum(-1)
    acc = 1 - 0.5 * l1
    torch.testing.assert_close(terms["tv"], (l1 * w).sum(-1).mean())
    torch.testing.assert_close(terms["ce"], (-pd.log().gather(-1, y.unsqueeze(-1)).squeeze(-1) * w).sum(-1).mean())
    bce = -(acc * F.logsigmoid(c) + (1 - acc) * F.logsigmoid(-c))
    torch.testing.assert_close(terms["conf"], (bce * w).sum(-1).mean())
    torch.testing.assert_close(w, torch.exp(-torch.arange(3.0) / 3))


def test_sample_anchors_respects_docs_and_labels():
    labels = torch.tensor([[5, 6, 7, -100, 8, 9, 10, 11, 12, -100]])
    docs = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1, 1, 1]])
    rows, pos = sample_anchors(labels, docs, 3, 10)
    assert sorted(pos.tolist()) == [0, 4, 5, 6]


def _backbone(**over):
    torch.manual_seed(2)
    model = Transformer(_cfg(**over)).eval()
    for prm in model.parameters():
        prm.requires_grad_(False)
    return model


def test_backbone_frozen_and_drafter_trains():
    backbone = _backbone()
    drafter = DSparkDrafter(backbone.cfg)
    ids = torch.randint(1, 97, (2, 40))
    labels = torch.cat([ids[:, 1:], torch.full((2, 1), -100)], dim=1)
    head_before = backbone.head.weight.clone()
    terms = dspark_training_losses(drafter, backbone, ids, labels, n_anchors=4)
    (terms["total"] + 1e-4 * terms["balance"]).backward()
    assert all(p.grad is None for p in backbone.parameters())
    named = dict(drafter.named_parameters())
    for n in ("blocks.0.attn.wq_b.weight", "main_proj.weight", "markov_head.embed.weight", "confidence_head.proj.weight", "blocks.2.ffn.experts.0.w1.weight"):
        assert named[n].grad is not None, n
    assert torch.equal(head_before, backbone.head.weight)
    opt = build_miniseek_v_optimizer(drafter, lr=1e-3, cfg=backbone.cfg)
    opt.step()
    assert 0.0 <= float(terms["accept"]) <= 1.0


def test_scheduler_early_stop():
    from algo.model.miniseek_v.dspark import schedule_prefix

    sps = {1: 100, 2: 95, 3: 90, 4: 60, 5: 50, 6: 40}
    # Θ: 100 → 1.9·95=180.5 → 2.62·90=235.8 → 2.692·60=161.5（下降即停）
    assert schedule_prefix(torch.tensor([0.9, 0.8, 0.1, 0.05, 0.01]), sps.__getitem__) == 2
    assert schedule_prefix(torch.tensor([0.9, 0.8, 0.7]), lambda b: 1.0) == 3
    assert schedule_prefix(torch.tensor([0.01, 0.9]), lambda b: 100.0 / b) == 0


def test_acceptance_preserves_target_distribution():
    from algo.model.miniseek_v.dspark import accept_or_resample

    g = torch.Generator().manual_seed(0)
    p_d = torch.tensor([0.6, 0.3, 0.1])
    p_t = torch.tensor([0.2, 0.3, 0.5])
    counts = torch.zeros(3)
    for _ in range(20000):
        x = int(torch.multinomial(p_d, 1, generator=g))
        _, y = accept_or_resample(x, p_d, p_t, g)
        counts[y] += 1
    torch.testing.assert_close(counts / counts.sum(), p_t, atol=0.02, rtol=0)


def _spec_models(seed=3):
    cfg = _cfg(index_topk=64, candidate_topk_blocks=32)
    torch.manual_seed(seed)
    backbone = Transformer(cfg).eval()
    drafter = DSparkDrafter(cfg).eval()
    return backbone, drafter


def test_greedy_spec_equals_greedy_with_rejections():
    backbone, drafter = _spec_models()
    ids = torch.randint(1, 97, (1, 12))
    plain = list(backbone.generate_stream(ids, max_new_tokens=30, temperature=0.0))
    stats = {}
    spec = list(backbone.generate_stream(ids, max_new_tokens=30, temperature=0.0, drafter=drafter, stats=stats))
    assert spec == plain
    assert stats["cycles"] > 0 and stats["drafted"] > stats["accepted"]  # 随机草稿头：大多被拒绝，走回滚


def test_greedy_spec_all_accepted_path(monkeypatch):
    import algo.model.miniseek_v.dspark as dsp

    backbone, drafter = _spec_models(4)
    ids = torch.randint(1, 97, (1, 10))
    P, G = ids.size(1), backbone.cfg.dspark_block_size
    plain = list(backbone.generate_stream(ids, max_new_tokens=40, temperature=0.0))

    def oracle(drafter_, backbone_, anchor, n, dcache, temperature):
        # anchor 位于位置 n；它之后的贪心 token 依次是 plain[n−P+1 ...]
        nxt = plain[n - P + 1 : n - P + 1 + G]
        nxt = nxt + [0] * (G - len(nxt))
        toks = torch.tensor([nxt])
        probs = torch.nn.functional.one_hot(toks, backbone_.cfg.vocab_size).float()
        return toks, probs, torch.full((1, G), 10.0)

    monkeypatch.setattr(dsp, "draft_block", oracle)
    stats = {}
    spec = list(backbone.generate_stream(ids, max_new_tokens=40, temperature=0.0, drafter=drafter, stats=stats))
    assert spec == plain
    assert stats["accepted"] >= stats["drafted"] - G  # 除了序列末尾，全部接受


def test_snapshot_restore_roundtrip():
    from algo.model.miniseek_v.cache import InferenceCache, ced_prefill, decode_chunk
    from algo.model.miniseek_v.dspark import _restore, _snapshot

    backbone, _ = _spec_models(5)
    cache = InferenceCache.create(backbone, 1)
    ced_prefill(backbone, torch.randint(1, 97, (1, 9)), cache)
    snap = _snapshot(cache)
    decode_chunk(backbone, torch.randint(1, 97, (1, 4)), cache)
    _restore(cache, snap)
    assert cache.pos == 9
    for lc, (wkv, wpos, kst, sst) in zip(cache.layers, snap[0]):
        assert torch.equal(lc.win_kv, wkv) and torch.equal(lc.win_pos, wpos)
        if kst is not None:
            assert torch.equal(lc.kv_state, kst)


def test_sts_calibration_reduces_ece():
    from algo.dspark_calibrate import cumulative_ece, fit_sts

    g = torch.Generator().manual_seed(1)
    true_logits = torch.randn(4000, 3, generator=g)
    accept = torch.bernoulli(torch.sigmoid(true_logits), generator=g)
    survive = torch.cumprod(accept, dim=1)
    overconfident = true_logits * 3.0
    temps = fit_sts(overconfident, survive)
    assert cumulative_ece(overconfident, survive, temps) < cumulative_ece(overconfident, survive, [1.0, 1.0, 1.0])
    assert all(t > 1.0 for t in temps)


@pytest.mark.slow
def test_train_dspark_cli_smoke(tmp_path):
    backbone = Transformer(_cfg(vocab_size=6400, max_seq_len=64))
    ckpt = tmp_path / "pt.pt"
    torch.save({"model": mvt.state_for_ckpt(backbone), "config": backbone.cfg.to_dict(), "step": 1}, ckpt)
    data = tmp_path / "pt.jsonl"
    data.write_text("\n".join(json.dumps({"text": "深度学习是机器学习的分支。" * 4}, ensure_ascii=False) for _ in range(8)), encoding="utf-8")
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.run(
        [sys.executable, "-m", "algo.train_dspark", "--backbone", str(ckpt), "--data", str(data), "--out-dir",
         str(tmp_path / "ds"), "--max-steps", "2", "--batch-size", "2", "--anchors", "4", "--log-every", "1"],
        cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8",
    )
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    assert "accept=" in proc.stdout
    blob = torch.load(tmp_path / "ds" / "dspark.pt", map_location="cpu", weights_only=False)
    assert blob["stage"] == "dspark" and "dspark" in blob
