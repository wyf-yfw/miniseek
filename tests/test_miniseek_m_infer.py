import pytest
import torch

from algo.model.miniseek_m import FlashConfig, Transformer
from algo.model.miniseek_m.cache import InferenceCache, LayerCache, ced_prefill, compressor_append, decode_chunk


def _model(seed=0, **over):
    # top-k 与候选池覆盖全部条目：排除 ReLU 打分并列时任意取舍，专门检验缓存与 CED 机制
    base = {**FlashConfig.tiny().to_dict(), "index_topk": 64, "candidate_topk_blocks": 32, **over}
    torch.manual_seed(seed)
    model = Transformer(FlashConfig.from_dict(base)).eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("attn_sink"):
                p.normal_(0, 1.0)
    return model


def _floor(T, P, W):
    floor = torch.zeros(1, T, dtype=torch.long)
    s = max(P - W, 0)
    floor[0, s:P] = s
    return floor


def _full_greedy(model, ids, steps, *, P, use_candidates=False):
    seq = ids
    out = []
    for _ in range(steps):
        logits, _, _ = model(seq, swa_floor=_floor(seq.size(1), P, model.cfg.window_size), use_candidate_pool=use_candidates)
        out.append(logits[:, -1])
        seq = torch.cat([seq, logits[:, -1].argmax(-1, keepdim=True)], dim=1)
    return seq, out


def _cached_greedy(model, ids, steps, *, use_candidates=False):
    cache = InferenceCache.create(model, ids.size(0), model.cfg.max_seq_len)
    logits = ced_prefill(model, ids, cache, use_candidates=use_candidates)
    seq, out = ids, [logits]
    for _ in range(steps - 1):
        nxt = logits.argmax(-1, keepdim=True)
        seq = torch.cat([seq, nxt], dim=1)
        logits = decode_chunk(model, nxt, cache, use_candidates=use_candidates)[:, -1]
        out.append(logits)
    seq = torch.cat([seq, logits.argmax(-1, keepdim=True)], dim=1)
    return seq, out, cache


@pytest.mark.parametrize("use_candidates", [False, True])
def test_cached_equals_full_short_prompt(use_candidates):
    model = _model()
    ids = torch.randint(1, 97, (1, 6))  # P ≤ 窗口：Bounded Replay 覆盖整段 prompt，应与全量完全一致
    steps = 24  # 远超窗口 8，环形缓存多次绕回
    full_seq, full = _full_greedy(model, ids, steps, P=6, use_candidates=use_candidates)
    seq, cached, _ = _cached_greedy(model, ids, steps, use_candidates=use_candidates)
    assert torch.equal(seq, full_seq)
    for a, b in zip(cached, full):
        torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("use_candidates", [False, True])
def test_ced_prefill_matches_replay_forward(use_candidates):
    model = _model(1)
    P = 21
    ids = torch.randint(1, 97, (1, P))
    full_seq, full = _full_greedy(model, ids, 12, P=P, use_candidates=use_candidates)
    seq, cached, cache = _cached_greedy(model, ids, 12, use_candidates=use_candidates)
    assert torch.equal(seq, full_seq)
    for a, b in zip(cached, full):
        torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-4)


def test_decoder_window_only_holds_replay_segment():
    model = _model(2)
    W, P = model.cfg.window_size, 30
    cache = InferenceCache.create(model, 1, model.cfg.max_seq_len)
    ced_prefill(model, torch.randint(1, 97, (1, P)), cache)
    for i in range(model.cfg.n_encoder_layers, model.cfg.n_layers):
        pos = cache.layers[i].win_pos[0]
        assert int(pos.min()) == P - W and int(pos.max()) == P - 1


def test_compressor_append_split_equivalence():
    model = _model(3)
    attn = model.layers[2].attn  # m = 2
    x = torch.randn(1, 13, model.cfg.dim)

    def fresh():
        return InferenceCache.create(model, 1, 64).layers[2]

    one = fresh()
    compressor_append(attn, x, 0, one)
    parts = fresh()
    p = 0
    for n in (3, 1, 4, 5):
        compressor_append(attn, x[:, p : p + n], p, parts)
        p += n
    torch.testing.assert_close(parts.ckv[:, :6], one.ckv[:, :6])
    torch.testing.assert_close(parts.ik[:, :6], one.ik[:, :6])
    torch.testing.assert_close(parts.kv_state[:, :1], one.kv_state[:, :1])


def test_generate_stream_interface():
    model = _model(4)
    ids = torch.randint(1, 97, (1, 5))
    toks = list(model.generate_stream(ids, max_new_tokens=6, temperature=0.0))
    assert len(toks) == 6 and all(isinstance(t, int) for t in toks)
    stop = list(model.generate_stream(ids, max_new_tokens=6, temperature=0.0, eos_token_id=toks[0]))
    assert stop == toks[:1]
    out = model.generate(ids, max_new_tokens=4, temperature=0.0)
    assert out.shape == (1, 9) and out[0, 5:].tolist() == toks[:4]
    dbg = list(model.generate_stream_debug(ids, max_new_tokens=2, temperature=0.7, candidates_k=3))
    assert len(dbg) == 2 and len(dbg[0][1]["ids"]) == 3
