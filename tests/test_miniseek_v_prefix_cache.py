import torch

import algo.model.miniseek_v.cache as cache_mod
from algo import miniseek_v_train as mvt
from algo.model import load_model_from_blob
from algo.model.miniseek_v import FlashConfig, Transformer
from algo.model.miniseek_v.cache import PrefixCache


def _model(seed=0, token_map=None, **over):
    base = {**FlashConfig.tiny().to_dict(), "index_topk": 64, "candidate_topk_blocks": 32, **over}
    torch.manual_seed(seed)
    return Transformer(FlashConfig.from_dict(base), token_map=token_map).eval()


def test_prefix_hit_skips_global_kv(monkeypatch):
    model = _model()
    pc = PrefixCache()
    ids1 = torch.randint(1, 97, (1, 20))
    toks1 = list(model.generate_stream(ids1, max_new_tokens=5, temperature=0.0, prefix_cache=pc))
    assert len(toks1) == 5 and pc.cache is not None and len(pc.tokens) == 24

    calls = []
    orig = cache_mod.compressor_append

    def spy(attn, x, p0, lc):
        calls.append(p0)
        return orig(attn, x, p0, lc)

    monkeypatch.setattr(cache_mod, "compressor_append", spy)
    ids2 = torch.tensor([pc.tokens + [5, 6, 7, 8, 9, 10]])
    toks2 = list(model.generate_stream(ids2, max_new_tokens=3, temperature=0.0, prefix_cache=pc))
    assert pc.hits == 1 and len(toks2) == 3
    assert calls and min(calls) >= 24  # 前缀 24 个位置的全局 KV 直接复用


def test_prefix_miss_resets():
    model = _model(1)
    pc = PrefixCache()
    list(model.generate_stream(torch.randint(1, 97, (1, 12)), max_new_tokens=2, temperature=0.0, prefix_cache=pc))
    other = torch.tensor([[0] + list(range(1, 12))])
    list(model.generate_stream(other, max_new_tokens=2, temperature=0.0, prefix_cache=pc))
    assert pc.hits == 0


def test_prefix_exact_when_context_within_window():
    model = _model(2)
    pc = PrefixCache()
    ids1 = torch.randint(1, 97, (1, 4))
    list(model.generate_stream(ids1, max_new_tokens=2, temperature=0.0, prefix_cache=pc))
    ids2 = torch.tensor([pc.tokens + [11, 12]])  # 总长 ≤ 窗口：重放覆盖全部，结果应与从头一致
    with_cache = list(model.generate_stream(ids2, max_new_tokens=6, temperature=0.0, prefix_cache=pc))
    fresh = list(model.generate_stream(ids2, max_new_tokens=6, temperature=0.0))
    assert pc.hits == 1 and with_cache == fresh


def test_load_blob_with_compressed_token_map():
    token_map = [i // 2 for i in range(97)]
    model = _model(3, token_map=token_map, engram_compressed_vocab_size=49)
    blob = {"model": mvt.state_for_ckpt(model), "config": model.cfg.to_dict(), "stage": "sft"}
    loaded, cfg = load_model_from_blob(blob)
    loaded.eval()
    assert isinstance(cfg, FlashConfig) and loaded.infer_candidate_pool is True
    ids = torch.randint(0, 97, (2, 24))
    assert torch.equal(loaded.hasher(ids), model.hasher(ids))
    torch.testing.assert_close(loaded(ids)[0], model(ids)[0])
