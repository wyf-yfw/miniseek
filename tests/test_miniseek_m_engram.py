import torch

from algo.model.miniseek_m.config import FlashConfig
from algo.model.miniseek_m.engram import Engram, EngramLayout, NgramHasher, build_compressed_token_map
from tests.miniseek_m_ref import official as ref


def _layout(cfg):
    return EngramLayout.from_config(cfg)


def test_primes_and_layout_match_official():
    cfg = FlashConfig.tiny()
    ours = _layout(cfg)
    expect = ref.EngramLayout.build(
        cfg.engram_layer_ids, cfg.engram_max_ngram_size, cfg.engram_n_heads, cfg.engram_head_dim, cfg.engram_vocab_size
    )
    assert ours.primes == expect.primes
    assert ours.n_cols == 6
    assert ours.num_embeddings == tuple(sum(sum(o) for o in layer) for layer in expect.primes)


def test_hash_ids_match_official():
    cfg = FlashConfig.tiny()
    layout = _layout(cfg)
    hasher = NgramHasher(cfg, layout)
    torch.manual_seed(0)
    ids = torch.randint(0, cfg.vocab_size, (2, 20))
    expect = ref.ngram_hash(
        ids,
        ref.EngramLayout.build(cfg.engram_layer_ids, 4, cfg.engram_n_heads, cfg.engram_head_dim, cfg.engram_vocab_size),
        cfg.vocab_size,
        cfg.engram_pad_id,
    )
    assert torch.equal(hasher(ids), expect)


def test_doc_boundary_blocks_lookback():
    cfg = FlashConfig.tiny()
    hasher = NgramHasher(cfg, _layout(cfg))
    torch.manual_seed(1)
    ids = torch.randint(1, cfg.vocab_size, (1, 18))
    doc_ids = torch.tensor([[0] * 7 + [1] * 11])
    packed = hasher(ids, doc_ids)
    assert torch.equal(packed[:, 7:], hasher(ids[:, 7:]))
    assert torch.equal(packed[:, :7], hasher(ids[:, :7]))


def _toy_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {"[PAD]": 0, "The": 1, "the": 2, "THE": 3, "é": 4, "e": 5, "x": 6}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[PAD]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="[PAD]")


def test_compressed_map_merges_case_and_accents():
    tok = _toy_tokenizer()
    lookup, size = build_compressed_token_map(tok)
    assert (lookup, size) == ref.build_compressed_token_map(tok)
    assert lookup[1] == lookup[2] == lookup[3]
    assert lookup[4] == lookup[5]
    assert len({lookup[0], lookup[1], lookup[4], lookup[6]}) == 4


def _engram(seed=0):
    cfg = FlashConfig.tiny()
    layout = _layout(cfg)
    torch.manual_seed(seed)
    return cfg, layout, Engram(cfg, 1, layout)


def test_engram_forward_matches_official():
    cfg, layout, eng = _engram()
    with torch.no_grad():
        eng.wkv.weight.normal_(0, 0.3)
        eng.q_weight.uniform_(0.5, 1.5)
    x = torch.randn(2, 9, cfg.hc_mult, cfg.dim)
    hash_ids = torch.randint(0, layout.num_embeddings[0], (2, 9, layout.n_cols))
    expect = ref.engram_forward(
        x, hash_ids, eng.embed.weight, eng.wkv.weight, eng.q_weight, eng.k_weight, cfg.norm_eps, cfg.dim, cfg.hc_mult
    )
    torch.testing.assert_close(eng(x, hash_ids), expect, atol=1e-5, rtol=1e-5)


def test_engram_writes_and_trains_from_init():
    cfg, layout, eng = _engram(1)
    x = torch.randn(2, 9, cfg.hc_mult, cfg.dim)
    hash_ids = torch.randint(0, layout.num_embeddings[0], (2, 9, layout.n_cols))
    out = eng(x, hash_ids)
    assert not torch.equal(out, x)
    out.square().mean().backward()
    assert eng.embed.weight.grad.abs().sum() > 0 and eng.q_weight.grad.abs().sum() > 0


def test_hasher_rejects_wrong_compressed_size():
    cfg = FlashConfig.from_dict({**FlashConfig.tiny().to_dict(), "engram_compressed_vocab_size": 5})
    try:
        NgramHasher(cfg, _layout(cfg), token_map=list(range(cfg.vocab_size)))
    except ValueError:
        return
    raise AssertionError("应拒绝与配置不一致的压缩词表")
