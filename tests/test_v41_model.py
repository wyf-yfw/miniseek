import torch

from algo.model import ModelConfig, build_model, config_from_dict
from algo.model.dense import DenseLM
from algo.model.v41 import FlashConfig, Transformer, bounded_replay_floor
from algo.model.v41.mhc import hc_pre


def _model(cfg=None, seed=0):
    torch.manual_seed(seed)
    return Transformer(cfg or FlashConfig.tiny())


def test_forward_backward_tiny():
    model = _model()
    ids = torch.randint(0, 97, (2, 40))
    logits, ce, extras = model(ids, labels=ids)
    assert logits.shape == (2, 40, 97) and logits.dtype == torch.float32
    assert torch.isfinite(ce) and extras["balance"] > 0 and torch.isfinite(extras["indexer_kl"])
    (ce + 1e-4 * extras["balance"] + extras["indexer_kl"]).backward()
    named = dict(model.named_parameters())
    for name in (
        "embed.weight",
        "head.weight",
        "layers.2.attn.compressor.wgate.weight",
        "layers.6.attn.indexer.wq_b.weight",
        "layers.4.attn.indexer.wk.weight",
        "layers.0.hc_attn_fn",
        "layers.0.hc_ffn_scale",
        "layers.1.engram.embed.weight",
        "layers.3.engram.wkv.weight",
        "layers.7.ffn.shared_experts.w2.weight",
        "layers.5.attn.attn_sink",
    ):
        assert named[name].grad is not None and named[name].grad.abs().sum() > 0, name


def test_ced_decoder_kv_from_encoder_output():
    model = _model(seed=1)
    cfg = model.cfg
    assert [layer.attn.compressor is not None for layer in model.layers[4:]] == [True, False, False, False]
    seen = {}
    model.layers[4].register_forward_pre_hook(lambda m, args: seen.setdefault("block_in", args))
    model.layers[4].attn.compressor.register_forward_pre_hook(lambda m, args: seen.setdefault("comp_in", args[0]))
    model(torch.randint(0, 97, (1, 30)))
    x, pre_mix = seen["block_in"][0], seen["block_in"][1]
    expect = model.layers[4].attn_norm(hc_pre(x, pre_mix))
    torch.testing.assert_close(seen["comp_in"], expect)
    assert cfg.layer_modes()[4] == "Full" and cfg.is_decoder(4)


def test_doc_isolation_full_model():
    cfg = FlashConfig.from_dict({**FlashConfig.tiny().to_dict(), "index_topk": 64, "candidate_topk_blocks": 16})
    model = _model(cfg, seed=2).eval()
    torch.manual_seed(3)
    ids = torch.randint(0, 97, (1, 40))
    doc_ids = torch.tensor([[0] * 16 + [1] * 24])
    packed, _, _ = model(ids, doc_ids=doc_ids)
    alone, _, _ = model(ids[:, 16:])
    torch.testing.assert_close(packed[:, 16:], alone, atol=1e-4, rtol=1e-4)


def test_swa_floor_only_affects_decoder():
    model = _model(seed=4).eval()
    seen = []
    model.layers[4].register_forward_pre_hook(lambda m, args: seen.append(args[0].clone()))
    ids = torch.randint(0, 97, (1, 30))
    floor = torch.zeros(1, 30, dtype=torch.long)
    floor[0, 20:] = 20
    a, _, _ = model(ids)
    b, _, _ = model(ids, swa_floor=floor)
    assert torch.equal(seen[0], seen[1])
    assert not torch.allclose(a[:, 20:], b[:, 20:])
    torch.testing.assert_close(a[:, :20], b[:, :20])


def test_replay_floor_from_labels():
    labels = torch.full((1, 12), -100)
    labels[0, 5:8] = 7  # 第一段回答：prompt 末位是 5 → P=6
    labels[0, 10:] = 7  # 第二段回答：P=11
    floor = bounded_replay_floor(labels, None, window=3)
    assert floor[0].tolist() == [0, 0, 0, 3, 3, 3, 0, 0, 8, 8, 8, 0]


def test_single_pass_shift():
    model = _model(seed=5).eval()
    seen = {}
    layer = model.layers[2]
    layer.attn.register_forward_pre_hook(lambda m, args: seen.setdefault("attn", []).append(args[0].clone()))
    layer.ffn.register_forward_pre_hook(lambda m, args: seen.setdefault("ffn", []).append(args[0].clone()))
    ids = torch.randint(0, 97, (1, 20))
    model(ids)
    with torch.no_grad():
        layer.hc_attn_fn.add_(torch.randn_like(layer.hc_attn_fn))
    model(ids)
    assert torch.equal(seen["attn"][0], seen["attn"][1])
    assert not torch.allclose(seen["ffn"][0], seen["ffn"][1])


def test_expert_bias_update_and_probe():
    model = _model(seed=6)
    model.train()
    model(torch.randint(0, 97, (2, 16)))
    model.update_expert_bias()
    assert any(layer.ffn.gate.bias.abs().sum() > 0 for layer in model.layers)
    counts, n_tok = model.probe_expert_load(torch.randint(0, 97, (2, 16)))
    assert n_tok == 32 and int(counts.sum()) == 32 * 2 * 8


def test_act_ckpt_matches_plain():
    model = _model(seed=7)
    model.train()
    ids = torch.randint(0, 97, (2, 24))
    _, ce, ex = model(ids, labels=ids)
    (ce + ex["indexer_kl"]).backward()
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad()
    model.act_ckpt = True
    _, ce2, ex2 = model(ids, labels=ids)
    (ce2 + ex2["indexer_kl"]).backward()
    torch.testing.assert_close(ce2, ce)
    for n, p in model.named_parameters():
        if n in grads:
            torch.testing.assert_close(p.grad, grads[n], atol=1e-5, rtol=1e-4, msg=n)


def test_f01_param_budget():
    with torch.device("meta"):
        model = Transformer(FlashConfig())
    n = model.num_params()
    print(f"MiniSeek-F0.1 params = {n:,}")
    assert n <= 200_000_000


def test_build_model_dispatch():
    assert isinstance(build_model(FlashConfig.tiny()), Transformer)
    assert isinstance(build_model(ModelConfig.tiny()), DenseLM)
    assert isinstance(config_from_dict(FlashConfig.tiny().to_dict()), FlashConfig)
    assert isinstance(config_from_dict(ModelConfig().to_dict()), ModelConfig)
