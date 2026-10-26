from __future__ import annotations

from pathlib import Path

import torch

from model.config import ModelConfig
from model.dense import DenseLM
from eval.miniseek_lm import MiniSeekLM, load_miniseek_blob


def test_load_training_blob(tmp_path):
    cfg = ModelConfig.tiny()
    model = DenseLM(cfg)
    ckpt = tmp_path / "t.pt"
    torch.save({"config": cfg.to_dict(), "model": model.state_dict(), "step": 1}, ckpt)
    m, tok, cfg2 = load_miniseek_blob(ckpt)
    assert cfg2.n_layers == cfg.n_layers
    assert m is not None
    assert tok is not None


def test_loglikelihood_and_generate(tmp_path):
    cfg = ModelConfig.tiny()
    model = DenseLM(cfg)
    ckpt = tmp_path / "t.pt"
    torch.save({"config": cfg.to_dict(), "model": model.state_dict(), "step": 1}, ckpt)
    lm = MiniSeekLM(ckpt=str(ckpt), device="cpu")
    from lm_eval.api.instance import Instance

    req = Instance(
        request_type="loglikelihood",
        doc={},
        arguments=("Hello", " world"),
        idx=0,
    )
    out = lm.loglikelihood([req])
    assert len(out) == 1
    ll, is_greedy = out[0]
    assert isinstance(ll, float)
    assert isinstance(is_greedy, bool)

    gen_req = Instance(
        request_type="generate_until",
        doc={},
        arguments=("1+1=", {"until": ["\n"], "max_gen_toks": 8}),
        idx=0,
    )
    texts = lm.generate_until([gen_req])
    assert len(texts) == 1
    assert isinstance(texts[0], str)


def test_generate_until_slides_window_past_max_length(tmp_path):
    """Context is already max_length-1; appending without a window exceeds DenseLM."""
    cfg = ModelConfig.tiny()
    model = DenseLM(cfg)
    ckpt = tmp_path / "t.pt"
    torch.save({"config": cfg.to_dict(), "model": model.state_dict(), "step": 1}, ckpt)
    lm = MiniSeekLM(ckpt=str(ckpt), device="cpu")
    limit = 8
    lm._max_length = limit
    lm.model.cfg.max_seq_len = limit
    eos = lm.eot_token_id
    token = 3 if eos != 3 else 4
    ctx_len = limit - 1
    lm.tok_encode = lambda *args, **kwargs: list(range(1, ctx_len + 1))
    seen: list[int] = []
    decoded: list[list[int]] = []

    def fake_forward(input_ids, labels=None):
        length = int(input_ids.shape[-1])
        seen.append(length)
        if length > limit:
            raise ValueError(f"seq_len {length} > max_seq_len {limit}")
        logits = torch.zeros(1, length, cfg.vocab_size)
        logits[0, -1, token] = 5
        return logits, None

    def fake_decode(tokens, skip_special_tokens=True):
        decoded.append(list(tokens))
        return "x" * len(tokens)

    lm.model.forward = fake_forward
    lm.tok_decode = fake_decode
    from lm_eval.api.instance import Instance

    gen_req = Instance(
        request_type="generate_until",
        doc={},
        arguments=("prompt", {"until": [], "max_gen_toks": 4}),
        idx=0,
    )
    texts = lm.generate_until([gen_req], disable_tqdm=True)
    assert texts == ["xxxx"]
    assert seen
    assert max(seen) <= limit
    assert seen[-1] == limit
    assert decoded[-1] == [token, token, token, token]


def test_loglikelihood_does_not_feed_past_max_length(tmp_path):
    cfg = ModelConfig.tiny()
    model = DenseLM(cfg)
    ckpt = tmp_path / "t.pt"
    torch.save({"config": cfg.to_dict(), "model": model.state_dict(), "step": 1}, ckpt)
    lm = MiniSeekLM(ckpt=str(ckpt), device="cpu")
    limit = 8
    lm._max_length = limit
    lm.model.cfg.max_seq_len = limit
    seen: list[int] = []

    def fake_forward(input_ids, labels=None):
        length = int(input_ids.shape[-1])
        seen.append(length)
        if length > limit:
            raise ValueError(f"seq_len {length} > max_seq_len {limit}")
        logits = torch.zeros(1, length, cfg.vocab_size)
        return logits, None

    lm.model.forward = fake_forward
    context = [1] * 30
    continuation = [2, 3, 4, 5]
    out = lm._loglikelihood_tokens([(("c", "a"), context, continuation)], disable_tqdm=True)
    assert len(out) == 1
    assert seen == [limit]
