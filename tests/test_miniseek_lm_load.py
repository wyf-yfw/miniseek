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
