"""CPU smoke：forward + 可选 tiny backward；参数量落在合理区间."""

from __future__ import annotations

import torch

from algo.model.config import ModelConfig
from algo.model.dense import DenseLM, count_parameters
from algo.model.tokenizer import MINIMIND_VOCAB_SIZE, detect_vocab_size, load_tokenizer


def test_config_head_dim():
    cfg = ModelConfig()
    assert cfg.d_model == cfg.n_heads * cfg.head_dim


def test_tokenizer_vocab_default():
    tok = load_tokenizer(allow_fallback=True)
    vs = detect_vocab_size(tok)
    assert vs == MINIMIND_VOCAB_SIZE or vs > 1000


def test_forward_and_param_count():
    cfg = ModelConfig(vocab_size=MINIMIND_VOCAB_SIZE, max_seq_len=128)
    model = DenseLM(cfg)
    n = count_parameters(model)
    # MiniMind vocab=6400 + Dense 规格 ≈ 25–35M；若改用 32k 词表可到 ~58M
    assert 15_000_000 <= n <= 80_000_000, f"unexpected param count: {n}"

    b, t = 2, 16
    ids = torch.randint(0, cfg.vocab_size, (b, t))
    labels = torch.randint(0, cfg.vocab_size, (b, t))
    logits, loss = model(ids, labels=labels)
    assert logits.shape == (b, t, cfg.vocab_size)
    assert loss is not None and torch.isfinite(loss)

    loss.backward()
    grad_ok = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert grad_ok


def test_generate_runs():
    cfg = ModelConfig(vocab_size=MINIMIND_VOCAB_SIZE, max_seq_len=64)
    model = DenseLM(cfg)
    model.eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 4))
    out = model.generate(ids, max_new_tokens=3, temperature=1.0, top_k=10)
    assert out.shape[1] == 4 + 3
