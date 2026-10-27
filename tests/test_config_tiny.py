# tests/test_config_tiny.py
from __future__ import annotations

import torch

from algo.model import build_model
from algo.model.config import ModelConfig
from algo.model.dense import DenseLM, count_parameters
from algo.model.tokenizer import MINIMIND_VOCAB_SIZE


def test_tiny_preset_dims():
    cfg = ModelConfig.tiny()
    assert cfg.arch == "tiny"
    assert cfg.d_model == 64
    assert cfg.n_layers == 2
    assert cfg.n_heads == 1
    assert cfg.head_dim == 64
    assert cfg.intermediate_size == 256
    assert cfg.vocab_size == MINIMIND_VOCAB_SIZE
    assert cfg.d_model == cfg.n_heads * cfg.head_dim


def test_tiny_build_and_forward():
    cfg = ModelConfig.tiny().with_vocab_size(MINIMIND_VOCAB_SIZE)
    model = build_model(cfg)
    assert isinstance(model, DenseLM)
    n = count_parameters(model)
    assert 100_000 <= n <= 2_000_000, n
    ids = torch.randint(0, cfg.vocab_size, (2, 8))
    logits, loss = model(ids, labels=ids)
    assert logits.shape == (2, 8, cfg.vocab_size)
    assert loss is not None and torch.isfinite(loss)
