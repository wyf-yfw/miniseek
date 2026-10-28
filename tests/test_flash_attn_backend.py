"""FlashAttention / SDPA backend selection smoke tests."""

from __future__ import annotations

import torch

from algo.model.attn_backend import (
    causal_attention,
    configure_attn_backend,
    flash_attn_func_or_none,
)
from algo.model.config import ModelConfig
from algo.model.dense import CausalSelfAttention, DenseLM


def test_config_use_flash_attn_default_on():
    cfg = ModelConfig()
    assert cfg.use_flash_attn is True
    tiny = ModelConfig.tiny()
    assert tiny.use_flash_attn is True


def test_configure_attn_backend_off_and_on():
    name_off = configure_attn_backend(False)
    assert "off" in name_off or "sdpa" in name_off
    name_on = configure_attn_backend(True)
    assert isinstance(name_on, str) and len(name_on) > 0
    # flash_attn package optional; either name is fine
    _ = flash_attn_func_or_none()


def test_causal_attention_cpu_flash_flag():
    configure_attn_backend(True)
    b, h, t, d = 2, 2, 8, 32
    q = torch.randn(b, h, t, d)
    k = torch.randn(b, h, t, d)
    v = torch.randn(b, h, t, d)
    y = causal_attention(q, k, v, head_dim=d, use_flash=True)
    assert y.shape == (b, h, t, d)
    assert torch.isfinite(y).all()


def test_causal_self_attention_respects_flag():
    cfg = ModelConfig.tiny()
    cfg = ModelConfig.from_dict({**cfg.to_dict(), "use_flash_attn": False})
    attn = CausalSelfAttention(cfg)
    assert attn.use_flash_attn is False
    x = torch.randn(2, 8, cfg.d_model)
    # rope cis buffer shape (max_seq, head_dim/2) complex
    from algo.model.dense import precompute_rope_cis

    cis = precompute_rope_cis(cfg.head_dim, cfg.max_seq_len)
    y = attn(x, cis)
    assert y.shape == x.shape


def test_dense_forward_with_flash_flag_on():
    configure_attn_backend(True)
    cfg = ModelConfig.tiny()
    cfg = ModelConfig.from_dict({**cfg.to_dict(), "use_flash_attn": True})
    model = DenseLM(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 8))
    logits, loss = model(ids, labels=ids)
    assert logits.shape == (2, 8, cfg.vocab_size)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
