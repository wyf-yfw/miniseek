import pytest
import torch

from algo.model.miniseek_m.config import FlashConfig
from algo.model.miniseek_m.rope import apply_rotary_emb, layer_freqs, precompute_freqs_cis, rotate_tail
from tests.miniseek_m_ref import official as ref


@pytest.mark.parametrize("original_seq_len", [0, 64])
def test_freqs_match_official(original_seq_len):
    args = (32, 128, original_seq_len, 160000.0, 16.0, 32, 1)
    torch.testing.assert_close(precompute_freqs_cis(*args), ref.precompute_freqs_cis(*args))


@pytest.mark.parametrize("shape", [(2, 5, 16), (2, 5, 3, 16)])
@pytest.mark.parametrize("inverse", [False, True])
def test_rotate_matches_official(shape, inverse):
    torch.manual_seed(0)
    x = torch.randn(*shape)
    freqs = ref.precompute_freqs_cis(16, 5, 0, 10000.0, 1.0, 32, 1)
    expect = ref.apply_rotary_emb_(x.clone(), freqs, inverse)
    torch.testing.assert_close(apply_rotary_emb(x, freqs, inverse), expect)


def test_inverse_rotation_roundtrip():
    torch.manual_seed(0)
    x = torch.randn(2, 6, 4, 24)
    freqs = precompute_freqs_cis(8, 6, 0, 10000.0, 1.0, 32, 1)
    y = rotate_tail(rotate_tail(x, freqs, 8), freqs, 8, inverse=True)
    torch.testing.assert_close(y, x, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(rotate_tail(x, freqs, 8)[..., :16], x[..., :16])


def test_layer_freqs_theta_by_mode():
    cfg = FlashConfig(original_seq_len=0)
    swa = layer_freqs(cfg, 0, 4)
    csa = layer_freqs(cfg, 2, 4)
    torch.testing.assert_close(swa, precompute_freqs_cis(32, 4, 0, 10000.0, 16.0, 32, 1))
    torch.testing.assert_close(csa, precompute_freqs_cis(32, 4, 0, 160000.0, 16.0, 32, 1))


def test_flash_config_roundtrip():
    cfg = FlashConfig()
    again = FlashConfig.from_dict(cfg.to_dict())
    assert again == cfg
    assert isinstance(again.compress_ratios, tuple)


def test_f01_defaults():
    cfg = FlashConfig()
    assert cfg.release_name == "MiniSeek-M0.1"
    assert cfg.layer_modes() == ["SWA", "SWA", "Full", "Reuse", "Full", "Reuse", "Reindex", "Reuse"]
    assert [cfg.kv_source_for(i) for i in range(8)] == [None, None, 2, 2, 4, 4, 4, 4]
    assert [cfg.is_decoder(i) for i in range(8)] == [False] * 4 + [True] * 4
    assert (cfg.dim, cfg.n_heads, cfg.head_dim, cfg.window_size, cfg.index_topk) == (512, 16, 128, 128, 64)
    assert (cfg.n_routed_experts, cfg.n_activated_experts, cfg.moe_inter_dim) == (64, 6, 176)
    assert cfg.engram_layer_ids == (1, 3) and cfg.tie_word_embeddings is False


def test_tiny_keeps_layer_pattern():
    assert FlashConfig.tiny().layer_modes() == FlashConfig().layer_modes()


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        FlashConfig(kv_source_layers=(3, 4))
