import torch
import torch.nn.functional as F

from algo.model.v41.mhc import hc_mixes, hc_post, hc_pre, hc_split_sinkhorn, identity_pre_mix
from tests.v41_ref import official as ref


def test_sinkhorn_matches_official():
    torch.manual_seed(0)
    mixes = torch.randn(3, 5, 24)
    scale = torch.rand(3)
    base = torch.randn(24)
    ours = hc_split_sinkhorn(mixes, scale, base, 4, 20, 1e-6)
    expect = ref.hc_split_sinkhorn(mixes, scale, base, 4, 20, 1e-6)
    for a, b in zip(ours, expect):
        torch.testing.assert_close(a, b)


def test_mixes_match_official():
    torch.manual_seed(1)
    x = torch.randn(2, 6, 4, 16)
    fn, scale, base = torch.randn(24, 64), torch.rand(3), torch.randn(24)
    ours = hc_mixes(x, fn, scale, base, 1e-20, 4, 20, 1e-6)
    expect = ref.hc_mixes(x, fn, scale, base, 1e-20, 4, 20, 1e-6)
    for a, b in zip(ours, expect):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)


def test_comb_doubly_stochastic():
    torch.manual_seed(2)
    # 20 轮只是近似收敛（mHC 论文 Fig.7），这里用常规尺度的 logits
    _, _, comb = hc_split_sinkhorn(torch.randn(100, 24), torch.ones(3), torch.zeros(24))
    torch.testing.assert_close(comb.sum(-2), torch.ones(100, 4), atol=1e-4, rtol=0)
    torch.testing.assert_close(comb.sum(-1), torch.ones(100, 4), atol=1e-2, rtol=0)
    assert torch.all(comb >= 0)


def test_pre_post_match_official():
    torch.manual_seed(3)
    x = torch.randn(2, 5, 4, 8)
    y = torch.randn(2, 5, 8)
    pre, post, comb = hc_split_sinkhorn(torch.randn(2, 5, 24), torch.rand(3), torch.randn(24))
    torch.testing.assert_close(hc_pre(x, pre), ref.hc_pre(x, pre))
    torch.testing.assert_close(hc_post(y, x, post, comb), ref.hc_post(y, x, post, comb), atol=1e-6, rtol=1e-6)


def _rms(x):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-20)


def test_zero_scale_is_prenorm():
    """scale=0、base=0：多流残差等价普通 Pre-Norm（4 份始终相同，RMSNorm 吸收尺度）。"""
    torch.manual_seed(4)
    d, hc = 16, 4
    h = torch.randn(2, 7, d)
    w1, w2 = torch.randn(d, d), torch.randn(d, d)
    f1 = lambda u: torch.tanh(F.linear(u, w1))  # noqa: E731
    f2 = lambda u: torch.tanh(F.linear(u, w2))  # noqa: E731

    plain = h + f1(_rms(h))
    plain = plain + f2(_rms(plain))
    plain = _rms(plain)

    fn = torch.randn(24, hc * d)
    zero_scale, zero_base = torch.zeros(3), torch.zeros(24)
    x = h.unsqueeze(2).repeat(1, 1, hc, 1)
    pre_mix = identity_pre_mix(x, hc)
    for f in (f1, f2):
        pre, post, comb = hc_mixes(x, fn, zero_scale, zero_base, 1e-20, hc, 20, 1e-6)
        y = f(_rms(hc_pre(x, pre_mix)))
        x = hc_post(y, x, post, comb)
        pre_mix = pre
    out = _rms(hc_pre(x, pre_mix))
    torch.testing.assert_close(out, plain, atol=1e-4, rtol=1e-4)
