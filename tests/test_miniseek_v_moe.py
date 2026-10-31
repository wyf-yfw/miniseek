import torch

from algo.model.miniseek_v.config import FlashConfig
from algo.model.miniseek_v.moe import Gate, MoE, sequence_balance_loss
from tests.miniseek_v_ref import official as ref


def _moe(seed=0):
    torch.manual_seed(seed)
    moe = MoE(FlashConfig.tiny())
    with torch.no_grad():
        moe.gate.weight.normal_(0, 0.5)
        moe.gate.bias.normal_(0, 0.1)
    return moe


def test_gate_matches_official():
    moe = _moe()
    x = torch.randn(30, 32)
    w, idx, _ = moe.gate(x)
    ew, eidx = ref.gate_forward(x, moe.gate.weight, moe.gate.bias, 2, "sqrtsoftplus", 1.5)
    assert torch.equal(idx, eidx)
    torch.testing.assert_close(w, ew)


def test_moe_matches_official():
    moe = _moe(1)
    x = torch.randn(2, 9, 32)
    y, _ = moe(x)
    experts = [(e.w1.weight, e.w2.weight, e.w3.weight) for e in moe.experts]
    s = moe.shared_experts
    expect = ref.moe_forward(
        x, moe.gate.weight, moe.gate.bias, experts, (s.w1.weight, s.w2.weight, s.w3.weight), 2, 10.0, 1.5
    )
    torch.testing.assert_close(y, expect, atol=1e-5, rtol=1e-5)


def test_bias_only_changes_selection():
    gate = Gate(8, 4, 2, route_scale=1.5)
    with torch.no_grad():
        gate.bias.copy_(torch.tensor([0.0, 0.0, 0.0, 100.0]))
    w, idx, scores = gate(torch.randn(16, 8))
    assert torch.all((idx == 3).any(-1))
    raw = scores.gather(1, idx)
    torch.testing.assert_close(w, raw / raw.sum(-1, keepdim=True) * 1.5)


def test_balance_loss_hand_example():
    scores = torch.tensor([[[1.0, 3.0], [2.0, 2.0]]])  # s' = [[.25,.75],[.5,.5]] → P = [.375,.625]
    balanced = torch.tensor([[[0], [1]]])  # counts [1,1] → f = 2/(1*2)·[1,1] = [1,1]
    skewed = torch.tensor([[[0], [0]]])  # counts [2,0] → f = [2,0]
    torch.testing.assert_close(sequence_balance_loss(scores, balanced, 2), torch.tensor(1.0))
    torch.testing.assert_close(sequence_balance_loss(scores, skewed, 2), torch.tensor(0.75))


def test_bias_update_direction_and_accumulation():
    moe = _moe(2)
    moe.train()
    with torch.no_grad():
        moe.gate.bias.zero_()
    moe(torch.randn(1, 5, 32))
    first = moe._step_counts.clone()
    moe(torch.randn(1, 7, 32))
    total = moe._step_counts.clone()
    assert int(total.sum()) == (5 + 7) * 2 and torch.all(total >= first)
    moe.update_expert_bias(0.001)
    expected = total.float().sum() / moe.n_routed
    torch.testing.assert_close(moe.gate.bias, 0.001 * torch.sign(expected - total.float()))
    assert moe._step_counts is None


def test_balance_loss_gradient_flows_to_router():
    moe = _moe(3)
    _, bal = moe(torch.randn(2, 6, 32))
    bal.backward()
    assert moe.gate.weight.grad is not None and moe.gate.weight.grad.abs().sum() > 0
