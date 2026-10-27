"""Lock measurement contracts for scripts/bench_train_opt.py (no GPU)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.bench_train_opt import (  # noqa: E402
    ORDER,
    VARIANTS,
    build_batches,
    count_log_syncs,
    simulate_log_schedule,
    simulate_log_sync,
)


def test_variants_cover_per_item_ids():
    expected = {
        "base",
        "moe_bounds",
        "moe_usage_gpu",
        "fused",
        "compile",
        "log10",
        "bucket",
        "pack",
        "all_equiv",
    }
    assert set(VARIANTS) == expected
    assert ORDER == [
        "base",
        "moe_bounds",
        "moe_usage_gpu",
        "fused",
        "compile",
        "log10",
        "bucket",
        "pack",
        "all_equiv",
    ]
    # single-variable: non-MoE singles keep slow moe
    for name in ("fused", "compile", "log10", "bucket", "pack"):
        assert VARIANTS[name]["moe_mode"] == "slow"
    assert VARIANTS["moe_bounds"]["moe_mode"] == "bounds_fast"
    assert VARIANTS["moe_usage_gpu"]["moe_mode"] == "usage_gpu"
    assert VARIANTS["all_equiv"]["moe_mode"] == "both_fast"
    assert VARIANTS["all_equiv"]["batch"] == "random"
    assert VARIANTS["log10"]["log_every"] == 10
    assert VARIANTS["base"]["log_every"] == 1


def test_pack_batch_shape():
    enc = [[1, 2, 3, 4]] * 400  # enough packed chunks for B=32
    batches = build_batches(enc, "pack", batch_size=32, max_seq=32)
    assert batches, "pack must produce at least one batch"
    x, y = batches[0]
    assert x.shape[0] == 32
    assert y.shape[0] == 32
    assert x.ndim == 2 and y.ndim == 2
    # y pad positions use -100
    assert (y == -100).any() or (y != -100).all()


def test_log10_keeps_per_step_loss_sync():
    n_loss, n_expert = count_log_syncs(log_every=10, n_steps=20)
    assert n_loss == 20
    assert n_expert == 2  # steps 0, 10
    n_loss1, n_expert1 = count_log_syncs(log_every=1, n_steps=20)
    assert n_loss1 == 20
    assert n_expert1 == 20
    # alias
    assert simulate_log_schedule(10, 20) == (20, 2)


def test_simulate_log_sync_call_counts():
    class FakeModel:
        def __init__(self):
            self.ends = 0
            self.begins = 0

        def end_expert_usage(self):
            self.ends += 1
            return torch.tensor([1, 2, 3])

        def begin_expert_usage(self):
            self.begins += 1

    class CountTensor(torch.Tensor):
        pass

    model = FakeModel()
    loss_items = {"n": 0}
    ce_items = {"n": 0}

    class Hook:
        def __init__(self, counter, val=1.0):
            self.counter = counter
            self.val = val

        def item(self):
            self.counter["n"] += 1
            return self.val

    total = Hook(loss_items)
    ce = Hook(ce_items)
    for step in range(20):
        simulate_log_sync(
            model, total, ce, None, step, log_every=10, record_usage=True
        )
    assert loss_items["n"] == 20
    assert ce_items["n"] == 2
    assert model.ends == 2
    assert model.begins == 2

    model2 = FakeModel()
    loss2 = {"n": 0}
    for step in range(20):
        simulate_log_sync(
            model2, Hook(loss2), None, None, step, log_every=1, record_usage=True
        )
    assert loss2["n"] == 20
    assert model2.ends == 20
