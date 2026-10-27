# tests/test_valid_metrics.py
from __future__ import annotations

import math

import pytest
import torch

from algo.model import build_model
from algo.model.config import ModelConfig
from algo.train_pretrain import evaluate_valid
from api import tasks_store as ts
from api import task_runner as tr


def _tiny_cfg(vocab: int = 128, max_seq_len: int = 32) -> ModelConfig:
    base = ModelConfig.tiny().to_dict()
    base.update(
        {
            "vocab_size": vocab,
            "max_seq_len": max_seq_len,
        }
    )
    return ModelConfig.from_dict(base)


def test_evaluate_valid_returns_finite():
    cfg = _tiny_cfg()
    model = build_model(cfg).eval()
    B, T, V = 2, 16, cfg.vocab_size
    x = torch.randint(0, V, (B, T))
    y = x.clone()

    class Loader:
        def __iter__(self):
            yield x, y

    out = evaluate_valid(model, Loader(), torch.device("cpu"))
    assert out["tokens"] > 0
    assert out["loss"] == out["loss"]  # not NaN
    assert math.isfinite(out["loss"])
    assert out["ppl"] >= 1.0
    assert math.isfinite(out["ppl"])


@pytest.fixture()
def tasks_root(tmp_path, monkeypatch):
    root = tmp_path / "tasks"
    monkeypatch.setattr(ts, "TASKS_ROOT", root)
    return root


def test_parse_valid_line_appends(tasks_root):
    tid = "t-valid"
    (tasks_root / tid).mkdir(parents=True)
    ts.write_metrics(tid, {"loss": [], "valid_loss": [], "expert_load": None})
    state = {"loss_bucket": 0}
    tr._parse_and_append_metrics(
        tid,
        "valid step=500 loss=3.2100 ppl=24.78 tokens=102400",
        plot_every=500,
        state=state,
    )
    m = ts.read_metrics(tid)
    assert m["valid_loss"][-1]["step"] == 500
    assert abs(m["valid_loss"][-1]["loss"] - 3.21) < 1e-6
    assert abs(m["valid_loss"][-1]["ppl"] - 24.78) < 1e-6
    assert m["valid_loss"][-1].get("tokens") == 102400


def test_build_train_cmd_includes_valid_flags(tasks_root):
    t = ts.create_task(
        name="valid-cmd",
        arch="tiny",
        mode="pretrain",
        hparams={
            "batch_size": 2,
            "max_steps": 4,
            "valid_path": "data/valid_mix.jsonl",
            "valid_every": 20,
            "valid_max_samples": 64,
        },
    )
    cmd = tr.build_train_cmd(t)
    assert "--valid" in cmd
    i = cmd.index("--valid")
    assert cmd[i + 1] == "data/valid_mix.jsonl"
    assert "--valid-every" in cmd
    assert cmd[cmd.index("--valid-every") + 1] == "20"
    assert "--valid-max-samples" in cmd
    assert cmd[cmd.index("--valid-max-samples") + 1] == "64"


def test_build_train_cmd_skips_valid_when_every_zero(tasks_root):
    t = ts.create_task(
        name="valid-off",
        arch="tiny",
        mode="pretrain",
        hparams={"valid_path": "data/valid_mix.jsonl", "valid_every": 0},
    )
    cmd = tr.build_train_cmd(t)
    assert "--valid" not in cmd
    assert "--valid-every" not in cmd
