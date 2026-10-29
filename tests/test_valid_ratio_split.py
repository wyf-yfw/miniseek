"""Deterministic train/valid split by ratio (no disk rewrite)."""
from __future__ import annotations

from algo.train_pretrain import split_samples_by_ratio
from api import tasks_store as ts
from api import task_runner as tr


def test_split_deterministic_and_disjoint():
    samples = [f"row-{i}" for i in range(1000)]
    tr1, va1 = split_samples_by_ratio(samples, 0.01, 42)
    tr2, va2 = split_samples_by_ratio(samples, 0.01, 42)
    assert tr1 == tr2 and va1 == va2
    assert set(tr1).isdisjoint(set(va1))
    assert len(tr1) + len(va1) == 1000
    assert abs(len(va1) / 1000 - 0.01) < 0.005
    # different seed -> different split (almost surely)
    tr3, va3 = split_samples_by_ratio(samples, 0.01, 99)
    assert va3 != va1


def test_split_ratio_zero_returns_all_train():
    samples = ["a", "b", "c"]
    tr, va = split_samples_by_ratio(samples, 0.0, 42)
    assert tr == samples
    assert va == []


def test_build_train_cmd_valid_ratio(tasks_root=None, tmp_path=None, monkeypatch=None):
    # fixture-style via pytest below
    pass


def test_build_cmd_prefers_ratio(tmp_path, monkeypatch):
    root = tmp_path / "tasks"
    monkeypatch.setattr(ts, "TASKS_ROOT", root)
    t = ts.create_task(
        name="ratio-cmd",
        arch="tiny",
        mode="pretrain",
        hparams={
            "batch_size": 2,
            "max_steps": 4,
            "valid_path": "data/valid_mix.jsonl",
            "valid_ratio": 0.01,
            "valid_seed": 7,
            "valid_every": 20,
            "valid_max_samples": 64,
        },
    )
    cmd = tr.build_train_cmd(t)
    assert "--valid-ratio" in cmd
    assert cmd[cmd.index("--valid-ratio") + 1] == "0.01"
    assert cmd[cmd.index("--valid-seed") + 1] == "7"
    assert "--valid" not in cmd  # 独立 valid 文件不再传入


def test_build_cmd_ignores_path_when_ratio_zero(tmp_path, monkeypatch):
    root = tmp_path / "tasks"
    monkeypatch.setattr(ts, "TASKS_ROOT", root)
    t = ts.create_task(
        name="path-cmd",
        arch="tiny",
        mode="pretrain",
        hparams={
            "valid_path": "data/valid_mix.jsonl",
            "valid_ratio": 0,
            "valid_every": 20,
            "valid_max_samples": 64,
        },
    )
    cmd = tr.build_train_cmd(t)
    assert "--valid" not in cmd
    assert "--valid-ratio" not in cmd


def test_build_cmd_sft_uses_ratio(tmp_path, monkeypatch):
    root = tmp_path / "tasks"
    monkeypatch.setattr(ts, "TASKS_ROOT", root)
    ckpt = tmp_path / "last.pt"
    ckpt.write_bytes(b"x")
    t = ts.create_task(
        name="sft-ratio",
        arch="tiny",
        mode="sft",
        hparams={
            "valid_path": "data/valid_mix.jsonl",
            "valid_ratio": 0.02,
            "valid_seed": 3,
            "valid_every": 10,
            "init_ckpt": str(ckpt),
        },
    )
    cmd = tr.build_train_cmd(t)
    assert "--valid-ratio" in cmd
    assert cmd[cmd.index("--valid-ratio") + 1] == "0.02"
    assert "--valid" not in cmd
