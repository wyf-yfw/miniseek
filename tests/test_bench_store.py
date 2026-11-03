from __future__ import annotations

import pytest

from api import bench_store as bs


def test_create_and_list(tmp_path, monkeypatch):
    monkeypatch.setattr(bs, "RUNS_ROOT", tmp_path / "runs")
    row = bs.create_run(
        ckpt_path="tasks/demo/checkpoints/x.pt",
        tasks=["ceval-valid", "gsm8k"],
        limit=2,
        device="cpu",
    )
    assert row["status"] == "draft"
    assert row["tasks"] == ["ceval-valid", "gsm8k"]
    assert (tmp_path / "runs" / row["id"] / "run.json").is_file()
    assert bs.list_runs()[0]["id"] == row["id"]


def test_reject_unknown_task(tmp_path, monkeypatch):
    monkeypatch.setattr(bs, "RUNS_ROOT", tmp_path / "runs")
    with pytest.raises(ValueError, match="task"):
        bs.create_run(ckpt_path="x.pt", tasks=["mmlu"], limit=None, device="cpu")


def test_create_default_is_tiny_suite(tmp_path, monkeypatch):
    monkeypatch.setattr(bs, "RUNS_ROOT", tmp_path / "runs")
    row = bs.create_run(ckpt_path="x.pt", device="cpu")
    assert row["tasks"][0] == "tinyMMLU"
    assert "tinyGSM8k" in row["tasks"]
    assert "ceval-valid" not in row["tasks"]


def test_accept_tiny_tasks(tmp_path, monkeypatch):
    monkeypatch.setattr(bs, "RUNS_ROOT", tmp_path / "runs")
    row = bs.create_run(
        ckpt_path="x.pt",
        tasks=["tinyMMLU", "tinyArc"],
        limit=None,
        device="cpu",
    )
    assert row["tasks"] == ["tinyMMLU", "tinyArc"]
