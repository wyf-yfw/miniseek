from __future__ import annotations

import pytest

from web import bench_store as bs


def test_create_and_list(tmp_path, monkeypatch):
    monkeypatch.setattr(bs, "RUNS_ROOT", tmp_path / "runs")
    row = bs.create_run(
        ckpt_path="checkpoints/x.pt",
        tasks=["ceval-valid", "gsm8k"],
        limit=2,
        device="cpu",
    )
    assert row["status"] == "queued"
    assert row["tasks"] == ["ceval-valid", "gsm8k"]
    assert (tmp_path / "runs" / row["id"] / "run.json").is_file()
    assert bs.list_runs()[0]["id"] == row["id"]


def test_reject_unknown_task(tmp_path, monkeypatch):
    monkeypatch.setattr(bs, "RUNS_ROOT", tmp_path / "runs")
    with pytest.raises(ValueError, match="task"):
        bs.create_run(ckpt_path="x.pt", tasks=["mmlu"], limit=None, device="cpu")
