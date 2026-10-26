# tests/test_tasks_store.py
from __future__ import annotations

import json
from pathlib import Path

import pytest

from web import tasks_store as ts


@pytest.fixture()
def tasks_root(tmp_path, monkeypatch):
    root = tmp_path / "tasks"
    monkeypatch.setattr(ts, "TASKS_ROOT", root)
    return root


def test_create_layout(tasks_root):
    data = ts.create_task(
        name="t1",
        arch="tiny",
        mode="pretrain",
        hparams={"batch_size": 2, "max_steps": 2},
    )
    d = ts.task_dir(data["id"])
    assert (d / "task.json").is_file()
    assert (d / "metrics.json").is_file()
    assert (d / "checkpoints").is_dir()
    assert data["status"] == "draft"
    assert data["out_dir"].endswith("checkpoints")
    loaded = ts.load_task(data["id"])
    assert loaded["name"] == "t1"


def test_list_and_delete(tasks_root):
    a = ts.create_task(name="a", arch="dense", mode="pretrain", hparams={})
    b = ts.create_task(name="b", arch="tiny", mode="pretrain", hparams={})
    rows = ts.list_tasks()
    assert {r["id"] for r in rows} == {a["id"], b["id"]}
    ts.delete_task(a["id"])
    assert not ts.task_dir(a["id"]).exists()
