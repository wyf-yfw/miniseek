# tests/test_tasks_store.py
from __future__ import annotations

import json
from pathlib import Path

import pytest

from api import tasks_store as ts


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


def test_reject_duplicate_name(tasks_root):
    ts.create_task(name="same", arch="tiny", mode="pretrain", hparams={})
    try:
        ts.create_task(name="same", arch="tiny", mode="pretrain", hparams={})
        assert False, "expected ValueError"
    except ValueError as e:
        assert "同名" in str(e)


def test_allocate_unique_name(tasks_root):
    ts.create_task(name="tiny-pretrain", arch="tiny", mode="pretrain", hparams={})
    assert ts.allocate_unique_name("tiny-pretrain") == "tiny-pretrain-2"
    ts.create_task(name="tiny-pretrain-2", arch="tiny", mode="pretrain", hparams={})
    assert ts.allocate_unique_name("tiny-pretrain") == "tiny-pretrain-3"
