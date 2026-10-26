# tests/test_task_runner_unit.py
from __future__ import annotations

import pytest

from web import tasks_store as ts
from web import task_runner as tr


@pytest.fixture()
def tasks_root(tmp_path, monkeypatch):
    root = tmp_path / "tasks"
    monkeypatch.setattr(ts, "TASKS_ROOT", root)
    return root


def test_reconcile_dead_pid(tasks_root, monkeypatch):
    t = ts.create_task(name="x", arch="tiny", mode="pretrain", hparams={"batch_size": 1})
    t["status"] = "running"
    t["pid"] = 999999  # almost surely dead
    ts.save_task(t)
    monkeypatch.setattr(tr, "pid_alive", lambda pid: False)
    out = tr.reconcile_task(ts.load_task(t["id"]))
    assert out["status"] in {"finished", "failed", "stopped"}
    assert out["pid"] is None


def test_start_refuses_second(tasks_root, monkeypatch):
    a = ts.create_task(name="a", arch="tiny", mode="pretrain", hparams={"max_steps": 1})
    b = ts.create_task(name="b", arch="tiny", mode="pretrain", hparams={"max_steps": 1})
    a["status"] = "running"
    a["pid"] = 1
    ts.save_task(a)
    monkeypatch.setattr(tr, "pid_alive", lambda pid: True)
    monkeypatch.setattr(tr, "_spawn_training", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not spawn")))
    with pytest.raises(RuntimeError, match="running"):
        tr.start_task(b["id"])


def test_pid_alive_false_when_owned_exited(tasks_root):
    class FakeProc:
        pid = 424242

        def poll(self):
            return 0

    tr._owned_procs["owned-exit"] = FakeProc()
    try:
        assert tr.pid_alive(424242) is False
    finally:
        tr._owned_procs.pop("owned-exit", None)


def test_pid_alive_true_when_owned_running(tasks_root, monkeypatch):
    class FakeProc:
        pid = 434343

        def poll(self):
            return None

    tr._owned_procs["owned-run"] = FakeProc()
    try:
        # Avoid /proc and kill for this fake pid
        monkeypatch.setattr(tr.os, "kill", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no kill")))
        assert tr.pid_alive(434343) is True
    finally:
        tr._owned_procs.pop("owned-run", None)
