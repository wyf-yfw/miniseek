# tests/test_task_runner_unit.py
from __future__ import annotations

import subprocess
import sys

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


def test_spawn_training_uses_text_log_and_win32_flags(tmp_path, monkeypatch):
    log_path = tmp_path / "train.log"
    captured: dict = {}

    class FakeProc:
        pid = 7

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return FakeProc()

    monkeypatch.setattr(tr.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(tr.sys, "platform", "win32")
    monkeypatch.setattr(tr.subprocess, "CREATE_NO_WINDOW", 0x08000000, raising=False)
    monkeypatch.setattr(tr.subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200, raising=False)
    proc = tr._spawn_training([sys.executable, "-u", "-c", "print(1)"], log_path)
    assert proc.pid == 7
    stdout = captured["kwargs"]["stdout"]
    assert stdout.encoding == "utf-8"
    assert stdout.errors == "replace"
    assert stdout.closed
    assert captured["kwargs"]["stderr"] is subprocess.STDOUT
    assert captured["kwargs"]["env"]["PYTHONUNBUFFERED"] == "1"
    assert "-u" in captured["cmd"]
    assert captured["kwargs"]["creationflags"] == 0x08000000 | 0x00000200


def test_spawn_training_child_writes_log(tmp_path):
    log_path = tmp_path / "nested" / "train.log"
    proc = tr._spawn_training(
        [sys.executable, "-u", "-c", "print('spawn-ok', flush=True)"],
        log_path,
    )
    assert proc.wait(timeout=20) == 0
    assert "spawn-ok" in log_path.read_text(encoding="utf-8")


def test_pid_alive_win32_does_not_kill(monkeypatch):
    """os.kill(pid, 0) on Windows is TerminateProcess. Liveness checks must not use it."""
    killed: list = []
    monkeypatch.setattr(tr.sys, "platform", "win32")
    monkeypatch.setattr(tr.os, "kill", lambda *a, **k: killed.append(a))
    if hasattr(tr, "_win_pid_alive"):
        monkeypatch.setattr(tr, "_win_pid_alive", lambda pid: pid == 51515)
        assert tr.pid_alive(51515) is True
        assert tr.pid_alive(51516) is False
    else:
        tr.pid_alive(51515)
    assert killed == []
