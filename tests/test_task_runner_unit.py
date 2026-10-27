# tests/test_task_runner_unit.py
from __future__ import annotations

import pytest

from api import tasks_store as ts
from api import task_runner as tr


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


import subprocess
import sys
from unittest.mock import MagicMock, patch


def test_spawn_training_uses_new_session_on_posix(tasks_root, tmp_path, monkeypatch):
    if sys.platform == "win32":
        pytest.skip("POSIX session test")
    log = tmp_path / "t.log"
    captured = {}

    def fake_popen(*args, **kwargs):
        captured.update(kwargs)
        m = MagicMock()
        m.pid = 55501
        m.poll.return_value = None
        return m

    monkeypatch.setattr(tr.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(tr.os, "open", lambda *a, **k: 3)
    monkeypatch.setattr(tr.os, "close", lambda *a, **k: None)
    tr._spawn_training([sys.executable, "-c", "pass"], log)
    assert captured.get("start_new_session") is True


def test_stop_task_uses_killpg_when_session_leader(tasks_root, monkeypatch):
    if sys.platform == "win32":
        pytest.skip("POSIX killpg test")
    t = ts.create_task(name="killpg", arch="tiny", mode="pretrain", hparams={"max_steps": 1})
    t["status"] = "running"
    t["pid"] = 77701
    ts.save_task(t)

    class FakeProc:
        pid = 77701
        def poll(self):
            return 0
        def wait(self, timeout=None):
            return 0

    tr._owned_procs[t["id"]] = FakeProc()
    calls = []

    def fake_killpg(pgid, sig):
        calls.append((pgid, sig))

    monkeypatch.setattr(tr, "pid_alive", lambda pid: len(calls) == 0)
    monkeypatch.setattr(tr.os, "killpg", fake_killpg, raising=False)
    # Ensure module has killpg attribute for patch path used by implementation
    if not hasattr(tr.os, "killpg"):
        monkeypatch.setattr(tr.os, "killpg", fake_killpg)
    out = tr.stop_task(t["id"])
    assert out["status"] == "stopped"
    assert calls and calls[0][0] == 77701



def test_build_train_cmd_sft_moe_allowed(tasks_root):
    """SFT + MoE must build a cmd (no ValueError) and pass --arch moe."""
    t = ts.create_task(
        name="sft-moe",
        arch="moe",
        mode="sft",
        hparams={"max_steps": 2, "batch_size": 2, "aux_coef": 0.02},
    )
    cmd = tr.build_train_cmd(ts.load_task(t["id"]))
    assert "algo.train_sft" in cmd
    assert "--arch" in cmd
    assert cmd[cmd.index("--arch") + 1] == "moe"
    assert "--aux-coef" in cmd
    assert cmd[cmd.index("--aux-coef") + 1] == "0.02"


def test_build_train_cmd_speed_defaults(tasks_root):
    t = ts.create_task(
        name="spd",
        arch="moe",
        mode="pretrain",
        hparams={"max_steps": 2, "batch_size": 2},
    )
    cmd = tr.build_train_cmd(ts.load_task(t["id"]))
    assert "--compile" in cmd
    assert "--no-compile" not in cmd
    assert "--fused-adamw" in cmd
    assert "--batch-mode" in cmd
    assert cmd[cmd.index("--batch-mode") + 1] == "pack"


def test_build_train_cmd_speed_opt_out(tasks_root):
    t = ts.create_task(
        name="spd2",
        arch="dense",
        mode="pretrain",
        hparams={
            "max_steps": 2,
            "compile": False,
            "fused_adamw": False,
            "batch_mode": "bucket",
        },
    )
    cmd = tr.build_train_cmd(ts.load_task(t["id"]))
    assert "--no-compile" in cmd
    assert "--no-fused-adamw" in cmd
    assert cmd[cmd.index("--batch-mode") + 1] == "bucket"
