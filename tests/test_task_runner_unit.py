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
    t = ts.create_task(name="x", arch="miniseek-v", mode="pretrain", hparams={"batch_size": 1})
    t["status"] = "running"
    t["pid"] = 999999  # almost surely dead
    ts.save_task(t)
    monkeypatch.setattr(tr, "pid_alive", lambda pid: False)
    out = tr.reconcile_task(ts.load_task(t["id"]))
    assert out["status"] in {"finished", "failed", "stopped"}
    assert out["pid"] is None


def test_start_refuses_second(tasks_root, monkeypatch):
    a = ts.create_task(name="a", arch="miniseek-v", mode="pretrain", hparams={"max_steps": 1})
    b = ts.create_task(name="b", arch="miniseek-v", mode="pretrain", hparams={"max_steps": 1})
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
    t = ts.create_task(name="killpg", arch="miniseek-v", mode="pretrain", hparams={"max_steps": 1})
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



def test_build_train_cmd_sft_m(tasks_root, tmp_path):
    """SFT + M builds a cmd with --arch miniseek-v, aux coef and post-train switches."""
    ckpt = tmp_path / "pretrain.pt"
    ckpt.write_bytes(b"x")
    t = ts.create_task(
        name="sft-m",
        arch="miniseek-v",
        mode="sft",
        hparams={
            "max_steps": 2,
            "batch_size": 2,
            "aux_coef": 0.02,
            "init_ckpt": str(ckpt),
            "candidate_pool": False,
            "act_ckpt": True,
        },
    )
    cmd = tr.build_train_cmd(ts.load_task(t["id"]))
    assert "algo.train_sft" in cmd
    assert cmd[cmd.index("--arch") + 1] == "miniseek-v"
    assert cmd[cmd.index("--aux-coef") + 1] == "0.02"
    assert cmd[cmd.index("--init-ckpt") + 1] == str(ckpt)
    assert "--no-candidate-pool" in cmd and "--act-ckpt" in cmd
    assert "--no-compile" in cmd  # M 默认不编译


def test_build_train_cmd_rejects_removed_moe(tasks_root):
    t = ts.create_task(name="old-moe", arch="moe", mode="pretrain", hparams={"max_steps": 2})
    try:
        tr.build_train_cmd(ts.load_task(t["id"]))
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_build_train_cmd_sft_requires_init_ckpt(tasks_root):
    t = ts.create_task(
        name="sft-no-init",
        arch="miniseek-v",
        mode="sft",
        hparams={"max_steps": 2, "init_ckpt": "", "resume": ""},
    )
    try:
        tr.build_train_cmd(ts.load_task(t["id"]))
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "Init Ckpt" in str(exc)


def test_build_train_cmd_sft_resume_skips_init(tasks_root, tmp_path):
    resume = tmp_path / "sft_last.pt"
    resume.write_bytes(b"x")
    t = ts.create_task(
        name="sft-resume",
        arch="miniseek-v",
        mode="sft",
        hparams={"max_steps": 2, "init_ckpt": "", "resume": str(resume)},
    )
    cmd = tr.build_train_cmd(ts.load_task(t["id"]))
    assert "--resume" in cmd
    assert "--init-ckpt" not in cmd


def test_build_train_cmd_speed_defaults(tasks_root):
    t = ts.create_task(
        name="spd",
        arch="miniseek-v",
        mode="pretrain",
        hparams={"max_steps": 2, "batch_size": 2},
    )
    cmd = tr.build_train_cmd(ts.load_task(t["id"]))
    # MiniSeek-V 默认不 compile（MoE/稀疏索引会 graph-break）
    assert "--no-compile" in cmd
    assert "--fused-adamw" in cmd
    assert "--batch-mode" in cmd
    assert cmd[cmd.index("--batch-mode") + 1] == "pack"


def test_build_train_cmd_speed_opt_out(tasks_root):
    t = ts.create_task(
        name="spd2",
        arch="miniseek-v",
        mode="pretrain",
        hparams={
            "max_steps": 2,
            "compile": False,
            "fused_adamw": False,
            "flash_attn": False,
            "batch_mode": "bucket",
        },
    )
    cmd = tr.build_train_cmd(ts.load_task(t["id"]))
    assert "--no-compile" in cmd
    assert "--no-fused-adamw" in cmd
    assert "--no-flash-attn" in cmd
    assert cmd[cmd.index("--batch-mode") + 1] == "bucket"


def test_build_train_cmd_sft_batch_mode(tasks_root, tmp_path):
    ckpt = tmp_path / "base.pt"
    ckpt.write_bytes(b"x")
    t = ts.create_task(
        name="sft-bm",
        arch="miniseek-v",
        mode="sft",
        hparams={
            "max_steps": 2,
            "init_ckpt": str(ckpt),
            "batch_mode": "bucket",
        },
    )
    cmd = tr.build_train_cmd(ts.load_task(t["id"]))
    assert "--batch-mode" in cmd
    assert cmd[cmd.index("--batch-mode") + 1] == "bucket"


def test_build_train_cmd_dspark_requires_backbone(tasks_root):
    t = ts.create_task(
        name="dspark-no-bb",
        arch="miniseek-v",
        mode="dspark",
        hparams={"max_steps": 100, "backbone": ""},
    )
    with pytest.raises(ValueError, match="Backbone Checkpoint"):
        tr.build_train_cmd(ts.load_task(t["id"]))


def test_build_train_cmd_dspark_success(tasks_root, tmp_path):
    bb = tmp_path / "backbone.pt"
    bb.write_bytes(b"x")
    t = ts.create_task(
        name="dspark-task",
        arch="miniseek-v",
        mode="dspark",
        hparams={
            "backbone": str(bb),
            "batch_size": 4,
            "max_seq_len": 256,
            "max_steps": 500,
            "anchors": 16,
            "calibrate": True,
            "sync_to_backbone": False,
        },
    )
    cmd = tr.build_train_cmd(ts.load_task(t["id"]))
    assert "algo.train_dspark" in cmd
    assert cmd[cmd.index("--backbone") + 1] == str(bb)
    assert cmd[cmd.index("--batch-size") + 1] == "4"
    assert cmd[cmd.index("--max-seq-len") + 1] == "256"
    assert cmd[cmd.index("--max-steps") + 1] == "500"
    assert cmd[cmd.index("--anchors") + 1] == "16"
    assert "--calibrate" in cmd
    assert "--no-sync-to-backbone" in cmd


def test_step_re_matches_dspark_log():
    line = "step=100 loss=2.8765 tv=0.3421 conf=0.7654 accept=0.650 ppl=17.75 lr=2.50e-04"
    m = tr.STEP_RE.search(line)
    assert m is not None
    assert m.group(1) == "100"
    assert m.group(2) == "2.8765"
    assert m.group(3) == "17.75"
    assert m.group(4) == "2.50e-04"

