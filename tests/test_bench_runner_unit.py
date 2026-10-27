from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from api import bench_store as bs
from api import bench_runner as br


@pytest.fixture()
def runs_root(tmp_path, monkeypatch):
    root = tmp_path / "runs"
    monkeypatch.setattr(bs, "RUNS_ROOT", root)
    return root


def test_bench_spawn_start_new_session(runs_root, tmp_path, monkeypatch):
    if sys.platform == "win32":
        pytest.skip("POSIX session test")
    ckpt = tmp_path / "a.pt"
    ckpt.write_bytes(b"x")
    run = bs.create_run(ckpt_path=str(ckpt), tasks=["gsm8k"], limit=1)
    captured = {}

    def fake_popen(*args, **kwargs):
        captured.update(kwargs)
        m = MagicMock()
        m.pid = 88801
        m.poll.return_value = None
        return m

    monkeypatch.setattr(br.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(br, "choose_device", lambda **k: "cpu")
    monkeypatch.setattr(br, "pid_alive", lambda pid: True)
    out = br.start_run(run["id"], training_busy=False)
    assert out["status"] == "running"
    assert captured.get("start_new_session") is True
