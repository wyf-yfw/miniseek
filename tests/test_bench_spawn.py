from __future__ import annotations

import subprocess
import sys

from web import bench_store as bs
from web import bench_runner as br


def test_start_run_popen_uses_text_log_and_win32_flags(tmp_path, monkeypatch):
    monkeypatch.setattr(bs, "RUNS_ROOT", tmp_path / "runs")
    ckpt = tmp_path / "a.pt"
    ckpt.write_bytes(b"x")
    captured: dict = {}

    class FakeProc:
        pid = 77

        def poll(self):
            return None

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return FakeProc()

    monkeypatch.setattr(br.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(br.sys, "platform", "win32")
    monkeypatch.setattr(br.subprocess, "CREATE_NO_WINDOW", 0x08000000, raising=False)
    monkeypatch.setattr(br.subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200, raising=False)
    monkeypatch.setattr(br, "choose_device", lambda **kwargs: "cpu")
    run = br.start_run(
        ckpt_path=str(ckpt),
        tasks=["gsm8k"],
        limit=1,
        training_busy=False,
    )
    assert run["status"] == "running"
    assert run["pid"] == 77
    stdout = captured["kwargs"]["stdout"]
    assert stdout.encoding == "utf-8"
    assert stdout.errors == "replace"
    assert stdout.closed
    assert captured["kwargs"]["stderr"] is subprocess.STDOUT
    assert captured["kwargs"]["env"]["PYTHONUNBUFFERED"] == "1"
    cmd = captured["cmd"]
    assert "-u" in cmd
    assert "-m" in cmd
    assert "eval.run_benchmark" in cmd
    assert captured["kwargs"]["creationflags"] == 0x08000000 | 0x00000200
    br._owned_procs.pop(run["id"], None)
