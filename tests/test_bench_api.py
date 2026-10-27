from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import web.bench_store as bs


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(bs, "RUNS_ROOT", tmp_path / "runs")
    monkeypatch.setenv("MINISEEK_MONOLITH", "1")
    ckpt = tmp_path / "a.pt"
    ckpt.write_bytes(b"x")
    from web_ui import app
    return TestClient(app), ckpt


def test_post_rejects_bad_task(client):
    c, ckpt = client
    r = c.post("/api/bench/tasks", json={"ckpt_path": str(ckpt), "tasks": ["mmlu"]})
    assert r.status_code == 400


def test_post_start_and_conflict(client):
    c, ckpt = client
    fake_proc = MagicMock()
    fake_proc.pid = 4242
    fake_proc.poll.return_value = None
    with patch("web.bench_runner.subprocess.Popen", return_value=fake_proc):
        with patch("web.bench_runner.pid_alive", return_value=True):
            with patch("web.bench_runner.choose_device", return_value="cpu"):
                r1 = c.post(
                    "/api/bench/tasks",
                    json={"ckpt_path": str(ckpt), "tasks": ["gsm8k"], "limit": 1},
                )
                assert r1.status_code == 200, r1.text
                rid = r1.json()["id"]
                s1 = c.post(f"/api/bench/tasks/{rid}/start")
                assert s1.status_code == 200, s1.text
                assert s1.json()["status"] == "running"
                r2 = c.post(
                    "/api/bench/tasks",
                    json={"ckpt_path": str(ckpt), "tasks": ["gsm8k"]},
                )
                assert r2.status_code == 200, r2.text
                rid2 = r2.json()["id"]
                s2 = c.post(f"/api/bench/tasks/{rid2}/start")
                assert s2.status_code == 409
