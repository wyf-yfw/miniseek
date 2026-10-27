# tests/test_worker_app.py
from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from web import tasks_store as ts
from web import bench_store as bs


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(ts, "TASKS_ROOT", tmp_path / "tasks")
    monkeypatch.setattr(bs, "RUNS_ROOT", tmp_path / "runs")
    from web.worker_app import app
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"ok": True}


def test_task_start_stop_proxy(client, monkeypatch):
    t = ts.create_task(name="w", arch="tiny", mode="pretrain", hparams={"max_steps": 1})

    def fake_start(tid):
        task = ts.load_task(tid)
        task["status"] = "running"
        task["pid"] = 12345
        ts.save_task(task)
        return task

    def fake_stop(tid):
        task = ts.load_task(tid)
        task["status"] = "stopped"
        task["pid"] = None
        ts.save_task(task)
        return task

    monkeypatch.setattr("web.worker_app.start_task", fake_start)
    monkeypatch.setattr("web.worker_app.stop_task", fake_stop)
    r = client.post(f"/v1/tasks/{t['id']}/start")
    assert r.status_code == 200
    assert r.json()["status"] == "running"
    r2 = client.post(f"/v1/tasks/{t['id']}/stop")
    assert r2.status_code == 200
    assert r2.json()["status"] == "stopped"


def test_task_start_conflict_409(client, monkeypatch):
    t = ts.create_task(name="c", arch="tiny", mode="pretrain", hparams={})
    monkeypatch.setattr(
        "web.worker_app.start_task",
        lambda tid: (_ for _ in ()).throw(RuntimeError("another task is already running: x")),
    )
    r = client.post(f"/v1/tasks/{t['id']}/start")
    assert r.status_code == 409


def test_bench_start_passes_training_busy(client, tmp_path, monkeypatch):
    ckpt = tmp_path / "a.pt"
    ckpt.write_bytes(b"x")
    run = bs.create_run(ckpt_path=str(ckpt), tasks=["gsm8k"], limit=1)
    seen = {}

    def fake_start(rid, *, training_busy):
        seen["busy"] = training_busy
        run_d = bs.load_run(rid)
        run_d["status"] = "running"
        run_d["pid"] = 9
        bs.save_run(run_d)
        return run_d

    monkeypatch.setattr("web.worker_app.bench_start_run", fake_start)
    r = client.post(f"/v1/bench/{run['id']}/start", json={"training_busy": True})
    assert r.status_code == 200
    assert seen["busy"] is True
