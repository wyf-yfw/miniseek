# tests/test_worker_proxy_api.py
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api import tasks_store as ts
from api import bench_store as bs


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(ts, "TASKS_ROOT", tmp_path / "tasks")
    monkeypatch.setattr(bs, "RUNS_ROOT", tmp_path / "runs")
    monkeypatch.delenv("MINISEEK_MONOLITH", raising=False)
    monkeypatch.setenv("MINISEEK_WORKER_URL", "http://127.0.0.1:8001")
    # Force non-monolith path
    monkeypatch.setenv("MINISEEK_MONOLITH", "0")
    import importlib
    import api.app as web_ui
    importlib.reload(web_ui)
    return TestClient(web_ui.app)


def test_api_start_forwards_to_worker(client, monkeypatch):
    t = ts.create_task(name="p", arch="tiny", mode="pretrain", hparams={})
    monkeypatch.setattr(
        "api.app._worker_request",
        lambda method, path, **kw: {"id": t["id"], "status": "running", "pid": 42},
    )
    r = client.post(f"/api/tasks/{t['id']}/start")
    assert r.status_code == 200
    assert r.json()["status"] == "running"


def test_api_start_maps_worker_409(client, monkeypatch):
    t = ts.create_task(name="q", arch="tiny", mode="pretrain", hparams={})
    from fastapi import HTTPException

    def boom(method, path, **kw):
        raise HTTPException(status_code=409, detail="another task is already running: x")

    monkeypatch.setattr("api.app._worker_request", boom)
    r = client.post(f"/api/tasks/{t['id']}/start")
    assert r.status_code == 409


def test_api_stop_forwards_to_worker(client, monkeypatch):
    t = ts.create_task(name="s", arch="tiny", mode="pretrain", hparams={})
    monkeypatch.setattr(
        "api.app._worker_request",
        lambda method, path, **kw: {"id": t["id"], "status": "stopped", "pid": None},
    )
    r = client.post(f"/api/tasks/{t['id']}/stop")
    assert r.status_code == 200
    assert r.json()["status"] == "stopped"


def test_api_bench_start_forwards_with_training_busy(client, tmp_path, monkeypatch):
    ckpt = tmp_path / "a.pt"
    ckpt.write_bytes(b"x")
    run = bs.create_run(ckpt_path=str(ckpt), tasks=["gsm8k"], limit=1)
    seen = {}

    def fake(method, path, **kw):
        seen["method"] = method
        seen["path"] = path
        seen["json"] = kw.get("json")
        return {"id": run["id"], "status": "running", "pid": 9}

    monkeypatch.setattr("api.app._worker_request", fake)
    r = client.post(f"/api/bench/tasks/{run['id']}/start")
    assert r.status_code == 200
    assert r.json()["status"] == "running"
    assert seen["method"] == "POST"
    assert seen["path"] == f"/v1/bench/{run['id']}/start"
    assert seen["json"] == {"training_busy": False}
