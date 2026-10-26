from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from web import tasks_store as ts


@pytest.fixture()
def client(tmp_path, monkeypatch):
    root = tmp_path / "tasks"
    monkeypatch.setattr(ts, "TASKS_ROOT", root)
    from web_ui import app
    return TestClient(app)


def test_create_and_get_task(client):
    r = client.post(
        "/api/tasks",
        json={
            "name": "smoke",
            "arch": "tiny",
            "mode": "pretrain",
            "hparams": {"max_steps": 2, "batch_size": 2},
        },
    )
    assert r.status_code == 200, r.text
    tid = r.json()["id"]
    g = client.get(f"/api/tasks/{tid}")
    assert g.status_code == 200
    assert g.json()["status"] == "draft"
    assert g.json()["arch"] == "tiny"


def test_list_includes_created(client):
    r = client.post("/api/tasks", json={"name": "a", "arch": "tiny", "mode": "pretrain", "hparams": {}})
    tid = r.json()["id"]
    rows = client.get("/api/tasks").json()["tasks"]
    assert any(x["id"] == tid for x in rows)


def test_old_train_endpoints_gone(client):
    assert client.post("/api/train/start", json={}).status_code == 410
    assert client.get("/api/train/processes").status_code == 410
