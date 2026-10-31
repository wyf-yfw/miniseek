from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api import tasks_store as ts


@pytest.fixture()
def client(tmp_path, monkeypatch):
    root = tmp_path / "tasks"
    monkeypatch.setattr(ts, "TASKS_ROOT", root)
    monkeypatch.setenv("MINISEEK_MONOLITH", "1")
    from api.app import app
    return TestClient(app)


def test_create_and_get_task(client):
    r = client.post(
        "/api/tasks",
        json={
            "name": "smoke",
            "arch": "miniseek-v",
            "mode": "pretrain",
            "hparams": {"max_steps": 2, "batch_size": 2},
        },
    )
    assert r.status_code == 200, r.text
    tid = r.json()["id"]
    g = client.get(f"/api/tasks/{tid}")
    assert g.status_code == 200
    assert g.json()["status"] == "draft"
    assert g.json()["arch"] == "miniseek-v"


def test_list_includes_created(client):
    r = client.post("/api/tasks", json={"name": "a", "arch": "miniseek-v", "mode": "pretrain", "hparams": {}})
    tid = r.json()["id"]
    rows = client.get("/api/tasks").json()["tasks"]
    assert any(x["id"] == tid for x in rows)


def test_old_train_endpoints_gone(client):
    assert client.post("/api/train/start", json={}).status_code == 410
    assert client.get("/api/train/processes").status_code == 410


def test_create_sft_requires_init_ckpt(client):
    r = client.post(
        "/api/tasks",
        json={
            "name": "sft-bad",
            "arch": "miniseek-v",
            "mode": "sft",
            "hparams": {"init_ckpt": "", "resume": ""},
        },
    )
    assert r.status_code == 400, r.text
    assert "Init Ckpt" in r.json()["detail"]


def test_create_sft_ok_with_init_ckpt(client):
    r = client.post(
        "/api/tasks",
        json={
            "name": "sft-ok",
            "arch": "miniseek-v",
            "mode": "sft",
            "hparams": {"init_ckpt": "/tmp/fake_pretrain.pt"},
        },
    )
    assert r.status_code == 200, r.text
    assert r.json()["hparams"]["init_ckpt"] == "/tmp/fake_pretrain.pt"


def test_patch_sft_clears_init_rejected(client):
    r = client.post(
        "/api/tasks",
        json={
            "name": "sft-patch",
            "arch": "miniseek-v",
            "mode": "sft",
            "hparams": {"init_ckpt": "/tmp/fake_pretrain.pt"},
        },
    )
    tid = r.json()["id"]
    bad = client.patch(
        f"/api/tasks/{tid}",
        json={"hparams": {"init_ckpt": "", "resume": ""}},
    )
    assert bad.status_code == 400, bad.text
    assert "Init Ckpt" in bad.json()["detail"]

