from __future__ import annotations

from pathlib import Path

import torch
from fastapi.testclient import TestClient

from algo.model.config import ModelConfig
from algo.model.dense import DenseLM
from api.app import app


def test_arch_design_dense():
    client = TestClient(app)
    r = client.get("/api/arch/design", params={"arch": "dense"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["config"]["n_layers"] == 8
    assert body["meta"]["source"] == "design"
    assert len([b for b in body["blocks"] if b["kind"] == "layer"]) == 8


def test_arch_design_post_tiny():
    client = TestClient(app)
    r = client.post("/api/arch/design", json={"arch": "tiny"})
    assert r.status_code == 200
    assert r.json()["config"]["arch"] == "tiny"


def test_arch_design_bad():
    client = TestClient(app)
    assert client.get("/api/arch/design", params={"arch": "nope"}).status_code == 400


def test_arch_inspect_404():
    client = TestClient(app)
    r = client.post("/api/arch/inspect", json={"ckpt_path": "/no/such.pt"})
    assert r.status_code == 404


def test_arch_inspect_tiny(tmp_path: Path):
    cfg = ModelConfig.tiny()
    model = DenseLM(cfg)
    p = tmp_path / "tiny.pt"
    torch.save({"config": cfg.to_dict(), "model": model.state_dict(), "step": 3, "loss": 1.1}, p)
    client = TestClient(app)
    r = client.post("/api/arch/inspect", json={"ckpt_path": str(p)})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["config"]["n_layers"] == 2


def test_arch_design_v41_flash():
    client = TestClient(app)
    r = client.get("/api/arch/design", params={"arch": "m-ref"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["meta"]["source"] == "reference"
    assert body["meta"]["diagram"] == "fig3"
    assert body["fig3"]["encoder"]["n_layers"] == 20
    assert body["fig3"]["decoder"]["n_layers"] == 20
    assert any(f["id"] == "ced" for f in body["kv_flows"])
