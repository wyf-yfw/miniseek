from __future__ import annotations

from pathlib import Path

import torch
from fastapi.testclient import TestClient

from algo.model.miniseek_v import FlashConfig, Transformer
from api.app import app


def test_arch_design_miniseek_m():
    client = TestClient(app)
    r = client.get("/api/arch/design", params={"arch": "miniseek-v"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["config"]["arch"] == "miniseek-v"
    assert body["meta"]["source"] == "design"
    assert body["meta"]["diagram"] == "fig3"


def test_arch_design_post_miniseek_m():
    client = TestClient(app)
    r = client.post("/api/arch/design", json={"arch": "miniseek-v"})
    assert r.status_code == 200
    assert r.json()["config"]["arch"] == "miniseek-v"


def test_arch_design_bad():
    client = TestClient(app)
    assert client.get("/api/arch/design", params={"arch": "nope"}).status_code == 400


def test_arch_inspect_404():
    client = TestClient(app)
    r = client.post("/api/arch/inspect", json={"ckpt_path": "/no/such.pt"})
    assert r.status_code == 404


def test_arch_inspect_miniseek_m(tmp_path: Path):
    cfg = FlashConfig.tiny()
    model = Transformer(cfg)
    p = tmp_path / "m.pt"
    torch.save({"config": cfg.to_dict(), "model": model.state_dict(), "step": 3, "loss": 1.1}, p)
    client = TestClient(app)
    r = client.post("/api/arch/inspect", json={"ckpt_path": str(p)})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["config"]["arch"] == "miniseek-v"
    assert body["meta"]["diagram"] == "fig3"


def test_arch_design_m_ref():
    client = TestClient(app)
    r = client.get("/api/arch/design", params={"arch": "m-ref"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["meta"]["source"] == "reference"
    assert body["meta"]["diagram"] == "fig3"
    assert body["fig3"]["encoder"]["n_layers"] == 20
    assert body["fig3"]["decoder"]["n_layers"] == 20
    assert any(f["id"] == "ced" for f in body["kv_flows"])
