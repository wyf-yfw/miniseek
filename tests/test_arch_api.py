from __future__ import annotations

from pathlib import Path

import torch
from fastapi.testclient import TestClient

from model.config import ModelConfig
from model.dense import DenseLM
from web_ui import app


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
    assert len([b for b in body["blocks"] if b["kind"] == "layer"]) == 2
    assert body["total_params"] > 0
