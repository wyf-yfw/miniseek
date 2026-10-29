# tests/test_arch_inspect.py
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from algo.model.miniseek_m import FlashConfig, Transformer
from api.arch_inspect import inspect_blob, inspect_path, load_ckpt_blob


def _tiny_blob():
    cfg = FlashConfig.tiny()
    model = Transformer(cfg)
    return {
        "config": cfg.to_dict(),
        "model": model.state_dict(),
        "step": 12,
        "loss": 2.5,
    }


def test_inspect_miniseek_m_fig3():
    blob = _tiny_blob()
    out = inspect_blob(blob, path="m.pt")
    assert out["config"]["arch"] == "miniseek-m"
    assert out["meta"]["diagram"] == "fig3"
    assert out["fig3"]["encoder"]["n_layers"] == 4


def test_inspect_path_missing():
    with pytest.raises(FileNotFoundError):
        inspect_path("/no/such/ckpt.pt")


def test_load_and_inspect_roundtrip(tmp_path: Path):
    blob = _tiny_blob()
    p = tmp_path / "t.pt"
    torch.save(blob, p)
    loaded = load_ckpt_blob(p)
    assert "model" in loaded and "config" in loaded
    out = inspect_path(p)
    assert out["meta"]["step"] == 12
    assert out["meta"]["diagram"] == "fig3"


def test_reject_dense_arch():
    blob = {"config": {"arch": "dense"}, "model": {"x": torch.zeros(1)}}
    with pytest.raises(ValueError, match="MiniSeek-M"):
        inspect_blob(blob)
