"""Old MiniSeek-M / miniseek-m checkpoints normalize to MiniSeek-V1."""

from algo.model.miniseek_v import FlashConfig
from algo.model.miniseek_v.config import normalize_flash_arch


def test_legacy_arch_normalizes_to_v():
    for arch in ("miniseek-m", "m", "v41", "f", "flash"):
        assert normalize_flash_arch(arch) == "miniseek-v"
    cfg = FlashConfig.from_dict({"arch": "miniseek-m", "model_family": "MiniSeek-M", "model_version": "0.1"})
    assert cfg.arch == "miniseek-v"
    assert cfg.model_family == "MiniSeek-V"
    assert cfg.model_version == "1"
    assert cfg.release_name == "MiniSeek-V1"
    d = cfg.to_dict()
    assert d["arch"] == "miniseek-v" and d["model_family"] == "MiniSeek-V"
