from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.pt_clean_job import (
    cleanup_job_intermediates,
    dj_work_dir,
    preflight_disk,
    preview_samples,
    required_free_bytes,
    run_builtin,
    DiskSpaceError,
    _write_recipe,
)


def test_required_free_bytes_scales_and_floors():
    assert required_free_bytes(1 << 20) == 4 * (1 << 20)
    big = 12 * (1 << 30)
    assert required_free_bytes(big) == max(4 * big, 50 * (1 << 30))


def test_preview_and_builtin(tmp_path):
    src = tmp_path / "in.jsonl"
    lines = [
        {"text": "短"},  # length drop
        {"text": "这是一段足够长的中文预训练样本文本，用于通过长度与语言过滤。" * 2},
        {"text": "这是一段足够长的中文预训练样本文本，用于通过长度与语言过滤。" * 2},  # dedup
        {"text": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},  # lang/rep
        {"text": "这是一段足够长的中文预训练样本文本加微信免费领现金刷单赚钱测试。" * 2},  # sensitive
    ]
    src.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in lines) + "\n", encoding="utf-8")
    ops = {
        "length": {"enabled": True, "min_len": 50, "max_len": 100000},
        "language": {"enabled": True, "lang": "zh", "min_score": 0.3},
        "repetition": {"enabled": True, "rep_len": 10, "max_ratio": 0.5},
        "exact_dedup": {"enabled": True, "lowercase": False},
        "sensitive": {"enabled": True, "wordlist_path": ""},
    }
    prev = preview_samples(str(src), ops, limit=20)
    assert prev["kept"] >= 1
    assert prev["dropped"] >= 1

    job_dir = tmp_path / "job"
    job_dir.mkdir()
    job = {
        "id": "t1",
        "input_path": str(src),
        "operators": ops,
        "write_dropped": False,
    }
    (job_dir / "job.json").write_text(json.dumps(job), encoding="utf-8")
    report = run_builtin(job_dir, job)
    assert report["engine"] == "builtin"
    assert report["kept"] >= 1
    assert (job_dir / "clean.jsonl").is_file()
    assert not (job_dir / "dropped.jsonl").is_file()
    prog_path = job_dir / "progress.json"
    assert prog_path.is_file()
    prog = json.loads(prog_path.read_text(encoding="utf-8"))
    assert prog.get("phase") == "done"
    assert prog.get("pct") == 100.0
    assert prog.get("kept") == report["kept"]
    assert prog.get("input_rows") == report["input_rows"]


def test_builtin_can_write_dropped(tmp_path):
    src = tmp_path / "in.jsonl"
    src.write_text(
        json.dumps({"text": "这是一段足够长的中文预训练样本文本，用于通过长度过滤。" * 2}, ensure_ascii=False)
        + "\n"
        + json.dumps({"text": "短"}, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    job = {
        "id": "t2",
        "input_path": str(src),
        "operators": {
            "length": {"enabled": True, "min_len": 50, "max_len": 100000},
            "language": {"enabled": False},
            "repetition": {"enabled": False},
            "exact_dedup": {"enabled": False},
            "sensitive": {"enabled": False},
        },
        "write_dropped": True,
    }
    (job_dir / "job.json").write_text(json.dumps(job), encoding="utf-8")
    report = run_builtin(job_dir, job)
    assert report["dropped"] >= 1
    assert (job_dir / "dropped.jsonl").is_file()


def test_cleanup_removes_dj_work(tmp_path):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / "clean.jsonl").write_text("{}\n", encoding="utf-8")
    (job_dir / "report.json").write_text("{}", encoding="utf-8")
    (job_dir / "clean.log").write_text("ok\n", encoding="utf-8")
    work = dj_work_dir(job_dir)
    (work / "hf_datasets" / "x").mkdir(parents=True)
    (work / "hf_datasets" / "x" / "cache.arrow").write_bytes(b"abc")
    (job_dir / "stray.tmp").write_text("x", encoding="utf-8")
    removed = cleanup_job_intermediates(job_dir)
    assert not work.exists()
    assert (job_dir / "clean.jsonl").is_file()
    assert (job_dir / "report.json").is_file()
    assert "stray.tmp" in removed or not (job_dir / "stray.tmp").exists()


def test_recipe_disables_hf_cache(tmp_path):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    src = tmp_path / "in.jsonl"
    src.write_text("{}\n", encoding="utf-8")
    job = {
        "id": "r1",
        "input_path": str(src),
        "operators": {
            "length": {"enabled": True, "min_len": 1, "max_len": 10},
            "language": {"enabled": False},
            "repetition": {"enabled": False},
            "exact_dedup": {"enabled": False},
            "sensitive": {"enabled": False},
        },
    }
    recipe = _write_recipe(job_dir, job, job_dir / "clean.jsonl")
    text = recipe.read_text(encoding="utf-8")
    assert "use_cache: false" in text
    assert "ds_cache_dir:" in text
    assert ".dj_work" in text


def test_preflight_disk_raises(tmp_path, monkeypatch):
    src = tmp_path / "in.jsonl"
    src.write_bytes(b"x" * 1000)
    job_dir = tmp_path / "job"
    job_dir.mkdir()

    import scripts.pt_clean_job as mod

    monkeypatch.setattr(mod, "free_bytes_for", lambda _p: 10)
    monkeypatch.setattr(mod, "required_free_bytes", lambda _n: 1000)
    with pytest.raises(DiskSpaceError):
        preflight_disk(src, anchor=job_dir)
