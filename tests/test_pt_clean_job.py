from __future__ import annotations

import json
from pathlib import Path

from scripts.pt_clean_job import preview_samples, run_builtin


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
    }
    (job_dir / "job.json").write_text(json.dumps(job), encoding="utf-8")
    report = run_builtin(job_dir, job)
    assert report["engine"] == "builtin"
    assert report["kept"] >= 1
    assert (job_dir / "clean.jsonl").is_file()
    assert (job_dir / "dropped.jsonl").is_file()
