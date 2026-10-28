"""Resume progress totals: this-run denominator; legacy absolute / inflate still parsed."""
from __future__ import annotations

from pathlib import Path

import api.app as app


def _write_log(task_dir: Path, body: str) -> None:
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "train.log").write_text(body, encoding="utf-8")


def test_legacy_sft_resume_total_is_start_plus_budget(tmp_path, monkeypatch):
    tid = "t-legacy-resume"
    root = tmp_path / "tasks"
    monkeypatch.setattr(app.ts, "task_dir", lambda task_id: root / task_id)
    app._train_totals_cache.clear()
    _write_log(
        root / tid,
        "\n".join(
            [
                "[sft] valid=/tmp/v.jsonl every=100 samples=512 max_samples=512",
                "[sft] batch_mode=pack samples=229126 steps/epoch≈2538 total_steps=2538 warmup=200",
                "[sft] resumed from /tmp/last.pt step=3578 best_loss=1.7",
                "epoch=0 step=4205 loss=1.57",
            ]
        )
        + "\n",
    )
    info = app._train_totals(tid)
    assert info["start_step"] == 3578
    assert info["total_steps"] == 3578 + 2538
    assert info["samples"] == 229126


def test_new_resume_this_run_from_zero(tmp_path, monkeypatch):
    tid = "t-new-resume-zero"
    root = tmp_path / "tasks"
    monkeypatch.setattr(app.ts, "task_dir", lambda task_id: root / task_id)
    app._train_totals_cache.clear()
    _write_log(
        root / tid,
        "\n".join(
            [
                "[train] resumed from /tmp/last.pt lifetime_step=106586 best_loss=2.6",
                "[train] max-steps=1000 为本轮步数，进度/LR 从 0→1000（ckpt lifetime 从 106586 起记）",
                "[train] samples=100 start_step=0 total_steps=1000 warmup=200 "
                "lr=0.001→1e-05 lifetime_base=106586",
                "epoch=0 step=1 loss=2.5 lr=5.00e-06",
            ]
        )
        + "\n",
    )
    info = app._train_totals(tid)
    assert info["start_step"] == 0
    assert info["total_steps"] == 1000


def test_old_absolute_resume_not_double_counted(tmp_path, monkeypatch):
    tid = "t-old-absolute"
    root = tmp_path / "tasks"
    monkeypatch.setattr(app.ts, "task_dir", lambda task_id: root / task_id)
    app._train_totals_cache.clear()
    _write_log(
        root / tid,
        "\n".join(
            [
                "[sft] resumed from /tmp/last.pt step=3578 best_loss=1.7",
                "[sft] batch_mode=pack samples=229126 steps/epoch≈2538 "
                "start_step=3578 total_steps=6116 warmup=200",
                "epoch=0 step=4205 loss=1.57",
            ]
        )
        + "\n",
    )
    info = app._train_totals(tid)
    assert info["start_step"] == 3578
    assert info["total_steps"] == 6116


def test_fresh_run_unchanged(tmp_path, monkeypatch):
    tid = "t-fresh"
    root = tmp_path / "tasks"
    monkeypatch.setattr(app.ts, "task_dir", lambda task_id: root / task_id)
    app._train_totals_cache.clear()
    _write_log(
        root / tid,
        "[sft] samples=1000 steps/epoch≈10 start_step=0 total_steps=10 warmup=1\n"
        "epoch=0 step=3 loss=2.0\n",
    )
    info = app._train_totals(tid)
    assert info["total_steps"] == 10
