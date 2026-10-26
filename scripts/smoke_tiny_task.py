#!/usr/bin/env python3
"""End-to-end: create Tiny task → start → wait → assert ckpt + metrics."""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from web import tasks_store as ts
from web import task_runner as tr


def _p(*args, **kwargs) -> None:
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)


def main() -> int:
    data = ts.create_task(
        name="smoke-tiny",
        arch="tiny",
        mode="pretrain",
        hparams={
            "data_path": str(ROOT / "data" / "pretrain_t2t_mini.jsonl"),
            "batch_size": 2,
            "max_steps": 2,
            "max_seq_len": 64,
            "warmup_steps": 0,
            "epochs": 1,
            "log_every": 1,
            "plot_every": 1,
            "lr": 1e-4,
            "min_lr": 3e-5,
            "grad_clip": 1.0,
            "save_loss_thresholds": "",
            "stable_steps": 200,
            "save_every": 0,
        },
    )
    tid = data["id"]
    _p(f"[smoke] created {tid} dir={ts.task_dir(tid)}")
    started = tr.start_task(tid)
    _p(f"[smoke] started pid={started.get('pid')} status={started.get('status')}")
    deadline = time.time() + 120
    while time.time() < deadline:
        t = tr.reconcile_task(ts.load_task(tid))
        if t.get("status") != "running":
            _p(f"[smoke] ended status={t.get('status')} exit={t.get('exit_code')}")
            break
        time.sleep(0.5)
    else:
        _p("[smoke] TIMEOUT still running", file=sys.stderr)
        tr.stop_task(tid)
        return 1

    t = ts.load_task(tid)
    last = Path(t["out_dir"]) / "last.pt"
    metrics = ts.read_metrics(tid)
    ok = last.is_file() and len(metrics.get("loss") or []) >= 1
    _p(f"[smoke] last.pt={last.is_file()} loss_points={len(metrics.get('loss') or [])}")
    if not ok:
        log = (ts.task_dir(tid) / "train.log").read_text(encoding="utf-8", errors="replace")
        _p(log[-2000:], file=sys.stderr)
        return 1
    if t.get("status") == "failed":
        _p(f"[smoke] task failed: {t.get('error')}", file=sys.stderr)
        return 1
    _p("[smoke] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
