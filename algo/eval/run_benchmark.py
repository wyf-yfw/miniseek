from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from algo.eval.bench_results import ALLOWED_TASKS, normalize_lm_eval_results


def simple_evaluate(**kwargs):
    from lm_eval import evaluator
    import datasets

    # cmmlu / ceval still ship dataset-loading scripts; datasets>=2.14 needs this.
    _orig_load = datasets.load_dataset

    def _load_dataset(*args, **kw):
        kw.setdefault("trust_remote_code", True)
        return _orig_load(*args, **kw)

    datasets.load_dataset = _load_dataset  # type: ignore[assignment]
    try:
        return evaluator.simple_evaluate(**kwargs)
    finally:
        datasets.load_dataset = _orig_load  # type: ignore[assignment]


def _write_progress(path: Optional[Path], data: Dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def run_bench(
    *,
    ckpt: str,
    tasks: List[str],
    out_dir: Path,
    device: str,
    limit: Optional[int],
    batch_size: int,
) -> dict:
    for t in tasks:
        if t not in ALLOWED_TASKS:
            raise ValueError(f"unsupported task: {t}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = Path(os.environ["BENCH_PROGRESS_PATH"]) if os.environ.get("BENCH_PROGRESS_PATH") else (out_dir / "progress.json")
    os.environ["BENCH_PROGRESS_PATH"] = str(progress_path)

    import algo.eval.miniseek_lm  # noqa: F401  # register model

    merged_metrics: Dict[str, Any] = {}
    suite_total = len(tasks)
    for i, task in enumerate(tasks):
        _write_progress(progress_path, {
            "suite_done": i,
            "suite_total": suite_total,
            "current_task": task,
            "sample_done": 0,
            "sample_total": 0,
        })
        print(f"[bench] suite {i+1}/{suite_total} task={task}", flush=True)
        raw = simple_evaluate(
            model="miniseek",
            model_args=f"ckpt={ckpt},device={device}",
            tasks=[task],
            batch_size=batch_size,
            limit=limit,
            device=device,
        )
        if isinstance(raw, tuple):
            raw = raw[0]
        norm = normalize_lm_eval_results(raw, [task])
        merged_metrics.update(norm.get("metrics") or {})
        _write_progress(progress_path, {
            "suite_done": i + 1,
            "suite_total": suite_total,
            "current_task": task,
            "sample_done": 0,
            "sample_total": 0,
        })
        print(f"[bench] task done {task} metrics={norm.get('metrics')}", flush=True)

    payload = {
        "ckpt_path": ckpt,
        "tasks": tasks,
        "metrics": merged_metrics,
        "limit": limit,
        "device": device,
    }
    (out_dir / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_progress(progress_path, {
        "suite_done": suite_total,
        "suite_total": suite_total,
        "current_task": None,
        "sample_done": 0,
        "sample_total": 0,
    })
    print(f"[bench] done metrics={payload.get('metrics')}", flush=True)
    return payload


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="MiniSeek lm-eval runner")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tasks", required=True, help="comma-separated")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--device", default="cpu")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=1)
    args = p.parse_args(argv)
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    print(f"[bench] start ckpt={args.ckpt} tasks={tasks} device={args.device} limit={args.limit}", flush=True)
    run_bench(
        ckpt=args.ckpt,
        tasks=tasks,
        out_dir=args.out,
        device=args.device,
        limit=args.limit,
        batch_size=args.batch_size,
    )
    return 0


if __name__ == "__main__":
    import os as _os

    code = main()
    # datasets/lm-eval may leave non-daemon threads; hard-exit so Web UI can reconcile.
    _os._exit(int(code))
