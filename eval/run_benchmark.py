from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

from eval.bench_results import ALLOWED_TASKS, normalize_lm_eval_results


def simple_evaluate(**kwargs):
    from lm_eval import evaluator

    return evaluator.simple_evaluate(**kwargs)


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
    import eval.miniseek_lm  # noqa: F401  # register model

    raw = simple_evaluate(
        model="miniseek",
        model_args=f"ckpt={ckpt},device={device}",
        tasks=tasks,
        batch_size=batch_size,
        limit=limit,
        device=device,
    )
    if isinstance(raw, tuple):
        raw = raw[0]
    norm = normalize_lm_eval_results(raw, tasks)
    payload = {
        "ckpt_path": ckpt,
        "tasks": tasks,
        "metrics": norm["metrics"],
        "limit": limit,
        "device": device,
    }
    (out_dir / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
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
    print(
        f"[bench] start ckpt={args.ckpt} tasks={','.join(tasks)} device={args.device} limit={args.limit}",
        flush=True,
    )
    payload = run_bench(
        ckpt=args.ckpt,
        tasks=tasks,
        out_dir=args.out,
        device=args.device,
        limit=args.limit,
        batch_size=args.batch_size,
    )
    print(
        f"[bench] done metrics={json.dumps(payload['metrics'], ensure_ascii=False)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
