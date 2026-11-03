from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# tiny* ≈100 题/场景，适合小模型日常门禁；full 仍保留原 C-Eval/CMMLU/GSM8K。
TINY_TASKS: Tuple[str, ...] = (
    "tinyMMLU",
    "tinyHellaswag",
    "tinyArc",
    "tinyWinogrande",
    "tinyTruthfulQA",
    "tinyGSM8k",
)
FULL_TASKS: Tuple[str, ...] = ("ceval-valid", "cmmlu", "gsm8k")
ALLOWED_TASKS: Tuple[str, ...] = TINY_TASKS + FULL_TASKS
DEFAULT_TASKS: Tuple[str, ...] = TINY_TASKS

BENCH_PRESETS: Dict[str, Dict[str, Any]] = {
    "tiny": {
        "id": "tiny",
        "label": "tiny（推荐）",
        "hint": "tinyBenchmarks：每场景约 100 题，估大榜分数，适合小模型日常评测。",
        "tasks": list(TINY_TASKS),
    },
    "full": {
        "id": "full",
        "label": "完整（慢）",
        "hint": "C-Eval valid + CMMLU + GSM8K 全量；题难、耗时长，发版深测再用。",
        "tasks": list(FULL_TASKS),
    },
}

_ACC_KEYS = (
    "acc_norm,none",
    "acc_norm",
    "acc,none",
    "acc",
    "exact_match,flexible-extract",
    "exact_match,strict-match",
    "exact_match",
)
_ERR_KEYS = (
    "acc_norm_stderr,none",
    "acc_norm_stderr",
    "acc_stderr,none",
    "acc_stderr",
    "exact_match_stderr,flexible-extract",
    "exact_match_stderr,strict-match",
    "exact_match_stderr",
)


def _pick(d: dict, keys: tuple[str, ...]) -> Optional[float]:
    for k in keys:
        if k in d and d[k] is not None:
            # tinyBenchmarks IRT postprocess may emit "N/A" stderr
            if isinstance(d[k], str):
                continue
            try:
                return float(d[k])
            except (TypeError, ValueError):
                continue
    return None


def normalize_lm_eval_results(raw: Dict[str, Any], tasks: List[str]) -> Dict[str, Any]:
    results = raw.get("results") or {}
    metrics: Dict[str, Any] = {}
    for t in tasks:
        block = results.get(t) or {}
        metrics[t] = {
            "acc": _pick(block, _ACC_KEYS),
            "acc_stderr": _pick(block, _ERR_KEYS),
        }
    return {"metrics": metrics}


def load_tiny_compare_baselines() -> Dict[str, Any]:
    """Side-by-side tinyBenchmarks reference scores for the Eval page compare table."""
    path = Path(__file__).resolve().parent / "tiny_compare_baselines.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    tasks = list(raw.get("tasks") or TINY_TASKS)
    models = []
    for m in raw.get("models") or []:
        scores = m.get("scores_pct") or {}
        # normalize GSM8k_flexible -> tinyGSM8k for table columns
        if "tinyGSM8k" not in scores and "tinyGSM8k_flexible" in scores:
            scores = dict(scores)
            scores["tinyGSM8k"] = scores.pop("tinyGSM8k_flexible")
        row = {
            "id": m.get("id"),
            "label": m.get("label"),
            "subtitle": m.get("subtitle") or "",
            "source": m.get("source"),
            "scores_pct": scores if scores else None,
        }
        models.append(row)
    return {
        "suite": raw.get("suite") or "tinyBenchmarks",
        "tasks": tasks,
        "metric_note": raw.get("metric_note") or "",
        "models": models,
    }

