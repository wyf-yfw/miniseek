from __future__ import annotations

from typing import Any, Dict, List, Optional

ALLOWED_TASKS = ("ceval-valid", "cmmlu", "gsm8k")

_ACC_KEYS = (
    "acc,none",
    "acc",
    "exact_match,flexible-extract",
    "exact_match,strict-match",
    "exact_match",
)
_ERR_KEYS = (
    "acc_stderr,none",
    "acc_stderr",
    "exact_match_stderr,flexible-extract",
    "exact_match_stderr,strict-match",
    "exact_match_stderr",
)


def _pick(d: dict, keys: tuple[str, ...]) -> Optional[float]:
    for k in keys:
        if k in d and d[k] is not None:
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
