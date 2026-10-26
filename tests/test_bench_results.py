from __future__ import annotations

import pytest

from eval.bench_results import normalize_lm_eval_results


def test_normalize_acc_and_gsm8k():
    raw = {
        "results": {
            "ceval-valid": {"acc,none": 0.25, "acc_stderr,none": 0.01},
            "gsm8k": {
                "exact_match,flexible-extract": 0.02,
                "exact_match_stderr,flexible-extract": 0.005,
            },
        }
    }
    out = normalize_lm_eval_results(raw, ["ceval-valid", "gsm8k"])
    assert out["metrics"]["ceval-valid"]["acc"] == pytest.approx(0.25)
    assert out["metrics"]["ceval-valid"]["acc_stderr"] == pytest.approx(0.01)
    assert out["metrics"]["gsm8k"]["acc"] == pytest.approx(0.02)


def test_missing_task_gets_none_acc():
    out = normalize_lm_eval_results({"results": {}}, ["cmmlu"])
    assert out["metrics"]["cmmlu"]["acc"] is None
