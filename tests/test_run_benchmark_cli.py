from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from algo.eval.run_benchmark import run_bench


def test_run_bench_writes_results(tmp_path):
    out = tmp_path / "run1"
    out.mkdir()
    fake = {
        "results": {
            "ceval-valid": {"acc,none": 0.5, "acc_stderr,none": 0.1},
        }
    }
    with patch("eval.run_benchmark.simple_evaluate", return_value=fake):
        run_bench(
            ckpt=str(tmp_path / "dummy.pt"),
            tasks=["ceval-valid"],
            out_dir=out,
            device="cpu",
            limit=2,
            batch_size=1,
        )
    data = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert data["metrics"]["ceval-valid"]["acc"] == 0.5
