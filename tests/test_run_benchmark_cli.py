from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from eval.run_benchmark import main, run_bench


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


def test_main_prints_flushed_start_and_done(tmp_path):
    out = tmp_path / "run1"
    fake = {
        "results": {
            "ceval-valid": {"acc,none": 0.25, "acc_stderr,none": 0.01},
        }
    }
    with patch("eval.run_benchmark.simple_evaluate", return_value=fake):
        with patch("builtins.print") as mock_print:
            rc = main(
                [
                    "--ckpt",
                    str(tmp_path / "dummy.pt"),
                    "--tasks",
                    "ceval-valid",
                    "--out",
                    str(out),
                    "--device",
                    "cpu",
                    "--limit",
                    "2",
                ]
            )
    assert rc == 0
    assert (out / "results.json").is_file()
    texts = [str(call.args[0]) for call in mock_print.call_args_list if call.args]
    start = next(t for t in texts if t.startswith("[bench] start"))
    done = next(t for t in texts if t.startswith("[bench] done"))
    assert texts.index(start) < texts.index(done)
    assert "ceval-valid" in start
    assert "0.25" in done
    start_call = next(c for c in mock_print.call_args_list if str(c.args[0]).startswith("[bench] start"))
    done_call = next(c for c in mock_print.call_args_list if str(c.args[0]).startswith("[bench] done"))
    assert start_call.kwargs.get("flush") is True
    assert done_call.kwargs.get("flush") is True
