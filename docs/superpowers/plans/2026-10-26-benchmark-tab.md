# Benchmark Tab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a「评测」Web tab that runs lm-eval on a chosen MiniSeek checkpoint for `ceval-valid`, `cmmlu`, and `gsm8k`, and shows accuracy plus run history.

**Architecture:** Background Popen jobs under `eval/runs/<id>/` (store + runner mirroring training tasks). CLI `python -m eval.run_benchmark` drives lm-evaluation-harness via a custom MiniSeek LM wrapper that loads `.pt` / HF dirs the same way as `InferenceEngine`. FastAPI `/api/bench/*` + fifth tab in `web/index.html`.

**Tech Stack:** Python 3, FastAPI, PyTorch, `lm-eval`, existing DenseLM/MoELM + tokenizer, pytest, vanilla HTML/JS.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-10-26-benchmark-tab-design.md`
- Suite whitelist only: `ceval-valid`, `cmmlu`, `gsm8k` (Agent/MATH out of scope)
- At most one `running` bench run; POST start returns 409 if another is running
- Device: `cuda` only if CUDA available and no training task `status=running`; else `cpu`; record `device` in `run.json`
- Reuse InferenceEngine blob loading rules for `.pt` and HF `pytorch_model.bin`+`config.json`
- Tabs order: `训练 | 对话 | 续写 | 架构 | 评测`
- Primary worktree: `/workspace/miniseek`; sync to `D:\miniseek` on wangyifan6 after UI verification
- Do not auto-edit `leaderboard.md` / `stage-N.md` in v1

## File map

| File | Responsibility |
|------|----------------|
| `eval/miniseek_lm.py` | lm-eval registered model: load ckpt, `loglikelihood`, `generate_until` |
| `eval/run_benchmark.py` | CLI: args → lm_eval → `results.json` |
| `eval/bench_results.py` | Normalize lm-eval raw JSON → `metrics` |
| `web/bench_store.py` | `eval/runs/` CRUD |
| `web/bench_runner.py` | start/stop Popen, reconcile |
| `web_ui.py` | `/api/bench/*` |
| `web/index.html` | 评测 tab |
| `requirements.txt` | `lm-eval>=0.4.5` |
| `eval/README.md` | CLI + Web notes |
| `tests/test_bench_store.py` | store tests |
| `tests/test_bench_results.py` | normalize tests |
| `tests/test_bench_api.py` | API tests |
| `tests/test_miniseek_lm_load.py` | load tiny ckpt into wrapper |

---

### Task 1: Store + result normalizer

**Files:**
- Create: `web/bench_store.py`
- Create: `eval/bench_results.py`
- Create: `tests/test_bench_store.py`
- Create: `tests/test_bench_results.py`

**Interfaces:**
- Consumes: none
- Produces:
  - `ALLOWED_TASKS = ("ceval-valid", "cmmlu", "gsm8k")` in `web/bench_store.py` (re-exported or duplicated as constant in `eval/bench_results.py` only if needed — prefer import from one place: put `ALLOWED_TASKS` in `eval/bench_results.py` and import in store)
  - `create_run(...) -> dict`, `list_runs()`, `load_run()`, `save_run()`, `run_dir()`
  - `normalize_lm_eval_results(raw: dict, tasks: list[str]) -> dict` with `metrics[task] = {acc, acc_stderr}`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_bench_store.py
from __future__ import annotations

import pytest

from web import bench_store as bs


def test_create_and_list(tmp_path, monkeypatch):
    monkeypatch.setattr(bs, "RUNS_ROOT", tmp_path / "runs")
    row = bs.create_run(
        ckpt_path="checkpoints/x.pt",
        tasks=["ceval-valid", "gsm8k"],
        limit=2,
        device="cpu",
    )
    assert row["status"] == "queued"
    assert row["tasks"] == ["ceval-valid", "gsm8k"]
    assert (tmp_path / "runs" / row["id"] / "run.json").is_file()
    assert bs.list_runs()[0]["id"] == row["id"]


def test_reject_unknown_task(tmp_path, monkeypatch):
    monkeypatch.setattr(bs, "RUNS_ROOT", tmp_path / "runs")
    with pytest.raises(ValueError, match="task"):
        bs.create_run(ckpt_path="x.pt", tasks=["mmlu"], limit=None, device="cpu")
```

```python
# tests/test_bench_results.py
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
```

- [ ] **Step 2: Run tests — expect FAIL**

Run: `cd /workspace/miniseek && .venv/bin/pytest tests/test_bench_store.py tests/test_bench_results.py -v`
Expected: import errors / FAIL

- [ ] **Step 3: Implement**

```python
# eval/bench_results.py
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
```

```python
# web/bench_store.py
from __future__ import annotations

import json
import secrets
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from eval.bench_results import ALLOWED_TASKS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_ROOT = PROJECT_ROOT / "eval" / "runs"


def new_run_id() -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{secrets.token_hex(2)}"


def run_dir(run_id: str) -> Path:
    return RUNS_ROOT / run_id


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def create_run(
    *,
    ckpt_path: str,
    tasks: List[str],
    limit: Optional[int],
    device: str,
) -> Dict[str, Any]:
    cleaned: List[str] = []
    for t in tasks:
        t = (t or "").strip()
        if t not in ALLOWED_TASKS:
            raise ValueError(f"unsupported task: {t}")
        if t not in cleaned:
            cleaned.append(t)
    if not cleaned:
        raise ValueError("tasks must be non-empty")
    rid = new_run_id()
    d = run_dir(rid)
    d.mkdir(parents=True, exist_ok=False)
    now = datetime.now().isoformat(timespec="seconds")
    data = {
        "id": rid,
        "status": "queued",
        "ckpt_path": ckpt_path,
        "tasks": cleaned,
        "limit": limit,
        "device": device,
        "pid": None,
        "created_at": now,
        "started_at": None,
        "ended_at": None,
        "exit_code": None,
        "error": None,
        "scores": None,
    }
    _write_json(d / "run.json", data)
    return data


def save_run(data: Dict[str, Any]) -> None:
    rid = data["id"]
    _write_json(run_dir(rid) / "run.json", data)


def load_run(run_id: str) -> Dict[str, Any]:
    path = run_dir(run_id) / "run.json"
    if not path.is_file():
        raise FileNotFoundError(run_id)
    return _read_json(path)


def list_runs() -> List[Dict[str, Any]]:
    if not RUNS_ROOT.is_dir():
        return []
    rows = []
    for p in RUNS_ROOT.iterdir():
        jp = p / "run.json"
        if jp.is_file():
            rows.append(_read_json(jp))
    rows.sort(key=lambda r: r.get("created_at") or "", reverse=True)
    return rows
```

- [ ] **Step 4: Run tests — expect PASS**

Run: `cd /workspace/miniseek && .venv/bin/pytest tests/test_bench_store.py tests/test_bench_results.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /workspace/miniseek
git add eval/bench_results.py web/bench_store.py tests/test_bench_store.py tests/test_bench_results.py
git commit -m "feat(bench): store + lm-eval results normalizer"
```

---

### Task 2: MiniSeek lm-eval model wrapper (load + loglikelihood smoke)

**Files:**
- Create: `eval/miniseek_lm.py`
- Create: `tests/test_miniseek_lm_load.py`
- Modify: `requirements.txt` (add `lm-eval>=0.4.5`)

**Interfaces:**
- Consumes: `ModelConfig`, `DenseLM`/`MoELM`, `load_tokenizer`, torch
- Produces:
  - `load_miniseek_blob(path: str | Path) -> tuple[nn.Module, tokenizer, ModelConfig]` (same path rules as InferenceEngine)
  - `class MiniSeekLM` registered with `@register_model("miniseek")` implementing lm-eval `LM` API: `loglikelihood`, `generate_until`, `loglikelihood_rolling` (rolling may raise `NotImplementedError` if unused)

Implementation notes:
- Subclass `lm_eval.api.model.TemplateLM` or `LM` per installed lm-eval version; if import path differs, adapt in this task after `pip install lm-eval`.
- `loglikelihood(requests)`: for each `(context, continuation)` encode, forward once, sum log-probs of continuation tokens; return `[(ll, is_greedy)]`.
- `generate_until(requests)`: greedy or temp=0 generation until stop sequences / max_gen_toks (needed for GSM8K).
- Constructor args via lm-eval `model_args`: `ckpt=...`, `device=cpu|cuda`.

- [ ] **Step 1: Add dependency and write load test**

```text
# append to requirements.txt
lm-eval>=0.4.5
```

```python
# tests/test_miniseek_lm_load.py
from __future__ import annotations

from pathlib import Path

import torch

from model.config import ModelConfig
from model.dense import DenseLM
from eval.miniseek_lm import load_miniseek_blob


def test_load_training_blob(tmp_path):
    cfg = ModelConfig.tiny()
    model = DenseLM(cfg)
    ckpt = tmp_path / "t.pt"
    torch.save({"config": cfg.to_dict(), "model": model.state_dict(), "step": 1}, ckpt)
    m, tok, cfg2 = load_miniseek_blob(ckpt)
    assert cfg2.n_layers == cfg.n_layers
    assert m is not None
    assert tok is not None
```

- [ ] **Step 2: Run test — expect FAIL**

Run: `cd /workspace/miniseek && .venv/bin/pip install 'lm-eval>=0.4.5' -q && .venv/bin/pytest tests/test_miniseek_lm_load.py -v`
Expected: FAIL (module missing)

- [ ] **Step 3: Implement `load_miniseek_blob` + minimal `MiniSeekLM`**

```python
# eval/miniseek_lm.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, List, Optional, Tuple

import torch
import torch.nn.functional as F

from model.config import ModelConfig
from model.dense import DenseLM
from model.tokenizer import load_tokenizer


def load_miniseek_blob(path: str | Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_dir() and (path / "pytorch_model.bin").exists():
        weight_path = path / "pytorch_model.bin"
        cfg_path = path / "config.json"
    elif path.name == "pytorch_model.bin":
        weight_path = path
        cfg_path = path.parent / "config.json"
    else:
        weight_path = None
        cfg_path = None
    if weight_path is not None:
        sd = torch.load(weight_path, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "model" in sd:
            blob = sd
        else:
            cfg_raw = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path and cfg_path.exists() else {}
            allowed = set(ModelConfig.__annotations__)
            cfg_dict = {k: v for k, v in cfg_raw.items() if k in allowed}
            blob = {"model": sd, "config": cfg_dict}
    else:
        blob = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ModelConfig.from_dict(blob["config"])
    if getattr(cfg, "arch", "dense") == "moe" or getattr(cfg, "use_moe", False):
        from model.moe import MoELM
        model = MoELM(cfg)
    else:
        model = DenseLM(cfg)
    model.load_state_dict(blob["model"], strict=True)
    model.eval()
    tok = load_tokenizer(allow_fallback=True)
    return model, tok, cfg


# Register MiniSeekLM with lm_eval after implementing loglikelihood / generate_until.
# Use @register_model("miniseek") from lm_eval.api.registry.
# Keep batch_size=1 initially for correctness on tiny models.
```

Fill `MiniSeekLM` completely in this task: inspect installed lm-eval `LM` / `TemplateLM` signatures (`pip show` + read site-packages) and implement required abstract methods. Include a unit test that calls `loglikelihood` on one synthetic pair if feasible without downloading datasets.

- [ ] **Step 4: Run tests — expect PASS**

Run: `.venv/bin/pytest tests/test_miniseek_lm_load.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add requirements.txt eval/miniseek_lm.py tests/test_miniseek_lm_load.py
git commit -m "feat(bench): MiniSeek lm-eval model loader"
```

---

### Task 3: CLI `eval.run_benchmark`

**Files:**
- Create: `eval/run_benchmark.py`
- Modify: `eval/README.md`
- Create: `tests/test_run_benchmark_cli.py` (arg parse + normalize path dry-run; optional skip if no lm-eval datasets)

**Interfaces:**
- Consumes: `MiniSeekLM`, `normalize_lm_eval_results`
- Produces: module runnable as `python -m eval.run_benchmark`
  - args: `--ckpt`, `--tasks` (comma), `--out` (run dir), `--device`, `--limit` (optional), `--batch_size` (default 1)
  - writes `results.json` into `--out` with `ckpt_path`, `tasks`, `metrics`, `raw` summary
  - exit 0 on success

- [ ] **Step 1: Write CLI smoke test that mocks simple_evaluate**

```python
# tests/test_run_benchmark_cli.py
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from eval.run_benchmark import run_bench


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
```

- [ ] **Step 2: Run — expect FAIL**

Run: `.venv/bin/pytest tests/test_run_benchmark_cli.py -v`

- [ ] **Step 3: Implement**

```python
# eval/run_benchmark.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

from eval.bench_results import ALLOWED_TASKS, normalize_lm_eval_results


def simple_evaluate(**kwargs):
    # Imported lazily so tests can patch this symbol on the module.
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
    # Import registers model
    import eval.miniseek_lm  # noqa: F401

    raw = simple_evaluate(
        model="miniseek",
        model_args=f"ckpt={ckpt},device={device}",
        tasks=tasks,
        batch_size=batch_size,
        limit=limit,
        device=device,
    )
    # lm-eval may return (results, samples) or dict depending on version — normalize to dict with "results"
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
    raise SystemExit(main())
```

Update `eval/README.md`: document default suite including GSM8K, CLI example with `--limit 8`, and note Web tab uses same CLI.

- [ ] **Step 4: pytest PASS**

- [ ] **Step 5: Commit**

```bash
git add eval/run_benchmark.py eval/README.md tests/test_run_benchmark_cli.py
git commit -m "feat(bench): CLI run_benchmark for lm-eval suite"
```

---

### Task 4: `bench_runner` + FastAPI `/api/bench/*`

**Files:**
- Create: `web/bench_runner.py`
- Modify: `web_ui.py`
- Create: `tests/test_bench_api.py`

**Interfaces:**
- Consumes: `bench_store`, training reconcile helper `_any_training_running` (or `web.task_runner` list)
- Produces:
  - `start_run(ckpt_path, tasks, limit) -> dict` (create + Popen)
  - `stop_run(run_id) -> dict`
  - `reconcile_run(run) -> dict` (pid dead → finished/failed; load scores from results.json)
  - Routes:
    - `GET /api/bench/runs`
    - `POST /api/bench/runs` body `{ckpt_path, tasks, limit?}`
    - `GET /api/bench/runs/{id}`
    - `GET /api/bench/runs/{id}/logs?tail=200`
    - `GET /api/bench/runs/{id}/results`
    - `POST /api/bench/runs/{id}/stop`

Popen command:

```python
cmd = [
    sys.executable, "-m", "eval.run_benchmark",
    "--ckpt", ckpt_path,
    "--tasks", ",".join(tasks),
    "--out", str(run_dir),
    "--device", device,
]
if limit is not None:
    cmd += ["--limit", str(limit)]
```

stdout/stderr → `run.log`. On reconcile finished, read `results.json` into `run["scores"]`.

- [ ] **Step 1: API tests with tmp RUNS_ROOT and mocked Popen**

```python
# tests/test_bench_api.py
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import web.bench_store as bs
import web_ui


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(bs, "RUNS_ROOT", tmp_path / "runs")
    ckpt = tmp_path / "a.pt"
    ckpt.write_bytes(b"x")
    return TestClient(web_ui.app), ckpt


def test_post_rejects_bad_task(client):
    c, ckpt = client
    r = c.post("/api/bench/runs", json={"ckpt_path": str(ckpt), "tasks": ["mmlu"]})
    assert r.status_code == 400


def test_post_start_and_conflict(client, monkeypatch):
    c, ckpt = client
    fake_proc = MagicMock()
    fake_proc.pid = 4242
    fake_proc.poll.return_value = None
    with patch("web.bench_runner.subprocess.Popen", return_value=fake_proc):
        r1 = c.post(
            "/api/bench/runs",
            json={"ckpt_path": str(ckpt), "tasks": ["gsm8k"], "limit": 1},
        )
    assert r1.status_code == 200
    assert r1.json()["status"] == "running"
    r2 = c.post(
        "/api/bench/runs",
        json={"ckpt_path": str(ckpt), "tasks": ["gsm8k"]},
    )
    assert r2.status_code == 409
```

- [ ] **Step 2: FAIL then implement runner + routes**

Mirror patterns in `web/task_runner.py` (`pid_alive`, owned procs, SIGTERM). Wire routes near other `/api/` blocks in `web_ui.py`. Missing ckpt → 404; bad tasks → 400.

- [ ] **Step 3: pytest PASS**

Run: `.venv/bin/pytest tests/test_bench_api.py tests/test_bench_store.py -v`

- [ ] **Step 4: Commit**

```bash
git add web/bench_runner.py web_ui.py tests/test_bench_api.py
git commit -m "feat(bench): API start/stop/list for eval runs"
```

---

### Task 5: Web「评测」Tab UI

**Files:**
- Modify: `web/index.html`

**Interfaces:**
- Consumes: `/api/bench/*`, `/api/pick-file`, optional `ENGINE` loaded ckpt display if exposed
- Produces: fifth tab `data-tab="bench"`

UI pieces:
1. Tab button「评测」; expand `.page-tabs` to 5 columns / wider max-width
2. Panel: ckpt input + Browse (`pick-file` key e.g. `bench-ckpt`), checkboxes for three tasks (default checked), limit number input, 开始 / 停止
3. Scores table + status pill
4. History list from `GET /api/bench/runs`; click loads detail + logs
5. Poll every 2s while status is `running`

- [ ] **Step 1: Add tab shell + JS fetch helpers** (follow 续写/架构 patterns in same file)
- [ ] **Step 2: Manual/visual check on box if possible; else static review**
- [ ] **Step 3: Commit**

```bash
git add web/index.html
git commit -m "feat(bench): 评测 tab UI for lm-eval suite"
```

---

### Task 6: Sync Windows + smoke

**Files:** none new (deploy)

- [ ] **Step 1:** Copy changed files to `D:\miniseek` on machine `ea181723-6900-45bd-a649-0c19c5dc5d77` (or git pull if remote tracking). Install `lm-eval` in Windows venv.
- [ ] **Step 2:** Restart `web_ui` on `:7860`
- [ ] **Step 3:** CLI smoke (limit 1) on a real ckpt if datasets downloadable:

```bash
python -m eval.run_benchmark --ckpt <path-to-pt> --tasks gsm8k --limit 1 --out eval/runs/_smoke --device cuda
```

If HF dataset download blocked, still verify API create+fail path and UI renders; document blocker.

- [ ] **Step 4:** Push commits to origin when user-ready

```bash
git push
```

---

## Self-review

1. **Spec coverage:** Suite / tab / API / runs dir / concurrency / device / no Agent / no leaderboard auto-write → Tasks 1–6.  
2. **Placeholders:** MiniSeekLM abstract methods must be fully filled in Task 2 against installed lm-eval (inspect site-packages; no TBD left in commit).  
3. **Types:** `run.json` fields consistent across store/runner/API/UI (`scores` = metrics dict).  
4. **YAGNI:** MATH/Agent not in UI.  

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-10-26-benchmark-tab.md`.

**1. Subagent-Driven (recommended)** — fresh subagent per task, review between tasks  
**2. Inline Execution** — this session with executing-plans checkpoints  

Which approach?
