# Task-driven Web UI + Tiny Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor MiniSeek Web training into per-task isolated directories (config/log/metrics/checkpoints), drop process scan/attach, and add a Tiny DenseLM smoke arch for CPU end-to-end tests.

**Architecture:** Filesystem is the source of truth under `tasks/<id>/`. A `TaskStore` owns JSON CRUD; a `TaskRunner` starts/stops one training subprocess, writing pid into `task.json` and logs/metrics into that folder. Web UI shows a left task list and right detail pane. `arch=tiny` uses the same `DenseLM` with fixed small `ModelConfig`.

**Tech Stack:** Python 3, FastAPI, existing `web_ui.py` / `web/index.html`, PyTorch `DenseLM`, pytest.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-10-26-task-driven-web-ui-design.md`
- Concurrent trainings: at most one `status=running`
- Metadata format: `task.json` only (not YAML)
- Full isolation: each task has `task.json`, `train.log`, `metrics.json`, `checkpoints/`
- No `/proc` process listing / attach UI
- Tiny: DenseLM, `d_model=64`, `n_layers=2`, `n_heads=1`, `head_dim=64`, `intermediate_size=256`, `vocab_size=6400`
- Do not migrate old `checkpoints/*` into tasks; new runs only
- Primary worktree: repo root (box `/workspace/miniseek` and/or `D:\miniseek` on wangyifan6); keep both in sync if editing both

## File map

| File | Responsibility |
|------|----------------|
| `model/config.py` | Add `arch` literal `tiny` + `ModelConfig.tiny()` preset |
| `model/__init__.py` | `build_model`: tiny → DenseLM |
| `train_pretrain.py` | Add `--arch dense\|moe\|tiny` (and MoE path if missing); build via `build_model` + tiny preset |
| `web/tasks_store.py` | NEW: task id, paths, load/save/list/delete, metrics helpers |
| `web/task_runner.py` | NEW: start/stop/reconcile pid, tail log → metrics, single-running guard |
| `web_ui.py` | Replace TrainingManager process-scan APIs with `/api/tasks*`; wire runner |
| `web/index.html` | Left task sidebar + detail; remove attach/process UI; Tiny in arch select |
| `tests/test_config_tiny.py` | Tiny config + forward param band |
| `tests/test_tasks_store.py` | Directory CRUD + single-running helper |
| `tests/test_task_runner_unit.py` | Pid reconcile / refuse second start (mocked) |

---

### Task 1: Tiny ModelConfig preset

**Files:**
- Modify: `model/config.py`
- Modify: `model/__init__.py`
- Create: `tests/test_config_tiny.py`

**Interfaces:**
- Produces: `Arch = Literal["dense", "moe", "tiny"]`; `ModelConfig.tiny() -> ModelConfig` with fixed dims; `build_model(cfg)` returns `DenseLM` for tiny

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config_tiny.py
from __future__ import annotations

import torch

from model import build_model
from model.config import ModelConfig
from model.dense import DenseLM, count_parameters
from model.tokenizer import MINIMIND_VOCAB_SIZE


def test_tiny_preset_dims():
    cfg = ModelConfig.tiny()
    assert cfg.arch == "tiny"
    assert cfg.d_model == 64
    assert cfg.n_layers == 2
    assert cfg.n_heads == 1
    assert cfg.head_dim == 64
    assert cfg.intermediate_size == 256
    assert cfg.vocab_size == MINIMIND_VOCAB_SIZE
    assert cfg.d_model == cfg.n_heads * cfg.head_dim


def test_tiny_build_and_forward():
    cfg = ModelConfig.tiny().with_vocab_size(MINIMIND_VOCAB_SIZE)
    model = build_model(cfg)
    assert isinstance(model, DenseLM)
    n = count_parameters(model)
    assert 100_000 <= n <= 2_000_000, n
    ids = torch.randint(0, cfg.vocab_size, (2, 8))
    logits, loss = model(ids, labels=ids)
    assert logits.shape == (2, 8, cfg.vocab_size)
    assert loss is not None and torch.isfinite(loss)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /workspace/miniseek && .venv/bin/python -m pytest tests/test_config_tiny.py -v`
Expected: FAIL (`tiny` / `ModelConfig.tiny` missing)

- [ ] **Step 3: Implement preset**

In `model/config.py`:
- Change `Arch = Literal["dense", "moe", "tiny"]`
- In `__post_init__`, treat `arch == "tiny"` like dense for `use_moe=False`; set `model_family` to `"MiniSeek-D"` if needed
- Add:

```python
@classmethod
def tiny(cls) -> "ModelConfig":
    return cls(
        model_family="MiniSeek-D",
        model_version="0.0-tiny",
        arch="tiny",
        d_model=64,
        n_layers=2,
        n_heads=1,
        head_dim=64,
        intermediate_size=256,
        vocab_size=6400,
        max_seq_len=256,
    )
```

In `model/__init__.py` `build_model`:

```python
if getattr(cfg, "arch", "dense") == "moe" or getattr(cfg, "use_moe", False):
    from .moe import MoELM
    return MoELM(cfg)
return DenseLM(cfg)  # dense and tiny
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_config_tiny.py tests/test_smoke.py -v`
Expected: PASS (existing smoke still uses default dense)

- [ ] **Step 5: Commit**

```bash
git add model/config.py model/__init__.py tests/test_config_tiny.py
git commit -m "feat: ModelConfig.tiny() DenseLM smoke preset"
```

---

### Task 2: train_pretrain `--arch` including tiny

**Files:**
- Modify: `train_pretrain.py`
- Test: manual CLI smoke (and optional pytest calling argparse builder if extracted)

**Interfaces:**
- Consumes: `ModelConfig.tiny()`, `build_model`
- Produces: CLI `--arch {dense,moe,tiny}` (default `dense`); tiny uses preset then `with_vocab_size` / `max_seq_len` from args

- [ ] **Step 1: Confirm current gap**

Run: `.venv/bin/python train_pretrain.py --help | rg arch || true`
Expected: no `--arch` today (box copy); web_ui already passes it — must add.

- [ ] **Step 2: Add argparse + model build**

Near other `add_argument` calls:

```python
p.add_argument(
    "--arch",
    type=str,
    default="dense",
    choices=["dense", "moe", "tiny"],
    help="model architecture",
)
p.add_argument("--aux-coef", type=float, default=0.01, help="MoE load-balance coef")
```

Replace hardcoded:

```python
cfg = ModelConfig(max_seq_len=args.max_seq_len).with_vocab_size(vocab)
model = DenseLM(cfg).to(device)
```

with:

```python
from model import build_model

if args.arch == "tiny":
    cfg = ModelConfig.tiny()
    cfg = ModelConfig.from_dict({**cfg.to_dict(), "max_seq_len": args.max_seq_len})
else:
    cfg = ModelConfig(arch=args.arch, max_seq_len=args.max_seq_len)
cfg = cfg.with_vocab_size(vocab)
model = build_model(cfg).to(device)
# If MoE training needs aux_coef, pass into train loop the same way existing MoE code expects (match wyf/main if present).
```

If MoE train loop is not on this branch yet, keep `moe` choice but raise a clear error until MoE train path exists — or port the MoE branch from the machine that already trains MoE. Do not silently ignore `--arch miniseek-moe`.

- [ ] **Step 3: CPU smoke 2 steps**

Run:

```bash
.venv/bin/python -u train_pretrain.py --arch tiny --max-steps 2 --batch-size 2 \
  --max-seq-len 64 --out-dir /tmp/miniseek-tiny-smoke --data data/pretrain_t2t_mini.jsonl \
  --warmup-steps 0 --log-every 1
```

Expected: finishes, writes `/tmp/miniseek-tiny-smoke/last.pt` (or uses built-in smoke jsonl if data missing).

- [ ] **Step 4: Commit**

```bash
git add train_pretrain.py
git commit -m "feat: train_pretrain --arch dense|moe|tiny"
```

---

### Task 3: TaskStore (filesystem CRUD)

**Files:**
- Create: `web/tasks_store.py`
- Create: `web/__init__.py` (empty)
- Create: `tests/test_tasks_store.py`

**Interfaces:**
- Produces:
  - `TASKS_ROOT: Path` default `PROJECT_ROOT / "tasks"`
  - `new_task_id() -> str` like `20261026-114500-a3f2`
  - `task_dir(task_id) -> Path`
  - `create_task(*, name, arch, mode, hparams) -> dict` status `draft`, creates dirs + `task.json` + empty `metrics.json`
  - `load_task(task_id) -> dict`
  - `save_task(data: dict) -> None`
  - `list_tasks() -> list[dict]` summary rows
  - `delete_task(task_id) -> None` (caller enforces not running)
  - `write_metrics(task_id, metrics: dict) -> None`
  - `read_metrics(task_id) -> dict`
  - `find_running_task_id() -> str | None` (status==running in json only)

- [ ] **Step 1: Failing tests**

```python
# tests/test_tasks_store.py
from __future__ import annotations

import json
from pathlib import Path

import pytest

from web import tasks_store as ts


@pytest.fixture()
def tasks_root(tmp_path, monkeypatch):
    root = tmp_path / "tasks"
    monkeypatch.setattr(ts, "TASKS_ROOT", root)
    return root


def test_create_layout(tasks_root):
    data = ts.create_task(
        name="t1",
        arch="tiny",
        mode="pretrain",
        hparams={"batch_size": 2, "max_steps": 2},
    )
    d = ts.task_dir(data["id"])
    assert (d / "task.json").is_file()
    assert (d / "metrics.json").is_file()
    assert (d / "checkpoints").is_dir()
    assert data["status"] == "draft"
    assert data["out_dir"].endswith("checkpoints")
    loaded = ts.load_task(data["id"])
    assert loaded["name"] == "t1"


def test_list_and_delete(tasks_root):
    a = ts.create_task(name="a", arch="dense", mode="pretrain", hparams={})
    b = ts.create_task(name="b", arch="tiny", mode="pretrain", hparams={})
    rows = ts.list_tasks()
    assert {r["id"] for r in rows} == {a["id"], b["id"]}
    ts.delete_task(a["id"])
    assert not ts.task_dir(a["id"]).exists()
```

- [ ] **Step 2: Run — expect FAIL**

Run: `.venv/bin/python -m pytest tests/test_tasks_store.py -v`

- [ ] **Step 3: Implement `web/tasks_store.py`**

```python
from __future__ import annotations

import json
import secrets
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TASKS_ROOT = PROJECT_ROOT / "tasks"


def new_task_id() -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{secrets.token_hex(2)}"


def task_dir(task_id: str) -> Path:
    return TASKS_ROOT / task_id


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def create_task(*, name: str, arch: str, mode: str, hparams: Dict[str, Any]) -> Dict[str, Any]:
    tid = new_task_id()
    d = task_dir(tid)
    ckpt = d / "checkpoints"
    ckpt.mkdir(parents=True, exist_ok=False)
    now = datetime.now().isoformat(timespec="seconds")
    data = {
        "id": tid,
        "name": name or f"{arch}-{mode}-{tid}",
        "status": "draft",
        "arch": arch,
        "mode": mode,
        "pid": None,
        "created_at": now,
        "started_at": None,
        "ended_at": None,
        "hparams": hparams or {},
        "out_dir": str(ckpt),
        "exit_code": None,
        "error": None,
    }
    _write_json(d / "task.json", data)
    _write_json(d / "metrics.json", {"loss": [], "expert_load": None})
    (d / "train.log").write_text("", encoding="utf-8")
    return data


def load_task(task_id: str) -> Dict[str, Any]:
    path = task_dir(task_id) / "task.json"
    if not path.exists():
        raise FileNotFoundError(task_id)
    return _read_json(path)


def save_task(data: Dict[str, Any]) -> None:
    tid = data["id"]
    _write_json(task_dir(tid) / "task.json", data)


def list_tasks() -> List[Dict[str, Any]]:
    if not TASKS_ROOT.exists():
        return []
    rows = []
    for p in TASKS_ROOT.iterdir():
        jf = p / "task.json"
        if jf.is_file():
            t = _read_json(jf)
            rows.append({
                "id": t["id"],
                "name": t.get("name"),
                "status": t.get("status"),
                "arch": t.get("arch"),
                "mode": t.get("mode"),
                "created_at": t.get("created_at"),
                "started_at": t.get("started_at"),
                "ended_at": t.get("ended_at"),
            })
    rows.sort(key=lambda r: r.get("created_at") or "", reverse=True)
    return rows


def delete_task(task_id: str) -> None:
    import shutil
    d = task_dir(task_id)
    if not d.exists():
        raise FileNotFoundError(task_id)
    shutil.rmtree(d)


def write_metrics(task_id: str, metrics: Dict[str, Any]) -> None:
    _write_json(task_dir(task_id) / "metrics.json", metrics)


def read_metrics(task_id: str) -> Dict[str, Any]:
    path = task_dir(task_id) / "metrics.json"
    if not path.exists():
        return {"loss": [], "expert_load": None}
    return _read_json(path)


def find_running_task_id() -> Optional[str]:
    for row in list_tasks():
        if row.get("status") == "running":
            return row["id"]
    return None
```

- [ ] **Step 4: Run tests — PASS**

Run: `.venv/bin/python -m pytest tests/test_tasks_store.py -v`

- [ ] **Step 5: Commit**

```bash
git add web/__init__.py web/tasks_store.py tests/test_tasks_store.py
git commit -m "feat: TaskStore filesystem CRUD under tasks/"
```

---

### Task 4: TaskRunner start/stop + single-running + pid reconcile

**Files:**
- Create: `web/task_runner.py`
- Create: `tests/test_task_runner_unit.py`
- Modify: reuse log-parse helpers from `web_ui.py` (extract `_parse_train_line` / STEP regex into shared functions or import carefully)

**Interfaces:**
- Consumes: `tasks_store.*`
- Produces:
  - `pid_alive(pid: int) -> bool` cross-platform (`os.kill(pid, 0)`)
  - `reconcile_task(task: dict) -> dict` if status running but pid dead → `finished`/`failed`
  - `start_task(task_id: str) -> dict` builds cmd like current `TrainingManager.start`, forces `out_dir` to task checkpoints, writes pid, tails log into task metrics
  - `stop_task(task_id: str) -> dict`
  - Raises `RuntimeError` / custom if another running

- [ ] **Step 1: Unit test refuse second running + reconcile**

```python
# tests/test_task_runner_unit.py
from __future__ import annotations

import pytest

from web import tasks_store as ts
from web import task_runner as tr


@pytest.fixture()
def tasks_root(tmp_path, monkeypatch):
    root = tmp_path / "tasks"
    monkeypatch.setattr(ts, "TASKS_ROOT", root)
    return root


def test_reconcile_dead_pid(tasks_root, monkeypatch):
    t = ts.create_task(name="x", arch="tiny", mode="pretrain", hparams={"batch_size": 1})
    t["status"] = "running"
    t["pid"] = 999999  # almost surely dead
    ts.save_task(t)
    monkeypatch.setattr(tr, "pid_alive", lambda pid: False)
    out = tr.reconcile_task(ts.load_task(t["id"]))
    assert out["status"] in {"finished", "failed", "stopped"}
    assert out["pid"] is None


def test_start_refuses_second(tasks_root, monkeypatch):
    a = ts.create_task(name="a", arch="tiny", mode="pretrain", hparams={"max_steps": 1})
    b = ts.create_task(name="b", arch="tiny", mode="pretrain", hparams={"max_steps": 1})
    a["status"] = "running"
    a["pid"] = 1
    ts.save_task(a)
    monkeypatch.setattr(tr, "pid_alive", lambda pid: True)
    monkeypatch.setattr(tr, "_spawn_training", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not spawn")))
    with pytest.raises(RuntimeError, match="running"):
        tr.start_task(b["id"])
```

- [ ] **Step 2: Run — FAIL**

- [ ] **Step 3: Implement runner skeleton**

Implement `web/task_runner.py` with:
- `pid_alive`
- `reconcile_task`
- `build_train_cmd(task) -> list[str]` copied from current `TrainingManager.start` command assembly; map `hparams` keys; always `--out-dir` = `task["out_dir"]`; pass `--arch` from `task["arch"]`; for tiny set defaults `max_steps`/`batch_size` if missing
- `_spawn_training(cmd, log_path) -> Popen` redirect stdout/stderr to `train.log`
- `start_task` / `stop_task`
- Background thread: tail `train.log`, parse loss lines (copy regexes from `web_ui.py`), append to `metrics.json` via `write_metrics`

Move or duplicate STEP/LOSS regex constants from `web_ui.py` into `task_runner.py` to avoid circular imports.

- [ ] **Step 4: Tests PASS**

- [ ] **Step 5: Commit**

```bash
git add web/task_runner.py tests/test_task_runner_unit.py
git commit -m "feat: TaskRunner start/stop with single-running guard"
```

---

### Task 5: FastAPI `/api/tasks` and remove process scan

**Files:**
- Modify: `web_ui.py` (routes + drop `list_processes`/`attach` endpoints; keep infer/chat)
- Optional: thin wrappers so SSE logs stream from selected task

**Interfaces:**
- Produces routes exactly as spec:
  - `GET/POST /api/tasks`
  - `GET/PATCH/DELETE /api/tasks/{id}`
  - `POST /api/tasks/{id}/start|stop`
  - `GET /api/tasks/{id}/logs` (SSE or poll)
  - `GET /api/tasks/{id}/metrics`
  - `GET /api/tasks/{id}/checkpoints`
- Remove: `GET /api/train/processes`, `POST /api/train/attach`
- Replace `POST /api/train/start|stop` with task routes (or keep as deprecated 410)

- [ ] **Step 1: Add route handlers calling store/runner**

Example shapes:

```python
@app.get("/api/tasks")
def api_list_tasks():
    rows = []
    for row in list_tasks():
        full = load_task(row["id"])
        rows.append(reconcile_task(full) and load_task(row["id"]) or row)
    # simpler: reconcile each then list again
    ...

@app.post("/api/tasks")
def api_create_task(body: TaskCreate):
    return create_task(name=body.name, arch=body.arch, mode=body.mode, hparams=body.hparams)

@app.post("/api/tasks/{task_id}/start")
def api_start(task_id: str):
    try:
        return start_task(task_id)
    except RuntimeError as e:
        raise HTTPException(409, str(e))
```

Extend `/api/defaults` with `"tiny": { "arch": "tiny", "batch_size": 4, "max_steps": 20, "epochs": 1, "max_seq_len": 128, "warmup_steps": 0, ... }`.

Checkpoint listing for a task: `sorted((task_dir(id)/"checkpoints").glob("*.pt"))`.

- [ ] **Step 2: Manual API smoke with TestClient**

```python
# quick in tests/test_tasks_api.py or a shell snippet
from fastapi.testclient import TestClient
from web_ui import app
client = TestClient(app)
r = client.post("/api/tasks", json={"name":"smoke","arch":"tiny","mode":"pretrain","hparams":{"max_steps":2,"batch_size":2}})
assert r.status_code == 200
tid = r.json()["id"]
assert client.get(f"/api/tasks/{tid}").json()["status"] == "draft"
```

- [ ] **Step 3: Delete old process endpoints and global `STATE_FILE`/`LOG_FILE` usage from start path**

Keep files ignored if present; do not write them anymore.

- [ ] **Step 4: Commit**

```bash
git add web_ui.py tests/test_tasks_api.py
git commit -m "feat: /api/tasks CRUD start/stop; drop process attach"
```

---

### Task 6: Web UI sidebar + Tiny option

**Files:**
- Modify: `web/index.html`

**Interfaces:**
- Consumes: `/api/tasks*` JSON
- UI: left list + create; right detail form/Start/Stop/loss/log/ckpt; arch select includes Tiny; remove process `<select>` and `#btn-attach`

- [ ] **Step 1: Layout**

Add CSS for `.layout { display:flex }`, `.task-sidebar { width:240px; ... }`, `.task-main { flex:1 }`.

Sidebar HTML:

```html
<aside class="task-sidebar">
  <button id="btn-new-task" type="button">新建 Task</button>
  <ul id="task-list"></ul>
</aside>
```

- [ ] **Step 2: JS**

- `refreshTasks()` → `GET /api/tasks`, render list with status dot
- click → `selectTask(id)` → `GET /api/tasks/{id}` + metrics + wire form
- `btn-new-task` → POST draft with defaults for selected arch (include tiny)
- Start/Stop → POST start/stop then refresh
- SSE/log: point stream URL to `/api/tasks/{id}/logs` (or poll metrics every 2s if SSE deferred)
- Remove attach/process refresh code paths
- Arch radio/select: dense | moe | tiny; on tiny apply defaults from `/api/defaults`

- [ ] **Step 3: Chat ckpt select**

`refreshCheckpoints()` uses `/api/tasks/{selectedId}/checkpoints` when a task is selected; still allow manual path if you keep a text input (optional). Default selected = current task’s `best.pt` or `last.pt`.

- [ ] **Step 4: Manual browser check on wangyifan6**

Start: `.venv\Scripts\python -m uvicorn web_ui:app --host 127.0.0.1 --port 7860` from `D:\miniseek` (after sync). Create Tiny task → Start → see loss → load ckpt in chat.

- [ ] **Step 5: Commit**

```bash
git add web/index.html
git commit -m "feat: task sidebar UI and Tiny arch option"
```

---

### Task 7: End-to-end Tiny CPU verification + docs touch

**Files:**
- Possibly `README.md` one paragraph on `tasks/`
- Ensure `.gitignore` has `/tasks/` (generated) unless user wants to keep drafts — **recommend ignore `tasks/` contents**

- [ ] **Step 1: gitignore**

Add:

```
tasks/
```

- [ ] **Step 2: E2E script (optional `scripts/smoke_tiny_task.py`)**

Creates task via store, `start_task`, waits until not running, asserts `checkpoints/last.pt` exists and metrics loss length ≥ 1.

- [ ] **Step 3: Run full pytest**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: PASS

- [ ] **Step 4: Sync to `D:\miniseek` if development was on box**

Copy changed files (`model/`, `train_pretrain.py`, `web/`, `web_ui.py`, tests, docs) via CopyFromBox / git pull.

- [ ] **Step 5: Final commit**

```bash
git add .gitignore scripts/smoke_tiny_task.py README.md
git commit -m "chore: ignore tasks/; add tiny task smoke script"
```

---

## Spec coverage checklist

| Spec item | Task |
|-----------|------|
| `tasks/<id>/` layout + task.json fields | 3 |
| metrics.json + train.log + checkpoints | 3, 4 |
| Single running | 4, 5 |
| No process scan/attach | 5, 6 |
| API table | 5 |
| Sidebar UI | 6 |
| Tiny dims + DenseLM | 1, 2 |
| Chat uses task ckpts by default | 6 |
| Windows-friendly pid check | 4 (`os.kill(pid,0)`) |
| Old ckpt migration out of scope | noted; no task |

## Self-review notes

- No TBD left for required behavior; MoE train path on this branch may need porting when `--arch miniseek-moe` is selected — Task 2 calls that out explicitly.
- Names aligned: `create_task`, `start_task`, `reconcile_task`, `ModelConfig.tiny()`.
- UI SSE may temporarily poll metrics if extracting SSE is large; still must show loss for the selected task.
