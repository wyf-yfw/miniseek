# Three-dir frontend/api/algo Reorg Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move MiniSeek into `frontend/`, `api/`, and `algo/` packages, delete root `checkpoints/`, update imports/systemd/docs with no compatibility shims.

**Architecture:** Mechanical git moves + import rewrites. Repo root stays WorkingDirectory for systemd and data/tasks. `web_ui.py` becomes `api/app.py`; `web/*` (minus frontend files) becomes `api/*`; model/train/eval/generate become `algo/*`. Train subprocess invokes `algo/train_*.py` by path under PROJECT_ROOT.

**Tech Stack:** Python 3.10+, FastAPI/uvicorn, pytest, systemd user units on wyf.

## Global Constraints

- Layout names: `frontend` / `api` / `algo` only (no monorepo packages/).
- No compatibility shims for `web_ui`, `web.*`, or root `train_*.py`.
- Keep at root: `data/`, `tasks/`, `tokenizer/`, `deploy/`, `tests/`, `docs/`, `scripts/`, `release/`.
- **Delete** root `checkpoints/` entirely (≈4.8G); Task weights live in `tasks/<id>/checkpoints/`.
- Never use Cursor CloudAgent for this work; edit on wyf (and sync Windows after).
- Do not interrupt a running train mid-cutover without stopping it first via worker Stop.

## File map (after)

```
frontend/__init__.py
frontend/index.html
frontend/frontend_server.py
api/__init__.py
api/app.py                 # was web_ui.py
api/worker_app.py
api/worker_client.py
api/task_runner.py
api/bench_runner.py
api/tasks_store.py
api/bench_store.py
api/arch_inspect.py
algo/__init__.py
algo/model/                # was model/
algo/train_pretrain.py
algo/train_sft.py
algo/generate.py
algo/eval/                 # was eval/
```

Removed from root after move: `web_ui.py`, `web/`, `model/`, `train_pretrain.py`, `train_sft.py`, `generate.py`, `eval/`, `checkpoints/`.

`bench_store.RUNS_ROOT` stays `PROJECT_ROOT / "eval" / "runs"` → after move use `PROJECT_ROOT / "algo" / "eval" / "runs"` **or** keep runs at `PROJECT_ROOT / "eval" / "runs"` by setting RUNS_ROOT to `PROJECT_ROOT / "eval" / "runs"` and creating a thin root `eval/runs` — **choose: keep RUNS_ROOT = PROJECT_ROOT / "eval" / "runs"** and leave an empty root `eval/` only for runs, OR move runs under `algo/eval/runs`. Spec says eval goes into algo; **use `PROJECT_ROOT / "algo" / "eval" / "runs"`** and update any docs.

---

### Task 1: Scaffold packages + git mv core trees

**Files:**
- Create: `frontend/__init__.py`, `api/__init__.py`, `algo/__init__.py`
- Move: `web/index.html` → `frontend/index.html`; `web/frontend_server.py` → `frontend/frontend_server.py`
- Move: remaining `web/*.py` → `api/`; `web_ui.py` → `api/app.py`
- Move: `model/` → `algo/model/`; `train_pretrain.py` → `algo/train_pretrain.py`; `train_sft.py` → `algo/train_sft.py`; `generate.py` → `algo/generate.py`; `eval/` → `algo/eval/`

**Interfaces:**
- Consumes: none
- Produces: package layout as in File map (imports still broken until Task 2–3)

- [ ] **Step 1: Stop services if train idle (wyf)**

```bash
systemctl --user is-active miniseek-web miniseek-api miniseek-worker
# If a train is running, stop via API/worker first; else:
systemctl --user stop miniseek-web miniseek-api miniseek-worker
```

Expected: inactive (dead) or confirm no owned train PIDs.

- [ ] **Step 2: Create package dirs and empty inits**

```bash
cd /home/wyf/miniseek
mkdir -p frontend api algo
touch frontend/__init__.py api/__init__.py algo/__init__.py
```

- [ ] **Step 3: git mv frontend pieces**

```bash
git mv web/index.html frontend/index.html
git mv web/frontend_server.py frontend/frontend_server.py
```

- [ ] **Step 4: git mv api pieces**

```bash
git mv web_ui.py api/app.py
for f in worker_app.py worker_client.py task_runner.py bench_runner.py tasks_store.py bench_store.py arch_inspect.py; do
  git mv "web/$f" "api/$f"
done
# drop empty web package leftovers
rm -f web/__init__.py
rmdir web 2>/dev/null || rm -rf web
```

- [ ] **Step 5: git mv algo pieces**

```bash
git mv model algo/model
git mv train_pretrain.py algo/train_pretrain.py
git mv train_sft.py algo/train_sft.py
git mv generate.py algo/generate.py
git mv eval algo/eval
```

- [ ] **Step 6: Commit scaffold (imports still broken — OK)**

```bash
git add -A
git commit -m "refactor: scaffold frontend/api/algo layout (moves only)"
```

---

### Task 2: Rewrite api + frontend imports and PROJECT_ROOT

**Files:**
- Modify: `api/app.py`, `api/*.py`, `frontend/frontend_server.py`
- Test: `tests/test_frontend_proxy.py`, `tests/test_worker_app.py`, `tests/test_task_runner_unit.py`, `tests/test_worker_proxy_api.py`, `tests/test_arch_api.py`, `tests/test_bench_api.py`

**Interfaces:**
- Consumes: layout from Task 1
- Produces:
  - `api.app:app` FastAPI app
  - `frontend.frontend_server:app`
  - `api.worker_app:app`
  - `PROJECT_ROOT = Path(__file__).resolve().parents[1]` in api/* and frontend/*
  - `build_train_cmd` uses `PROJECT_ROOT / "algo" / script`

- [ ] **Step 1: Write failing import smoke test**

Create `tests/test_package_layout.py`:

```python
def test_import_entrypoints():
    import api.app
    import api.worker_app
    import frontend.frontend_server
    import algo.model.config
    assert api.app.app is not None
    assert api.worker_app.app is not None
    assert frontend.frontend_server.app is not None
```

- [ ] **Step 2: Run to verify fail/error**

```bash
cd /home/wyf/miniseek && python -m pytest tests/test_package_layout.py -v
```

Expected: FAIL (ImportError from old `from web` / `from model` inside modules).

- [ ] **Step 3: Rewrite imports in api/**

In every `api/*.py`:
- `from web` / `import web` → `from api` / `import api`
- `from model` → `from algo.model`
- In `api/app.py`: former `from web_ui` N/A; keep `app = FastAPI...`
- `PROJECT_ROOT = Path(__file__).resolve().parents[1]`
- `WEB_DIR` for monolith static fallback if still present → `PROJECT_ROOT / "frontend"` (or remove static serve from api if only frontend serves HTML)
- Defaults that pointed at `checkpoints/...`:

```python
DEFAULT_PRETRAIN_OUT = str(PROJECT_ROOT / "tasks" / "_default_pretrain")
DEFAULT_SFT_OUT = str(PROJECT_ROOT / "tasks" / "_default_sft")
DEFAULT_MOE_OUT = str(PROJECT_ROOT / "tasks" / "_default_moe")
```

(Note: task create still forces `out_dir` via store; these are UI defaults only.)

- Update `list_checkpoints_global` to scan `tasks/**/checkpoints/*.pt` instead of `checkpoints/**`:

```python
def list_checkpoints_global() -> List[str]:
    files = glob.glob(str(PROJECT_ROOT / "tasks" / "**" / "checkpoints" / "*.pt"), recursive=True)
    ...
```

- [ ] **Step 4: Fix task_runner train script path**

In `api/task_runner.py` `build_train_cmd`:

```python
script = "train_sft.py" if mode == "sft" else "train_pretrain.py"
...
cmd = [
    sys.executable,
    "-u",
    str(PROJECT_ROOT / "algo" / script),
    ...
]
```

- [ ] **Step 5: Fix bench_store RUNS_ROOT and bench_runner eval CLI path**

```python
# api/bench_store.py
RUNS_ROOT = PROJECT_ROOT / "algo" / "eval" / "runs"
```

In `api/bench_runner.py`, ensure `eval.run_benchmark` or script path uses `algo/eval/...` (match existing invocation style; if it calls `python -m eval.run_benchmark`, change to `python -m algo.eval.run_benchmark`).

- [ ] **Step 6: Fix frontend_server**

```python
PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = Path(__file__).resolve().parent  # frontend/
```

Update `tests/test_frontend_proxy.py` to `import frontend.frontend_server as fs`.

- [ ] **Step 7: Update all tests that import web_ui / web / model**

Replace:
- `from web_ui import app` → `from api.app import app`
- `import web_ui` → `import api.app as web_ui` **or** rewrite monkeypatch targets to `api.app._worker_request`
- `from web import X` → `from api import X`
- `from web.Y` → `from api.Y`
- `from model` → `from algo.model`
- `from train_pretrain import` → `from algo.train_pretrain import`

- [ ] **Step 8: Run targeted tests**

```bash
python -m pytest tests/test_package_layout.py tests/test_frontend_proxy.py tests/test_worker_app.py tests/test_task_runner_unit.py tests/test_worker_proxy_api.py tests/test_arch_api.py tests/test_bench_api.py -v
```

Expected: PASS (or only failures that belong to algo import Task 3).

- [ ] **Step 9: Commit**

```bash
git add api frontend tests
git commit -m "refactor: point api/frontend imports at new packages"
```

---

### Task 3: Rewrite algo imports + pyproject

**Files:**
- Modify: `algo/train_pretrain.py`, `algo/train_sft.py`, `algo/generate.py`, `algo/eval/**/*.py`, `algo/model/**` if needed
- Modify: `pyproject.toml`
- Test: `tests/test_moe.py`, `tests/test_config_tiny.py`, `tests/test_miniseek_lm_load.py`, `tests/test_arch_inspect.py`, `tests/test_run_benchmark_cli.py`

**Interfaces:**
- Consumes: `algo.model.*`
- Produces: `from algo.model...` throughout algo; setuptools includes `algo*`, `api*`, `frontend*`

- [ ] **Step 1: Update pyproject packages**

```toml
[tool.setuptools.packages.find]
include = ["algo*", "api*", "frontend*"]
```

Keep `pythonpath = ["."]`.

- [ ] **Step 2: Rewrite algo internal imports**

Replace `from model` / `import model` with `from algo.model` / `import algo.model` in:
- `algo/train_pretrain.py`, `algo/train_sft.py`, `algo/generate.py`, `algo/eval/*`

If eval uses relative imports already, keep them inside `algo.eval`.

- [ ] **Step 3: Fix any hardcoded `checkpoints/` defaults in train CLIs**

Search and change default `--out-dir` from `checkpoints/...` to a task-style path or require explicit `--out-dir` (CLI smoke uses explicit out-dir in tests).

- [ ] **Step 4: Run model/eval tests**

```bash
python -m pytest tests/test_moe.py tests/test_config_tiny.py tests/test_arch_inspect.py tests/test_miniseek_lm_load.py tests/test_run_benchmark_cli.py tests/test_package_layout.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add algo pyproject.toml tests
git commit -m "refactor: algo package imports and setuptools includes"
```

---

### Task 4: Delete root checkpoints/ + scrub docs/deploy

**Files:**
- Delete: `checkpoints/` (local, gitignored if so — still remove on disk)
- Modify: `deploy/miniseek-*.service`, `deploy/README.md`, `deploy/CUTOVER-2026-10-27.md`, `README.md`, `README_en.md`, `.gitignore` if it only mentions checkpoints
- Modify: any docs referencing `python train_pretrain.py` or `web_ui:app`

**Interfaces:**
- Consumes: final module paths from Tasks 2–3
- Produces: systemd ExecStart lines below

- [ ] **Step 1: Update systemd units**

`deploy/miniseek-web.service`:

```
ExecStart=/usr/bin/python3 -m uvicorn frontend.frontend_server:app --host 0.0.0.0 --port 7860
```

`deploy/miniseek-api.service`:

```
ExecStart=/usr/bin/python3 -m uvicorn api.app:app --host 127.0.0.1 --port 8000
```

`deploy/miniseek-worker.service`:

```
ExecStart=/usr/bin/python3 -m uvicorn api.worker_app:app --host 127.0.0.1 --port 8001
```

WorkingDirectory remains `/home/wyf/miniseek`.

- [ ] **Step 2: Update README train/web commands**

Document:
- `python -m uvicorn frontend.frontend_server:app --port 7860`
- `python algo/train_pretrain.py ...` or `python -m algo.train_pretrain`
- Remove root `checkpoints/` instructions; say weights under `tasks/<id>/checkpoints/`

- [ ] **Step 3: Delete checkpoints directory on wyf**

```bash
rm -rf /home/wyf/miniseek/checkpoints
# verify
test ! -e /home/wyf/miniseek/checkpoints && du -sh /home/wyf/miniseek/tasks | head
```

Expected: `checkpoints` gone; `tasks` still present.

- [ ] **Step 4: Full pytest**

```bash
cd /home/wyf/miniseek && python -m pytest -q
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add deploy README.md README_en.md docs .gitignore
git commit -m "chore: drop root checkpoints; update deploy and docs for three-dir layout"
```

---

### Task 5: wyf cutover + Windows sync

**Files:**
- Modify (on machine): `~/.config/systemd/user/miniseek-*.service` copies if installed from deploy/

- [ ] **Step 1: Install units and restart**

```bash
cp /home/wyf/miniseek/deploy/miniseek-web.service ~/.config/systemd/user/
cp /home/wyf/miniseek/deploy/miniseek-api.service ~/.config/systemd/user/
cp /home/wyf/miniseek/deploy/miniseek-worker.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user restart miniseek-worker miniseek-api miniseek-web
systemctl --user is-active miniseek-web miniseek-api miniseek-worker
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:7860/
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8000/api/defaults
```

Expected: all `active`; HTTP 200.

- [ ] **Step 2: Acceptance — restart web does not kill worker train**

Start a tiny train via UI/API; `systemctl --user restart miniseek-web`; confirm task still running; then Stop via API.

- [ ] **Step 3: Push and sync D:\\miniseek**

```bash
# wyf
git push origin HEAD
# Windows
cd D:\\miniseek && git pull --ff-only
```

- [ ] **Step 4: Final commit only if leftover doc fixes**

```bash
git status -sb
```

---

## Spec coverage checklist

| Spec item | Task |
|-----------|------|
| frontend/ from index + frontend_server | 1 |
| api/ from web_ui + web runners | 1–2 |
| algo/ from model/train/eval/generate | 1, 3 |
| root keeps data/tasks/tokenizer/deploy | 1 (unchanged) |
| delete checkpoints/ | 4 |
| systemd new module paths | 4–5 |
| no compat shims | 2–3 |
| pytest + three-service acceptance | 4–5 |
| README update | 4 |
