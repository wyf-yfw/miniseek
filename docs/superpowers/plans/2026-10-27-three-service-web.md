# Three-Service Web (frontend / API / worker) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split MiniSeek Web into three local services so restarting frontend or API never kills running train/bench processes; worker alone owns Popen lifecycle.

**Architecture:** Frontend (`:7860`) serves static `web/` and reverse-proxies `/api/*` to API (`127.0.0.1:8000`). API keeps CRUD/SSE/chat and HTTP-forwards start/stop to worker (`127.0.0.1:8001`). Worker calls `task_runner` / `bench_runner` with `start_new_session=True` (POSIX) and stops via process-group signals; disk (`task.json` / `eval/runs`) remains source of truth.

**Tech Stack:** FastAPI + uvicorn, httpx (already in requirements), Starlette StaticFiles + StreamingResponse proxy (no Caddy/nginx required on wyf), systemd user units, pytest + TestClient.

## Global Constraints

- Do not change training algorithms, hparam defaults, loss curve, or live-metrics SSE semantics.
- Do not change browser `/api/...` paths or `tasks/<id>/` / `eval/runs/<id>/` schemas (except optional reconcile fields already allowed by spec).
- `:8000` and `:8001` bind **127.0.0.1 only**; browser never talks to them directly.
- Windows (wangyifan6) may keep monolith via `MINISEEK_MONOLITH=1` (or missing worker); this plan does not require Windows systemd.
- No Redis/Celery/k8s; still at most one running train and one running bench.
- Coding only on wyf (`machineId 6ad0a143-39b5-4473-8b0a-7c7f4f2dedd2`); never Cursor CloudAgent; do not push unless the user asks.
- Spec: `docs/superpowers/specs/2026-10-27-three-service-web-design.md`.

---

## File map

| Path | Role |
|------|------|
| `web/task_runner.py` | Add POSIX `start_new_session` + Windows process-group spawn; stop via `killpg` / CTRL_BREAK; export `reattach_running_tails()`, `stop_all_owned()` for worker lifespan |
| `web/bench_runner.py` | Same spawn/stop isolation; export `reattach_running()`, `stop_all_owned()` |
| `web/worker_app.py` | **New** FastAPI worker: `/health`, `/v1/tasks/{id}/start\|stop`, `/v1/bench/{id}/start\|stop`; lifespan reattach + shutdown stop-all |
| `web_ui.py` | Proxy start/stop to worker when not monolith; keep CRUD/SSE/chat; optionally stop serving `/` static when behind frontend (keep `/` for monolith) |
| `web/frontend_server.py` | **New** static `web/` on `:7860` + streaming `/api` reverse proxy to API (SSE-safe) |
| `web/worker_client.py` | **New** thin httpx helper used by API (`forward_worker`) |
| `tests/test_task_runner_unit.py` | Spawn flags + killpg stop path (mocked Popen) |
| `tests/test_bench_runner_unit.py` | **New** bench spawn/stop isolation unit tests |
| `tests/test_worker_app.py` | **New** TestClient for worker routes |
| `tests/test_tasks_api.py` / `tests/test_worker_proxy_api.py` | API start/stop forwards to worker (mocked httpx) |
| `tests/test_frontend_proxy.py` | **New** static + `/api` proxy smoke |
| `deploy/miniseek-web.service` | Frontend unit `:7860` |
| `deploy/miniseek-api.service` | API unit `127.0.0.1:8000` |
| `deploy/miniseek-worker.service` | Worker unit `127.0.0.1:8001` |
| `deploy/README.md` | Cutover from `miniseek-webui` |

---

### Task 1: Process isolation in runners

**Files:**
- Modify: `web/task_runner.py` (`_spawn_training`, `stop_task`; add helpers)
- Modify: `web/bench_runner.py` (`start_run` Popen, `stop_run`; add helpers)
- Test: `tests/test_task_runner_unit.py`
- Create: `tests/test_bench_runner_unit.py`

**Interfaces:**
- Consumes: existing `start_task` / `stop_task` / `start_run` / `stop_run`, `_owned_procs`
- Produces:
  - `_spawn_kwargs() -> dict` shared flags for Popen
  - `_terminate_pid(pid: int, *, owned: Optional[subprocess.Popen] = None, timeout_s: float = 5.0) -> Optional[int]`
  - `reattach_running_tails() -> list[str]` (task ids reattached)
  - `stop_all_owned() -> None` (train)
  - `bench_runner.reattach_running() -> list[str]`
  - `bench_runner.stop_all_owned() -> None`

- [ ] **Step 1: Write failing tests for spawn flags + terminate**

Append to `tests/test_task_runner_unit.py`:

```python
import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest

from web import task_runner as tr


def test_spawn_training_uses_new_session_on_posix(tasks_root, tmp_path, monkeypatch):
    if sys.platform == "win32":
        pytest.skip("POSIX session test")
    log = tmp_path / "t.log"
    captured = {}

    def fake_popen(*args, **kwargs):
        captured.update(kwargs)
        m = MagicMock()
        m.pid = 55501
        m.poll.return_value = None
        return m

    monkeypatch.setattr(tr.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(tr.os, "open", lambda *a, **k: 3)
    monkeypatch.setattr(tr.os, "close", lambda *a, **k: None)
    tr._spawn_training([sys.executable, "-c", "pass"], log)
    assert captured.get("start_new_session") is True


def test_stop_task_uses_killpg_when_session_leader(tasks_root, monkeypatch):
    if sys.platform == "win32":
        pytest.skip("POSIX killpg test")
    t = ts.create_task(name="killpg", arch="tiny", mode="pretrain", hparams={"max_steps": 1})
    t["status"] = "running"
    t["pid"] = 77701
    ts.save_task(t)

    class FakeProc:
        pid = 77701
        def poll(self):
            return 0
        def wait(self, timeout=None):
            return 0

    tr._owned_procs[t["id"]] = FakeProc()
    calls = []

    def fake_killpg(pgid, sig):
        calls.append((pgid, sig))

    monkeypatch.setattr(tr, "pid_alive", lambda pid: len(calls) == 0)
    monkeypatch.setattr(tr.os, "killpg", fake_killpg, raising=False)
    # Ensure module has killpg attribute for patch path used by implementation
    if not hasattr(tr.os, "killpg"):
        monkeypatch.setattr(tr.os, "killpg", fake_killpg)
    out = tr.stop_task(t["id"])
    assert out["status"] == "stopped"
    assert calls and calls[0][0] == 77701
```

Create `tests/test_bench_runner_unit.py`:

```python
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from web import bench_store as bs
from web import bench_runner as br


@pytest.fixture()
def runs_root(tmp_path, monkeypatch):
    root = tmp_path / "runs"
    monkeypatch.setattr(bs, "RUNS_ROOT", root)
    return root


def test_bench_spawn_start_new_session(runs_root, tmp_path, monkeypatch):
    if sys.platform == "win32":
        pytest.skip("POSIX session test")
    ckpt = tmp_path / "a.pt"
    ckpt.write_bytes(b"x")
    run = bs.create_run(ckpt_path=str(ckpt), tasks=["gsm8k"], limit=1)
    captured = {}

    def fake_popen(*args, **kwargs):
        captured.update(kwargs)
        m = MagicMock()
        m.pid = 88801
        m.poll.return_value = None
        return m

    monkeypatch.setattr(br.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(br, "choose_device", lambda **k: "cpu")
    monkeypatch.setattr(br, "pid_alive", lambda pid: True)
    out = br.start_run(run["id"], training_busy=False)
    assert out["status"] == "running"
    assert captured.get("start_new_session") is True
```

(If `bs.create_run` name differs, use the same create path as `tests/test_bench_api.py` / `bench_store.create_run` — inspect `web/bench_store.py` and match the real factory name.)

- [ ] **Step 2: Run tests — expect FAIL**

```bash
cd /home/wyf/miniseek && python3 -m pytest tests/test_task_runner_unit.py::test_spawn_training_uses_new_session_on_posix tests/test_task_runner_unit.py::test_stop_task_uses_killpg_when_session_leader tests/test_bench_runner_unit.py::test_bench_spawn_start_new_session -v
```

Expected: FAIL (`start_new_session` missing / `killpg` not called).

- [ ] **Step 3: Implement spawn/stop helpers in `web/task_runner.py`**

Near `_spawn_training`, replace Popen body and add terminate helpers:

```python
def _popen_isolation_kwargs() -> dict:
    kwargs: dict = {}
    if sys.platform == "win32":
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(
            subprocess, "CREATE_NEW_PROCESS_GROUP", 0
        )
        kwargs["creationflags"] = flags
    else:
        kwargs["start_new_session"] = True
    return kwargs


def _terminate_pid(
    pid: int,
    *,
    owned: Optional[subprocess.Popen] = None,
    timeout_s: float = 5.0,
) -> Optional[int]:
    """SIGTERM process group (POSIX session leader) or CTRL_BREAK (Windows); then SIGKILL."""
    pid_i = int(pid)
    try:
        if sys.platform == "win32":
            os.kill(pid_i, signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
        else:
            os.killpg(pid_i, signal.SIGTERM)
    except (OSError, ProcessLookupError, AttributeError):
        try:
            os.kill(pid_i, signal.SIGTERM)
        except OSError:
            pass

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if owned is not None and owned.poll() is not None:
            return owned.poll()
        if not pid_alive(pid_i):
            break
        time.sleep(0.1)
    else:
        try:
            if sys.platform == "win32":
                os.kill(pid_i, signal.SIGTERM)
            else:
                os.killpg(pid_i, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            try:
                os.kill(pid_i, signal.SIGKILL)
            except OSError:
                pass

    code = None
    if owned is not None:
        try:
            code = owned.wait(timeout=2)
        except Exception:  # noqa: BLE001
            code = owned.poll()
    return code


def _spawn_training(cmd: List[str], log_path: Path) -> subprocess.Popen:
    env = os.environ.copy()
    env.pop("all_proxy", None)
    env.pop("ALL_PROXY", None)
    env["PYTHONUNBUFFERED"] = "1"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_bytes(b"")
    fd = os.open(log_path, os.O_WRONLY | os.O_CREAT, 0o644)
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=fd,
            stderr=subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
            env=env,
            **_popen_isolation_kwargs(),
        )
    finally:
        os.close(fd)
    return proc
```

In `stop_task`, replace the `os.kill` + poll loop with:

```python
    owned = _owned_procs.get(task_id)
    code = _terminate_pid(int(pid), owned=owned)
    _owned_procs.pop(task_id, None)
```

Add reattach + stop-all at end of module:

```python
def reattach_running_tails() -> List[str]:
    """Worker boot: for disk status=running with live pid, restart log→metrics tail (no new Popen)."""
    attached: List[str] = []
    for row in ts.list_tasks():
        if row.get("status") != "running":
            continue
        task = ts.load_task(row["id"])
        pid = task.get("pid")
        if not pid_alive(pid):
            reconcile_task(task)
            continue
        task_id = task["id"]
        plot_every = int(
            (task.get("hparams") or {}).get("plot_every")
            or (1 if task.get("arch") == "tiny" else 500)
        )
        with _tail_lock:
            token = _tail_tokens.get(task_id, 0) + 1
            _tail_tokens[task_id] = token
        threading.Thread(
            target=_tail_log,
            args=(task_id, int(pid), token, max(1, plot_every)),
            daemon=True,
        ).start()
        attached.append(task_id)
    return attached


def stop_all_owned() -> None:
    """Worker shutdown hook: stop every running train task (owned or disk-running)."""
    ids = set(_owned_procs.keys())
    for row in ts.list_tasks():
        if row.get("status") == "running":
            ids.add(row["id"])
    for tid in list(ids):
        try:
            stop_task(tid)
        except Exception:  # noqa: BLE001
            pass
```

- [ ] **Step 4: Mirror isolation in `web/bench_runner.py`**

Import `_popen_isolation_kwargs` and `_terminate_pid` from `task_runner` (or duplicate thin wrappers to avoid circular imports — prefer import from `task_runner` since bench already imports `pid_alive`).

In `start_run`, replace `creationflags=...` block with `**_popen_isolation_kwargs()` (drop the win32-only creationflags local).

In `stop_run`, use `_terminate_pid` like train.

Add:

```python
def reattach_running() -> List[str]:
    """Mark live running benches as known; no log-tail thread today."""
    attached: List[str] = []
    for row in bs.list_runs():
        if row.get("status") != "running":
            continue
        run = reconcile_run(bs.load_run(row["id"]))
        if run.get("status") == "running" and pid_alive(run.get("pid")):
            attached.append(run["id"])
    return attached


def stop_all_owned() -> None:
    ids = set(_owned_procs.keys())
    for row in bs.list_runs():
        if row.get("status") == "running":
            ids.add(row["id"])
    for rid in list(ids):
        try:
            stop_run(rid)
        except Exception:  # noqa: BLE001
            pass
```

- [ ] **Step 5: Run unit tests — expect PASS**

```bash
cd /home/wyf/miniseek && python3 -m pytest tests/test_task_runner_unit.py tests/test_bench_runner_unit.py -v
```

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
cd /home/wyf/miniseek && git add web/task_runner.py web/bench_runner.py tests/test_task_runner_unit.py tests/test_bench_runner_unit.py && git commit -m "$(cat <<'MSG'
fix: detach train/bench process group from parent

MSG
)"
```

---

### Task 2: Worker FastAPI app

**Files:**
- Create: `web/worker_app.py`
- Create: `tests/test_worker_app.py`

**Interfaces:**
- Consumes: `start_task`, `stop_task`, `reattach_running_tails`, `stop_all_owned` from `task_runner`; `start_run`, `stop_run`, `reattach_running`, `stop_all_owned` from `bench_runner`
- Produces: FastAPI `app` with:
  - `GET /health` → `{ok: true}`
  - `POST /v1/tasks/{id}/start` → task dict (409/400 mapped)
  - `POST /v1/tasks/{id}/stop` → task dict
  - `POST /v1/bench/{id}/start` body optional `{training_busy?: bool}` → run dict
  - `POST /v1/bench/{id}/stop` → run dict

- [ ] **Step 1: Write failing TestClient tests**

```python
# tests/test_worker_app.py
from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from web import tasks_store as ts
from web import bench_store as bs


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(ts, "TASKS_ROOT", tmp_path / "tasks")
    monkeypatch.setattr(bs, "RUNS_ROOT", tmp_path / "runs")
    from web.worker_app import app
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"ok": True}


def test_task_start_stop_proxy(client, monkeypatch):
    t = ts.create_task(name="w", arch="tiny", mode="pretrain", hparams={"max_steps": 1})

    def fake_start(tid):
        task = ts.load_task(tid)
        task["status"] = "running"
        task["pid"] = 12345
        ts.save_task(task)
        return task

    def fake_stop(tid):
        task = ts.load_task(tid)
        task["status"] = "stopped"
        task["pid"] = None
        ts.save_task(task)
        return task

    monkeypatch.setattr("web.worker_app.start_task", fake_start)
    monkeypatch.setattr("web.worker_app.stop_task", fake_stop)
    r = client.post(f"/v1/tasks/{t['id']}/start")
    assert r.status_code == 200
    assert r.json()["status"] == "running"
    r2 = client.post(f"/v1/tasks/{t['id']}/stop")
    assert r2.status_code == 200
    assert r2.json()["status"] == "stopped"


def test_task_start_conflict_409(client, monkeypatch):
    t = ts.create_task(name="c", arch="tiny", mode="pretrain", hparams={})
    monkeypatch.setattr(
        "web.worker_app.start_task",
        lambda tid: (_ for _ in ()).throw(RuntimeError("another task is already running: x")),
    )
    r = client.post(f"/v1/tasks/{t['id']}/start")
    assert r.status_code == 409


def test_bench_start_passes_training_busy(client, tmp_path, monkeypatch):
    ckpt = tmp_path / "a.pt"
    ckpt.write_bytes(b"x")
    run = bs.create_run(ckpt_path=str(ckpt), tasks=["gsm8k"], limit=1)
    seen = {}

    def fake_start(rid, *, training_busy):
        seen["busy"] = training_busy
        run_d = bs.load_run(rid)
        run_d["status"] = "running"
        run_d["pid"] = 9
        bs.save_run(run_d)
        return run_d

    monkeypatch.setattr("web.worker_app.bench_start_run", fake_start)
    r = client.post(f"/v1/bench/{run['id']}/start", json={"training_busy": True})
    assert r.status_code == 200
    assert seen["busy"] is True
```

- [ ] **Step 2: Run — expect FAIL (module missing)**

```bash
cd /home/wyf/miniseek && python3 -m pytest tests/test_worker_app.py -v
```

Expected: FAIL `ModuleNotFoundError: web.worker_app`.

- [ ] **Step 3: Implement `web/worker_app.py`**

```python
"""MiniSeek worker control plane — owns train/bench Popen lifecycle."""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from web.task_runner import (
    reattach_running_tails,
    start_task,
    stop_all_owned as stop_all_train,
    stop_task,
)
from web.bench_runner import (
    reattach_running as reattach_bench,
    start_run as bench_start_run,
    stop_all_owned as stop_all_bench,
    stop_run as bench_stop_run,
)


class BenchStartBody(BaseModel):
    training_busy: bool = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    reattach_running_tails()
    reattach_bench()
    try:
        yield
    finally:
        stop_all_train()
        stop_all_bench()


app = FastAPI(title="MiniSeek Worker", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/v1/tasks/{task_id}/start")
def worker_start_task(task_id: str) -> dict:
    try:
        return start_task(task_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (ValueError, FileNotFoundError, NotImplementedError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/tasks/{task_id}/stop")
def worker_stop_task(task_id: str) -> dict:
    try:
        return stop_task(task_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/v1/bench/{run_id}/start")
def worker_start_bench(run_id: str, body: Optional[BenchStartBody] = None) -> dict:
    busy = bool(body.training_busy) if body is not None else False
    try:
        return bench_start_run(run_id, training_busy=busy)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ValueError, NotImplementedError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/v1/bench/{run_id}/stop")
def worker_stop_bench(run_id: str) -> dict:
    try:
        return bench_stop_run(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
cd /home/wyf/miniseek && python3 -m pytest tests/test_worker_app.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/wyf/miniseek && git add web/worker_app.py tests/test_worker_app.py && git commit -m "$(cat <<'MSG'
feat: add miniseek worker HTTP control plane

MSG
)"
```

---

### Task 3: API proxies start/stop to worker

**Files:**
- Create: `web/worker_client.py`
- Modify: `web_ui.py` (`api_start_task`, `api_stop_task`, `api_bench_start_task`, `api_bench_stop_task`; keep lifespan reconcile read-only)
- Create: `tests/test_worker_proxy_api.py`
- Modify lightly: `tests/test_tasks_api.py` only if imports break (CRUD tests must still pass)

**Interfaces:**
- Consumes: worker HTTP contract from Task 2
- Produces:
  - Env `MINISEEK_WORKER_URL` default `http://127.0.0.1:8001`
  - Env `MINISEEK_MONOLITH` (`1`/`true`/`yes`) → call local `start_task`/`stop_task` (Windows / single-process)
  - `worker_client.forward(method, path, *, json=None) -> httpx.Response` raising mapped errors
  - API routes still expose `/api/tasks/{id}/start|stop` and `/api/bench/tasks/{id}/start|stop`

- [ ] **Step 1: Write failing proxy tests**

```python
# tests/test_worker_proxy_api.py
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from web import tasks_store as ts


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(ts, "TASKS_ROOT", tmp_path / "tasks")
    monkeypatch.delenv("MINISEEK_MONOLITH", raising=False)
    monkeypatch.setenv("MINISEEK_WORKER_URL", "http://127.0.0.1:8001")
    # Force non-monolith path
    monkeypatch.setenv("MINISEEK_MONOLITH", "0")
    import importlib
    import web_ui
    importlib.reload(web_ui)
    return TestClient(web_ui.app)


def test_api_start_forwards_to_worker(client, monkeypatch):
    t = ts.create_task(name="p", arch="tiny", mode="pretrain", hparams={})
    fake = MagicMock()
    fake.status_code = 200
    fake.json.return_value = {"id": t["id"], "status": "running", "pid": 42}
    fake.raise_for_status = MagicMock()
    monkeypatch.setattr("web.worker_client.httpx.request", lambda *a, **k: fake)
    # Prefer patching the helper the route calls:
    monkeypatch.setattr(
        "web_ui._worker_request",
        lambda method, path, **kw: {"id": t["id"], "status": "running", "pid": 42},
    )
    r = client.post(f"/api/tasks/{t['id']}/start")
    assert r.status_code == 200
    assert r.json()["status"] == "running"


def test_api_start_maps_worker_409(client, monkeypatch):
    t = ts.create_task(name="q", arch="tiny", mode="pretrain", hparams={})
    from fastapi import HTTPException

    def boom(method, path, **kw):
        raise HTTPException(status_code=409, detail="another task is already running: x")

    monkeypatch.setattr("web_ui._worker_request", boom)
    r = client.post(f"/api/tasks/{t['id']}/start")
    assert r.status_code == 409
```

- [ ] **Step 2: Run — expect FAIL**

```bash
cd /home/wyf/miniseek && python3 -m pytest tests/test_worker_proxy_api.py -v
```

Expected: FAIL (`_worker_request` missing).

- [ ] **Step 3: Implement `web/worker_client.py`**

```python
"""HTTP client from API → worker."""
from __future__ import annotations

import os
from typing import Any, Optional

import httpx
from fastapi import HTTPException

DEFAULT_WORKER_URL = "http://127.0.0.1:8001"


def worker_base_url() -> str:
    return (os.environ.get("MINISEEK_WORKER_URL") or DEFAULT_WORKER_URL).rstrip("/")


def use_monolith() -> bool:
    flag = (os.environ.get("MINISEEK_MONOLITH") or "").strip().lower()
    if flag in {"1", "true", "yes", "on"}:
        return True
    if flag in {"0", "false", "no", "off"}:
        return False
    # Default: monolith only on Windows so local uvicorn still works without worker.
    import sys
    return sys.platform == "win32"


def forward(method: str, path: str, *, json: Optional[dict] = None, timeout: float = 60.0) -> Any:
    url = worker_base_url() + path
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.request(method, url, json=json)
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"worker unreachable: {exc}") from exc
    if resp.status_code >= 400:
        detail: Any
        try:
            payload = resp.json()
            detail = payload.get("detail", payload)
        except Exception:  # noqa: BLE001
            detail = resp.text
        raise HTTPException(status_code=resp.status_code, detail=detail)
    if resp.status_code == 204 or not resp.content:
        return {"ok": True}
    return resp.json()
```

- [ ] **Step 4: Wire `web_ui.py` start/stop**

Near imports, add:

```python
from web import worker_client as wc
```

Add helper:

```python
def _worker_request(method: str, path: str, **kwargs):
    return wc.forward(method, path, **kwargs)
```

Replace `api_start_task` / `api_stop_task`:

```python
@app.post("/api/tasks/{task_id}/start")
def api_start_task(task_id: str) -> dict:
    _get_task_or_404(task_id)
    if wc.use_monolith():
        try:
            return start_task(task_id)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (ValueError, FileNotFoundError, NotImplementedError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _worker_request("POST", f"/v1/tasks/{task_id}/start")


@app.post("/api/tasks/{task_id}/stop")
def api_stop_task(task_id: str) -> dict:
    _get_task_or_404(task_id)
    if wc.use_monolith():
        return stop_task(task_id)
    return _worker_request("POST", f"/v1/tasks/{task_id}/stop")
```

Replace bench start/stop similarly:

```python
@app.post("/api/bench/tasks/{run_id}/start")
def api_bench_start_task(run_id: str) -> dict:
    _get_bench_or_404(run_id)  # use existing helper name if different
    body = {"training_busy": _any_training_running()}
    if wc.use_monolith():
        try:
            run = bench_start_run(run_id, training_busy=body["training_busy"])
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (ValueError, NotImplementedError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return _bench_with_progress(run)
    run = _worker_request("POST", f"/v1/bench/{run_id}/start", json=body)
    return _bench_with_progress(run)


@app.post("/api/bench/tasks/{run_id}/stop")
def api_bench_stop_task(run_id: str) -> dict:
    _get_bench_or_404(run_id)
    if wc.use_monolith():
        return _bench_with_progress(bench_stop_run(run_id))
    return _bench_with_progress(_worker_request("POST", f"/v1/bench/{run_id}/stop"))
```

Keep API lifespan reconcile as **read-only disk repair for dead pids** (current behavior). Do **not** call `start_task` from API lifespan.

Document in a one-line comment above `use_monolith`: Windows default monolith; set `MINISEEK_MONOLITH=0` + run worker to split on Windows.

- [ ] **Step 5: Run proxy + existing API tests**

```bash
cd /home/wyf/miniseek && MINISEEK_MONOLITH=1 python3 -m pytest tests/test_tasks_api.py tests/test_worker_proxy_api.py tests/test_bench_api.py -v
```

Expected: PASS (CRUD tests use monolith; proxy tests use mocked `_worker_request`).

Note: if `test_bench_api.py` still patches `subprocess.Popen` through the API, set `MINISEEK_MONOLITH=1` in that module's fixture via `monkeypatch.setenv` so local Popen path remains, **or** update those tests to mock `_worker_request` — prefer env monolith in existing bench API fixture to minimize churn.

- [ ] **Step 6: Commit**

```bash
cd /home/wyf/miniseek && git add web/worker_client.py web_ui.py tests/test_worker_proxy_api.py tests/test_bench_api.py tests/test_tasks_api.py && git commit -m "$(cat <<'MSG'
feat: API forwards train/bench start-stop to worker

MSG
)"
```

---

### Task 4: Frontend static + reverse proxy

**Files:**
- Create: `web/frontend_server.py`
- Create: `tests/test_frontend_proxy.py`
- Do **not** change relative `/api` URLs in `web/index.html`

**Interfaces:**
- Consumes: API at `MINISEEK_API_URL` default `http://127.0.0.1:8000`
- Produces: FastAPI/Starlette app listening for systemd on `0.0.0.0:7860`:
  - `GET /` → `web/index.html`
  - static files under `web/` (no directory listing of secrets)
  - `/{path}` proxy for anything starting with `api/` → API, streaming body, hop-by-hop headers stripped; **no response buffering** for `text/event-stream`

- [ ] **Step 1: Write failing proxy test**

```python
# tests/test_frontend_proxy.py
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(monkeypatch, tmp_path):
    # Point WEB_DIR at a tiny fixture copy of index.html
    web_dir = tmp_path / "web"
    web_dir.mkdir()
    (web_dir / "index.html").write_text("<html>ok</html>", encoding="utf-8")
    monkeypatch.setenv("MINISEEK_API_URL", "http://127.0.0.1:8000")
    import importlib
    import web.frontend_server as fs
    monkeypatch.setattr(fs, "WEB_DIR", web_dir)
    importlib.reload(fs)
    monkeypatch.setattr(fs, "WEB_DIR", web_dir)
    return TestClient(fs.app)


def test_serves_index(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "ok" in r.text


def test_api_proxy_forwards(client, monkeypatch):
    import web.frontend_server as fs

    class FakeResp:
        status_code = 200
        headers = {"content-type": "application/json"}
        content = b'{"ok":true}'

        def iter_bytes(self):
            yield self.content

        def close(self):
            return None

    class FakeClient:
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def build_request(self, *a, **k):
            return object()
        def send(self, *a, **k):
            return FakeResp()

    monkeypatch.setattr(fs.httpx, "Client", lambda **kw: FakeClient())
    r = client.get("/api/defaults")
    assert r.status_code == 200
```

- [ ] **Step 2: Run — expect FAIL**

```bash
cd /home/wyf/miniseek && python3 -m pytest tests/test_frontend_proxy.py -v
```

Expected: FAIL missing module.

- [ ] **Step 3: Implement `web/frontend_server.py`**

```python
"""Static frontend + /api reverse proxy (SSE-safe)."""
from __future__ import annotations

import os
from pathlib import Path

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = Path(__file__).resolve().parent
API_URL = (os.environ.get("MINISEEK_API_URL") or "http://127.0.0.1:8000").rstrip("/")

HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "content-encoding",
    "content-length",
}

app = FastAPI(title="MiniSeek Web")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html", headers={"Cache-Control": "no-cache"})


@app.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
async def proxy_api(path: str, request: Request):
    url = f"{API_URL}/api/{path}"
    if request.url.query:
        url = f"{url}?{request.url.query}"
    headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in HOP_BY_HOP and k.lower() != "host"
    }
    body = await request.body()

    client = httpx.AsyncClient(timeout=None)
    req = client.build_request(request.method, url, headers=headers, content=body)
    upstream = await client.send(req, stream=True)

    out_headers = {
        k: v for k, v in upstream.headers.items() if k.lower() not in HOP_BY_HOP
    }
    # Disable intermediary buffering hints for SSE
    if "text/event-stream" in (upstream.headers.get("content-type") or ""):
        out_headers["Cache-Control"] = "no-cache"
        out_headers["X-Accel-Buffering"] = "no"

    async def stream():
        try:
            async for chunk in upstream.aiter_bytes():
                yield chunk
        finally:
            await upstream.aclose()
            await client.aclose()

    return StreamingResponse(
        stream(),
        status_code=upstream.status_code,
        headers=out_headers,
        media_type=upstream.headers.get("content-type"),
    )


# Static assets after routes so / and /api win
app.mount("/", StaticFiles(directory=str(WEB_DIR), html=False), name="static")
```

Keep `web_ui.py` `GET /` for monolith/Windows; on wyf three-service mode the frontend unit serves static instead.

- [ ] **Step 4: Run tests**

```bash
cd /home/wyf/miniseek && python3 -m pytest tests/test_frontend_proxy.py -v
```

Expected: PASS. Manually fix FakeClient to AsyncClient pattern if TestClient requires async mock — use `httpx.MockTransport` if simpler:

```python
def handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"ok": True})
transport = httpx.MockTransport(handler)
```

Adjust implementation/tests until green; do not leave TBD.

- [ ] **Step 5: Commit**

```bash
cd /home/wyf/miniseek && git add web/frontend_server.py tests/test_frontend_proxy.py && git commit -m "$(cat <<'MSG'
feat: frontend static server with /api reverse proxy

MSG
)"
```

---

### Task 5: systemd units + migrate docs

**Files:**
- Create: `deploy/miniseek-web.service`
- Create: `deploy/miniseek-api.service`
- Create: `deploy/miniseek-worker.service`
- Create: `deploy/README.md`

**Interfaces:**
- Consumes: apps from Tasks 2–4
- Produces: user units with WorkingDirectory `/home/wyf/miniseek`, Restart=on-failure; web After/Wants api; api After/Wants worker

- [ ] **Step 1: Write unit files**

`deploy/miniseek-worker.service`:

```ini
[Unit]
Description=MiniSeek Worker (train/bench Popen)
After=network-online.target

[Service]
Type=simple
WorkingDirectory=/home/wyf/miniseek
Environment=PATH=/home/wyf/.local/bin:/usr/bin:/bin
Environment=PYTHONIOENCODING=utf-8
Environment=MINISEEK_MONOLITH=0
ExecStart=/usr/bin/python3 -m uvicorn web.worker_app:app --host 127.0.0.1 --port 8001
Restart=on-failure
RestartSec=5
# Prefer graceful lifespan stop-all; avoid surprise cgroup kill of detached sessions
KillMode=mixed
TimeoutStopSec=120

[Install]
WantedBy=default.target
```

`deploy/miniseek-api.service`:

```ini
[Unit]
Description=MiniSeek API
After=network-online.target miniseek-worker.service
Wants=miniseek-worker.service

[Service]
Type=simple
WorkingDirectory=/home/wyf/miniseek
Environment=PATH=/home/wyf/.local/bin:/usr/bin:/bin
Environment=PYTHONIOENCODING=utf-8
Environment=MINISEEK_MONOLITH=0
Environment=MINISEEK_WORKER_URL=http://127.0.0.1:8001
ExecStart=/usr/bin/python3 -m uvicorn web_ui:app --host 127.0.0.1 --port 8000
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
```

`deploy/miniseek-web.service`:

```ini
[Unit]
Description=MiniSeek Web frontend (static + /api proxy)
After=network-online.target miniseek-api.service
Wants=miniseek-api.service

[Service]
Type=simple
WorkingDirectory=/home/wyf/miniseek
Environment=PATH=/home/wyf/.local/bin:/usr/bin:/bin
Environment=PYTHONIOENCODING=utf-8
Environment=MINISEEK_API_URL=http://127.0.0.1:8000
ExecStart=/usr/bin/python3 -m uvicorn web.frontend_server:app --host 0.0.0.0 --port 7860
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
```

- [ ] **Step 2: Write `deploy/README.md`**

Include exact cutover commands:

```bash
# From repo root on wyf
mkdir -p ~/.config/systemd/user
cp deploy/miniseek-*.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user disable --now miniseek-webui.service || true
systemctl --user enable --now miniseek-worker.service miniseek-api.service miniseek-web.service
systemctl --user status miniseek-worker miniseek-api miniseek-web --no-pager
curl -s http://127.0.0.1:8001/health
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8000/api/defaults
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:7860/
ss -ltnp | grep -E ':7860|:8000|:8001'
```

Acceptance notes: Funnel/public reverse proxy still points at `:7860`. Old `miniseek-webui.service` retired — do not enable both (port clash). Windows: `MINISEEK_MONOLITH=1 uvicorn web_ui:app --host 0.0.0.0 --port 7860`.

- [ ] **Step 3: Commit**

```bash
cd /home/wyf/miniseek && git add deploy/ && git commit -m "$(cat <<'MSG'
chore: systemd user units for web/api/worker

MSG
)"
```

---

### Task 6: wyf cutover + acceptance

**Files:** none required (ops); optional note `deploy/CUTOVER-2026-10-27.md` if useful

**Interfaces:**
- Consumes: units from Task 5; acceptance criteria 1–7 from spec

- [ ] **Step 1: Confirm no critical long train you must keep**

```bash
curl -s http://127.0.0.1:7860/api/status || true
systemctl --user is-active miniseek-webui.service || true
```

If a real long pretrain is `running` and user did not authorize kill, **stop and report** — do not cut over.

- [ ] **Step 2: Install units and switch**

```bash
cd /home/wyf/miniseek
mkdir -p ~/.config/systemd/user
cp deploy/miniseek-web.service deploy/miniseek-api.service deploy/miniseek-worker.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user disable --now miniseek-webui.service
systemctl --user enable --now miniseek-worker.service miniseek-api.service miniseek-web.service
sleep 2
systemctl --user is-active miniseek-worker miniseek-api miniseek-web
curl -s http://127.0.0.1:8001/health
curl -s http://127.0.0.1:8000/api/defaults | head -c 200; echo
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:7860/
ss -ltnp | grep -E '7860|8000|8001'
```

Expected: three `active`; health `{"ok":true}`; defaults JSON; `/` → `200`; `8000`/`8001` listeners on `127.0.0.1` only; `7860` on `0.0.0.0` or `*`.

- [ ] **Step 3: Tiny train smoke + restart isolation**

```bash
# Create + start tiny task via frontend port
TID=$(curl -s -X POST http://127.0.0.1:7860/api/tasks \
  -H 'Content-Type: application/json' \
  -d '{"name":"three-svc-smoke","arch":"tiny","mode":"pretrain","hparams":{"max_steps":30,"batch_size":2,"plot_every":1}}' \
  | python3 -c 'import sys,json; print(json.load(sys.stdin)["id"])')
curl -s -X POST "http://127.0.0.1:7860/api/tasks/${TID}/start" | python3 -m json.tool
PID=$(curl -s "http://127.0.0.1:7860/api/tasks/${TID}" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("pid"))')
echo "train_pid=$PID"
test -n "$PID" && test "$PID" != "None"
kill -0 "$PID"

systemctl --user restart miniseek-web.service
sleep 1
kill -0 "$PID"   # must still be alive

systemctl --user restart miniseek-api.service
sleep 2
kill -0 "$PID"   # must still be alive
curl -s "http://127.0.0.1:7860/api/tasks/${TID}" | python3 -c 'import sys,json; t=json.load(sys.stdin); assert t["status"]=="running", t'

curl -s -X POST "http://127.0.0.1:7860/api/tasks/${TID}/stop" | python3 -m json.tool
sleep 1
if kill -0 "$PID" 2>/dev/null; then echo "FAIL: pid still alive after Stop"; exit 1; else echo "Stop ok"; fi
```

Expected: pid survives web+api restarts; Stop kills process; `task.json` not `running`.

- [ ] **Step 4: Optional bench smoke (if time)**

Create a draft bench run against a tiny/finished ckpt if one exists; Start; `systemctl --user restart miniseek-web`; assert bench pid alive; Stop.

- [ ] **Step 5: Worker stop kills train (acceptance 4)**

```bash
# Start another tiny train, then:
systemctl --user stop miniseek-worker.service
# pid must die via lifespan stop_all; task status stopped/failed
systemctl --user start miniseek-worker.service
```

- [ ] **Step 6: Record results + commit note (no push)**

```bash
cd /home/wyf/miniseek && git status -sb
# If only docs/note changed:
# git add deploy/CUTOVER-2026-10-27.md && git commit -m "docs: three-service cutover acceptance on wyf"
```

Paste into the note: unit statuses, smoke task id, confirmation that `:8000`/`:8001` are localhost-only.

---

## Spec coverage checklist

| Spec acceptance / requirement | Task |
|-------------------------------|------|
| 1 Start pretrain/tiny → running + pid | Task 6 |
| 2 restart `miniseek-web` → train pid alive | Task 4–6 |
| 3 restart `miniseek-api` → train pid alive + running in UI | Task 3, 6 |
| 4 Stop UI→API→worker **or** stop/restart worker → train exits; restart web/api must **not** | Task 1–2 (killpg + lifespan), Task 6 |
| 5 Bench same isolation | Task 1–3, 6 |
| 6 Funnel/public still uses `:7860` create→Start→loss→Stop | Task 4–6 |
| 7 No `0.0.0.0:8000/8001` | Task 5 binds `127.0.0.1` |
| `start_new_session` + killpg | Task 1 |
| Worker `/health` + `/v1/...` | Task 2 |
| API forwards start/stop; CRUD/SSE stay on API | Task 3 |
| Frontend static + `/api` proxy SSE-safe | Task 4 |
| Three systemd user units; retire `miniseek-webui` | Task 5–6 |
| Worker crash → reattach tail on boot | Task 1 `reattach_running_tails`, Task 2 lifespan |
| Windows monolith optional | Task 3 `use_monolith()` |
| No algo / schema / browser path changes | Global Constraints |

## Self-review notes

- Placeholder scan: no TBD; bench `create_run` name must match `web/bench_store.py` at implement time (use existing test helpers).
- Type consistency: worker paths `/v1/tasks/{id}/start|stop`, `/v1/bench/{id}/start|stop`; API still `/api/tasks/...` and `/api/bench/tasks/...`.
- `stop_all_owned` on worker shutdown satisfies acceptance 4 without relying on systemd cgroup kill of detached sessions.
