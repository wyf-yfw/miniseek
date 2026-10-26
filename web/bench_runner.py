"""Background lm-eval runs under eval/runs/<id>/."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from web import bench_store as bs
from web.task_runner import pid_alive

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_owned_procs: Dict[str, subprocess.Popen] = {}


def _any_bench_running() -> Optional[str]:
    for row in bs.list_runs():
        if row.get("status") != "running":
            continue
        run = reconcile_run(bs.load_run(row["id"]))
        if run.get("status") == "running" and pid_alive(run.get("pid")):
            return run["id"]
    return None


def choose_device(*, training_busy: bool) -> str:
    if torch.cuda.is_available() and not training_busy:
        return "cuda"
    return "cpu"



def _ingest_results(run: Dict[str, Any]) -> bool:
    """If results.json present, attach scores. Return True when file exists."""
    results_path = bs.run_dir(run["id"]) / "results.json"
    if not results_path.is_file():
        return False
    try:
        payload = json.loads(results_path.read_text(encoding="utf-8"))
        run["scores"] = payload.get("metrics")
        return True
    except Exception as exc:  # noqa: BLE001
        run["error"] = run.get("error") or f"bad results.json: {exc}"
        return True


def reconcile_run(run: Dict[str, Any]) -> Dict[str, Any]:
    if run.get("status") != "running":
        return run
    pid = run.get("pid")
    owned = _owned_procs.get(run["id"])
    alive = pid_alive(pid)
    if owned is not None and owned.poll() is not None:
        alive = False
    has_results = (bs.run_dir(run["id"]) / "results.json").is_file()
    # Results already on disk: treat as done even if a worker thread keeps the PID alive.
    if alive and not has_results:
        return run
    if alive and has_results:
        try:
            if pid is not None:
                os.kill(int(pid), signal.SIGTERM)
        except OSError:
            pass
        alive = False
    _owned_procs.pop(run["id"], None)
    exit_code = run.get("exit_code")
    if owned is not None:
        try:
            exit_code = owned.poll()
        except Exception:  # noqa: BLE001
            pass
    if exit_code is None and has_results:
        exit_code = 0
    now = datetime.now().isoformat(timespec="seconds")
    run["pid"] = None
    run["ended_at"] = now
    run["exit_code"] = exit_code
    if _ingest_results(run) and (exit_code in (0, None)):
        if run.get("error") and "bad results.json" in str(run.get("error")):
            run["status"] = "failed"
        else:
            run["status"] = "finished"
            run["error"] = None
    elif exit_code == 0:
        run["status"] = "failed"
        run["error"] = run.get("error") or "exit 0 but results.json missing"
    else:
        run["status"] = "failed"
        run["error"] = run.get("error") or f"exit_code={exit_code}"
    bs.save_run(run)
    return run


def start_run(run_id: str, *, training_busy: bool) -> Dict[str, Any]:
    for row in bs.list_runs():
        if row.get("status") == "running":
            reconcile_run(bs.load_run(row["id"]))
    busy = _any_bench_running()
    if busy:
        raise RuntimeError(f"another bench run is already running: {busy}")

    run = bs.load_run(run_id)
    st = run.get("status")
    if st not in {"draft", "stopped", "failed", "finished"}:
        raise RuntimeError(f"只能从 draft/stopped/failed 启动，当前 status={st}")

    ckpt_path = (run.get("ckpt_path") or "").strip()
    if not ckpt_path:
        raise ValueError("ckpt_path 不能为空")
    ckpt = Path(ckpt_path)
    if not ckpt.exists():
        raise FileNotFoundError(str(ckpt_path))

    tasks = list(run.get("tasks") or [])
    if not tasks:
        raise ValueError("tasks must be non-empty")
    limit = run.get("limit")

    device = choose_device(training_busy=training_busy)
    out_dir = bs.run_dir(run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    # reset progress / results for a fresh start
    results_path = out_dir / "results.json"
    if results_path.is_file():
        try:
            results_path.unlink()
        except OSError:
            pass
    bs.write_progress(run_id, {
        "suite_done": 0,
        "suite_total": len(tasks),
        "current_task": None,
        "sample_done": 0,
        "sample_total": 0,
    })

    cmd = [
        sys.executable,
        "-u",
        "-m",
        "eval.run_benchmark",
        "--ckpt",
        str(ckpt_path),
        "--tasks",
        ",".join(tasks),
        "--out",
        str(out_dir),
        "--device",
        device,
    ]
    if limit is not None:
        cmd.extend(["--limit", str(int(limit))])

    log_path = out_dir / "run.log"
    log_path.write_bytes(b"")
    env = os.environ.copy()
    env.pop("all_proxy", None)
    env.pop("ALL_PROXY", None)
    env["PYTHONUNBUFFERED"] = "1"
    env["BENCH_PROGRESS_PATH"] = str(out_dir / "progress.json")
    env["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    creationflags = 0
    if sys.platform == "win32":
        creationflags = (
            getattr(subprocess, "CREATE_NO_WINDOW", 0)
            | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        )
    log_fh = open(log_path, "w", encoding="utf-8", errors="replace", buffering=1)
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
            env=env,
            creationflags=creationflags,
        )
    finally:
        log_fh.close()

    now = datetime.now().isoformat(timespec="seconds")
    run["status"] = "running"
    run["device"] = device
    run["pid"] = proc.pid
    run["started_at"] = now
    run["ended_at"] = None
    run["exit_code"] = None
    run["error"] = None
    run["scores"] = None
    bs.save_run(run)
    _owned_procs[run_id] = proc
    return bs.load_run(run_id)


def stop_run(run_id: str) -> Dict[str, Any]:
    run = bs.load_run(run_id)
    pid = run.get("pid")
    if run.get("status") != "running" or not pid_alive(pid):
        run = reconcile_run(run)
        if run.get("status") == "running":
            run["status"] = "stopped"
            run["pid"] = None
            run["ended_at"] = datetime.now().isoformat(timespec="seconds")
            bs.save_run(run)
        return run

    assert pid is not None
    try:
        os.kill(int(pid), signal.SIGTERM)
    except OSError:
        pass
    owned = _owned_procs.get(run_id)
    for _ in range(50):
        if owned is not None and owned.poll() is not None:
            break
        if not pid_alive(pid):
            break
        time.sleep(0.1)
    else:
        try:
            os.kill(int(pid), signal.SIGKILL)
        except OSError:
            pass

    owned = _owned_procs.pop(run_id, None)
    code = None
    if owned is not None:
        try:
            code = owned.wait(timeout=2)
        except Exception:  # noqa: BLE001
            code = owned.poll()

    run = bs.load_run(run_id)
    run["pid"] = None
    run["exit_code"] = code
    run["ended_at"] = datetime.now().isoformat(timespec="seconds")
    if _ingest_results(run) and not (run.get("error") and "bad results.json" in str(run.get("error"))):
        run["status"] = "finished"
        run["error"] = None
        if run.get("exit_code") is None:
            run["exit_code"] = 0
    else:
        run["status"] = "stopped"
    bs.save_run(run)
    return run


def read_log(run_id: str, *, tail: Optional[int] = None) -> str:
    path = bs.run_dir(run_id) / "run.log"
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    if tail is None or tail <= 0:
        return text
    lines = text.splitlines()
    return "\n".join(lines[-tail:])
