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


def reconcile_run(run: Dict[str, Any]) -> Dict[str, Any]:
    if run.get("status") != "running":
        return run
    pid = run.get("pid")
    if pid_alive(pid):
        return run
    owned = _owned_procs.pop(run["id"], None)
    exit_code = run.get("exit_code")
    if owned is not None:
        try:
            exit_code = owned.poll()
        except Exception:  # noqa: BLE001
            pass
    now = datetime.now().isoformat(timespec="seconds")
    run["pid"] = None
    run["ended_at"] = now
    run["exit_code"] = exit_code
    results_path = bs.run_dir(run["id"]) / "results.json"
    if results_path.is_file() and exit_code == 0:
        try:
            payload = json.loads(results_path.read_text(encoding="utf-8"))
            run["scores"] = payload.get("metrics")
            run["status"] = "finished"
        except Exception as exc:  # noqa: BLE001
            run["status"] = "failed"
            run["error"] = run.get("error") or f"bad results.json: {exc}"
    elif exit_code == 0:
        run["status"] = "failed"
        run["error"] = run.get("error") or "exit 0 but results.json missing"
    else:
        run["status"] = "failed"
        run["error"] = run.get("error") or f"exit_code={exit_code}"
    bs.save_run(run)
    return run


def start_run(
    *,
    ckpt_path: str,
    tasks: List[str],
    limit: Optional[int],
    training_busy: bool,
) -> Dict[str, Any]:
    for row in bs.list_runs():
        if row.get("status") == "running":
            reconcile_run(bs.load_run(row["id"]))
    busy = _any_bench_running()
    if busy:
        raise RuntimeError(f"another bench run is already running: {busy}")

    ckpt = Path(ckpt_path)
    if not ckpt.exists():
        raise FileNotFoundError(str(ckpt_path))

    device = choose_device(training_busy=training_busy)
    run = bs.create_run(
        ckpt_path=str(ckpt_path),
        tasks=tasks,
        limit=limit,
        device=device,
    )
    out_dir = bs.run_dir(run["id"])
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "eval.run_benchmark",
        "--ckpt",
        str(ckpt_path),
        "--tasks",
        ",".join(run["tasks"]),
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
    fd = os.open(log_path, os.O_WRONLY | os.O_CREAT, 0o644)
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=fd,
            stderr=subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
            env=env,
        )
    finally:
        os.close(fd)

    now = datetime.now().isoformat(timespec="seconds")
    run["status"] = "running"
    run["pid"] = proc.pid
    run["started_at"] = now
    run["ended_at"] = None
    run["exit_code"] = None
    run["error"] = None
    bs.save_run(run)
    _owned_procs[run["id"]] = proc
    return bs.load_run(run["id"])


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
    run["status"] = "stopped"
    run["pid"] = None
    run["exit_code"] = code
    run["ended_at"] = datetime.now().isoformat(timespec="seconds")
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
