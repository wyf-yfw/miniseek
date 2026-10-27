"""Background PT clean jobs under data/clean/<id>/."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from api import clean_store as cs
from api.task_runner import _popen_isolation_kwargs, _terminate_pid, pid_alive

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_owned_procs: Dict[str, subprocess.Popen] = {}


def _any_clean_running() -> Optional[str]:
    for row in cs.list_jobs():
        if row.get("status") != "running":
            continue
        job = reconcile_job(cs.load_job(row["id"]))
        if job.get("status") == "running" and pid_alive(job.get("pid")):
            return job["id"]
    return None


def _ingest_report(job: Dict[str, Any]) -> bool:
    report = cs.read_report(job["id"])
    if not report:
        return False
    job["engine"] = report.get("engine")
    job["stats"] = {
        "input_rows": report.get("input_rows"),
        "kept": report.get("kept"),
        "dropped": report.get("dropped"),
        "per_operator": report.get("per_operator"),
        "seconds": report.get("seconds"),
    }
    return True


def reconcile_job(job: Dict[str, Any]) -> Dict[str, Any]:
    if job.get("status") != "running":
        return job
    pid = job.get("pid")
    owned = _owned_procs.get(job["id"])
    alive = pid_alive(pid)
    if owned is not None and owned.poll() is not None:
        alive = False
    has_report = (cs.job_dir(job["id"]) / "report.json").is_file()
    has_clean = (cs.job_dir(job["id"]) / "clean.jsonl").is_file()
    if alive and not (has_report and has_clean):
        return job
    if alive and has_report and has_clean:
        try:
            if pid is not None:
                import signal
                os.kill(int(pid), signal.SIGTERM)
        except OSError:
            pass
        alive = False
    _owned_procs.pop(job["id"], None)
    exit_code = job.get("exit_code")
    if owned is not None:
        try:
            exit_code = owned.poll()
        except Exception:  # noqa: BLE001
            pass
    if exit_code is None and has_report:
        exit_code = 0
    now = datetime.now().isoformat(timespec="seconds")
    job["pid"] = None
    job["ended_at"] = now
    job["exit_code"] = exit_code
    if _ingest_report(job) and (exit_code in (0, None)):
        job["status"] = "finished"
        job["error"] = None
    elif exit_code == 0:
        job["status"] = "failed"
        job["error"] = job.get("error") or "exit 0 but report.json missing"
    else:
        job["status"] = "failed"
        job["error"] = job.get("error") or f"exit_code={exit_code}"
    cs.save_job(job)
    return job


def start_job(job_id: str) -> Dict[str, Any]:
    for row in cs.list_jobs():
        if row.get("status") == "running":
            reconcile_job(cs.load_job(row["id"]))
    busy = _any_clean_running()
    if busy:
        raise RuntimeError(f"another clean job is already running: {busy}")

    job = cs.load_job(job_id)
    st = job.get("status")
    if st not in {"draft", "stopped", "failed", "finished"}:
        raise RuntimeError(f"只能从 draft/stopped/failed/finished 启动，当前 status={st}")

    input_path = (job.get("input_path") or "").strip()
    if not input_path:
        raise ValueError("input_path 不能为空")
    if not Path(input_path).is_file():
        raise FileNotFoundError(input_path)

    out_dir = cs.job_dir(job_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("clean.jsonl", "dropped.jsonl", "report.json"):
        p = out_dir / name
        if p.is_file():
            try:
                p.unlink()
            except OSError:
                pass

    cmd = [
        sys.executable,
        "-u",
        str(PROJECT_ROOT / "scripts" / "pt_clean_job.py"),
        "--job-dir",
        str(out_dir),
    ]
    log_path = out_dir / "clean.log"
    log_path.write_bytes(b"")
    env = os.environ.copy()
    env.pop("all_proxy", None)
    env.pop("ALL_PROXY", None)
    env["PYTHONUNBUFFERED"] = "1"
    log_fh = open(log_path, "w", encoding="utf-8", errors="replace", buffering=1)
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
            env=env,
            **_popen_isolation_kwargs(),
        )
    finally:
        log_fh.close()

    now = datetime.now().isoformat(timespec="seconds")
    job["status"] = "running"
    job["pid"] = proc.pid
    job["started_at"] = now
    job["ended_at"] = None
    job["exit_code"] = None
    job["error"] = None
    job["stats"] = None
    job["engine"] = None
    cs.save_job(job)
    _owned_procs[job_id] = proc
    return cs.load_job(job_id)


def stop_job(job_id: str) -> Dict[str, Any]:
    job = cs.load_job(job_id)
    pid = job.get("pid")
    if job.get("status") != "running" or not pid_alive(pid):
        job = reconcile_job(job)
        if job.get("status") == "running":
            job["status"] = "stopped"
            job["pid"] = None
            job["ended_at"] = datetime.now().isoformat(timespec="seconds")
            cs.save_job(job)
        return job

    assert pid is not None
    owned = _owned_procs.get(job_id)
    code = _terminate_pid(int(pid), owned=owned)
    _owned_procs.pop(job_id, None)

    job = cs.load_job(job_id)
    job["pid"] = None
    job["exit_code"] = code
    job["ended_at"] = datetime.now().isoformat(timespec="seconds")
    if _ingest_report(job):
        job["status"] = "finished"
        job["error"] = None
        if job.get("exit_code") is None:
            job["exit_code"] = 0
    else:
        job["status"] = "stopped"
    cs.save_job(job)
    return job


def read_log(job_id: str, *, tail: Optional[int] = None) -> str:
    path = cs.job_dir(job_id) / "clean.log"
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    if tail is None or tail <= 0:
        return text
    lines = text.splitlines()
    return "\n".join(lines[-tail:])


def reattach_running() -> List[str]:
    attached: List[str] = []
    for row in cs.list_jobs():
        if row.get("status") != "running":
            continue
        job = reconcile_job(cs.load_job(row["id"]))
        if job.get("status") == "running" and pid_alive(job.get("pid")):
            attached.append(job["id"])
    return attached


def stop_all_owned() -> None:
    ids = set(_owned_procs.keys())
    for row in cs.list_jobs():
        if row.get("status") == "running":
            ids.add(row["id"])
    for jid in list(ids):
        try:
            stop_job(jid)
        except Exception:  # noqa: BLE001
            pass


def run_preview(input_path: str, operators: Dict[str, Any], *, limit: int = 200) -> Dict[str, Any]:
    path = Path(input_path)
    if not path.is_file():
        raise FileNotFoundError(str(path))
    limit = max(1, min(int(limit or 200), 5000))
    ops = cs._merge_operators(operators)
    import importlib.util
    mod_path = PROJECT_ROOT / "scripts" / "pt_clean_job.py"
    spec = importlib.util.spec_from_file_location("pt_clean_job", mod_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load pt_clean_job")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.preview_samples(str(path), ops, limit=limit)
