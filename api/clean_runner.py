"""Background PT clean jobs under data/clean/<id>/."""
from __future__ import annotations

import importlib.util
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


def _load_pt_clean_mod():
    mod_path = PROJECT_ROOT / "scripts" / "pt_clean_job.py"
    spec = importlib.util.spec_from_file_location("pt_clean_job", mod_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load pt_clean_job")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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
    # Wipe DJ leftovers even if the child died mid-run.
    try:
        mod = _load_pt_clean_mod()
        mod.cleanup_job_intermediates(cs.job_dir(job["id"]))
    except Exception:  # noqa: BLE001
        pass
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
    inp = Path(input_path)
    if not inp.is_file():
        raise FileNotFoundError(input_path)

    out_dir = cs.job_dir(job_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    mod = _load_pt_clean_mod()
    # Preflight: refuse before spawning if free disk << 4× input.
    try:
        info = mod.preflight_disk(inp, anchor=out_dir)
    except mod.DiskSpaceError as exc:
        raise ValueError(str(exc)) from exc

    for name in ("clean.jsonl", "dropped.jsonl", "report.json", "progress.json"):
        p = out_dir / name
        if p.is_file():
            try:
                p.unlink()
            except OSError:
                pass
    # Clear prior DJ/HF workdir so a restart does not pile onto old caches.
    mod.cleanup_job_intermediates(out_dir)

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
    # Even builtin path: keep process temp under the job dir when possible.
    tmp = out_dir / ".dj_work" / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    env.setdefault("TMPDIR", str(tmp))
    env.setdefault("TMP", str(tmp))
    env.setdefault("TEMP", str(tmp))
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
    job["disk_preflight"] = {
        "input_bytes": info["input_bytes"],
        "need_bytes": info["need_bytes"],
        "free_bytes": info["free_bytes"],
    }
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
    try:
        mod = _load_pt_clean_mod()
        mod.cleanup_job_intermediates(cs.job_dir(job_id))
    except Exception:  # noqa: BLE001
        pass
    if _ingest_report(job):
        job["status"] = "finished"
        job["error"] = None
        if job.get("exit_code") is None:
            job["exit_code"] = 0
    else:
        job["status"] = "stopped"
    cs.save_job(job)
    return job



def _parse_progress_from_log(job_id: str) -> Optional[Dict[str, Any]]:
    """Best-effort: last `progress rows=N kept=K dropped=D` line from clean.log."""
    import re

    text = read_log(job_id, tail=80)
    if not text:
        return None
    pat = re.compile(r"progress rows=(\d+)\s+kept=(\d+)\s+dropped=(\d+)")
    last = None
    for line in text.splitlines():
        m = pat.search(line)
        if m:
            last = {
                "input_rows": int(m.group(1)),
                "kept": int(m.group(2)),
                "dropped": int(m.group(3)),
            }
    return last


def job_progress(job: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return progress dict for UI. Prefer progress.json; else log + file-size estimate."""
    jid = job["id"]
    prog = cs.read_progress(jid)
    if prog and isinstance(prog, dict):
        out = dict(prog)
        out.setdefault("source", "progress.json")
        return out

    # Best-effort for jobs started before progress.json existed
    parsed = _parse_progress_from_log(jid)
    if not parsed and job.get("status") != "running":
        return None

    input_path = (job.get("input_path") or "").strip()
    total_bytes = 0
    disk = job.get("disk_preflight") or {}
    if disk.get("input_bytes"):
        total_bytes = int(disk["input_bytes"])
    elif input_path:
        try:
            total_bytes = int(Path(input_path).stat().st_size)
        except OSError:
            total_bytes = 0

    clean_path = cs.job_dir(jid) / "clean.jsonl"
    clean_bytes = 0
    if clean_path.is_file():
        try:
            clean_bytes = int(clean_path.stat().st_size)
        except OSError:
            clean_bytes = 0

    kept = (parsed or {}).get("kept")
    dropped = (parsed or {}).get("dropped")
    input_rows = (parsed or {}).get("input_rows")
    pct = None
    eta_sec = None
    # Estimate pct from output size vs expected kept fraction of input
    if total_bytes > 0 and kept is not None and dropped is not None:
        ratio = kept / max(1, kept + dropped)
        expected = max(1.0, total_bytes * max(0.05, min(1.0, ratio)))
        pct = round(min(99.5, 100.0 * clean_bytes / expected), 2) if clean_bytes else 0.0
    elif total_bytes > 0 and clean_bytes > 0:
        pct = round(min(99.5, 100.0 * clean_bytes / total_bytes), 2)

    started = job.get("started_at")
    elapsed_sec = None
    if started:
        try:
            from datetime import datetime as _dt
            t0 = _dt.fromisoformat(started)
            elapsed_sec = round((_dt.now() - t0).total_seconds(), 1)
            if pct is not None and 0 < pct < 100 and elapsed_sec and elapsed_sec > 0:
                eta_sec = round(elapsed_sec * (100.0 - pct) / pct, 1)
        except Exception:  # noqa: BLE001
            pass

    rate_rows_s = None
    if input_rows and elapsed_sec and elapsed_sec > 0:
        rate_rows_s = round(input_rows / elapsed_sec, 1)

    if parsed is None and clean_bytes <= 0 and job.get("status") != "running":
        return None

    return {
        "engine": job.get("engine") or "builtin",
        "phase": "streaming" if job.get("status") == "running" else (job.get("status") or "unknown"),
        "bytes_read": None,
        "total_bytes": total_bytes or None,
        "clean_bytes": clean_bytes,
        "input_rows": input_rows,
        "kept": kept,
        "dropped": dropped,
        "pct": pct,
        "elapsed_sec": elapsed_sec,
        "rate_rows_s": rate_rows_s,
        "eta_sec": eta_sec,
        "source": "log+filesize",
    }


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
    mod = _load_pt_clean_mod()
    return mod.preview_samples(str(path), ops, limit=limit)
