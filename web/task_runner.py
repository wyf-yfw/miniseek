"""Per-task training process runner.

Starts/stops at most one training subprocess; pid and status live in task.json.
Logs and metrics stay under tasks/<id>/.
"""
from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from web import tasks_store as ts

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STEP_RE = re.compile(
    r"step=(\d+)\s+loss=([0-9.]+)(?:\s+ce=[0-9.]+)?(?:\s+aux=[0-9.]+)?\s+ppl=([0-9.]+)\s+lr=([0-9.eE+-]+)"
)
RESUME_RE = re.compile(r"resumed from .* step=(\d+)")

# In-memory tail tokens so a new start cancels an old tailer for same process manager.
_tail_lock = threading.Lock()
_tail_tokens: Dict[str, int] = {}
_owned_procs: Dict[str, subprocess.Popen] = {}


def pid_alive(pid: Optional[int]) -> bool:
    """True if process exists and is not a reaped/zombie exit.

    Prefer owned Popen.poll() (reaps on POSIX). os.kill(pid, 0) alone is
    not enough on Linux: zombies still pass kill(0).
    """
    if not pid:
        return False
    pid_i = int(pid)
    for proc in list(_owned_procs.values()):
        try:
            if proc.pid == pid_i:
                return proc.poll() is None
        except Exception:  # noqa: BLE001
            pass
    try:
        os.kill(pid_i, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    # Linux: zombie still answers kill(0); treat as dead for reconcile/tail.
    try:
        with open(f"/proc/{pid_i}/stat", "r", encoding="utf-8") as fh:
            data = fh.read()
        state = data.rsplit(")", 1)[-1].lstrip()[:1]
        if state == "Z":
            return False
    except OSError:
        pass
    return True


def reconcile_task(task: Dict[str, Any]) -> Dict[str, Any]:
    """If status=running but pid is dead, flip to finished/failed and clear pid."""
    if task.get("status") != "running":
        return task
    pid = task.get("pid")
    if pid_alive(pid):
        return task
    # Dead — prefer failed if we saw non-zero exit, else finished.
    exit_code = task.get("exit_code")
    owned = _owned_procs.pop(task["id"], None)
    if owned is not None:
        try:
            exit_code = owned.poll()
        except Exception:  # noqa: BLE001
            pass
    now = datetime.now().isoformat(timespec="seconds")
    if exit_code not in (None, 0):
        task["status"] = "failed"
        task["exit_code"] = exit_code
        task["error"] = task.get("error") or f"exit_code={exit_code}"
    else:
        task["status"] = "finished" if exit_code == 0 else "failed"
        if exit_code is None:
            # Unknown exit (orphaned / killed externally) — treat as failed
            task["status"] = "failed"
            task["error"] = task.get("error") or "process died without exit code"
        else:
            task["exit_code"] = exit_code
    task["pid"] = None
    task["ended_at"] = now
    ts.save_task(task)
    return task


def _hp(hparams: Dict[str, Any], key: str, default: Any) -> Any:
    if key in hparams and hparams[key] is not None and hparams[key] != "":
        return hparams[key]
    return default


def build_train_cmd(task: Dict[str, Any]) -> List[str]:
    hp = dict(task.get("hparams") or {})
    arch = (task.get("arch") or "dense").strip().lower()
    mode = (task.get("mode") or "pretrain").strip().lower()
    if mode not in {"pretrain", "sft"}:
        raise ValueError(f"未知训练模式: {mode}")
    if arch not in {"dense", "moe", "tiny"}:
        raise ValueError(f"未知架构: {arch}")
    if mode == "sft" and arch == "moe":
        raise ValueError("SFT 暂不支持 MoE")

    # Tiny defaults for smoke-friendly CPU runs
    if arch == "tiny":
        hp.setdefault("batch_size", 4)
        hp.setdefault("max_steps", 20)
        hp.setdefault("epochs", 1)
        hp.setdefault("max_seq_len", 128)
        hp.setdefault("warmup_steps", 0)
        hp.setdefault("log_every", 1)
        hp.setdefault("plot_every", 1)

    script = "train_sft.py" if mode == "sft" else "train_pretrain.py"
    data_path = str(_hp(hp, "data_path", PROJECT_ROOT / "data" / "pretrain_t2t_mini.jsonl"))
    out_dir = str(task["out_dir"])

    cmd = [
        sys.executable,
        "-u",
        str(PROJECT_ROOT / script),
        "--data",
        data_path,
        "--out-dir",
        out_dir,
        "--batch-size",
        str(int(_hp(hp, "batch_size", 4))),
        "--max-seq-len",
        str(int(_hp(hp, "max_seq_len", 512))),
        "--lr",
        str(float(_hp(hp, "lr", 1e-4))),
        "--min-lr",
        str(float(_hp(hp, "min_lr", 3e-5))),
        "--warmup-steps",
        str(int(_hp(hp, "warmup_steps", 200))),
        "--epochs",
        str(int(_hp(hp, "epochs", 1))),
        "--grad-clip",
        str(float(_hp(hp, "grad_clip", 1.0))),
        "--save-loss-thresholds",
        str(_hp(hp, "save_loss_thresholds", "20,15,12,10,9,8,7,6,5,4,3")),
        "--stable-steps",
        str(int(_hp(hp, "stable_steps", 200))),
        "--save-every",
        str(int(_hp(hp, "save_every", 0))),
        "--log-every",
        str(max(1, int(_hp(hp, "log_every", 10)))),
    ]
    max_steps = _hp(hp, "max_steps", None)
    if max_steps is not None and int(max_steps) > 0:
        cmd.extend(["--max-steps", str(int(max_steps))])

    if mode == "pretrain":
        cmd.extend(["--arch", arch, "--aux-coef", str(float(_hp(hp, "aux_coef", 0.01)))])

    resume = str(_hp(hp, "resume", "") or "").strip()
    init_ckpt = str(_hp(hp, "init_ckpt", "") or "").strip()
    if mode == "sft":
        if resume and Path(resume).exists():
            cmd.extend(["--resume", resume])
        elif init_ckpt and Path(init_ckpt).exists():
            cmd.extend(["--init-ckpt", init_ckpt])
    else:
        if resume and Path(resume).exists():
            cmd.extend(["--resume", resume])
    return cmd


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
        )
    finally:
        os.close(fd)
    return proc


def _parse_and_append_metrics(task_id: str, line: str, plot_every: int, state: Dict[str, Any]) -> None:
    resumed = RESUME_RE.search(line)
    if resumed:
        state["loss_bucket"] = int(resumed.group(1)) // max(1, plot_every)
    matched = STEP_RE.search(line)
    if not matched:
        return
    step = int(matched.group(1))
    loss = float(matched.group(2))
    every = max(1, plot_every)
    bucket = step // every
    if bucket < 1 or bucket <= state.get("loss_bucket", 0):
        return
    state["loss_bucket"] = bucket
    metrics = ts.read_metrics(task_id)
    points = list(metrics.get("loss") or [])
    points.append({"step": step, "loss": loss, "ppl": float(matched.group(3)), "lr": float(matched.group(4))})
    if len(points) > 5000:
        points = points[-5000:]
    metrics["loss"] = points
    ts.write_metrics(task_id, metrics)


def _tail_log(task_id: str, pid: int, token: int, plot_every: int) -> None:
    log_path = ts.task_dir(task_id) / "train.log"
    state: Dict[str, Any] = {"loss_bucket": 0}
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as fh:
            while True:
                with _tail_lock:
                    if _tail_tokens.get(task_id) != token:
                        return
                line = fh.readline()
                if line:
                    clean = line.replace("\x00", "").strip()
                    if clean:
                        _parse_and_append_metrics(task_id, clean, plot_every, state)
                    continue
                if not pid_alive(pid):
                    rest = fh.read().replace("\x00", "")
                    for extra in rest.splitlines():
                        extra = extra.strip()
                        if extra:
                            _parse_and_append_metrics(task_id, extra, plot_every, state)
                    task = ts.load_task(task_id)
                    if task.get("status") == "running" and task.get("pid") == pid:
                        owned = _owned_procs.pop(task_id, None)
                        code = None
                        if owned is not None:
                            try:
                                code = owned.wait(timeout=0.5)
                            except Exception:  # noqa: BLE001
                                code = owned.poll()
                        task["exit_code"] = code
                        task["pid"] = None
                        task["ended_at"] = datetime.now().isoformat(timespec="seconds")
                        if code == 0:
                            task["status"] = "finished"
                        elif task.get("status") == "stopped":
                            pass
                        else:
                            task["status"] = "failed" if code not in (0, None) else "finished"
                            if code not in (0, None):
                                task["error"] = f"exit_code={code}"
                            if code is None:
                                task["status"] = "finished"
                        # Don't overwrite user-initiated stopped
                        cur = ts.load_task(task_id)
                        if cur.get("status") == "stopped":
                            return
                        ts.save_task(task)
                    return
                time.sleep(0.05)
    except Exception as exc:  # noqa: BLE001
        try:
            task = ts.load_task(task_id)
            if task.get("status") == "running":
                task["error"] = f"log tail failed: {exc}"
                ts.save_task(task)
        except Exception:  # noqa: BLE001
            pass


def start_task(task_id: str) -> Dict[str, Any]:
    # Reconcile any stale running first
    for row in ts.list_tasks():
        if row.get("status") == "running":
            reconcile_task(ts.load_task(row["id"]))

    running_id = ts.find_running_task_id()
    if running_id and running_id != task_id:
        raise RuntimeError(f"another task is already running: {running_id}")
    if running_id == task_id and pid_alive(ts.load_task(task_id).get("pid")):
        raise RuntimeError(f"task already running: {task_id}")

    task = ts.load_task(task_id)
    if task.get("status") == "running" and pid_alive(task.get("pid")):
        raise RuntimeError(f"task already running: {task_id}")

    cmd = build_train_cmd(task)
    log_path = ts.task_dir(task_id) / "train.log"
    # Reset metrics for a fresh run
    ts.write_metrics(task_id, {"loss": [], "expert_load": None})

    proc = _spawn_training(cmd, log_path)
    now = datetime.now().isoformat(timespec="seconds")
    task["status"] = "running"
    task["pid"] = proc.pid
    task["started_at"] = now
    task["ended_at"] = None
    task["exit_code"] = None
    task["error"] = None
    ts.save_task(task)
    _owned_procs[task_id] = proc

    plot_every = int((task.get("hparams") or {}).get("plot_every") or (1 if task.get("arch") == "tiny" else 200))
    with _tail_lock:
        token = _tail_tokens.get(task_id, 0) + 1
        _tail_tokens[task_id] = token
    threading.Thread(
        target=_tail_log,
        args=(task_id, proc.pid, token, max(1, plot_every)),
        daemon=True,
    ).start()
    return ts.load_task(task_id)


def stop_task(task_id: str) -> Dict[str, Any]:
    task = ts.load_task(task_id)
    pid = task.get("pid")
    if task.get("status") != "running" or not pid_alive(pid):
        task = reconcile_task(task)
        if task.get("status") == "running":
            task["status"] = "stopped"
            task["pid"] = None
            task["ended_at"] = datetime.now().isoformat(timespec="seconds")
            ts.save_task(task)
        return task

    assert pid is not None
    try:
        os.kill(int(pid), signal.SIGTERM)
    except OSError:
        pass
    owned = _owned_procs.get(task_id)
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

    owned = _owned_procs.pop(task_id, None)
    code = None
    if owned is not None:
        try:
            code = owned.wait(timeout=2)
        except Exception:  # noqa: BLE001
            code = owned.poll()

    with _tail_lock:
        _tail_tokens[task_id] = _tail_tokens.get(task_id, 0) + 1  # cancel tailer

    task = ts.load_task(task_id)
    task["status"] = "stopped"
    task["pid"] = None
    task["exit_code"] = code
    task["ended_at"] = datetime.now().isoformat(timespec="seconds")
    ts.save_task(task)
    return task
