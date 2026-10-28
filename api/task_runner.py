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

from api import tasks_store as ts

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STEP_RE = re.compile(
    r"step=(\d+)\s+loss=([0-9.]+)(?:\s+ce=[0-9.]+)?(?:\s+aux=[0-9.]+)?\s+ppl=([0-9.]+)\s+lr=([0-9.eE+-]+)"
)
RESUME_RE = re.compile(r"resumed from .* (?:lifetime_step|step)=(\d+)")
VALID_RE = re.compile(
    r"valid\s+step=(\d+)\s+loss=([0-9.eE+-]+)\s+ppl=([0-9.eE+-]+)"
    r"(?:\s+tokens=(\d+))?"
)

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


def validate_sft_init_ckpt(mode: str, hparams: Optional[Dict[str, Any]] = None, *, require_exists: bool = False) -> None:
    """SFT must fine-tune from a pretrained ckpt unless Resume is set.

    Resume takes priority when non-empty; otherwise Init Ckpt is required.
    When require_exists=True (start/build), the chosen path must exist on disk.
    """
    mode_l = (mode or "").strip().lower()
    if mode_l != "sft":
        return
    hp = dict(hparams or {})
    resume = str(hp.get("resume") or "").strip()
    init_ckpt = str(hp.get("init_ckpt") or "").strip()
    if resume:
        if require_exists and not Path(resume).exists():
            raise ValueError(f"SFT Resume 路径不存在: {resume}")
        return
    if not init_ckpt:
        raise ValueError(
            "SFT 必须填写 Init Ckpt（从预训练 .pt 微调）；若断点续训请改填 Resume"
        )
    if require_exists and not Path(init_ckpt).exists():
        raise ValueError(f"SFT Init Ckpt 路径不存在: {init_ckpt}")


def build_train_cmd(task: Dict[str, Any]) -> List[str]:
    hp = dict(task.get("hparams") or {})
    arch = (task.get("arch") or "dense").strip().lower()
    mode = (task.get("mode") or "pretrain").strip().lower()
    if mode not in {"pretrain", "sft"}:
        raise ValueError(f"未知训练模式: {mode}")
    if arch not in {"dense", "moe", "tiny"}:
        raise ValueError(f"未知架构: {arch}")
    # Tiny defaults for smoke-friendly CPU runs
    if arch == "tiny":
        hp.setdefault("batch_size", 4)
        hp.setdefault("max_steps", 20)
        hp.setdefault("epochs", 1)
        hp.setdefault("max_seq_len", 128)
        hp.setdefault("warmup_steps", 0)
        hp.setdefault("log_every", 1)
        hp.setdefault("plot_every", 1)

    module = "algo.train_sft" if mode == "sft" else "algo.train_pretrain"
    _default_data = PROJECT_ROOT / "data" / (
        "SFT/sft_miniseek_v5_10g.jsonl" if mode == "sft" else "PT/pretrain_mix_12g_clean.jsonl"
    )
    data_path = str(_hp(hp, "data_path", _default_data))
    out_dir = str(task["out_dir"])

    cmd = [
        sys.executable,
        "-u",
        "-m",
        module,
        "--data",
        data_path,
        "--out-dir",
        out_dir,
        "--batch-size",
        str(int(_hp(hp, "batch_size", 32 if mode == "pretrain" and arch != "tiny" else 4))),
        "--max-seq-len",
        str(int(_hp(hp, "max_seq_len", 512))),
        "--lr",
        str(float(_hp(hp, "lr", 5e-5))),
        "--min-lr",
        str(float(_hp(hp, "min_lr", 1e-5))),
        "--warmup-steps",
        str(int(_hp(hp, "warmup_steps", 200 if mode == "pretrain" else 100))),
        "--epochs",
        str(int(_hp(hp, "epochs", 1))),
        "--grad-clip",
        str(float(_hp(hp, "grad_clip", 1.0))),
        "--save-loss-thresholds",
        str(_hp(hp, "save_loss_thresholds", "")),
        "--stable-steps",
        str(int(_hp(hp, "stable_steps", 50 if mode == "pretrain" else 200))),
        "--save-every",
        str(int(_hp(hp, "save_every", 0))),
        "--log-every",
        str(max(1, int(_hp(hp, "log_every", 10)))),
    ]
    max_steps = _hp(hp, "max_steps", None)
    if max_steps is not None and int(max_steps) > 0:
        cmd.extend(["--max-steps", str(int(max_steps))])

    # arch / aux_coef apply to both pretrain and SFT (MoE SFT supported)
    cmd.extend(["--arch", arch, "--aux-coef", str(float(_hp(hp, "aux_coef", 0.01)))])

    resume = str(_hp(hp, "resume", "") or "").strip()
    init_ckpt = str(_hp(hp, "init_ckpt", "") or "").strip()
    if mode == "sft":
        validate_sft_init_ckpt(mode, hp, require_exists=True)
        if resume:
            cmd.extend(["--resume", resume])
        else:
            cmd.extend(["--init-ckpt", init_ckpt])
    else:
        if resume and Path(resume).exists():
            cmd.extend(["--resume", resume])

    valid_every = int(_hp(hp, "valid_every", 20 if arch == "tiny" else 500))
    valid_max_samples = int(_hp(hp, "valid_max_samples", 512))
    valid_ratio = float(_hp(hp, "valid_ratio", 0) or 0)
    valid_seed = int(_hp(hp, "valid_seed", 42))
    # 验证只从训练集按比例抽，不接受独立 valid 文件（hparams.valid_path 忽略）。
    if valid_every > 0 and valid_ratio > 0:
        cmd.extend(
            [
                "--valid-ratio",
                str(valid_ratio),
                "--valid-seed",
                str(valid_seed),
                "--valid-every",
                str(valid_every),
                "--valid-max-samples",
                str(valid_max_samples),
            ]
        )

    # Speed opts (bench-positive): defaults match train_pretrain / train_sft
    compile_on = _hp(hp, "compile", True)
    if isinstance(compile_on, str):
        compile_on = compile_on.strip().lower() not in {"0", "false", "no", "off"}
    cmd.append("--compile" if bool(compile_on) else "--no-compile")

    fused_on = _hp(hp, "fused_adamw", True)
    if isinstance(fused_on, str):
        fused_on = fused_on.strip().lower() not in {"0", "false", "no", "off"}
    cmd.append("--fused-adamw" if bool(fused_on) else "--no-fused-adamw")

    batch_mode = str(_hp(hp, "batch_mode", "pack")).strip().lower()
    if batch_mode not in {"random", "bucket", "pack"}:
        batch_mode = "pack"
    cmd.extend(["--batch-mode", batch_mode])

    return cmd


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


def _parse_and_append_metrics(task_id: str, line: str, plot_every: int, state: Dict[str, Any]) -> None:
    resumed = RESUME_RE.search(line)
    if resumed:
        # 续训进度/曲线从本轮 0 起；不要用 ckpt lifetime 把 bucket 抬高，否则点画不上。
        if "lifetime_step=" in line:
            state["loss_bucket"] = 0
        else:
            state["loss_bucket"] = int(resumed.group(1)) // max(1, plot_every)
    vm = VALID_RE.search(line)
    if vm:
        step = int(vm.group(1))
        metrics = ts.read_metrics(task_id)
        points = list(metrics.get("valid_loss") or [])
        point = {
            "step": step,
            "loss": float(vm.group(2)),
            "ppl": float(vm.group(3)),
        }
        if vm.group(4) is not None:
            point["tokens"] = int(vm.group(4))
        points.append(point)
        if len(points) > 5000:
            points = points[-5000:]
        metrics["valid_loss"] = points
        ts.write_metrics(task_id, metrics)
        return
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



def rebuild_metrics_from_log(task_id: str, plot_every: int, max_points: int = 5000) -> Dict[str, Any]:
    """Resample train/valid loss from train.log. Does not rewrite metrics.json.

    plot_every: keep one train point per N steps (bucket = step // plot_every).
    Valid lines are kept as logged (usually already sparse).
    """
    every = max(1, int(plot_every))
    max_points = max(50, min(int(max_points), 20000))
    log_path = ts.task_dir(task_id) / "train.log"
    loss: List[Dict[str, Any]] = []
    valid: List[Dict[str, Any]] = []
    loss_bucket = 0
    if not log_path.exists():
        return {"loss": [], "valid_loss": [], "expert_load": None, "plot_every": every}
    with open(log_path, "r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.replace("\x00", "").strip()
            if not line:
                continue
            resumed = RESUME_RE.search(line)
            if resumed:
                # 新日志 lifetime_step=：本轮从 0 画曲线；旧日志 step= 仍跳过已画 bucket。
                if "lifetime_step=" in line:
                    loss_bucket = 0
                else:
                    loss_bucket = int(resumed.group(1)) // every
                continue
            vm = VALID_RE.search(line)
            if vm:
                point = {
                    "step": int(vm.group(1)),
                    "loss": float(vm.group(2)),
                    "ppl": float(vm.group(3)),
                }
                if vm.group(4) is not None:
                    point["tokens"] = int(vm.group(4))
                valid.append(point)
                continue
            matched = STEP_RE.search(line)
            if not matched:
                continue
            step = int(matched.group(1))
            bucket = step // every
            if bucket < 1 or bucket <= loss_bucket:
                continue
            loss_bucket = bucket
            loss.append(
                {
                    "step": step,
                    "loss": float(matched.group(2)),
                    "ppl": float(matched.group(3)),
                    "lr": float(matched.group(4)),
                }
            )
    # Prefer latest window if too dense
    if len(loss) > max_points:
        loss = loss[-max_points:]
    if len(valid) > max_points:
        valid = valid[-max_points:]
    # Preserve expert_load from on-disk metrics if present
    disk = ts.read_metrics(task_id)
    return {
        "loss": loss,
        "valid_loss": valid,
        "expert_load": disk.get("expert_load"),
        "plot_every": every,
        "source": "train.log",
    }


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

    validate_sft_init_ckpt(
        str(task.get("mode") or ""),
        task.get("hparams") or {},
        require_exists=True,
    )
    cmd = build_train_cmd(task)
    log_path = ts.task_dir(task_id) / "train.log"
    # Reset metrics for a fresh run
    ts.write_metrics(task_id, {"loss": [], "valid_loss": [], "expert_load": None})

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

    plot_every = int((task.get("hparams") or {}).get("plot_every") or (1 if task.get("arch") == "tiny" else 500))
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
    owned = _owned_procs.get(task_id)
    code = _terminate_pid(int(pid), owned=owned)
    _owned_procs.pop(task_id, None)

    with _tail_lock:
        _tail_tokens[task_id] = _tail_tokens.get(task_id, 0) + 1  # cancel tailer

    task = ts.load_task(task_id)
    task["status"] = "stopped"
    task["pid"] = None
    task["exit_code"] = code
    task["ended_at"] = datetime.now().isoformat(timespec="seconds")
    ts.save_task(task)
    return task

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
