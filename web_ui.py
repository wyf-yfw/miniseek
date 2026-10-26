#!/usr/bin/env python3
"""MiniSeek 控制台。

Gradio 的定时轮询盖不住训练日志：子进程把 stdout 重定向到文件时会块缓冲，
页面上要攒够一大块才刷新一次。这里改成 FastAPI 页面，训练进程每打印一行，
后台立刻通过 SSE 推到浏览器。
"""

from __future__ import annotations

import asyncio
import glob
import json
import os
from datetime import datetime
import re
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

os.environ.pop("all_proxy", None)
os.environ.pop("ALL_PROXY", None)

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from model.config import ModelConfig
from model.dense import DenseLM
from model.tokenizer import detect_vocab_size, load_tokenizer

PROJECT_ROOT = Path(__file__).parent.resolve()
WEB_DIR = PROJECT_ROOT / "web"
DEFAULT_PRETRAIN_DATA = str(PROJECT_ROOT / "data" / "pretrain_t2t_mini.jsonl")
DEFAULT_SFT_DATA = str(PROJECT_ROOT / "data" / "sft_t2t_mini.jsonl")
DEFAULT_PRETRAIN_OUT = str(PROJECT_ROOT / "checkpoints" / "stage0")
DEFAULT_SFT_OUT = str(PROJECT_ROOT / "checkpoints" / "sft")
DEFAULT_MOE_OUT = str(PROJECT_ROOT / "checkpoints" / "stage1")
# 兼容旧字段
DEFAULT_DATA = DEFAULT_PRETRAIN_DATA
DEFAULT_CKPT_DIR = DEFAULT_PRETRAIN_OUT
LOG_FILE = PROJECT_ROOT / "train_web.log"
STATE_FILE = PROJECT_ROOT / "train_web.json"
TRAIN_SCRIPTS = ("train_pretrain.py", "train_sft.py")
STEP_RE = re.compile(
    # MoE logs insert ce=/aux= between loss and ppl; dense has none.
    r"step=(\d+)\s+loss=([0-9.]+)(?:\s+ce=[0-9.]+)?(?:\s+aux=[0-9.]+)?\s+ppl=([0-9.]+)\s+lr=([0-9.eE+-]+)"
)
RESUME_RE = re.compile(r"resumed from .* step=(\d+)")


def _training_pid_alive(pid: Optional[int]) -> bool:
    """进程还在，且确实是本仓库的训练脚本。pid 被系统复用时不会误认。

    只认「python … train_*.py」真训练进程；带 while/pgrep 的等待 shell
    即使命令行里写了脚本名，也不算。
    """
    if not pid:
        return False
    stat_path = Path(f"/proc/{pid}/stat")
    cmd_path = Path(f"/proc/{pid}/cmdline")
    try:
        stat = stat_path.read_text()
        raw = cmd_path.read_bytes()
        parts = [p.decode(errors="replace") for p in raw.split(b"\x00") if p]
        cmdline = " ".join(parts)
    except OSError:
        return False
    rparen = stat.rfind(")")
    if rparen < 0 or rparen + 2 >= len(stat):
        return False
    if stat[rparen + 2] in {"Z", "X"}:
        return False
    # shell / 等待循环：命令行含 while/pgrep/grep 等，不算训练
    low = cmdline.lower()
    if any(tok in low for tok in (" while ", "pgrep ", "grep -", "/bin/zsh", "/bin/bash", "/usr/bin/zsh")):
        # 允许 python 路径里碰巧有这些字时仍继续往下；但纯 shell 直接否
        exe0 = Path(parts[0]).name if parts else ""
        if exe0 in {"zsh", "bash", "sh", "dash"} or parts and parts[0].endswith(("/zsh", "/bash", "/sh")):
            return False
    # 必须是 python 解释器在跑 train_*.py
    if not parts:
        return False
    exe = Path(parts[0]).name
    if not (exe.startswith("python") or exe == "python3"):
        # 也可能是 /usr/bin/python3 -u train_xxx.py
        if not any(Path(p).name.startswith("python") for p in parts[:3]):
            return False
    joined = " ".join(parts)
    return any(
        (f" {name}" in f" {joined}") or joined.endswith(name) or f"/{name}" in joined
        for name in TRAIN_SCRIPTS
    )


def _cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ").decode(errors="replace").strip()
    except OSError:
        return ""


def _stdout_path(pid: int) -> Optional[str]:
    try:
        target = os.readlink(f"/proc/{pid}/fd/1")
    except OSError:
        return None
    if target.startswith("/") and Path(target).is_file():
        return target
    return None


def _flag(cmd: str, name: str) -> str:
    parts = cmd.split()
    if name not in parts:
        return ""
    idx = parts.index(name)
    if idx + 1 >= len(parts):
        return ""
    return parts[idx + 1]


def _proc_start_text(pid: int) -> str:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        fields = stat[stat.rfind(")") + 2 :].split()
        ticks = int(fields[19])
        clk = os.sysconf("SC_CLK_TCK")
        btime = 0
        for line in Path("/proc/stat").read_text().splitlines():
            if line.startswith("btime "):
                btime = int(line.split()[1])
                break
        if not btime:
            return ""
        return datetime.fromtimestamp(btime + ticks / clk).strftime("%m-%d %H:%M:%S")
    except (OSError, IndexError, ValueError):
        return ""


def _put(queue: asyncio.Queue, msg: Tuple[str, dict]) -> None:
    try:
        queue.put_nowait(msg)
    except asyncio.QueueFull:
        pass


class TrainingManager:
    """后台训练进程。日志按行读出，再推给所有 SSE 连接。"""

    def __init__(self) -> None:
        self.process: Optional[subprocess.Popen] = None
        self.log_file = LOG_FILE
        self.lines: Deque[str] = deque(maxlen=2000)
        self.current_step = 0
        self.current_loss: Optional[float] = None
        self.current_ppl: Optional[float] = None
        self.current_lr: Optional[float] = None
        self.plot_every = 200
        self.loss_points: List[Tuple[int, float]] = []
        self._loss_bucket = 0
        self._lock = threading.Lock()
        self._subscribers: List[asyncio.Queue] = []
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._tail_token = 0
        self._starting = False
        self._pid: Optional[int] = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def is_running(self) -> bool:
        return _training_pid_alive(self._pid)

    def try_adopt(self) -> None:
        """只恢复上一次手动接入的进程。没选过的，留给页面上的列表。"""
        if not STATE_FILE.exists():
            return
        try:
            data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            pid = int(data.get("pid") or 0)
            plot_every = max(1, int(data.get("plot_every") or 200))
        except (OSError, ValueError, TypeError):
            return
        if _training_pid_alive(pid):
            try:
                self.attach(pid, plot_every)
            except LookupError:
                return

    def list_processes(self) -> List[dict]:
        rows: List[dict] = []
        attached = self._pid if _training_pid_alive(self._pid) else None
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            pid = int(entry.name)
            if not _training_pid_alive(pid):
                continue
            cmd = _cmdline(pid)
            out_dir = _flag(cmd, "--out-dir")
            data = _flag(cmd, "--data")
            started = _proc_start_text(pid)
            log_path = _stdout_path(pid)
            out_name = Path(out_dir).name if out_dir else "训练"
            data_name = Path(data).name if data else ""
            label = f"PID {pid} · {out_name}"
            if data_name:
                label += f" · {data_name}"
            if started:
                label += f" · {started}"
            if not log_path:
                label += " · 无日志文件"
            rows.append({
                "pid": pid,
                "cmd": cmd,
                "label": label,
                "log_path": log_path or "",
                "log_readable": bool(log_path),
                "started": started,
                "attached": pid == attached,
            })
        rows.sort(key=lambda row: row["pid"])
        return rows

    def attach(self, pid: int, plot_every: int) -> str:
        """把页面接到一个已经在跑的训练进程上，不重启它。"""
        if not _training_pid_alive(pid):
            raise LookupError(f"训练进程 {pid} 已经不在了")
        log_path = _stdout_path(pid)
        if not log_path:
            raise LookupError(f"进程 {pid} 的输出不是日志文件，页面读不到")
        with self._lock:
            self._tail_token += 1
            token = self._tail_token
            self._pid = pid
            self.process = None
            self.log_file = Path(log_path)
            self.plot_every = max(1, int(plot_every))
            self.lines.clear()
            self.loss_points.clear()
            self._loss_bucket = 0
            self.current_step = 0
            self.current_loss = None
            self.current_ppl = None
            self.current_lr = None
            every = self.plot_every
            subs = list(self._subscribers)
        self._write_state()
        self._fanout(subs, ("reset", {"ok": True, "plot_every": every}))
        threading.Thread(target=self._tail, args=(token, pid), daemon=True).start()
        self._broadcast_status()
        return f"已接入训练进程 (PID: {pid})"

    def _write_state(self) -> None:
        if not self._pid:
            return
        STATE_FILE.write_text(
            json.dumps({"pid": self._pid, "plot_every": self.plot_every}),
            encoding="utf-8",
        )

    def _clear_state(self, pid: int) -> None:
        try:
            data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            if int(data.get("pid") or 0) != pid:
                return
            STATE_FILE.unlink()
        except (OSError, ValueError, TypeError):
            return

    def start(self, req: "TrainRequest") -> str:
        running = self.list_processes()
        if running:
            return f"训练已在运行中 (PID: {running[0]['pid']})。请在列表里接入，或先终止该进程"
        self.log_file = LOG_FILE
        with self._lock:
            if self._starting or _training_pid_alive(self._pid):
                return f"训练已在运行中 (PID: {self._pid})。请在列表里接入，或先终止该进程"
            self._starting = True
            self._tail_token += 1
            token = self._tail_token
            self.lines.clear()
            self.current_step = 0
            self.current_loss = None
            self.current_ppl = None
            self.current_lr = None
            self.plot_every = max(1, int(req.plot_every))
            self.loss_points.clear()
            self._loss_bucket = 0
            plot_every = self.plot_every
            subs = list(self._subscribers)

        self._fanout(subs, ("reset", {"ok": True, "plot_every": plot_every}))

        mode = (req.mode or "pretrain").strip().lower()
        if mode not in {"pretrain", "sft"}:
            raise ValueError(f"未知训练模式: {mode}")
        script = "train_sft.py" if mode == "sft" else "train_pretrain.py"
        cmd = [
            sys.executable,
            "-u",
            str(PROJECT_ROOT / script),
            "--data",
            req.data_path.strip(),
            "--out-dir",
            req.out_dir.strip(),
            "--batch-size",
            str(int(req.batch_size)),
            "--max-seq-len",
            str(int(req.max_seq_len)),
            "--lr",
            str(float(req.lr)),
            "--min-lr",
            str(float(req.min_lr)),
            "--warmup-steps",
            str(int(req.warmup_steps)),
            "--epochs",
            str(int(req.epochs)),
            "--grad-clip",
            str(float(req.grad_clip)),
            "--save-loss-thresholds",
            req.save_loss_thresholds.strip(),
            "--stable-steps",
            str(int(req.stable_steps)),
            "--save-every",
            str(int(req.save_every)),
            "--log-every",
            str(max(1, int(req.log_every))),
        ]
        if req.max_steps and int(req.max_steps) > 0:
            cmd.extend(["--max-steps", str(int(req.max_steps))])
        arch = (req.arch or "dense").strip().lower()
        if arch not in {"dense", "moe"}:
            raise ValueError(f"未知架构: {arch}")
        if mode == "pretrain":
            cmd.extend(["--arch", arch, "--aux-coef", str(float(req.aux_coef))])
        elif arch == "moe":
            raise ValueError("SFT 暂不支持 MoE，请先用 Dense 或等 M0.1 SFT")
        resume = (req.resume or "").strip()
        init_ckpt = (req.init_ckpt or "").strip()
        if mode == "sft":
            # SFT：优先断点续训；否则用预训练 base 做 --init-ckpt
            if resume and Path(resume).exists():
                cmd.extend(["--resume", resume])
            elif init_ckpt and Path(init_ckpt).exists():
                cmd.extend(["--init-ckpt", init_ckpt])
            elif resume:
                raise FileNotFoundError(f"找不到 resume: {resume}")
            elif init_ckpt:
                raise FileNotFoundError(f"找不到 init_ckpt: {init_ckpt}")
        else:
            if resume and Path(resume).exists():
                cmd.extend(["--resume", resume])

        env = os.environ.copy()
        env.pop("all_proxy", None)
        env.pop("ALL_PROXY", None)
        env["PYTHONUNBUFFERED"] = "1"

        self.log_file.write_bytes(b"")
        fd = os.open(self.log_file, os.O_WRONLY | os.O_CREAT, 0o644)
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=fd,
                stderr=subprocess.STDOUT,
                cwd=str(PROJECT_ROOT),
                env=env,
            )
        except Exception:
            self._starting = False
            raise
        finally:
            os.close(fd)
        self._pid = self.process.pid
        self._starting = False
        self._write_state()

        threading.Thread(target=self._tail, args=(token, self._pid), daemon=True).start()
        self._broadcast_status()
        return f"训练已启动 (PID: {self._pid})"

    def stop(self) -> str:
        pid = self._pid
        if not _training_pid_alive(pid):
            return "当前没有正在运行的训练进程"
        assert pid is not None
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            return "当前没有正在运行的训练进程"
        for _ in range(50):
            if not _training_pid_alive(pid):
                break
            time.sleep(0.1)
        else:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
        if self.process is not None and self.process.pid == pid:
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
        return f"训练进程 (PID: {pid}) 已终止"

    def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        gpu = _gpu_text()
        with self._lock:
            running = _training_pid_alive(self._pid)
            queue.put_nowait(("hello", {
                "ok": True,
                "plot_every": self.plot_every,
                "loss": [{"step": step, "loss": loss} for step, loss in self.loss_points],
                "status": {
                    "running": running,
                    "pid": self._pid if running else None,
                    "step": self.current_step,
                    "loss": self.current_loss,
                    "ppl": self.current_ppl,
                    "lr": self.current_lr,
                    "gpu": gpu,
                },
            }))
            for line in self.lines:
                queue.put_nowait(("log", {"line": line}))
            self._subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        with self._lock:
            if queue in self._subscribers:
                self._subscribers.remove(queue)

    def status_dict(self) -> dict:
        with self._lock:
            payload = self._status_locked()
        payload["gpu"] = _gpu_text()
        return payload

    def _status_locked(self) -> dict:
        running = _training_pid_alive(self._pid)
        return {
            "running": running,
            "pid": self._pid if running else None,
            "step": self.current_step,
            "loss": self.current_loss,
            "ppl": self.current_ppl,
            "lr": self.current_lr,
        }

    def _tail(self, token: int, pid: int) -> None:
        """跟着日志文件走。训练进程不挂在当前服务上时，刷新页面也能继续读。"""
        try:
            with open(self.log_file, "r", encoding="utf-8", errors="replace") as fh:
                while True:
                    if token != self._tail_token:
                        return
                    line = fh.readline()
                    if token != self._tail_token:
                        return
                    if line:
                        clean = line.replace("\x00", "").strip()
                        if clean:
                            self._emit_log(clean)
                        continue
                    if not _training_pid_alive(pid):
                        rest = fh.read().replace("\x00", "")
                        for extra in rest.splitlines():
                            if token != self._tail_token:
                                return
                            extra = extra.strip()
                            if extra:
                                self._emit_log(extra)
                        if token != self._tail_token:
                            return
                        code = self.process.returncode if self.process is not None and self.process.pid == pid else None
                        if code is None and self.process is not None and self.process.pid == pid:
                            try:
                                code = self.process.wait(timeout=0.2)
                            except subprocess.TimeoutExpired:
                                code = None
                        suffix = f" (exit={code})" if code is not None else ""
                        self._emit_log(f"[web] 训练进程已结束{suffix}")
                        with self._lock:
                            if self._pid == pid:
                                self._pid = None
                                self.process = None
                        self._clear_state(pid)
                        self._broadcast_status()
                        return
                    time.sleep(0.05)
        except Exception as exc:
            self._emit_log(f"[web] 日志读取失败: {exc}")

    def _record_loss_locked(self, step: int, loss: float) -> Optional[Tuple[int, float]]:
        """每 plot_every 步记一个点。续训时对齐到已完成的区间，不补历史。"""
        every = max(1, self.plot_every)
        bucket = step // every
        if bucket < 1 or bucket <= self._loss_bucket:
            return None
        self._loss_bucket = bucket
        point = (step, loss)
        self.loss_points.append(point)
        if len(self.loss_points) > 5000:
            del self.loss_points[:-5000]
        return point

    def _emit_log(self, line: str) -> None:
        status_payload = None
        point: Optional[Tuple[int, float]] = None
        with self._lock:
            self.lines.append(line)
            resumed = RESUME_RE.search(line)
            if resumed:
                self._loss_bucket = int(resumed.group(1)) // max(1, self.plot_every)
            matched = STEP_RE.search(line)
            if matched:
                step = int(matched.group(1))
                if self.current_step and step + 20 < self.current_step:
                    self.loss_points.clear()
                    self._loss_bucket = 0
                self.current_step = step
                self.current_loss = float(matched.group(2))
                self.current_ppl = float(matched.group(3))
                self.current_lr = float(matched.group(4))
                point = self._record_loss_locked(self.current_step, self.current_loss)
                status_payload = self._status_locked()
            every = self.plot_every
            subs = list(self._subscribers)
        self._fanout(subs, ("log", {"line": line}))
        if status_payload is not None:
            status_payload["gpu"] = _gpu_text()
            self._fanout(subs, ("status", status_payload))
        if point is not None:
            self._fanout(subs, ("loss", {
                "step": point[0],
                "loss": point[1],
                "every": every,
            }))

    def _broadcast_status(self) -> None:
        payload = self.status_dict()
        with self._lock:
            subs = list(self._subscribers)
        self._fanout(subs, ("status", payload))

    def _fanout(self, subs: List[asyncio.Queue], msg: Tuple[str, dict]) -> None:
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        for queue in subs:
            try:
                loop.call_soon_threadsafe(_put, queue, msg)
            except RuntimeError:
                return


def _gpu_text() -> str:
    if not torch.cuda.is_available():
        return "无 GPU"
    free, total = torch.cuda.mem_get_info(0)
    used = (total - free) / (1024**3)
    return f"{torch.cuda.get_device_name(0)} | 显存 {used:.1f}/{total / (1024**3):.1f} GB"


manager = TrainingManager()


def format_chat_messages(messages: List[dict]) -> str:
    """与 train_sft.format_chat 对齐：每轮 <|im_start|>role\ncontent<|im_end|>。"""
    parts: List[str] = []
    for turn in messages:
        role = (turn.get("role") or "").strip().lower()
        if role not in {"user", "assistant", "system"}:
            continue
        content = turn.get("content")
        if content is None:
            content = ""
        content = str(content).strip()
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    return "\n".join(parts)


class InferenceEngine:
    def __init__(self) -> None:
        self.model: Optional[DenseLM] = None
        self.tokenizer = None
        self.loaded_ckpt: Optional[str] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._gen_lock = threading.Lock()

    def load_model(self, ckpt_path: str) -> str:
        if not ckpt_path or not Path(ckpt_path).exists():
            raise HTTPException(status_code=404, detail=f"找不到 checkpoint: {ckpt_path}")
        if self.tokenizer is None:
            self.tokenizer = load_tokenizer(allow_fallback=True)
        blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = ModelConfig.from_dict(blob["config"])
        if getattr(cfg, "arch", "dense") == "moe" or getattr(cfg, "use_moe", False):
            from model.moe import MoELM
            model = MoELM(cfg)
        else:
            model = DenseLM(cfg)
        model.load_state_dict(blob["model"], strict=True)
        # 训练占 GPU / 显存紧张时直接 CPU，避免把 Web UI OOM 掉
        device = torch.device("cpu")
        if torch.cuda.is_available():
            train_busy = _training_pid_alive(manager._pid)
            if not train_busy:
                # Web UI 重启后可能未 attach，再扫一遍训练进程
                for proc in manager.list_processes():
                    train_busy = True
                    break
            try:
                free, total = torch.cuda.mem_get_info(0)
            except Exception:
                free, total = 0, 1
            # 训练中一律 CPU；空闲时也至少留约 4GB
            if (not train_busy) and free > 4 * (1024 ** 3):
                device = torch.device("cuda")
        if device.type == "cuda":
            try:
                model.to(device).eval()
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    torch.cuda.empty_cache()
                    device = torch.device("cpu")
                    model.to(device).eval()
                else:
                    raise
        else:
            model.to(device).eval()
        self.model = model
        self.device = device
        self.loaded_ckpt = ckpt_path
        step = blob.get("step", "?")
        loss = blob.get("loss", "?")
        where = "GPU" if device.type == "cuda" else "CPU"
        return f"已加载 {Path(ckpt_path).name} (Step: {step}, Loss: {loss}, {where})"

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        top_k: int = 50,
        seed: int = 42,
    ) -> str:
        if self.model is None:
            raise HTTPException(status_code=400, detail="请先加载一个 Checkpoint")
        if not prompt.strip():
            raise HTTPException(status_code=400, detail="请输入 prompt")

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        t0 = time.time()
        with self._gen_lock:
            out = self.model.generate(
                input_ids,
                max_new_tokens=int(max_new_tokens),
                temperature=float(temperature),
                top_k=int(top_k),
                eos_token_id=eos_id,
            )
        cost = time.time() - t0
        gen_tokens = out.shape[1] - len(ids)
        speed = gen_tokens / max(cost, 1e-6)
        text = self.tokenizer.decode(out[0].tolist(), skip_special_tokens=True)
        return (
            f"{text}\n\n---\n生成 {gen_tokens} tokens | 耗时 {cost:.2f}s | "
            f"{speed:.1f} tokens/s"
        )

    def build_chat_prompt(self, messages: List[dict]) -> str:
        cleaned: List[dict] = []
        for m in messages:
            role = (m.get("role") or "").strip().lower()
            if role not in {"user", "assistant", "system"}:
                continue
            content = str(m.get("content") or "").strip()
            if not content and role != "assistant":
                continue
            cleaned.append({"role": role, "content": content})
        if not cleaned or cleaned[-1]["role"] != "user":
            raise HTTPException(status_code=400, detail="最后一条消息必须是 user")
        return format_chat_messages(cleaned) + "\n<|im_start|>assistant\n"

    def stream_chat(
        self,
        messages: List[dict],
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 50,
        seed: int = 42,
    ):
        if self.model is None:
            raise HTTPException(status_code=400, detail="请先加载一个 Checkpoint")
        prompt = self.build_chat_prompt(messages)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # chat 模板已含 <|im_start|>，不要再自动塞 bos
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        t0 = time.time()
        pieces: List[str] = []
        n = 0
        with self._gen_lock:
            for token in self.model.generate_stream(
                input_ids,
                max_new_tokens=int(max_new_tokens),
                temperature=float(temperature),
                top_k=int(top_k),
                eos_token_id=eos_id,
            ):
                n += 1
                if eos_id is not None and token == int(eos_id):
                    break
                piece = self.tokenizer.decode([token], skip_special_tokens=False)
                # 跳过纯特殊符碎片
                if piece in {"<|im_start|>", "<|im_end|>", "<|endoftext|>"}:
                    continue
                pieces.append(piece)
                yield {"event": "token", "data": {"text": piece}}
        cost = time.time() - t0
        full = "".join(pieces)
        speed = n / max(cost, 1e-6)
        yield {
            "event": "done",
            "data": {
                "text": full,
                "tokens": n,
                "seconds": round(cost, 2),
                "tokens_per_s": round(speed, 1),
            },
        }


infer_engine = InferenceEngine()


class TrainRequest(BaseModel):
    mode: str = "pretrain"  # pretrain | sft
    arch: str = "dense"  # dense | moe（预训练）；SFT 暂仅 dense
    aux_coef: float = 0.01
    data_path: str
    out_dir: str
    batch_size: int = 8
    max_seq_len: int = 512
    lr: float = 1e-4
    min_lr: float = 3e-5
    warmup_steps: int = 200
    epochs: int = 1
    max_steps: int = 0
    grad_clip: float = 1.0
    save_loss_thresholds: str = "20,15,12,10,9,8,7,6,5,4,3"
    stable_steps: int = 200
    save_every: int = 0
    resume: str = ""
    init_ckpt: str = ""  # SFT：从预训练 base 初始化
    log_every: int = Field(default=1, ge=1)
    plot_every: int = Field(default=200, ge=1)


class AttachRequest(BaseModel):
    pid: int
    plot_every: int = Field(default=200, ge=1)


class LoadRequest(BaseModel):
    ckpt_path: str


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 64
    temperature: float = 0.8
    top_k: int = 50
    seed: int = 42


def list_checkpoints() -> List[str]:
    files = glob.glob(str(PROJECT_ROOT / "checkpoints" / "**" / "*.pt"), recursive=True)
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files


def env_text() -> str:
    lines = []
    tok = load_tokenizer(allow_fallback=True)
    is_fallback = "fallback" in getattr(tok, "name_or_path", "").lower()
    if is_fallback:
        lines.append("分词器: 当前为 FallbackTokenizer（不能正式训练）")
    else:
        lines.append(f"分词器: {getattr(tok, 'name_or_path', './tokenizer')} (词表 {len(tok)})")

    for label, p in [
        ("预训练数据", Path(DEFAULT_PRETRAIN_DATA)),
        ("SFT 数据", Path(DEFAULT_SFT_DATA)),
        ("验证集", PROJECT_ROOT / "data" / "valid.jsonl"),
    ]:
        if p.exists():
            lines.append(f"{label}: {p.name} ({p.stat().st_size / (1024 * 1024):.1f} MB)")
        else:
            lines.append(f"{label}: 缺少 {p.name}")
    lines.append(_gpu_text())
    return "\n".join(lines)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@asynccontextmanager
async def lifespan(app: FastAPI):
    manager.bind_loop(asyncio.get_running_loop())
    manager.try_adopt()
    yield


app = FastAPI(title="MiniSeek", lifespan=lifespan)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(
        WEB_DIR / "index.html",
        headers={"Cache-Control": "no-cache"},
    )


@app.get("/api/defaults")
def defaults() -> dict:
    best = PROJECT_ROOT / "checkpoints" / "stage0" / "best.pt"
    return {
        "mode": "pretrain",
        "arch": "dense",
        "pretrain": {
            "arch": "dense",
            "data_path": DEFAULT_PRETRAIN_DATA,
            "out_dir": DEFAULT_PRETRAIN_OUT,
            "lr": 1e-4,
            "min_lr": 3e-5,
            "warmup_steps": 200,
            "save_loss_thresholds": "20,15,12,10,9,8,7,6,5,4,3",
            "init_ckpt": "",
            "resume": "",
            "aux_coef": 0.01,
        },
        "pretrain_moe": {
            "arch": "moe",
            "data_path": DEFAULT_PRETRAIN_DATA,
            "out_dir": DEFAULT_MOE_OUT,
            "lr": 1e-4,
            "min_lr": 3e-5,
            "warmup_steps": 200,
            "save_loss_thresholds": "20,15,12,10,9,8,7,6,5,4,3",
            "init_ckpt": "",
            "resume": "",
            "aux_coef": 0.01,
        },
        "sft": {
            "arch": "dense",
            "data_path": DEFAULT_SFT_DATA,
            "out_dir": DEFAULT_SFT_OUT,
            "lr": 5e-5,
            "min_lr": 1e-5,
            "warmup_steps": 100,
            "save_loss_thresholds": "5,4,3,2.5,2,1.5,1",
            "init_ckpt": str(best) if best.exists() else "",
            "resume": "",
            "aux_coef": 0.01,
        },
        # 兼容旧前端
        "data_path": DEFAULT_PRETRAIN_DATA,
        "out_dir": DEFAULT_PRETRAIN_OUT,
    }


@app.get("/api/status")
def status() -> dict:
    return manager.status_dict()


@app.get("/api/checkpoints")
def checkpoints() -> dict:
    return {"files": list_checkpoints()}


@app.get("/api/env")
def environment() -> dict:
    return {"text": env_text()}


@app.post("/api/train/start")
def train_start(req: TrainRequest) -> dict:
    if not req.data_path.strip() or not req.out_dir.strip():
        raise HTTPException(status_code=400, detail="数据路径和输出目录不能为空")
    if int(req.batch_size) < 1 or int(req.max_seq_len) < 2:
        raise HTTPException(status_code=400, detail="batch size 和序列长度不合法")
    mode = (req.mode or "pretrain").strip().lower()
    if mode == "sft":
        data_p = Path(req.data_path.strip())
        if not data_p.exists():
            raise HTTPException(status_code=400, detail=f"找不到 SFT 数据: {data_p}")
        if not (req.resume or "").strip() and not (req.init_ckpt or "").strip():
            raise HTTPException(
                status_code=400,
                detail="SFT 需要选择「从 Checkpoint 初始化」（一般用 stage0/best.pt），或断点续训",
            )
    try:
        message = manager.start(req)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if message.startswith("训练已在运行中"):
        raise HTTPException(status_code=409, detail=message)
    return {"message": message, "status": manager.status_dict()}


@app.get("/api/train/processes")
def train_processes() -> dict:
    attached = manager._pid if _training_pid_alive(manager._pid) else None
    return {"attached_pid": attached, "processes": manager.list_processes()}


@app.post("/api/train/attach")
def train_attach(req: AttachRequest) -> dict:
    try:
        message = manager.attach(req.pid, req.plot_every)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"message": message, "status": manager.status_dict()}


@app.post("/api/train/stop")
def train_stop() -> dict:
    return {"message": manager.stop(), "status": manager.status_dict()}


@app.get("/api/logs/stream")
async def log_stream() -> StreamingResponse:
    queue = manager.subscribe()

    async def events():
        try:
            while True:
                try:
                    event, data = await asyncio.wait_for(queue.get(), timeout=15)
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
                    continue
                yield _sse(event, data)
        finally:
            manager.unsubscribe(queue)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/infer/load")
def infer_load(req: LoadRequest) -> dict:
    return {"message": infer_engine.load_model(req.ckpt_path)}


@app.post("/api/infer/generate")
def infer_generate(req: GenerateRequest) -> dict:
    return {"text": infer_engine.generate(
        req.prompt,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
        seed=req.seed,
    )}


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    max_new_tokens: int = 256
    temperature: float = 0.8
    top_k: int = 50
    seed: int = 42


@app.post("/api/infer/chat/stream")
async def infer_chat_stream(req: ChatRequest) -> StreamingResponse:
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages 不能为空")
    payload = [m.model_dump() for m in req.messages]

    def event_iter():
        try:
            for item in infer_engine.stream_chat(
                payload,
                max_new_tokens=req.max_new_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
                seed=req.seed,
            ):
                yield _sse(item["event"], item["data"])
        except HTTPException as exc:
            detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
            yield _sse("error", {"message": detail})
        except Exception as exc:  # noqa: BLE001
            yield _sse("error", {"message": str(exc)})

    return StreamingResponse(
        event_iter(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "7860"))
    print(f"Starting Miniseek Web UI on http://0.0.0.0:{port}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
