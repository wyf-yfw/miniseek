#!/usr/bin/env python3
"""MiniSeek 控制台 — task 驱动训练 + 对话/续写。

训练以 tasks/<id>/ 为真相源；不再扫描 /proc、不再 attach。
"""

from __future__ import annotations

import asyncio
import glob
import json
import os
import re
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.pop("all_proxy", None)
os.environ.pop("ALL_PROXY", None)

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from model.config import ModelConfig
from model.dense import DenseLM
from model.tokenizer import load_tokenizer
from web import tasks_store as ts
from web import bench_store as bs
from web.task_runner import pid_alive, reconcile_task, start_task, stop_task
from web.bench_runner import (
    reconcile_run,
    start_run as bench_start_run,
    stop_run as bench_stop_run,
    read_log as bench_read_log,
)
from web.arch_inspect import inspect_design as arch_inspect_design, inspect_path as arch_inspect_path

PROJECT_ROOT = Path(__file__).parent.resolve()
WEB_DIR = PROJECT_ROOT / "web"
DEFAULT_PRETRAIN_DATA = str(PROJECT_ROOT / "data" / "pretrain_t2t_mini.jsonl")
DEFAULT_SFT_DATA = str(PROJECT_ROOT / "data" / "sft_t2t_mini.jsonl")
DEFAULT_PRETRAIN_OUT = str(PROJECT_ROOT / "checkpoints" / "stage0")
DEFAULT_SFT_OUT = str(PROJECT_ROOT / "checkpoints" / "sft")
DEFAULT_MOE_OUT = str(PROJECT_ROOT / "checkpoints" / "stage1")


def _gpu_text() -> str:
    if not torch.cuda.is_available():
        return "无 GPU"
    free, total = torch.cuda.mem_get_info(0)
    used = (total - free) / (1024**3)
    return f"{torch.cuda.get_device_name(0)} | 显存 {used:.1f}/{total / (1024**3):.1f} GB"


def _any_training_running() -> bool:
    for row in ts.list_tasks():
        if row.get("status") != "running":
            continue
        t = reconcile_task(ts.load_task(row["id"]))
        if t.get("status") == "running" and pid_alive(t.get("pid")):
            return True
    return False


def format_chat_messages(messages: List[dict]) -> str:
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
        path = Path(ckpt_path)
        # HF export dir or pytorch_model.bin → normalize to training blob shape
        if path.is_dir() and (path / "pytorch_model.bin").exists():
            weight_path = path / "pytorch_model.bin"
            cfg_path = path / "config.json"
        elif path.name == "pytorch_model.bin":
            weight_path = path
            cfg_path = path.parent / "config.json"
        else:
            weight_path = None
            cfg_path = None
        if weight_path is not None:
            import json
            sd = torch.load(weight_path, map_location="cpu", weights_only=False)
            if isinstance(sd, dict) and "model" in sd:
                blob = sd
            else:
                cfg_raw = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path and cfg_path.exists() else {}
                allowed = set(ModelConfig.__annotations__)
                cfg_dict = {k: v for k, v in cfg_raw.items() if k in allowed}
                blob = {
                    "model": sd,
                    "config": cfg_dict,
                    "step": cfg_raw.get("train_step", "?"),
                    "loss": cfg_raw.get("train_loss", "?"),
                }
        else:
            blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = ModelConfig.from_dict(blob["config"])
        if getattr(cfg, "arch", "dense") == "moe" or getattr(cfg, "use_moe", False):
            from model.moe import MoELM
            model = MoELM(cfg)
        else:
            model = DenseLM(cfg)
        model.load_state_dict(blob["model"], strict=True)
        device = torch.device("cpu")
        if torch.cuda.is_available():
            train_busy = _any_training_running()
            try:
                free, total = torch.cuda.mem_get_info(0)
            except Exception:
                free, total = 0, 1
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

        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        t0 = time.time()
        token_ids: List[int] = []
        prev_text = ""
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
                token_ids.append(int(token))
                full = self.tokenizer.decode(token_ids, skip_special_tokens=False)
                for special in ("<|im_start|>", "<|im_end|>", "<|endoftext|>"):
                    full = full.replace(special, "")
                if "\ufffd" in full:
                    continue
                if full.startswith(prev_text):
                    piece = full[len(prev_text):]
                else:
                    piece = full
                prev_text = full
                if not piece:
                    continue
                yield {"event": "token", "data": {"text": piece}}
        cost = time.time() - t0
        speed = n / max(cost, 1e-6)
        yield {
            "event": "done",
            "data": {
                "text": prev_text,
                "tokens": n,
                "seconds": round(cost, 2),
                "tokens_per_s": round(speed, 1),
            },
        }



    def stream_continue(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_k: int = 50,
        seed: int = 42,
    ):
        """Raw prefix continuation for pretrained LMs (no chat template)."""
        if self.model is None:
            raise HTTPException(status_code=400, detail="请先加载一个 Checkpoint")
        if not str(prompt).strip():
            raise HTTPException(status_code=400, detail="请输入前缀")

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        t0 = time.time()
        token_ids: List[int] = []
        prev_text = ""
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
                token_ids.append(int(token))
                full = self.tokenizer.decode(token_ids, skip_special_tokens=False)
                for special in ("<|im_start|>", "<|im_end|>", "<|endoftext|>"):
                    full = full.replace(special, "")
                if "\ufffd" in full:
                    continue
                if full.startswith(prev_text):
                    piece = full[len(prev_text):]
                else:
                    piece = full
                prev_text = full
                if not piece:
                    continue
                yield {"event": "token", "data": {"text": piece}}
        cost = time.time() - t0
        speed = n / max(cost, 1e-6)
        yield {
            "event": "done",
            "data": {
                "text": prev_text,
                "tokens": n,
                "seconds": round(cost, 2),
                "tokens_per_s": round(speed, 1),
            },
        }


infer_engine = InferenceEngine()


class TaskCreate(BaseModel):
    name: str = ""
    arch: str = "dense"
    mode: str = "pretrain"
    hparams: Dict[str, Any] = Field(default_factory=dict)
    uniquify: bool = False


class TaskPatch(BaseModel):
    name: Optional[str] = None
    arch: Optional[str] = None
    mode: Optional[str] = None
    hparams: Optional[Dict[str, Any]] = None


class LoadRequest(BaseModel):
    ckpt_path: str


class ArchInspectRequest(BaseModel):
    ckpt_path: str


class ArchDesignRequest(BaseModel):
    arch: str = "dense"  # tiny | dense | moe


class BenchStartRequest(BaseModel):
    ckpt_path: str
    tasks: List[str] = Field(default_factory=lambda: ["ceval-valid", "cmmlu", "gsm8k"])
    limit: Optional[int] = None


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 64
    temperature: float = 0.8
    top_k: int = 50
    seed: int = 42


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    max_new_tokens: int = 256
    temperature: float = 0.8
    top_k: int = 50
    seed: int = 42


def list_checkpoints_global() -> List[str]:
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


def _get_task_or_404(task_id: str) -> dict:
    try:
        return ts.load_task(task_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"task not found: {task_id}") from exc


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Reconcile any stale running tasks on boot
    for row in ts.list_tasks():
        if row.get("status") == "running":
            reconcile_task(ts.load_task(row["id"]))
    for row in bs.list_runs():
        if row.get("status") == "running":
            reconcile_run(bs.load_run(row["id"]))
    yield


app = FastAPI(title="MiniSeek", lifespan=lifespan)



class PickFileBody(BaseModel):
    kind: str = "file"  # file | dir
    title: str = "选择文件"
    initial: str = ""
    filetypes: str = ""  # e.g. "JSONL|*.jsonl;Checkpoint|*.pt;All|*.*"


def _parse_filetypes(spec: str):
    """'JSONL|*.jsonl;Checkpoint|*.pt' -> tkinter filetypes list."""
    pairs = []
    for part in (spec or "").split(";"):
        part = part.strip()
        if not part:
            continue
        if "|" in part:
            label, pattern = part.split("|", 1)
        else:
            label, pattern = part, part
        pairs.append((label.strip() or pattern, pattern.strip()))
    if not pairs:
        pairs = [("All files", "*.*")]
    # Always allow all as last option if not present
    if not any(p[1] == "*.*" for p in pairs):
        pairs.append(("All files", "*.*"))
    return pairs


@app.post("/api/pick-file")
def api_pick_file(body: PickFileBody) -> dict:
    """Open a native OS file/folder dialog on the machine running this UI.

    Browser <input type=file> cannot expose absolute paths; for local training
    we need real filesystem paths, so the server pops the system dialog.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"无法打开系统文件对话框：{exc}") from exc

    initial = (body.initial or "").strip()
    init_dir = ""
    init_file = ""
    if initial:
        p = Path(initial)
        if p.is_dir():
            init_dir = str(p)
        else:
            init_dir = str(p.parent) if p.parent.exists() else ""
            init_file = p.name

    result: dict = {"path": None, "cancelled": True}

    def _open() -> None:
        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass
        root.update()
        try:
            if body.kind == "dir":
                path = filedialog.askdirectory(
                    parent=root,
                    title=body.title or "选择目录",
                    initialdir=init_dir or str(PROJECT_ROOT),
                )
            else:
                path = filedialog.askopenfilename(
                    parent=root,
                    title=body.title or "选择文件",
                    initialdir=init_dir or str(PROJECT_ROOT),
                    initialfile=init_file or "",
                    filetypes=_parse_filetypes(body.filetypes),
                )
        finally:
            root.destroy()
        if path:
            result["path"] = str(Path(path))
            result["cancelled"] = False

    # tkinter must run on a dedicated thread with its own loop on some setups;
    # call directly first (uvicorn worker thread is fine for local single-process).
    try:
        _open()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"文件选择失败：{exc}") from exc
    return result


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
        "dense": {
            "arch": "dense",
            "mode": "pretrain",
            "data_path": DEFAULT_PRETRAIN_DATA,
            "batch_size": 8,
            "max_seq_len": 512,
            "lr": 1e-4,
            "min_lr": 3e-5,
            "warmup_steps": 200,
            "epochs": 1,
            "max_steps": 0,
            "grad_clip": 1.0,
            "save_loss_thresholds": "20,15,12,10,9,8,7,6,5,4,3",
            "stable_steps": 200,
            "save_every": 0,
            "log_every": 1,
            "plot_every": 200,
            "init_ckpt": "",
            "resume": "",
            "aux_coef": 0.01,
        },
        "moe": {
            "arch": "moe",
            "mode": "pretrain",
            "data_path": DEFAULT_PRETRAIN_DATA,
            "batch_size": 8,
            "max_seq_len": 512,
            "lr": 1e-4,
            "min_lr": 3e-5,
            "warmup_steps": 200,
            "epochs": 1,
            "max_steps": 0,
            "grad_clip": 1.0,
            "save_loss_thresholds": "20,15,12,10,9,8,7,6,5,4,3",
            "stable_steps": 200,
            "save_every": 0,
            "log_every": 1,
            "plot_every": 200,
            "init_ckpt": "",
            "resume": "",
            "aux_coef": 0.01,
        },
        "tiny": {
            "arch": "tiny",
            "mode": "pretrain",
            "data_path": DEFAULT_PRETRAIN_DATA,
            "batch_size": 4,
            "max_seq_len": 128,
            "lr": 1e-4,
            "min_lr": 3e-5,
            "warmup_steps": 0,
            "epochs": 1,
            "max_steps": 20,
            "grad_clip": 1.0,
            "save_loss_thresholds": "20,15,12,10",
            "stable_steps": 200,
            "save_every": 0,
            "log_every": 1,
            "plot_every": 1,
            "init_ckpt": "",
            "resume": "",
            "aux_coef": 0.01,
        },
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
            "mode": "sft",
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
        "data_path": DEFAULT_PRETRAIN_DATA,
        "out_dir": DEFAULT_PRETRAIN_OUT,
    }


@app.get("/api/env")
def environment() -> dict:
    return {"text": env_text()}


@app.get("/api/checkpoints")
def checkpoints() -> dict:
    """Global fallback list (legacy paths). Prefer /api/tasks/{id}/checkpoints."""
    return {"files": list_checkpoints_global()}


# ---- Tasks API ----

@app.get("/api/tasks")
def api_list_tasks() -> dict:
    rows = []
    for row in ts.list_tasks():
        try:
            full = reconcile_task(ts.load_task(row["id"]))
            rows.append({
                "id": full["id"],
                "name": full.get("name"),
                "status": full.get("status"),
                "arch": full.get("arch"),
                "mode": full.get("mode"),
                "created_at": full.get("created_at"),
                "started_at": full.get("started_at"),
                "ended_at": full.get("ended_at"),
                "pid": full.get("pid"),
            })
        except FileNotFoundError:
            continue
    rows.sort(key=lambda r: r.get("created_at") or "", reverse=True)
    return {"tasks": rows}


@app.post("/api/tasks")
def api_create_task(body: TaskCreate) -> dict:
    arch = (body.arch or "dense").strip().lower()
    mode = (body.mode or "pretrain").strip().lower()
    if arch not in {"dense", "moe", "tiny"}:
        raise HTTPException(status_code=400, detail=f"未知架构: {arch}")
    if mode not in {"pretrain", "sft"}:
        raise HTTPException(status_code=400, detail=f"未知模式: {mode}")
    defaults_map = defaults()
    base = dict(defaults_map.get(arch) or defaults_map["dense"])
    base.update(body.hparams or {})
    base["arch"] = arch
    base.pop("out_dir", None)  # out_dir forced by store
    name = (body.name or "").strip() or f"{arch}-{mode}"
    try:
        return ts.create_task(name=name, arch=arch, mode=mode, hparams=base, uniquify=bool(body.uniquify))
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/api/tasks/{task_id}")
def api_get_task(task_id: str) -> dict:
    return reconcile_task(_get_task_or_404(task_id))


@app.patch("/api/tasks/{task_id}")
def api_patch_task(task_id: str, body: TaskPatch) -> dict:
    task = reconcile_task(_get_task_or_404(task_id))
    if body.name is not None:
        new_name = body.name.strip() or task["name"]
        if ts.name_taken(new_name, exclude_id=task_id):
            raise HTTPException(status_code=409, detail=f"已存在同名 Task：{new_name}")
        task["name"] = new_name
    if task.get("status") == "draft":
        if body.arch is not None:
            arch = body.arch.strip().lower()
            if arch not in {"dense", "moe", "tiny"}:
                raise HTTPException(status_code=400, detail=f"未知架构: {arch}")
            task["arch"] = arch
        if body.mode is not None:
            mode = body.mode.strip().lower()
            if mode not in {"pretrain", "sft"}:
                raise HTTPException(status_code=400, detail=f"未知模式: {mode}")
            task["mode"] = mode
        if body.hparams is not None:
            hp = dict(task.get("hparams") or {})
            hp.update(body.hparams)
            hp.pop("out_dir", None)
            task["hparams"] = hp
    elif body.hparams is not None or body.arch is not None or body.mode is not None:
        raise HTTPException(status_code=400, detail="只有 draft 可改 arch/mode/hparams")
    ts.save_task(task)
    return task


@app.delete("/api/tasks/{task_id}")
def api_delete_task(task_id: str) -> dict:
    task = reconcile_task(_get_task_or_404(task_id))
    if task.get("status") == "running" and pid_alive(task.get("pid")):
        raise HTTPException(status_code=409, detail="running task 不能删除，请先 Stop")
    ts.delete_task(task_id)
    return {"ok": True, "id": task_id}


@app.post("/api/tasks/{task_id}/start")
def api_start_task(task_id: str) -> dict:
    _get_task_or_404(task_id)
    try:
        return start_task(task_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (ValueError, FileNotFoundError, NotImplementedError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/tasks/{task_id}/stop")
def api_stop_task(task_id: str) -> dict:
    _get_task_or_404(task_id)
    return stop_task(task_id)


@app.get("/api/tasks/{task_id}/metrics")
def api_task_metrics(task_id: str) -> dict:
    _get_task_or_404(task_id)
    return ts.read_metrics(task_id)


@app.get("/api/tasks/{task_id}/checkpoints")
def api_task_checkpoints(task_id: str) -> dict:
    _get_task_or_404(task_id)
    ckpt_dir = ts.task_dir(task_id) / "checkpoints"
    files = sorted(ckpt_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return {"files": [str(p) for p in files]}


@app.get("/api/tasks/{task_id}/logs")
async def api_task_logs(task_id: str, tail: int = 200) -> StreamingResponse:
    """SSE: stream new lines from tasks/<id>/train.log; also emit metrics snapshots."""
    _get_task_or_404(task_id)
    log_path = ts.task_dir(task_id) / "train.log"

    async def events():
        # Initial dump of last `tail` lines
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
        except OSError:
            text = ""
        lines = [ln for ln in text.splitlines() if ln.strip()]
        for ln in lines[-max(1, tail):]:
            yield _sse("log", {"line": ln})
        metrics = ts.read_metrics(task_id)
        yield _sse("metrics", metrics)
        task = reconcile_task(ts.load_task(task_id))
        yield _sse("status", {
            "id": task["id"],
            "status": task.get("status"),
            "pid": task.get("pid"),
        })

        pos = log_path.stat().st_size if log_path.exists() else 0
        idle = 0
        while True:
            await asyncio.sleep(0.4)
            try:
                if not log_path.exists():
                    continue
                size = log_path.stat().st_size
                if size < pos:
                    pos = 0
                if size > pos:
                    with open(log_path, "r", encoding="utf-8", errors="replace") as fh:
                        fh.seek(pos)
                        chunk = fh.read()
                        pos = fh.tell()
                    for ln in chunk.splitlines():
                        clean = ln.replace("\x00", "").strip()
                        if clean:
                            yield _sse("log", {"line": clean})
                    metrics = ts.read_metrics(task_id)
                    yield _sse("metrics", metrics)
                    idle = 0
                else:
                    idle += 1
                    if idle % 10 == 0:
                        yield ": ping\n\n"
                        task = reconcile_task(ts.load_task(task_id))
                        yield _sse("status", {
                            "id": task["id"],
                            "status": task.get("status"),
                            "pid": task.get("pid"),
                        })
                        if task.get("status") not in {"running"}:
                            # Final metrics flush then end stream after a short grace
                            metrics = ts.read_metrics(task_id)
                            yield _sse("metrics", metrics)
                            if idle > 25:
                                break
            except Exception as exc:  # noqa: BLE001
                yield _sse("error", {"message": str(exc)})
                break

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# Deprecated train endpoints → 410
def _train_gone() -> None:
    raise HTTPException(status_code=410, detail="已改为 /api/tasks；请刷新页面")


@app.post("/api/train/start")
def train_start_gone() -> None:
    _train_gone()


@app.post("/api/train/stop")
def train_stop_gone() -> None:
    _train_gone()


@app.get("/api/train/processes")
def train_processes_gone() -> None:
    _train_gone()


@app.post("/api/train/attach")
def train_attach_gone() -> None:
    _train_gone()


@app.get("/api/status")
def status() -> dict:
    rid = ts.find_running_task_id()
    if rid:
        t = reconcile_task(ts.load_task(rid))
        return {
            "running": t.get("status") == "running",
            "pid": t.get("pid"),
            "task_id": rid,
            "gpu": _gpu_text(),
        }
    return {"running": False, "pid": None, "task_id": None, "gpu": _gpu_text()}






def _get_bench_or_404(run_id: str) -> dict:
    try:
        return reconcile_run(bs.load_run(run_id))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"bench run not found: {run_id}") from exc


@app.get("/api/bench/runs")
def api_bench_list() -> dict:
    rows = []
    for row in bs.list_runs():
        if row.get("status") == "running":
            row = reconcile_run(bs.load_run(row["id"]))
        rows.append(row)
    return {"runs": rows}


@app.post("/api/bench/runs")
def api_bench_start(req: BenchStartRequest) -> dict:
    path = (req.ckpt_path or "").strip()
    if not path:
        raise HTTPException(status_code=400, detail="ckpt_path 不能为空")
    try:
        return bench_start_run(
            ckpt_path=path,
            tasks=list(req.tasks or []),
            limit=req.limit,
            training_busy=_any_training_running(),
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"找不到 checkpoint: {path}")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/api/bench/runs/{run_id}")
def api_bench_get(run_id: str) -> dict:
    return _get_bench_or_404(run_id)


@app.get("/api/bench/runs/{run_id}/logs")
def api_bench_logs(run_id: str, tail: int = 200) -> dict:
    _get_bench_or_404(run_id)
    return {"text": bench_read_log(run_id, tail=tail)}


@app.get("/api/bench/runs/{run_id}/results")
def api_bench_results(run_id: str) -> dict:
    _get_bench_or_404(run_id)
    rp = bs.run_dir(run_id) / "results.json"
    if not rp.is_file():
        raise HTTPException(status_code=404, detail="results.json 尚未生成")
    import json as _json
    return _json.loads(rp.read_text(encoding="utf-8"))


@app.post("/api/bench/runs/{run_id}/stop")
def api_bench_stop(run_id: str) -> dict:
    _get_bench_or_404(run_id)
    return bench_stop_run(run_id)


@app.get("/api/arch/design")
def api_arch_design_get(arch: str = "dense") -> dict:
    try:
        return arch_inspect_design(arch)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/arch/design")
def api_arch_design(req: ArchDesignRequest) -> dict:
    try:
        return arch_inspect_design(req.arch)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/arch/inspect")
def api_arch_inspect(req: ArchInspectRequest) -> dict:
    path = (req.ckpt_path or "").strip()
    if not path:
        raise HTTPException(status_code=400, detail="ckpt_path 不能为空")
    try:
        return arch_inspect_path(path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"找不到 checkpoint: {path}")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"解析失败: {exc}") from exc


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



@app.post("/api/infer/continue/stream")
async def infer_continue_stream(req: GenerateRequest) -> StreamingResponse:
    """Prefix continuation stream for pretrained models (no chat template)."""

    def event_iter():
        try:
            for item in infer_engine.stream_continue(
                req.prompt,
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
