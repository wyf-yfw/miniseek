#!/usr/bin/env python3
"""MiniSeek 控制台 — task 驱动训练 + 对话/续写。

训练以 tasks/<id>/ 为真相源；不再扫描 /proc、不再 attach。
"""

from __future__ import annotations

import asyncio
import fnmatch
import glob
import json
import os
import re
import sys
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.pop("all_proxy", None)
os.environ.pop("ALL_PROXY", None)

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from algo.model import load_model_from_blob
from algo.model.attn_backend import configure_attn_backend
from algo.model.config import ModelConfig
from algo.model.dense import DenseLM
from algo.model.tokenizer import load_tokenizer
from algo.model.v41.cache import PrefixCache
from api import tasks_store as ts
from api import bench_store as bs
from api import clean_store as cs
from api.task_runner import (
    TRAIN_ARCHS,
    pid_alive,
    reconcile_task,
    rebuild_metrics_from_log,
    start_task,
    stop_task,
    validate_sft_init_ckpt,
)
from api.bench_runner import (
    reconcile_run,
    start_run as bench_start_run,
    stop_run as bench_stop_run,
    read_log as bench_read_log,
)
from api.clean_runner import (
    reconcile_job as reconcile_clean,
    start_job as clean_start_job,
    stop_job as clean_stop_job,
    read_log as clean_read_log,
    run_preview as clean_run_preview,
    job_progress as clean_job_progress,
)
from api.arch_inspect import inspect_design as arch_inspect_design, inspect_path as arch_inspect_path
from api import worker_client as wc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = PROJECT_ROOT / "frontend"
DEFAULT_PRETRAIN_DATA = str(PROJECT_ROOT / "data" / "PT" / "pretrain_mix_12g_clean.jsonl")
DEFAULT_SFT_DATA = str(PROJECT_ROOT / "data" / "SFT" / "sft_miniseek_v5_10g.jsonl")
DEFAULT_PRETRAIN_OUT = str(PROJECT_ROOT / "tasks" / "_default_pretrain")
DEFAULT_SFT_OUT = str(PROJECT_ROOT / "tasks" / "_default_sft")
DEFAULT_V41_OUT = str(PROJECT_ROOT / "tasks" / "_default_v41")


def _gpu_text() -> str:
    if not torch.cuda.is_available():
        return "无 GPU"
    free, total = torch.cuda.mem_get_info(0)
    used = (total - free) / (1024**3)
    return f"{torch.cuda.get_device_name(0)} | 显存 {used:.1f}/{total / (1024**3):.1f} GB"


def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        return None


_TRAIN_TOTALS_RE = re.compile(r"total_steps=(\d+)")
_TRAIN_SAMPLES_RE = re.compile(r"samples=(\d+)")
_TRAIN_START_STEP_RE = re.compile(r"start_step=(\d+)")
_TRAIN_RESUMED_STEP_RE = re.compile(r"resumed from .* (?:lifetime_step|step)=(\d+)")
# train.log uses "step=123"; do not use "step": (expert_load JSON) or start_step=/total_steps=
_TRAIN_STEP_RE = re.compile(r"(?<![A-Za-z0-9_])step=(\d+)\b")
_train_totals_cache: Dict[str, Dict[str, Optional[int]]] = {}


def _latest_train_step(task_id: str) -> Optional[int]:
    """Latest step=N from train.log tail — independent of plot_every / metrics.json."""
    log_path = ts.task_dir(task_id) / "train.log"
    try:
        with open(log_path, "rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - 262144))
            chunk = fh.read().decode("utf-8", errors="replace")
    except OSError:
        return None
    step: Optional[int] = None
    for m in _TRAIN_STEP_RE.finditer(chunk):
        step = int(m.group(1))
    return step


def _train_totals(task_id: str) -> Dict[str, Optional[int]]:
    """从 train.log 头部取 samples / total_steps / start_step。

    现行约定：续训也从本轮 0 起，``start_step=0``、``total_steps``=本轮 max_steps，
    进度分母不含 ckpt 历史步。旧日志若先打本轮 total、后打 resumed step=N，
    且 total<=start，则仍把分母修成 start+this_run（兼容历史）。
    更旧的「绝对 total=起点+本轮」日志：前端用 (step-start)/(total-start) 显示本轮进度。
    """
    hit = _train_totals_cache.get(task_id)
    if hit is not None:
        return hit
    info: Dict[str, Optional[int]] = {
        "total_steps": None,
        "samples": None,
        "start_step": None,
    }
    try:
        log_path = ts.task_dir(task_id) / "train.log"
        with open(log_path, "r", encoding="utf-8", errors="replace") as fh:
            for _ in range(800):
                line = fh.readline()
                if not line:
                    break
                # Stop once we leave the startup header (first real train step line).
                if line.startswith("epoch=") and " step=" in line and "loss=" in line:
                    break
                # Prefer train-set samples (last non-valid header hit), not valid=...samples=
                if "valid=" not in line:
                    m = _TRAIN_SAMPLES_RE.search(line)
                    if m:
                        info["samples"] = int(m.group(1))
                m = _TRAIN_TOTALS_RE.search(line)
                if m:
                    # Last header hit wins (post-resume corrected line).
                    info["total_steps"] = int(m.group(1))
                m = _TRAIN_START_STEP_RE.search(line)
                if m:
                    info["start_step"] = int(m.group(1))
                m = _TRAIN_RESUMED_STEP_RE.search(line)
                if m:
                    # Legacy "resumed ... step=N" may seed start_step; new logs use
                    # lifetime_step= and always print start_step=0 afterwards.
                    if info["start_step"] is None and "lifetime_step=" not in line:
                        info["start_step"] = int(m.group(1))
    except OSError:
        pass

    start = info.get("start_step")
    total = info.get("total_steps")
    if start is not None and total is not None and total <= start:
        # Legacy: total_steps was this-run budget only.
        info["total_steps"] = start + total

    if info["total_steps"] is not None:
        _train_totals_cache[task_id] = info
    return info



def _live_train_status(
    task_id: str,
    *,
    prev_step: Optional[int] = None,
    prev_mono: Optional[float] = None,
) -> Dict[str, Any]:
    """Enriched SSE status: gpu / elapsed / step / step_s (+ sliding snap fields)."""
    task = reconcile_task(ts.load_task(task_id))
    totals = _train_totals(task_id)
    metrics = ts.read_metrics(task_id)
    loss = metrics.get("loss") or []
    step: Optional[int] = int(loss[-1]["step"]) if loss else None
    log_step = _latest_train_step(task_id)
    if log_step is not None and (step is None or log_step > step):
        step = log_step
    started = _parse_iso(task.get("started_at"))
    ended = _parse_iso(task.get("ended_at"))
    now = datetime.now()
    if started is None:
        elapsed_sec = 0
    elif ended is not None:
        elapsed_sec = max(0, int((ended - started).total_seconds()))
    else:
        elapsed_sec = max(0, int((now - started).total_seconds()))

    step_s: Optional[float] = None
    mono = time.monotonic()
    if (
        step is not None
        and prev_step is not None
        and prev_mono is not None
        and step > prev_step
    ):
        dt = mono - prev_mono
        if dt >= 0.5:
            step_s = (step - prev_step) / dt
    if step_s is None and step is not None and elapsed_sec > 0:
        start = totals.get("start_step")
        if start is not None and step >= int(start):
            step_s = (step - int(start)) / max(elapsed_sec, 1)
        elif len(loss) >= 2:
            first = int(loss[0]["step"])
            avg = (step - first) / max(elapsed_sec, 1)
            step_s = step / max(elapsed_sec, 1) if avg < 0 else avg
        else:
            step_s = step / max(elapsed_sec, 1)

    return {
        "id": task["id"],
        "status": task.get("status"),
        "pid": task.get("pid"),
        "gpu": _gpu_text(),
        "started_at": task.get("started_at"),
        "ended_at": task.get("ended_at"),
        "elapsed_sec": elapsed_sec,
        "step": step,
        "step_s": None if step_s is None else round(float(step_s), 4),
        "total_steps": totals["total_steps"],
        "start_step": totals.get("start_step"),
        "samples": totals["samples"],
        "_mono": mono,
        "_step_snap": step,
    }


def _public_status(raw: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in raw.items() if not str(k).startswith("_")}


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
        self._cancel = threading.Event()
        self.prefix_cache = None
        self.drafter = None
        self.sps = None

    def stop_generation(self) -> None:
        """Signal the active stream_chat / stream_continue loop to break ASAP."""
        self._cancel.set()

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
                blob = {
                    "model": sd,
                    "config": cfg_raw,
                    "step": cfg_raw.get("train_step", "?"),
                    "loss": cfg_raw.get("train_loss", "?"),
                }
        else:
            blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model, cfg = load_model_from_blob(blob)
        configure_attn_backend(bool(getattr(cfg, "use_flash_attn", True)))
        # v41：多轮对话复用上一轮的全局 KV（单槽前缀缓存）；换权重时清空
        self.prefix_cache = PrefixCache() if getattr(model, "supports_prefix_cache", False) else None
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
        dspark_note = self._load_sibling_drafter(path, cfg, device)
        step = blob.get("step", "?")
        loss = blob.get("loss", "?")
        where = "GPU" if device.type == "cuda" else "CPU"
        return f"已加载 {Path(ckpt_path).name} (Step: {step}, Loss: {loss}, {where}){dspark_note}"

    def _load_sibling_drafter(self, path: Path, cfg, device: torch.device) -> str:
        """v41：checkpoint 同目录有 dspark.pt 时加载草稿头，对话/续写走 DSpark 投机解码。"""
        self.drafter, self.sps = None, None
        dpath = path.with_name("dspark.pt") if path.is_file() else None
        if dpath is None or not dpath.exists() or not getattr(self.model, "supports_prefix_cache", False):
            return ""
        try:
            from algo.model.v41.dspark import measure_sps
            from algo.train_dspark import load_drafter

            dblob = torch.load(dpath, map_location="cpu", weights_only=False)
            drafter = load_drafter(dblob)
            if drafter.cfg.dim != cfg.dim or drafter.cfg.vocab_size != cfg.vocab_size:
                return "（dspark.pt 与主干尺寸不符，未启用）"
            self.drafter = drafter.to(device).eval()
            self.sps = measure_sps(self.model, cfg.dspark_block_size + 1, device)
            return "，DSpark 投机解码已启用"
        except Exception as exc:  # noqa: BLE001
            self.drafter = None
            return f"（DSpark 加载失败：{exc}）"

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

    def expert_load(
        self,
        *,
        data_path: str,
        num_samples: int = 64,
        max_seq_len: int = 512,
        batch_size: int = 4,
        ckpt_path: str = "",
    ) -> dict:
        """Run a validation probe and return routed-expert assignment load."""
        import math

        ckpt_path = (ckpt_path or "").strip()
        if ckpt_path:
            if self.loaded_ckpt != ckpt_path or self.model is None:
                self.load_model(ckpt_path)
        if self.model is None:
            raise HTTPException(status_code=400, detail="请先加载一个 Checkpoint")

        cfg = getattr(self.model, "cfg", None)
        arch = getattr(cfg, "arch", "dense") if cfg is not None else "dense"
        use_moe = arch == "v41" or bool(getattr(cfg, "use_moe", False))
        if not use_moe:
            return {
                "arch": "dense",
                "ok": False,
                "message": "当前是 Dense 模型，无路由专家负载（N/A）",
            }
        if not hasattr(self.model, "probe_expert_load"):
            raise HTTPException(status_code=400, detail="该 MoE 权重不支持专家负载探针")

        data_p = Path(data_path.strip() or DEFAULT_PRETRAIN_DATA)
        if not data_p.is_absolute():
            data_p = (PROJECT_ROOT / data_p).resolve()
        if not data_p.exists():
            raise HTTPException(status_code=404, detail=f"找不到数据: {data_p}")

        num_samples = max(1, int(num_samples))
        max_seq_len = max(8, int(max_seq_len))
        batch_size = max(1, min(int(batch_size), 16))
        n_routed = int(getattr(cfg, "n_routed_experts", 0))
        top_k = int(getattr(cfg, "num_experts_per_tok", None) or getattr(cfg, "n_activated_experts", 2))
        n_layers = int(getattr(cfg, "n_layers", 0))
        n_shared = int(getattr(cfg, "n_shared_experts", 1))

        texts: List[str] = []
        with data_p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = obj.get("text") or obj.get("content") or ""
                if text:
                    texts.append(text)
                if len(texts) >= num_samples:
                    break
        if not texts:
            raise HTTPException(status_code=400, detail="探针数据为空")

        total_counts = torch.zeros(n_routed, dtype=torch.long)
        n_tokens = 0
        t0 = time.time()
        with self._gen_lock:
            for start in range(0, len(texts), batch_size):
                chunk = texts[start : start + batch_size]
                encoded = [
                    self.tokenizer.encode(t, add_special_tokens=True)[:max_seq_len]
                    for t in chunk
                ]
                encoded = [ids if ids else [0] for ids in encoded]
                max_len = max(len(ids) for ids in encoded)
                pad_id = int(getattr(self.tokenizer, "pad_token_id", 0) or 0)
                batch = torch.full(
                    (len(encoded), max_len), pad_id, dtype=torch.long, device=self.device
                )
                for i, ids in enumerate(encoded):
                    batch[i, : len(ids)] = torch.tensor(
                        ids, dtype=torch.long, device=self.device
                    )
                counts, tok = self.model.probe_expert_load(batch)
                total_counts += counts.cpu().long()
                n_tokens += int(tok)

        counts_list = [int(x) for x in total_counts.tolist()]
        total_assign = int(sum(counts_list))
        fractions = [
            (c / total_assign if total_assign > 0 else 0.0) for c in counts_list
        ]
        max_frac = max(fractions) if fractions else 0.0
        min_frac = min(fractions) if fractions else 0.0
        # Shannon entropy (bits) over assignment distribution
        entropy = 0.0
        for f in fractions:
            if f > 0:
                entropy -= f * math.log(f, 2)
        uniform = math.log(max(n_routed, 1), 2)
        cost = time.time() - t0
        return {
            "arch": arch,
            "ok": True,
            "ckpt": self.loaded_ckpt,
            "data_path": str(data_p),
            "num_samples": len(texts),
            "n_tokens": n_tokens,
            "n_routed": n_routed,
            "n_shared": n_shared,
            "top_k": top_k,
            "n_layers": n_layers,
            "counts": counts_list,
            "fractions": fractions,
            "total_assignments": total_assign,
            "max_frac": max_frac,
            "min_frac": min_frac,
            "max_expert": int(max(range(n_routed), key=lambda i: fractions[i])) if n_routed else -1,
            "min_expert": int(min(range(n_routed), key=lambda i: fractions[i])) if n_routed else -1,
            "entropy_bits": entropy,
            "entropy_uniform_bits": uniform,
            "seconds": round(cost, 2),
            "note": "shared 专家对每个 token 始终启用；下图为路由专家 top-k 槽位占比（跨层合计）",
            "message": (
                f"验证探针完成：{len(texts)} 条 / {n_tokens} tokens，"
                f"最大负载 {max_frac * 100:.1f}% (E{int(max(range(n_routed), key=lambda i: fractions[i])) if n_routed else -1})"
            ),
        }

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
        cancelled = False
        if self.drafter is not None:
            extra = {"drafter": self.drafter, "sps": self.sps}
        elif self.prefix_cache is not None:
            extra = {"prefix_cache": self.prefix_cache}
        else:
            extra = {}
        with self._gen_lock:
            self._cancel.clear()
            for token in self.model.generate_stream(
                input_ids,
                max_new_tokens=int(max_new_tokens),
                temperature=float(temperature),
                top_k=int(top_k),
                eos_token_id=eos_id,
                **extra,
            ):
                if self._cancel.is_set():
                    cancelled = True
                    break
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
                "cancelled": cancelled,
            },
        }


    @staticmethod
    def _escape_token_piece(s: str) -> str:
        """Readable repr for byte-level / incomplete UTF-8 pieces (never blank or bare U+FFFD)."""
        if s is None:
            return ""
        s = str(s)
        if not s:
            return "<empty>"
        out = []
        for ch in s:
            o = ord(ch)
            if ch == "\n":
                out.append("\\n")
            elif ch == "\t":
                out.append("\\t")
            elif ch == "\r":
                out.append("\\r")
            elif ch == " ":
                out.append("·")
            elif ch == "�" or o < 32 or o == 127:
                out.append(f"\\u{o:04x}")
            elif not ch.isprintable():
                out.append(f"\\u{o:04x}")
            else:
                out.append(ch)
        rendered = "".join(out)
        if all(c == "�" for c in s):
            return " ".join(f"\\u{ord(c):04x}" for c in s) or "<empty>"
        return rendered

    def _token_str(self, token_id: int) -> str:
        """Surface form for top-k / step label. Never return blank or lone U+FFFD."""
        tok = self.tokenizer
        tid = int(token_id)
        decoded = ""
        try:
            decoded = tok.decode([tid], skip_special_tokens=False)
        except Exception:
            decoded = ""
        raw = None
        try:
            raw = tok.convert_ids_to_tokens(tid)
            if raw is not None:
                raw = str(raw)
        except Exception:
            raw = None

        # Whole UTF-8 piece from a single id.
        if decoded and "�" not in decoded:
            if any(c in decoded for c in ("\n", "\t", "\r", " ")) or any(ord(c) < 32 for c in decoded):
                return self._escape_token_piece(decoded)
            return decoded

        # Byte-level BPE: vocab surface (bytes_to_unicode) or unicode escapes — never bare U+FFFD.
        if raw:
            return self._escape_token_piece(raw)
        if decoded:
            return self._escape_token_piece(decoded)
        return f"<id:{tid}>"

    def _format_topk(self, chosen_id: int, info: dict) -> list:
        rows = []
        ids = info.get("ids") or []
        logits = info.get("logits") or []
        probs = info.get("probs") or []
        for i, tid in enumerate(ids):
            tid = int(tid)
            rows.append({
                "id": tid,
                "token": self._token_str(tid),
                "logit": round(float(logits[i]) if i < len(logits) else 0.0, 4),
                "prob": round(float(probs[i]) if i < len(probs) else 0.0, 6),
                "chosen": tid == int(chosen_id),
            })
        if not any(r["chosen"] for r in rows):
            rows.append({
                "id": int(chosen_id),
                "token": self._token_str(int(chosen_id)),
                "logit": None,
                "prob": None,
                "chosen": True,
            })
        return rows

    def stream_chat_debug(
        self,
        messages: List[dict],
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 50,
        seed: int = 42,
        debug_topk: int = 10,
    ):
        if self.model is None:
            raise HTTPException(status_code=400, detail="请先加载一个 Checkpoint")
        if not hasattr(self.model, "generate_stream_debug"):
            raise HTTPException(status_code=400, detail="当前模型不支持逐步调试")
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
        candidates_k = max(1, min(int(debug_topk), 50))
        cancelled = False
        with self._gen_lock:
            self._cancel.clear()
            for token, info in self.model.generate_stream_debug(
                input_ids,
                max_new_tokens=int(max_new_tokens),
                temperature=float(temperature),
                top_k=int(top_k),
                eos_token_id=eos_id,
                candidates_k=candidates_k,
            ):
                if self._cancel.is_set():
                    cancelled = True
                    break
                n += 1
                topk_rows = self._format_topk(int(token), info)
                if eos_id is not None and token == int(eos_id):
                    yield {
                        "event": "token",
                        "data": {
                            "text": "",
                            "token_id": int(token),
                            "token": self._token_str(int(token)),
                            "incomplete_utf8": False,
                            "topk": topk_rows,
                            "eos": True,
                            "step": n,
                        },
                    }
                    break
                token_ids.append(int(token))
                full = self.tokenizer.decode(token_ids, skip_special_tokens=False)
                for special in ("<|im_start|>", "<|im_end|>", "<|endoftext|>"):
                    full = full.replace(special, "")
                token_disp = self._token_str(int(token))
                incomplete = "�" in full
                piece = ""
                if not incomplete:
                    # Complete UTF-8 so far: emit delta vs previously shown clean text.
                    if full.startswith(prev_text):
                        piece = full[len(prev_text):]
                    else:
                        piece = full
                    prev_text = full
                yield {
                    "event": "token",
                    "data": {
                        "text": piece,
                        "token_id": int(token),
                        "token": token_disp,
                        "incomplete_utf8": incomplete,
                        "topk": topk_rows,
                        "eos": False,
                        "step": n,
                    },
                }
        cost = time.time() - t0
        speed = n / max(cost, 1e-6)
        yield {
            "event": "done",
            "data": {
                "text": prev_text,
                "tokens": n,
                "seconds": round(cost, 2),
                "tokens_per_s": round(speed, 1),
                "cancelled": cancelled,
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
        cancelled = False
        extra = {"drafter": self.drafter, "sps": self.sps} if self.drafter is not None else {}
        with self._gen_lock:
            self._cancel.clear()
            for token in self.model.generate_stream(
                input_ids,
                max_new_tokens=int(max_new_tokens),
                temperature=float(temperature),
                top_k=int(top_k),
                eos_token_id=eos_id,
                **extra,
            ):
                if self._cancel.is_set():
                    cancelled = True
                    break
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
                "cancelled": cancelled,
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


class ExpertLoadRequest(BaseModel):
    ckpt_path: str = ""
    data_path: str = DEFAULT_PRETRAIN_DATA
    num_samples: int = Field(default=64, ge=1, le=2048)
    max_seq_len: int = Field(default=512, ge=8, le=4096)
    batch_size: int = Field(default=4, ge=1, le=16)


class ArchInspectRequest(BaseModel):
    ckpt_path: str


class ArchDesignRequest(BaseModel):
    arch: str = "dense"  # tiny | dense | v41 | v41-flash (reference)


class BenchCreate(BaseModel):
    name: str = ""
    ckpt_path: str = ""
    tasks: List[str] = Field(default_factory=lambda: ["ceval-valid", "cmmlu", "gsm8k"])
    limit: Optional[int] = None


class BenchPatch(BaseModel):
    name: Optional[str] = None
    ckpt_path: Optional[str] = None
    tasks: Optional[List[str]] = None
    limit: Optional[int] = None


class CleanCreate(BaseModel):
    name: str = ""
    input_path: str = ""
    operators: Optional[Dict[str, Any]] = None


class CleanPatch(BaseModel):
    name: Optional[str] = None
    input_path: Optional[str] = None
    operators: Optional[Dict[str, Any]] = None


class CleanPreviewRequest(BaseModel):
    input_path: str
    operators: Optional[Dict[str, Any]] = None
    limit: int = 200


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
    debug: bool = False
    debug_topk: int = 10


def list_checkpoints_global() -> List[str]:
    files = glob.glob(str(PROJECT_ROOT / "tasks" / "**" / "checkpoints" / "*.pt"), recursive=True)
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
    ]:
        if p.exists():
            lines.append(f"{label}: {p.name} ({p.stat().st_size / (1024 * 1024):.1f} MB)")
        else:
            lines.append(f"{label}: 缺少 {p.name}")
    lines.append("验证: 默认 valid_ratio=1% 从训练集划分（无需 valid_mix.jsonl）")
    lines.append(_gpu_text())
    return "\n".join(lines)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"



def _worker_request(method: str, path: str, **kwargs):
    return wc.forward(method, path, **kwargs)


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
    kind: str = "file"  # file | dir | any (kept for old clients)
    title: str = "选择文件"
    initial: str = ""
    filetypes: str = ""  # e.g. "JSONL|*.jsonl;Checkpoint|*.pt;All|*.*"


# In-page path picker may browse under the project and the user home.
FS_ALLOWED_ROOTS: List[Path] = [
    PROJECT_ROOT.resolve(),
    Path.home().resolve(),
]


def _parse_glob_patterns(spec: str) -> List[str]:
    """'JSONL|*.jsonl;Checkpoint|*.pt;All|*.*' -> ['*.jsonl', '*.pt', '*.*']."""
    patterns: List[str] = []
    for part in (spec or "").split(";"):
        part = part.strip()
        if not part:
            continue
        pattern = part.split("|", 1)[1].strip() if "|" in part else part
        if pattern and pattern not in patterns:
            patterns.append(pattern)
    return patterns or ["*.*"]


def _effective_patterns(patterns: List[str]) -> List[str]:
    """Prefer specific globs; only treat *.*/ * as allow-all when alone."""
    specific = [p for p in patterns if p not in {"*.*", "*", ""}]
    return specific or ["*.*"]


def _match_filename(name: str, patterns: List[str]) -> bool:
    eff = _effective_patterns(patterns)
    if any(p in {"*.*", "*", ""} for p in eff):
        return True
    return any(fnmatch.fnmatch(name, p) for p in eff)


def _is_under_allowed(path: Path) -> bool:
    try:
        resolved = path.resolve()
    except (OSError, RuntimeError):
        return False
    for root in FS_ALLOWED_ROOTS:
        try:
            resolved.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def _safe_fs_path(raw: Optional[str]) -> Path:
    """Resolve a user path and reject escapes outside allowed roots."""
    text = (raw or "").strip()
    if "\x00" in text:
        raise HTTPException(status_code=400, detail="非法路径")
    if not text:
        return PROJECT_ROOT.resolve()
    p = Path(text)
    try:
        p = (PROJECT_ROOT / p).resolve() if not p.is_absolute() else p.resolve()
    except (OSError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=f"非法路径：{exc}") from exc
    if not _is_under_allowed(p):
        raise HTTPException(status_code=403, detail="路径超出允许范围（项目目录或家目录）")
    return p


@app.get("/api/fs/list")
def api_fs_list(path: str = "", filetypes: str = "") -> dict:
    """List a directory for the in-page path picker. Never returns file contents."""
    target = _safe_fs_path(path)
    if target.exists() and target.is_file():
        target = target.parent
        if not _is_under_allowed(target):
            raise HTTPException(status_code=403, detail="路径超出允许范围（项目目录或家目录）")
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"路径不存在: {target}")
    if not target.is_dir():
        raise HTTPException(status_code=400, detail="不是目录")

    patterns = _parse_glob_patterns(filetypes)
    try:
        children = list(target.iterdir())
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=f"无权限读取: {exc}") from exc

    entries: List[dict] = []
    for child in children:
        name = child.name
        if name.startswith("."):
            continue
        try:
            is_dir = child.is_dir()
        except OSError:
            continue
        try:
            resolved = child.resolve()
        except (OSError, RuntimeError):
            continue
        if not _is_under_allowed(resolved):
            continue
        if not is_dir and not _match_filename(name, patterns):
            continue
        item: dict = {
            "name": name,
            "path": str(resolved),
            "is_dir": is_dir,
        }
        if not is_dir:
            try:
                item["size"] = child.stat().st_size
            except OSError:
                item["size"] = None
        entries.append(item)

    entries.sort(key=lambda e: (not e["is_dir"], e["name"].lower()))

    parent = target.parent
    parent_out: Optional[str] = None
    try:
        parent_res = parent.resolve()
        if parent_res != target.resolve() and _is_under_allowed(parent_res):
            parent_out = str(parent_res)
    except (OSError, RuntimeError):
        parent_out = None

    return {
        "path": str(target.resolve()),
        "parent": parent_out,
        "entries": entries,
        "roots": [str(r) for r in FS_ALLOWED_ROOTS],
    }


@app.post("/api/pick-file")
def api_pick_file(body: PickFileBody) -> dict:
    """Deprecated native dialog. Remote/Funnel clients use GET /api/fs/list + in-page picker."""
    raise HTTPException(
        status_code=410,
        detail="系统文件对话框已停用（远程/无图形界面不可用），请使用页面内路径选择器",
    )


@app.get("/")
def index() -> FileResponse:
    return FileResponse(
        WEB_DIR / "index.html",
        headers={"Cache-Control": "no-cache"},
    )


@app.get("/api/defaults")
def defaults() -> dict:
    best = PROJECT_ROOT / "tasks" / "_default_pretrain" / "best.pt"
    return {
        "mode": "pretrain",
        "arch": "dense",
        "dense": {
            "arch": "dense",
            "mode": "pretrain",
            "data_path": DEFAULT_PRETRAIN_DATA,
            "batch_size": 32,
            "max_seq_len": 512,
            "lr": 5e-5,
            "min_lr": 1e-5,
            "warmup_steps": 200,
            "epochs": 1,
            "max_steps": 0,
            "grad_clip": 1.0,
            "save_loss_thresholds": "",
            "stable_steps": 50,
            "save_every": 0,
            "log_every": 1,
            "batch_mode": "pack",
            "compile": True,
            "fused_adamw": True,
            "flash_attn": True,
            "plot_every": 500,
            "valid_every": 500,
            "valid_max_samples": 512,
            "valid_ratio": 0.01,
            "valid_seed": 42,
            "init_ckpt": "",
            "resume": "",
            "aux_coef": 0.0001,
        },
        "v41": {
            "arch": "v41",
            "mode": "pretrain",
            "data_path": DEFAULT_PRETRAIN_DATA,
            "batch_size": 8,
            "max_seq_len": 1024,
            "lr": 3e-4,
            "min_lr": 3e-5,
            "warmup_steps": 200,
            "epochs": 1,
            "max_steps": 0,
            "grad_clip": 1.0,
            "save_loss_thresholds": "",
            "stable_steps": 50,
            "save_every": 0,
            "log_every": 1,
            "batch_mode": "pack",
            "compile": False,
            "fused_adamw": True,
            "flash_attn": False,
            "plot_every": 500,
            "valid_every": 500,
            "valid_max_samples": 512,
            "valid_ratio": 0.01,
            "valid_seed": 42,
            "init_ckpt": "",
            "resume": "",
            "aux_coef": 0.0001,
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
            "save_loss_thresholds": "",
            "stable_steps": 200,
            "save_every": 0,
            "log_every": 1,
            "batch_mode": "pack",
            "compile": True,
            "fused_adamw": True,
            "flash_attn": True,
            "plot_every": 1,
            "valid_every": 20,
            "valid_max_samples": 512,
            "valid_ratio": 0.01,
            "valid_seed": 42,
            "init_ckpt": "",
            "resume": "",
            "aux_coef": 0.0001,
        },
        "pretrain": {
            "arch": "dense",
            "data_path": DEFAULT_PRETRAIN_DATA,
            "out_dir": DEFAULT_PRETRAIN_OUT,
            "batch_size": 32,
            "lr": 5e-5,
            "min_lr": 1e-5,
            "warmup_steps": 200,
            "stable_steps": 50,
            "plot_every": 500,
            "valid_every": 500,
            "valid_max_samples": 512,
            "valid_ratio": 0.01,
            "valid_seed": 42,
            "save_loss_thresholds": "",
            "init_ckpt": "",
            "resume": "",
            "aux_coef": 0.0001,
        },
        "pretrain_v41": {
            "arch": "v41",
            "data_path": DEFAULT_PRETRAIN_DATA,
            "out_dir": DEFAULT_V41_OUT,
            "batch_size": 8,
            "lr": 3e-4,
            "min_lr": 3e-5,
            "warmup_steps": 200,
            "stable_steps": 50,
            "plot_every": 500,
            "valid_every": 500,
            "valid_max_samples": 512,
            "valid_ratio": 0.01,
            "valid_seed": 42,
            "save_loss_thresholds": "",
            "init_ckpt": "",
            "resume": "",
            "aux_coef": 0.0001,
        },
        "sft": {
            "arch": "dense",
            "mode": "sft",
            "data_path": DEFAULT_SFT_DATA,
            "out_dir": DEFAULT_SFT_OUT,
            "lr": 5e-5,
            "min_lr": 1e-5,
            "warmup_steps": 100,
            "plot_every": 500,
            "valid_every": 500,
            "valid_max_samples": 512,
            "valid_ratio": 0.0,
            "valid_seed": 42,
            "save_loss_thresholds": "",
            "init_ckpt": str(best) if best.exists() else "",
            "resume": "",
            "aux_coef": 0.0001,
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
    if arch not in TRAIN_ARCHS:
        raise HTTPException(status_code=400, detail=f"未知架构: {arch}")
    if mode not in {"pretrain", "sft"}:
        raise HTTPException(status_code=400, detail=f"未知模式: {mode}")
    defaults_map = defaults()
    base = dict(defaults_map.get(arch) or defaults_map["dense"])
    base.update(body.hparams or {})
    base["arch"] = arch
    base["mode"] = mode
    base.pop("out_dir", None)  # out_dir forced by store
    try:
        validate_sft_init_ckpt(mode, base, require_exists=False)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
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
            if arch not in TRAIN_ARCHS:
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
        try:
            validate_sft_init_ckpt(
                str(task.get("mode") or ""),
                task.get("hparams") or {},
                require_exists=False,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
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


@app.get("/api/tasks/{task_id}/metrics")
def api_task_metrics(
    task_id: str,
    plot_every: Optional[int] = None,
    max_points: int = 5000,
) -> dict:
    _get_task_or_404(task_id)
    if plot_every is None:
        return ts.read_metrics(task_id)
    return rebuild_metrics_from_log(task_id, plot_every=plot_every, max_points=max_points)


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
        prev_step: Optional[int] = None
        prev_mono: Optional[float] = None
        last_status_ts = 0.0

        def emit_status(force: bool = False) -> Optional[str]:
            nonlocal prev_step, prev_mono, last_status_ts
            now_m = time.monotonic()
            if not force and (now_m - last_status_ts) < 4.0:
                return None
            raw = _live_train_status(task_id, prev_step=prev_step, prev_mono=prev_mono)
            snap = raw.get("_step_snap")
            mono = raw.get("_mono")
            if snap is not None:
                prev_step = int(snap)
            if mono is not None:
                prev_mono = float(mono)
            last_status_ts = now_m
            return _sse("status", _public_status(raw))

        # Initial dump of last `tail` lines
        try:
            text_log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
        except OSError:
            text_log = ""
        lines = [ln for ln in text_log.splitlines() if ln.strip()]
        for ln in lines[-max(1, tail):]:
            yield _sse("log", {"line": ln})
        metrics = ts.read_metrics(task_id)
        yield _sse("metrics", metrics)
        st = emit_status(force=True)
        if st:
            yield st

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
                    st = emit_status(force=False)
                    if st:
                        yield st
                    idle = 0
                else:
                    idle += 1
                    if idle % 10 == 0:
                        yield ": ping\n\n"
                        st = emit_status(force=True)
                        if st:
                            yield st
                        task = reconcile_task(ts.load_task(task_id))
                        if task.get("status") not in {"running"}:
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
        raise HTTPException(status_code=404, detail=f"bench task not found: {run_id}") from exc


def _bench_with_progress(run: dict) -> dict:
    out = dict(run)
    try:
        out["progress"] = bs.read_progress(run["id"])
    except Exception:  # noqa: BLE001
        out["progress"] = None
    return out


# ---- Bench Tasks API (symmetric with /api/tasks) ----

@app.get("/api/bench/tasks")
def api_bench_list_tasks() -> dict:
    rows = []
    for row in bs.list_runs():
        if row.get("status") == "running":
            row = reconcile_run(bs.load_run(row["id"]))
        rows.append({
            "id": row["id"],
            "name": row.get("name") or row["id"],
            "status": row.get("status"),
            "ckpt_path": row.get("ckpt_path"),
            "tasks": row.get("tasks"),
            "limit": row.get("limit"),
            "device": row.get("device"),
            "created_at": row.get("created_at"),
            "started_at": row.get("started_at"),
            "ended_at": row.get("ended_at"),
            "pid": row.get("pid"),
            "scores": row.get("scores"),
        })
    return {"tasks": rows}


@app.post("/api/bench/tasks")
def api_bench_create_task(body: BenchCreate) -> dict:
    try:
        return bs.create_run(
            name=body.name or "",
            ckpt_path=body.ckpt_path or "",
            tasks=list(body.tasks or ["ceval-valid", "cmmlu", "gsm8k"]),
            limit=body.limit,
            device="cpu",
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/bench/tasks/{run_id}")
def api_bench_get_task(run_id: str) -> dict:
    return _bench_with_progress(_get_bench_or_404(run_id))


@app.patch("/api/bench/tasks/{run_id}")
def api_bench_patch_task(run_id: str, body: BenchPatch) -> dict:
    _get_bench_or_404(run_id)
    try:
        kwargs = {}
        if body.name is not None:
            kwargs["name"] = body.name
        if body.ckpt_path is not None:
            kwargs["ckpt_path"] = body.ckpt_path
        if body.tasks is not None:
            kwargs["tasks"] = body.tasks
        if "limit" in body.model_fields_set:
            kwargs["limit"] = body.limit
        run = bs.patch_run(run_id, **kwargs)
        return _bench_with_progress(run)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc



@app.delete("/api/bench/tasks/{run_id}")
def api_bench_delete_task(run_id: str) -> dict:
    run = _get_bench_or_404(run_id)
    if run.get("status") == "running" and pid_alive(run.get("pid")):
        raise HTTPException(status_code=409, detail="running task 不能删除，请先 Stop")
    try:
        bs.delete_run(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"ok": True, "id": run_id}


@app.post("/api/bench/tasks/{run_id}/start")
def api_bench_start_task(run_id: str) -> dict:
    _get_bench_or_404(run_id)
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


@app.get("/api/bench/tasks/{run_id}/logs")
def api_bench_task_logs(run_id: str, tail: int = 200) -> dict:
    _get_bench_or_404(run_id)
    return {"text": bench_read_log(run_id, tail=tail)}


@app.get("/api/bench/tasks/{run_id}/results")
def api_bench_task_results(run_id: str) -> dict:
    _get_bench_or_404(run_id)
    rp = bs.run_dir(run_id) / "results.json"
    if not rp.is_file():
        raise HTTPException(status_code=404, detail="results.json 尚未生成")
    import json as _json
    return _json.loads(rp.read_text(encoding="utf-8"))


@app.get("/api/bench/tasks/{run_id}/progress")
def api_bench_task_progress(run_id: str) -> dict:
    _get_bench_or_404(run_id)
    return bs.read_progress(run_id)


def _bench_runs_gone() -> None:
    raise HTTPException(status_code=410, detail="已改为 /api/bench/tasks；请刷新页面")


@app.get("/api/bench/runs")
@app.post("/api/bench/runs")
def api_bench_runs_gone() -> None:
    _bench_runs_gone()


@app.get("/api/bench/runs/{run_id}")
@app.get("/api/bench/runs/{run_id}/logs")
@app.get("/api/bench/runs/{run_id}/results")
@app.post("/api/bench/runs/{run_id}/stop")
@app.delete("/api/bench/runs/{run_id}")
def api_bench_runs_id_gone(run_id: str) -> None:
    _bench_runs_gone()


def _get_clean_or_404(job_id: str) -> dict:
    try:
        return reconcile_clean(cs.load_job(job_id))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"clean job not found: {job_id}") from exc


@app.get("/api/clean/jobs")
def api_clean_list_jobs() -> dict:
    rows = []
    for row in cs.list_jobs():
        if row.get("status") == "running":
            row = reconcile_clean(cs.load_job(row["id"]))
        item = {
            "id": row["id"],
            "name": row.get("name") or row["id"],
            "status": row.get("status"),
            "input_path": row.get("input_path"),
            "created_at": row.get("created_at"),
            "started_at": row.get("started_at"),
            "ended_at": row.get("ended_at"),
            "pid": row.get("pid"),
            "stats": row.get("stats"),
            "engine": row.get("engine"),
        }
        if row.get("status") == "running":
            try:
                item["progress"] = clean_job_progress(row)
            except Exception:  # noqa: BLE001
                item["progress"] = None
        rows.append(item)
    return {"jobs": rows, "defaults": {
        "operators": cs.DEFAULT_OPERATORS,
        "input_path": str(PROJECT_ROOT / "data" / "pretrain_mix_12g.jsonl"),
        "engine_pref": cs.DEFAULT_ENGINE_PREF,
        "write_dropped": cs.DEFAULT_WRITE_DROPPED,
        "disk_hint": cs.DISK_HINT,
    }}


@app.post("/api/clean/jobs")
def api_clean_create_job(body: CleanCreate) -> dict:
    try:
        return cs.create_job(
            name=body.name or "",
            input_path=body.input_path or "",
            operators=body.operators,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/clean/jobs/{job_id}")
def api_clean_get_job(job_id: str) -> dict:
    job = _get_clean_or_404(job_id)
    out = dict(job)
    out["report"] = cs.read_report(job_id)
    try:
        out["progress"] = clean_job_progress(job)
    except Exception:  # noqa: BLE001
        out["progress"] = None
    return out


@app.get("/api/clean/jobs/{job_id}/progress")
def api_clean_job_progress(job_id: str) -> dict:
    job = _get_clean_or_404(job_id)
    prog = clean_job_progress(job)
    if prog is None:
        return {"id": job_id, "status": job.get("status"), "progress": None}
    return {"id": job_id, "status": job.get("status"), "progress": prog}


@app.patch("/api/clean/jobs/{job_id}")
def api_clean_patch_job(job_id: str, body: CleanPatch) -> dict:
    _get_clean_or_404(job_id)
    try:
        kwargs = {}
        if body.name is not None:
            kwargs["name"] = body.name
        if body.input_path is not None:
            kwargs["input_path"] = body.input_path
        if body.operators is not None:
            kwargs["operators"] = body.operators
        return cs.patch_job(job_id, **kwargs)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/api/clean/jobs/{job_id}")
def api_clean_delete_job(job_id: str) -> dict:
    job = _get_clean_or_404(job_id)
    if job.get("status") == "running" and pid_alive(job.get("pid")):
        raise HTTPException(status_code=409, detail="running job 不能删除，请先 Stop")
    try:
        cs.delete_job(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"ok": True, "id": job_id}


@app.post("/api/clean/jobs/{job_id}/start")
def api_clean_start_job(job_id: str) -> dict:
    """Start clean job. Prefer worker; if worker lacks /v1/clean (old binary while
    train holds it), fall back to local Popen with start_new_session so API
    restart still will not reap the clean child.
    """
    _get_clean_or_404(job_id)

    def _local():
        try:
            return clean_start_job(job_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (ValueError, NotImplementedError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    if wc.use_monolith():
        return _local()
    try:
        return _worker_request("POST", f"/v1/clean/{job_id}/start")
    except HTTPException as exc:
        if exc.status_code in {404, 502}:
            return _local()
        raise


@app.post("/api/clean/jobs/{job_id}/stop")
def api_clean_stop_job(job_id: str) -> dict:
    _get_clean_or_404(job_id)
    if wc.use_monolith():
        return clean_stop_job(job_id)
    try:
        return _worker_request("POST", f"/v1/clean/{job_id}/stop")
    except HTTPException as exc:
        if exc.status_code in {404, 502}:
            return clean_stop_job(job_id)
        raise


@app.get("/api/clean/jobs/{job_id}/logs")
def api_clean_job_logs(job_id: str, tail: int = 200) -> dict:
    _get_clean_or_404(job_id)
    return {"text": clean_read_log(job_id, tail=tail)}


@app.get("/api/clean/jobs/{job_id}/report")
def api_clean_job_report(job_id: str) -> dict:
    _get_clean_or_404(job_id)
    report = cs.read_report(job_id)
    if report is None:
        raise HTTPException(status_code=404, detail="report.json 尚未生成")
    return report


@app.post("/api/clean/preview")
def api_clean_preview(body: CleanPreviewRequest) -> dict:
    try:
        return clean_run_preview(body.input_path, body.operators or {}, limit=body.limit)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc



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


@app.post("/api/moe/expert-load")
def moe_expert_load(req: ExpertLoadRequest) -> dict:
    return infer_engine.expert_load(
        data_path=req.data_path,
        num_samples=req.num_samples,
        max_seq_len=req.max_seq_len,
        batch_size=req.batch_size,
        ckpt_path=req.ckpt_path,
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


def _single_turn_messages(messages: List[dict]) -> List[dict]:
    """对话平台只做单轮：保留 system + 最后一条 user，丢掉历史多轮。"""
    systems: List[dict] = []
    last_user: Optional[dict] = None
    for m in messages:
        role = (m.get("role") or "").strip().lower()
        content = str(m.get("content") or "")
        if role == "system" and content.strip():
            systems.append({"role": "system", "content": content})
        elif role == "user":
            last_user = {"role": "user", "content": content}
    if last_user is None:
        return systems
    return systems + [last_user]


@app.post("/api/infer/stop")
def infer_stop() -> dict:
    """Stop in-flight chat/continue generation so the next request is not blocked on _gen_lock."""
    infer_engine.stop_generation()
    return {"ok": True, "message": "已请求停止生成"}


@app.post("/api/infer/chat/stream")
async def infer_chat_stream(req: ChatRequest) -> StreamingResponse:
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages 不能为空")
    payload = _single_turn_messages([m.model_dump() for m in req.messages])
    if not payload or (payload[-1].get("role") or "").strip().lower() != "user":
        raise HTTPException(status_code=400, detail="messages 中需要至少一条 user")

    def event_iter():
        try:
            if req.debug:
                gen = infer_engine.stream_chat_debug(
                    payload,
                    max_new_tokens=req.max_new_tokens,
                    temperature=req.temperature,
                    top_k=req.top_k,
                    seed=req.seed,
                    debug_topk=req.debug_topk,
                )
            else:
                gen = infer_engine.stream_chat(
                    payload,
                    max_new_tokens=req.max_new_tokens,
                    temperature=req.temperature,
                    top_k=req.top_k,
                    seed=req.seed,
                )
            for item in gen:
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
