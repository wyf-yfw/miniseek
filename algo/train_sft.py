#!/usr/bin/env python3
"""SFT（监督微调）：jsonl 行 {"conversations":[{"role","content"},...]}。

把多轮对话拼成 MiniMind 风格 chat 文本，只对 assistant 回复算 loss。

示例::

    python -m algo.train_sft --data data/sft_t2t_mini.jsonl \\
        --init-ckpt tasks/_default_pretrain/best.pt --out-dir tasks/_default_sft \\
        --max-steps 100
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from algo.model.config import ModelConfig
from algo.model.dense import DenseLM
from algo.model.tokenizer import detect_vocab_size, load_tokenizer


def format_chat(conversations: List[dict]) -> str:
    """拼成 <|im_start|>role\\ncontent<|im_end|>\\n ..."""
    parts: List[str] = []
    for turn in conversations:
        role = (turn.get("role") or "").strip().lower()
        if role not in {"user", "assistant", "system"}:
            continue
        content = turn.get("content")
        if content is None:
            content = ""
        content = str(content).strip()
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    return "\n".join(parts)


class JsonlSFTDataset(Dataset):
    """conversations → input_ids / labels；user/system 段 labels=-100。"""

    def __init__(self, path: str, tokenizer, max_seq_len: int, max_samples: Optional[int] = None):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples: List[List[dict]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                conv = obj.get("conversations") or obj.get("messages")
                if not isinstance(conv, list) or not conv:
                    continue
                # 至少要有一轮 assistant，否则没法算 SFT loss
                if not any((t.get("role") or "").lower() == "assistant" for t in conv):
                    continue
                self.samples.append(conv)
                if max_samples is not None and len(self.samples) >= max_samples:
                    break
        if not self.samples:
            raise RuntimeError(f"SFT 数据为空或格式不对: {path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        conv = self.samples[idx]
        bos = getattr(self.tokenizer, "bos_token", "<|im_start|>") or "<|im_start|>"
        eos = getattr(self.tokenizer, "eos_token", "<|im_end|>") or "<|im_end|>"

        input_ids: List[int] = []
        labels: List[int] = []

        for turn in conv:
            role = (turn.get("role") or "").strip().lower()
            if role not in {"user", "assistant", "system"}:
                continue
            content = str(turn.get("content") or "").strip()
            # 整段（含 special tokens）一起编码，再决定哪些进 loss
            chunk = f"{bos}{role}\n{content}{eos}\n"
            ids = self.tokenizer.encode(chunk, add_special_tokens=False)
            if not ids:
                continue
            input_ids.extend(ids)
            if role == "assistant":
                labels.extend(ids)
            else:
                labels.extend([-100] * len(ids))

        # 截断到 max_seq_len；保证至少有一对可训练 token
        input_ids = input_ids[: self.max_seq_len]
        labels = labels[: self.max_seq_len]
        if len(input_ids) < 2:
            eos_id = int(getattr(self.tokenizer, "eos_token_id", 2) or 2)
            input_ids = [eos_id, eos_id]
            labels = [-100, eos_id]

        # next-token：x = ids[:-1], y = labels[1:]
        x = torch.tensor(input_ids[:-1], dtype=torch.long)
        y = torch.tensor(labels[1:], dtype=torch.long)
        return x, y


def collate_pad(batch, pad_id: int = 0):
    inputs, labels = zip(*batch)
    max_len = max(x.size(0) for x in inputs)
    b = len(inputs)
    x = torch.full((b, max_len), pad_id, dtype=torch.long)
    y = torch.full((b, max_len), -100, dtype=torch.long)
    for i, (inp, lab) in enumerate(zip(inputs, labels)):
        n = inp.size(0)
        x[i, :n] = inp
        y[i, :n] = lab
    return x, y


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_loss_thresholds(s: str) -> List[float]:
    s = (s or "").strip()
    if not s:
        return []
    vals = [float(x.strip()) for x in s.split(",") if x.strip()]
    return sorted(set(vals), reverse=True)


def build_ckpt(
    model: torch.nn.Module,
    opt: torch.optim.Optimizer,
    cfg: ModelConfig,
    global_step: int,
    vocab: int,
    loss: Optional[float],
) -> Dict[str, Any]:
    cpu_state = {k: v.detach().float().cpu() for k, v in model.state_dict().items()}
    return {
        "model": cpu_state,
        "optimizer": {k: v.cpu() if torch.is_tensor(v) else v for k, v in opt.state_dict().items()},
        "config": cfg.to_dict(),
        "step": global_step,
        "vocab_size": vocab,
        "loss": loss,
        "stage": "sft",
    }


def save_ckpt(path: Path, ckpt: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(ckpt, tmp)
    tmp.replace(path)


def prune_step_ckpts(out_dir: Path, keep_last: int) -> None:
    if keep_last <= 0:
        return
    files = sorted(out_dir.glob("step-*.pt"), key=lambda p: p.stat().st_mtime)
    for old in files[:-keep_last]:
        try:
            old.unlink()
        except OSError:
            pass


def lr_at_step(
    step: int,
    *,
    base_lr: float,
    min_lr: float,
    warmup_steps: int,
    total_steps: int,
) -> float:
    if total_steps < 1:
        total_steps = 1
    if warmup_steps > 0 and step <= warmup_steps:
        return base_lr * float(step) / float(max(1, warmup_steps))
    if step >= total_steps:
        return min_lr
    decay_steps = max(1, total_steps - warmup_steps)
    progress = float(step - warmup_steps) / float(decay_steps)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


def set_lr(opt: torch.optim.Optimizer, lr: float) -> None:
    for group in opt.param_groups:
        group["lr"] = lr


def main() -> None:
    p = argparse.ArgumentParser(description="miniseek SFT")
    p.add_argument("--data", type=str, default="data/sft_t2t_mini.jsonl")
    p.add_argument("--tokenizer", type=str, default=None)
    p.add_argument("--out-dir", type=str, default="tasks/_default_sft")
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-5, help="peak LR after warmup (SFT 通常低于预训练)")
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=0)
    p.add_argument("--keep-last", type=int, default=3)
    p.add_argument("--save-loss-thresholds", type=str, default="", help="可选 loss-*.pt 里程碑；默认空=只 best.pt+last.pt")
    p.add_argument("--stable-steps", type=int, default=200)
    p.add_argument("--save-best", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--init-ckpt",
        type=str,
        default=None,
        help="从预训练 base 权重初始化（只加载 model，不加载 optimizer）",
    )
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        help="从 SFT checkpoint 断点续训（含 optimizer）",
    )
    args = p.parse_args()

    thresholds = parse_loss_thresholds(args.save_loss_thresholds)
    hit_thresholds: Set[float] = set()
    streak: Dict[float, int] = {th: 0 for th in thresholds}

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[sft] device={device} weights=fp32 autocast={'bf16' if device.type == 'cuda' else 'off'}")

    tok = load_tokenizer(args.tokenizer, allow_fallback=True)
    vocab = detect_vocab_size(tok)
    pad_id = int(getattr(tok, "pad_token_id", 0) or 0)

    cfg = ModelConfig(max_seq_len=args.max_seq_len).with_vocab_size(vocab)
    model = DenseLM(cfg).to(device)

    if not Path(args.data).exists():
        raise SystemExit(f"[sft] 缺少数据文件: {args.data}")

    ds = JsonlSFTDataset(args.data, tok, max_seq_len=args.max_seq_len, max_samples=args.max_samples)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_pad(b, pad_id=pad_id),
        num_workers=0,
    )

    steps_per_epoch = max(1, math.ceil(len(ds) / max(1, args.batch_size)))
    if args.max_steps is not None:
        total_steps = args.max_steps
    else:
        total_steps = steps_per_epoch * max(1, args.epochs)
    print(
        f"[sft] samples={len(ds)} steps/epoch≈{steps_per_epoch} "
        f"total_steps={total_steps} warmup={args.warmup_steps} "
        f"lr={args.lr}→{args.min_lr} thresholds={thresholds} "
        f"stable_steps={args.stable_steps}"
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for th in thresholds:
        tag = f"{th:g}"
        if (out_dir / f"loss-{tag}.pt").exists():
            hit_thresholds.add(th)
            streak[th] = 0
    if hit_thresholds:
        print(f"[sft] existing milestones: {sorted(hit_thresholds, reverse=True)}")

    global_step = 0
    best_loss = float("inf")

    if args.init_ckpt and args.resume:
        print("[sft] WARN: 同时给了 --init-ckpt 和 --resume，以 --resume 为准")

    if args.resume:
        resume_path = Path(args.resume)
        blob = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(blob["model"])
        if "optimizer" in blob:
            try:
                opt.load_state_dict(blob["optimizer"])
            except Exception as e:  # noqa: BLE001
                print(f"[sft] WARN: optimizer 未恢复: {e}")
        global_step = int(blob.get("step", 0))
        if blob.get("loss") is not None:
            best_loss = float(blob["loss"])
        print(f"[sft] resumed from {resume_path} step={global_step} best_loss={best_loss}")
    elif args.init_ckpt:
        init_path = Path(args.init_ckpt)
        blob = torch.load(init_path, map_location="cpu", weights_only=False)
        model.load_state_dict(blob["model"], strict=True)
        print(f"[sft] init weights from {init_path} (pretrain step={blob.get('step', '?')})")

    model.train()
    t0 = time.time()
    last_loss: Optional[float] = None

    def save_milestone(step: int, loss_val: float, threshold: float) -> None:
        tag = f"{threshold:g}"
        path = out_dir / f"loss-{tag}.pt"
        ckpt = build_ckpt(model, opt, cfg, step, vocab, loss_val)
        save_ckpt(path, ckpt)
        print(
            f"[sft] milestone save loss<={threshold:g} stable={args.stable_steps} "
            f"@ step={step} loss={loss_val:.4f} -> {path}",
            flush=True,
        )

    def maybe_save_best(step: int, loss_val: float) -> None:
        nonlocal best_loss
        if not args.save_best:
            return
        if loss_val < best_loss:
            best_loss = loss_val
            ckpt = build_ckpt(model, opt, cfg, step, vocab, loss_val)
            save_ckpt(out_dir / "best.pt", ckpt)
            print(f"[sft] best save loss={loss_val:.4f} @ step={step} -> {out_dir / 'best.pt'}", flush=True)

    def maybe_save_step(step: int, loss_val: float) -> None:
        if args.save_every <= 0 or step % args.save_every != 0:
            return
        ckpt = build_ckpt(model, opt, cfg, step, vocab, loss_val)
        step_path = out_dir / f"step-{step}.pt"
        save_ckpt(step_path, ckpt)
        prune_step_ckpts(out_dir, args.keep_last)
        print(f"[sft] step save @ step={step} loss={loss_val:.4f} -> {step_path}", flush=True)

    for epoch in range(args.epochs):
        for input_ids, labels in loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            next_step = global_step + 1
            cur_lr = lr_at_step(
                next_step,
                base_lr=args.lr,
                min_lr=args.min_lr,
                warmup_steps=args.warmup_steps,
                total_steps=total_steps,
            )
            set_lr(opt, cur_lr)

            opt.zero_grad(set_to_none=True)
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    _, loss = model(input_ids, labels=labels)
            else:
                _, loss = model(input_ids, labels=labels)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            global_step += 1
            last_loss = float(loss.item())

            if global_step % args.log_every == 0 or global_step == 1:
                print(
                    f"epoch={epoch} step={global_step} loss={last_loss:.4f} "
                    f"ppl={math.exp(min(last_loss, 20)):.2f} lr={cur_lr:.2e}",
                    flush=True,
                )

            for th in thresholds:
                if th in hit_thresholds:
                    continue
                if last_loss <= th:
                    streak[th] = streak.get(th, 0) + 1
                    if streak[th] >= max(1, args.stable_steps):
                        hit_thresholds.add(th)
                        streak[th] = 0
                        save_milestone(global_step, last_loss, th)
                else:
                    if streak.get(th, 0) > 0:
                        streak[th] = 0

            maybe_save_best(global_step, last_loss)
            maybe_save_step(global_step, last_loss)

            if args.max_steps is not None and global_step >= args.max_steps:
                break
        if args.max_steps is not None and global_step >= args.max_steps:
            break

    if last_loss is None:
        last_loss = best_loss if best_loss < float("inf") else 0.0
    final_ckpt = build_ckpt(model, opt, cfg, global_step, vocab, last_loss)
    save_ckpt(out_dir / "last.pt", final_ckpt)
    print(f"[sft] final save last.pt @ step={global_step} loss={last_loss:.4f}")

    elapsed = time.time() - t0
    print(
        f"[sft] done steps={global_step} elapsed={elapsed:.1f}s "
        f"dir={out_dir} best_loss={best_loss if best_loss < float('inf') else 'n/a'} "
        f"milestones_hit={sorted(hit_thresholds, reverse=True)}"
    )


if __name__ == "__main__":
    main()
