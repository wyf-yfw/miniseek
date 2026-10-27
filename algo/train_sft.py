#!/usr/bin/env python3
"""SFT（监督微调）：jsonl 行 {"conversations":[{"role","content"},...]}。

把多轮对话拼成 MiniMind 风格 chat 文本，只对 assistant 回复算 loss。

示例::

    python -m algo.train_sft --data data/sft_miniseek_v1.jsonl \\
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
from functools import partial
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from algo.batching import (
    LengthBucketBatchSampler,
    PackedSFTBatchIter,
    truncate_keep_tail_with_eos,
)
from algo.model import build_model
from algo.model.config import ModelConfig
from algo.model.tokenizer import detect_vocab_size, load_tokenizer
from algo.train_pretrain import (
    JsonlTextDataset,
    _emit_expert_load_line,
    _unwrap_model,
    collate_pad as pretrain_collate_pad,
    evaluate_valid,
)


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

    def encode_conversation(self, conv: List[dict]) -> Tuple[List[int], List[int]]:
        """Encode one multi-turn sample to full ids/labels (keep-tail + trailing eos)."""
        bos = getattr(self.tokenizer, "bos_token", "<|im_start|>") or "<|im_start|>"
        eos = getattr(self.tokenizer, "eos_token", "<|im_end|>") or "<|im_end|>"
        eos_id = int(getattr(self.tokenizer, "eos_token_id", 2) or 2)

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

        input_ids, labels = truncate_keep_tail_with_eos(
            input_ids, self.max_seq_len, eos_id, labels=labels
        )
        assert labels is not None
        if len(input_ids) < 2:
            input_ids = [eos_id, eos_id]
            labels = [-100, eos_id]
        return input_ids, labels

    def approx_length(self, idx: int) -> int:
        """Char-length proxy for bucket sampling (same idea as pretrain)."""
        text = format_chat(self.samples[idx])
        return min(self.max_seq_len, max(2, len(text)))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        input_ids, labels = self.encode_conversation(self.samples[idx])
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
    p.add_argument("--data", type=str, default="data/sft_miniseek_v1.jsonl")
    p.add_argument("--tokenizer", type=str, default=None)
    p.add_argument("--out-dir", type=str, default="tasks/_default_sft")
    p.add_argument("--arch", choices=["dense", "moe", "tiny"], default="dense", help="model architecture")
    p.add_argument("--aux-coef", type=float, default=0.01, help="MoE load-balance aux weight")
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
    p.add_argument(
        "--valid",
        type=str,
        default="",
        help="验证集 jsonl（每行 text 字段）；空=不跑 valid",
    )
    p.add_argument(
        "--valid-every",
        type=int,
        default=0,
        help="每隔多少 step 跑一次 valid；0=关闭",
    )
    p.add_argument(
        "--valid-max-samples",
        type=int,
        default=512,
        help="valid 最多用多少条样本",
    )
    p.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="CUDA 上默认 torch.compile；--no-compile 关闭",
    )
    p.add_argument(
        "--fused-adamw",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="CUDA 上默认 AdamW(fused=True)；--no-fused-adamw 关闭",
    )
    p.add_argument(
        "--batch-mode",
        choices=["random", "bucket", "pack"],
        default="pack",
        help="组批方式：pack=拼接样本（默认，少 pad）；"
        "bucket=长度分桶；random=随机 pad。pack 不做跨样本 attention mask",
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

    if args.arch == "tiny":
        cfg = ModelConfig.tiny()
        cfg = ModelConfig.from_dict({**cfg.to_dict(), "max_seq_len": args.max_seq_len})
    else:
        cfg = ModelConfig(arch=args.arch, max_seq_len=args.max_seq_len)
    cfg = cfg.with_vocab_size(vocab)
    model = build_model(cfg).to(device)
    if args.compile:
        if device.type != "cuda":
            print("[sft] WARN: --compile ignored (需要 CUDA)")
        else:
            try:
                model = torch.compile(model)
                print("[sft] torch.compile enabled")
            except Exception as e:  # noqa: BLE001
                print(f"[sft] WARN: torch.compile 失败，继续未编译: {e}")
    print(
        f"[sft] arch={args.arch} release={cfg.release_name} "
        f"aux_coef={args.aux_coef if args.arch == 'moe' else 'n/a'}"
    )
    expert_window_steps = 0
    if args.arch == "moe":
        _unwrap_model(model).begin_expert_usage()

    if not Path(args.data).exists():
        raise SystemExit(f"[sft] 缺少数据文件: {args.data}")

    ds = JsonlSFTDataset(args.data, tok, max_seq_len=args.max_seq_len, max_samples=args.max_samples)
    batch_mode = str(args.batch_mode).lower()
    shuffle_seed = int(args.seed) % (2**31 - 1)
    packed_iter_factory = None
    if batch_mode == "pack":
        def packed_iter_factory(epoch: int):
            return PackedSFTBatchIter(
                ds.samples,
                ds.encode_conversation,
                max_seq_len=args.max_seq_len,
                batch_size=args.batch_size,
                pad_id=pad_id,
                seed=(shuffle_seed + epoch) % (2**31 - 1),
            )

        loader = None
        steps_per_epoch = PackedSFTBatchIter(
            ds.samples,
            ds.encode_conversation,
            max_seq_len=args.max_seq_len,
            batch_size=args.batch_size,
            pad_id=pad_id,
            seed=shuffle_seed,
        ).estimate_len()
    elif batch_mode == "bucket":
        lengths = [ds.approx_length(i) for i in range(len(ds))]
        batch_sampler = LengthBucketBatchSampler(
            lengths, args.batch_size, seed=shuffle_seed, drop_last=False
        )
        loader = DataLoader(
            ds,
            batch_sampler=batch_sampler,
            collate_fn=partial(collate_pad, pad_id=pad_id),
            num_workers=0,
        )
        steps_per_epoch = len(batch_sampler)
    else:
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=partial(collate_pad, pad_id=pad_id),
            num_workers=0,
        )
        steps_per_epoch = max(1, math.ceil(len(ds) / max(1, args.batch_size)))

    valid_loader = None
    valid_path = str(args.valid or "").strip()
    if int(args.valid_every) > 0 and valid_path:
        if Path(valid_path).exists():
            valid_ds = JsonlTextDataset(
                valid_path,
                tok,
                max_seq_len=args.max_seq_len,
                max_samples=int(args.valid_max_samples) if int(args.valid_max_samples) > 0 else None,
            )
            valid_loader = DataLoader(
                valid_ds,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=partial(pretrain_collate_pad, pad_id=pad_id),
                num_workers=0,
            )
            print(
                f"[sft] valid={valid_path} every={args.valid_every} "
                f"samples={len(valid_ds)} max_samples={args.valid_max_samples}",
                flush=True,
            )
        else:
            print(f"[sft] WARN: valid 文件不存在，跳过: {valid_path}", flush=True)

    if args.max_steps is not None:
        total_steps = args.max_steps
    else:
        total_steps = steps_per_epoch * max(1, args.epochs)
    print(
        f"[sft] batch_mode={batch_mode} samples={len(ds)} steps/epoch≈{steps_per_epoch} "
        f"total_steps={total_steps} warmup={args.warmup_steps} "
        f"lr={args.lr}→{args.min_lr} thresholds={thresholds} "
        f"stable_steps={args.stable_steps}",
        flush=True,
    )

    opt_kwargs = {"lr": args.lr, "weight_decay": 0.01}
    if bool(args.fused_adamw) and device.type == "cuda":
        try:
            opt = torch.optim.AdamW(model.parameters(), fused=True, **opt_kwargs)
            print("[sft] AdamW fused=True", flush=True)
        except (TypeError, RuntimeError) as e:
            opt = torch.optim.AdamW(model.parameters(), **opt_kwargs)
            print(f"[sft] WARN: fused AdamW 不可用，回退: {e}", flush=True)
    else:
        opt = torch.optim.AdamW(model.parameters(), **opt_kwargs)
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
        if batch_mode == "pack":
            assert packed_iter_factory is not None
            epoch_iter = packed_iter_factory(epoch)
        else:
            epoch_iter = loader
        for input_ids, labels in epoch_iter:
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

            def _step_forward():
                out = model(input_ids, labels=labels)
                if isinstance(out, tuple) and len(out) == 3:
                    _logits, ce, aux = out
                    total = ce + args.aux_coef * aux
                    return total, ce, aux
                _logits, ce = out
                return ce, ce, None

            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss, ce_loss, aux_loss = _step_forward()
            else:
                loss, ce_loss, aux_loss = _step_forward()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            global_step += 1
            last_loss = float(loss.item())
            if args.arch == "moe":
                expert_window_steps += 1

            if global_step % args.log_every == 0 or global_step == 1:
                if aux_loss is not None:
                    print(
                        f"epoch={epoch} step={global_step} loss={last_loss:.4f} "
                        f"ce={float(ce_loss.item()):.4f} aux={float(aux_loss.item()):.4f} "
                        f"ppl={math.exp(min(float(ce_loss.item()), 20)):.2f} lr={cur_lr:.2e}",
                        flush=True,
                    )
                else:
                    print(
                        f"epoch={epoch} step={global_step} loss={last_loss:.4f} "
                        f"ppl={math.exp(min(last_loss, 20)):.2f} lr={cur_lr:.2e}",
                        flush=True,
                    )
                if args.arch == "moe":
                    _emit_expert_load_line(
                        model=model,
                        step=global_step,
                        window_steps=expert_window_steps,
                        cfg=cfg,
                    )
                    expert_window_steps = 0

            if (
                valid_loader is not None
                and int(args.valid_every) > 0
                and global_step % int(args.valid_every) == 0
            ):
                stats = evaluate_valid(model, valid_loader, device)
                print(
                    f"valid step={global_step} loss={stats['loss']:.4f} "
                    f"ppl={stats['ppl']:.2f} tokens={int(stats['tokens'])}",
                    flush=True,
                )
                model.train()

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
