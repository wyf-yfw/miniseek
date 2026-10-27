#!/usr/bin/env python3
"""Stage-0 预训练：jsonl 行 {"text": "..."} 的 next-token 预测.

示例::

    python -m algo.train_pretrain --data data/pretrain_mix_5g.jsonl --max-steps 2
    python -m algo.train_pretrain --data data/pretrain_mix_5g.jsonl --epochs 1 --batch-size 32
    # 默认只写 best.pt + last.pt；可选 --save-loss-thresholds / --save-every（占盘，慎用）
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# 标准输出重定向到文件时，默认是块缓冲，Web 控制台会一次性看到一大段。
# 改成按行刷出，配合 python -u，日志能跟着训练步实时出现。
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

import argparse
import hashlib
import json
import math
import random
import time
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import torch
from torch.utils.data import DataLoader, Dataset

from algo.model import build_model
from algo.model.config import ModelConfig
from algo.model.tokenizer import detect_vocab_size, load_tokenizer


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """torch.compile 后取底层模块。"""
    return getattr(model, "_orig_mod", model)


def _emit_expert_load_line(
    *,
    model: torch.nn.Module,
    step: int,
    window_steps: int,
    cfg: ModelConfig,
) -> None:
    """结束本 log 窗口的 usage 记录，打印 expert_load=JSON，再开启下一窗口。"""
    raw = _unwrap_model(model)
    if not hasattr(raw, "end_expert_usage") or not hasattr(raw, "begin_expert_usage"):
        return
    counts_t = raw.end_expert_usage()
    counts = [int(x) for x in counts_t.tolist()]
    total = int(sum(counts))
    fractions = [(c / total if total > 0 else 0.0) for c in counts]
    n_routed = int(getattr(cfg, "n_routed_experts", len(counts)))
    if fractions:
        max_expert = int(max(range(len(fractions)), key=lambda i: fractions[i]))
        min_expert = int(min(range(len(fractions)), key=lambda i: fractions[i]))
        max_frac = float(fractions[max_expert])
        min_frac = float(fractions[min_expert])
    else:
        max_expert = min_expert = -1
        max_frac = min_frac = 0.0
    payload = {
        "step": int(step),
        "window_steps": int(window_steps),
        "n_routed": n_routed,
        "top_k": int(getattr(cfg, "num_experts_per_tok", 2)),
        "n_layers": int(getattr(cfg, "n_layers", 0)),
        "n_shared": int(getattr(cfg, "n_shared_experts", 1)),
        "counts": counts,
        "fractions": fractions,
        "total_assignments": total,
        "max_frac": max_frac,
        "min_frac": min_frac,
        "max_expert": max_expert,
        "min_expert": min_expert,
    }
    print("expert_load=" + json.dumps(payload, ensure_ascii=False, separators=(",", ":")), flush=True)
    raw.begin_expert_usage()



class JsonlTextDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_seq_len: int, max_samples: Optional[int] = None):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples: List[str] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = obj.get("text") or obj.get("content") or ""
                if text:
                    self.samples.append(text)
                if max_samples is not None and len(self.samples) >= max_samples:
                    break
        if not self.samples:
            raise RuntimeError(f"数据为空: {path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        text = self.samples[idx]
        ids = self.tokenizer.encode(text, add_special_tokens=True)
        eos_id = int(getattr(self.tokenizer, "eos_token_id", 2) or 2)
        # EOS comes from the dataset text (trailing <|im_end|>); do not append here.
        # Truncation must keep a trailing EOS when the sample ends with one.
        if len(ids) > self.max_seq_len:
            if ids and ids[-1] == eos_id:
                ids = ids[: self.max_seq_len - 1] + [eos_id]
            else:
                ids = ids[: self.max_seq_len]
        if len(ids) < 2:
            # Degenerate sample (empty / too short after encode): keep a valid pair.
            ids = [eos_id, eos_id]
        input_ids = torch.tensor(ids[:-1], dtype=torch.long)
        labels = torch.tensor(ids[1:], dtype=torch.long)
        return input_ids, labels


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


def run_salt(args: argparse.Namespace) -> int:
    """每次任务一个不同的打乱种子。

    同一个 out_dir + 同样参数可复现；换任务、换数据、换 resume 起点必然是不同排列。
    这样续训看到的是新的一遍随机数据，而不是把开头那段原样重放一遍。
    """
    key = f"{args.out_dir}|{args.data}|{args.seed}|{args.resume}"
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) % (2**31 - 1)


def parse_loss_thresholds(s: str) -> List[float]:
    """Parse comma-separated descending loss milestones, e.g. '20,15,12,10'."""
    s = (s or "").strip()
    if not s:
        return []
    vals = [float(x.strip()) for x in s.split(",") if x.strip()]
    # keep descending unique order for clarity
    vals = sorted(set(vals), reverse=True)
    return vals


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
    """Linear warmup then cosine decay to min_lr."""
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
    p = argparse.ArgumentParser(description="miniseek Stage-0 pretrain")
    p.add_argument("--data", type=str, default="data/pretrain_mix_5g.jsonl")
    p.add_argument("--tokenizer", type=str, default=None)
    p.add_argument("--out-dir", type=str, default="tasks/_default_pretrain")
    p.add_argument("--arch", choices=["dense", "moe", "tiny"], default="dense", help="model architecture")
    p.add_argument("--aux-coef", type=float, default=0.01, help="MoE load-balance aux weight")
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-5, help="peak LR after warmup")
    p.add_argument("--min-lr", type=float, default=1e-5, help="cosine floor LR")
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=None, help="smoke: --max-steps 2")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="每 N 步额外写 step-N.pt；默认 0=关闭（默认只 best.pt + last.pt）",
    )
    p.add_argument(
        "--keep-last",
        type=int,
        default=3,
        help="若开了 --save-every，最多保留最近 K 个 step-*.pt",
    )
    p.add_argument(
        "--save-loss-thresholds",
        type=str,
        default="",
        help="可选：loss 连续低于各阈值满 --stable-steps 后各存一次 loss-*.pt（逗号分隔降序）；"
        "默认空=不写里程碑，只保留 best.pt + last.pt（省磁盘）",
    )
    p.add_argument(
        "--stable-steps",
        type=int,
        default=50,
        help="里程碑/best：连续多少步满足条件才存盘；中途回升则计数清零",
    )
    p.add_argument(
        "--save-best",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="loss 连续低于已保存 best 满 --stable-steps 后才更新 best.pt（默认开）",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader worker 进程数；0=主进程读数据",
    )
    p.add_argument(
        "--compile",
        action="store_true",
        help="对模型 torch.compile（仅 CUDA；失败则告警并继续）",
    )
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        help="从 checkpoint 继续（含 optimizer）",
    )
    args = p.parse_args()

    thresholds = parse_loss_thresholds(args.save_loss_thresholds)
    hit_thresholds: Set[float] = set()
    # consecutive steps with loss <= each pending threshold
    streak: Dict[float, int] = {th: 0 for th in thresholds}

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 权重保持 fp32；CUDA 上用 autocast，避免整模 .to(bfloat16) 弄坏 RoPE complex buffer
    print(f"[train] device={device} weights=fp32 autocast={'bf16' if device.type == 'cuda' else 'off'}")

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
            print("[train] WARN: --compile ignored (需要 CUDA)")
        else:
            try:
                model = torch.compile(model)
                print("[train] torch.compile enabled")
            except Exception as e:  # noqa: BLE001
                print(f"[train] WARN: torch.compile 失败，继续未编译: {e}")
    print(
        f"[train] arch={args.arch} release={cfg.release_name} "
        f"aux_coef={args.aux_coef if args.arch == 'moe' else 'n/a'}"
    )
    expert_window_steps = 0
    if args.arch == "moe":
        _unwrap_model(model).begin_expert_usage()

    if not Path(args.data).exists():
        print(f"[train] WARN: 缺少 {args.data}，使用内置中文占位样本做 smoke")
        Path("data").mkdir(parents=True, exist_ok=True)
        smoke_path = Path("data/_smoke_train.jsonl")
        with open(smoke_path, "w", encoding="utf-8") as f:
            for t in [
                "深度学习是机器学习的一个分支。<|im_end|>",
                "大语言模型可以生成自然语言文本。<|im_end|>",
                "今天天气很好，适合出去散步。<|im_end|>",
                "北京是中国的首都。<|im_end|>",
            ]:
                f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
        args.data = str(smoke_path)
        args.max_samples = args.max_samples or 4
        args.max_steps = args.max_steps or 2

    ds = JsonlTextDataset(
        args.data, tok, max_seq_len=args.max_seq_len, max_samples=args.max_samples
    )
    # 打乱顺序按任务派生，避免每次续训都重放同一批开头样本
    salt = run_salt(args)
    shuffle_gen = torch.Generator()
    shuffle_gen.manual_seed((int(args.seed) + salt) % (2**31 - 1))
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        generator=shuffle_gen,
        collate_fn=partial(collate_pad, pad_id=pad_id),
        num_workers=max(0, int(args.num_workers)),
        pin_memory=(device.type == "cuda"),
    )

    steps_per_epoch = max(1, math.ceil(len(ds) / max(1, args.batch_size)))
    plan_steps = steps_per_epoch * max(1, args.epochs)
    total_steps = int(args.max_steps) if args.max_steps is not None else plan_steps

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # skip thresholds that already have a milestone file on disk
    for th in thresholds:
        tag = f"{th:g}"
        if (out_dir / f"loss-{tag}.pt").exists():
            hit_thresholds.add(th)
            streak[th] = 0
    if hit_thresholds:
        print(f"[train] existing milestones: {sorted(hit_thresholds, reverse=True)}")

    global_step = 0
    best_loss = float("inf")
    best_streak = 0
    pending_best = float("inf")
    if args.resume:
        resume_path = Path(args.resume)
        blob = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(blob["model"])
        if "optimizer" in blob:
            try:
                opt.load_state_dict(blob["optimizer"])
            except Exception as e:  # noqa: BLE001
                print(f"[train] WARN: optimizer 未恢复: {e}")
        global_step = int(blob.get("step", 0))
        if blob.get("loss") is not None:
            best_loss = float(blob["loss"])
        if args.max_steps is None:
            # 续训没给绝对步数时，"epochs" 表示再扫几遍数据：
            # 总预算 = 已跑 + 本次，LR 余弦才会覆盖整段而不是从头重算
            total_steps = global_step + plan_steps
        print(f"[train] resumed from {resume_path} step={global_step} best_loss={best_loss}")

    print(
        f"[train] samples={len(ds)} steps/epoch≈{steps_per_epoch} "
        f"start_step={global_step} total_steps={total_steps} "
        f"shuffle_salt={salt} warmup={args.warmup_steps} "
        f"lr={args.lr}→{args.min_lr} thresholds={thresholds} "
        f"stable_steps={args.stable_steps}"
    )
    if global_step >= total_steps:
        print(
            f"[train] WARN: 起点 step={global_step} 已不低于 total_steps={total_steps}，"
            f"本次不会再有有效训练步"
        )

    model.train()
    t0 = time.time()
    last_loss: Optional[float] = None

    def save_milestone(step: int, loss_val: float, threshold: float) -> None:
        # filename: loss-10.pt for 10.0, loss-9.5.pt for 9.5
        tag = f"{threshold:g}"
        path = out_dir / f"loss-{tag}.pt"
        ckpt = build_ckpt(model, opt, cfg, step, vocab, loss_val)
        save_ckpt(path, ckpt)
        print(
            f"[train] milestone save loss<={threshold:g} stable={args.stable_steps} "
            f"@ step={step} loss={loss_val:.4f} -> {path}",
            flush=True,
        )

    def maybe_save_best(step: int, loss_val: float) -> None:
        nonlocal best_loss, best_streak, pending_best
        if not args.save_best:
            return
        need = max(1, int(args.stable_steps))
        # 连续 stable_steps 步 loss 都 < 已记录 best_loss 才写盘；写后 best_loss=窗口内 min
        if loss_val < best_loss:
            best_streak += 1
            if loss_val < pending_best:
                pending_best = loss_val
            if best_streak >= need:
                best_loss = pending_best
                ckpt = build_ckpt(model, opt, cfg, step, vocab, pending_best)
                save_ckpt(out_dir / "best.pt", ckpt)
                print(
                    f"[train] best save loss={pending_best:.4f} stable={need} "
                    f"@ step={step} -> {out_dir / 'best.pt'}",
                    flush=True,
                )
                best_streak = 0
                pending_best = float("inf")
        else:
            if best_streak > 0:
                best_streak = 0
                pending_best = float("inf")

    def maybe_save_step(step: int, loss_val: float) -> None:
        if args.save_every <= 0 or step % args.save_every != 0:
            return
        ckpt = build_ckpt(model, opt, cfg, step, vocab, loss_val)
        step_path = out_dir / f"step-{step}.pt"
        save_ckpt(step_path, ckpt)
        prune_step_ckpts(out_dir, args.keep_last)
        print(f"[train] step save @ step={step} loss={loss_val:.4f} -> {step_path}", flush=True)

    for epoch in range(args.epochs):
        for input_ids, labels in loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            # schedule uses 1-indexed step after increment; set LR for upcoming step
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

            # loss milestones: need consecutive --stable-steps below each threshold
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
    # always write last.pt once at end
    final_ckpt = build_ckpt(model, opt, cfg, global_step, vocab, last_loss)
    save_ckpt(out_dir / "last.pt", final_ckpt)
    print(f"[train] final save last.pt @ step={global_step} loss={last_loss:.4f}")

    elapsed = time.time() - t0
    print(
        f"[train] done steps={global_step} elapsed={elapsed:.1f}s "
        f"dir={out_dir} best_loss={best_loss if best_loss < float('inf') else 'n/a'} "
        f"milestones_hit={sorted(hit_thresholds, reverse=True)}"
    )


if __name__ == "__main__":
    main()
