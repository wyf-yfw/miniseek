#!/usr/bin/env python3
"""SFT（监督微调）：jsonl 行 {"conversations":[{"role","content"},...]}。

把多轮对话拼成 MiniMind 风格 chat 文本，只对 assistant 回复算 loss。

示例::

    python -m algo.train_sft --data data/SFT/sft_miniseek_v5_10g.jsonl \\
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
    truncate_keep_head_with_eos,
)
from algo import miniseek_m_train as mmt
from algo.model.miniseek_m import FlashConfig, step_expert_bias
from algo.model.tokenizer import detect_vocab_size, load_tokenizer
from algo.train_pretrain import (
    _emit_expert_load_line,
    _unwrap_model,
    evaluate_valid,
    reservoir_subset,
    resolve_step_budget,
    split_samples_by_ratio,
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

    def __init__(
        self,
        path: Optional[str] = None,
        tokenizer=None,
        max_seq_len: int = 512,
        max_samples: Optional[int] = None,
        samples: Optional[List[List[dict]]] = None,
        seed: int = 0,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        if samples is not None:
            self.samples, self.n_seen = reservoir_subset(samples, max_samples, seed)
            if not self.samples:
                raise RuntimeError("SFT 数据为空: (samples)")
            return
        if not path:
            raise ValueError("JsonlSFTDataset 需要 path 或 samples")

        def _convs():
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
                    yield conv

        self.samples, self.n_seen = reservoir_subset(_convs(), max_samples, seed)
        if not self.samples:
            raise RuntimeError(f"SFT 数据为空或格式不对: {path}")

    def __len__(self) -> int:
        return len(self.samples)

    def encode_conversation(self, conv: List[dict]) -> Tuple[List[int], List[int]]:
        """Encode one multi-turn sample to full ids/labels (keep-head + trailing eos)."""
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

        input_ids, labels = truncate_keep_head_with_eos(
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
    cfg: FlashConfig,
    global_step: int,
    vocab: int,
    loss: Optional[float],
) -> Dict[str, Any]:
    return {
        "model": mmt.state_for_ckpt(model),
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
    p.add_argument("--data", type=str, default="data/SFT/sft_miniseek_v5_10g.jsonl")
    p.add_argument("--tokenizer", type=str, default=None)
    p.add_argument("--out-dir", type=str, default="tasks/_default_sft")
    p.add_argument("--arch", choices=["miniseek-m", "m", "v41"], default="miniseek-m", help="MiniSeek-M architecture")
    p.add_argument(
        "--aux-coef",
        type=float,
        default=0.0001,
        help="MoE 序列级均衡损失权重（DeepSeek-M 为 0.0001）。专家 bias 另按负载更新，不靠这个系数",
    )
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-5, help="peak LR after warmup (SFT 通常低于预训练)")
    p.add_argument("--min-lr", type=float, default=None, help="cosine floor LR（默认 m=lr×0.1，其余 1e-5）")
    p.add_argument(
        "--candidate-pool",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="M：分层稀疏索引候选池（论文在后训练引入，SFT 默认开）",
    )
    p.add_argument(
        "--bounded-replay",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="M：模拟 Decoder SWA Bounded Replay（回答起点前 window 个位置的滑窗截断，SFT 默认开）",
    )
    p.add_argument(
        "--act-ckpt",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="M：按层 activation checkpointing（显存不够时开）",
    )
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="本轮步数上限（新开/续训相同）；续训时进度与 LR 从 0 起算，不把 ckpt 旧步数并进分母。不传则只按 epochs",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="先从全量合格样本里按 --seed 均匀抽取这么多条，再按 epoch 打乱。不传则用全量",
    )
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
        "--valid-every",
        type=int,
        default=0,
        help="每隔多少 step 跑一次 valid；0=关闭",
    )
    p.add_argument(
        "--valid-ratio",
        type=float,
        default=0.0,
        help="从训练集按比例随机抽 valid（0=关闭；不读独立 valid 文件，不改写源 jsonl）",
    )
    p.add_argument(
        "--valid-seed",
        type=int,
        default=42,
        help="valid-ratio 划分随机种子（可复现）",
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
        default=None,
        help="CUDA 上 torch.compile；默认 m 关（MoE 分发与稀疏索引会 graph-break），其余开",
    )
    p.add_argument(
        "--fused-adamw",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="CUDA 上默认 AdamW(fused=True)；--no-fused-adamw 关闭",
    )
    p.add_argument(
        "--flash-attn",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="默认开 FlashAttention：优先 flash_attn，否则 torch SDPA flash；"
        "--no-flash-attn 关闭",
    )
    p.add_argument(
        "--batch-mode",
        choices=["random", "bucket", "pack"],
        default="pack",
        help="组批方式：pack=拼接样本（默认，少 pad）；"
        "bucket=长度分桶；random=随机 pad。pack 不做跨样本 attention mask",
    )
    args = p.parse_args()
    if args.arch in ("m", "v41"):
        args.arch = "miniseek-m"
    is_mm = True
    if args.compile is None:
        args.compile = False
    if args.min_lr is None:
        args.min_lr = args.lr * 0.1

    thresholds = parse_loss_thresholds(args.save_loss_thresholds)
    hit_thresholds: Set[float] = set()
    streak: Dict[float, int] = {th: 0 for th in thresholds}

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[sft] device={device} weights=fp32 autocast={'bf16' if device.type == 'cuda' else 'off'}")
    print("[sft] flash_attn=n/a（MiniSeek-M 用按索引 gather 的稀疏注意力，与官方 sparse_attn 同语义）")

    tok = load_tokenizer(args.tokenizer, allow_fallback=True)
    vocab = detect_vocab_size(tok)
    pad_id = int(getattr(tok, "pad_token_id", 0) or 0)

    cfg, model = mmt.build_miniseek_m(tok, vocab, pad_id, args.max_seq_len, device)
    mmt.unwrap(model).act_ckpt = bool(args.act_ckpt)
    print(mmt.describe(cfg, model), flush=True)
    print(
        f"[sft] MiniSeek-M post-train: candidate_pool={bool(args.candidate_pool)} "
        f"bounded_replay={bool(args.bounded_replay)}",
        flush=True,
    )
    if args.compile:
        if device.type != "cuda":
            print("[sft] WARN: --compile ignored (需要 CUDA)")
        else:
            try:
                model = torch.compile(model)
                print("[sft] torch.compile enabled")
            except Exception as e:  # noqa: BLE001
                print(f"[sft] WARN: torch.compile 失败，继续未编译: {e}")
    has_moe = is_mm
    print(
        f"[sft] arch={args.arch} release={cfg.release_name} "
        f"aux_coef={args.aux_coef if has_moe else 'n/a'}"
    )
    expert_window_steps = 0
    if has_moe:
        _unwrap_model(model).begin_expert_usage()
    pack_kw = {"doc_align": mmt.doc_align(cfg), "with_doc_ids": True} if is_mm else {}

    if not Path(args.data).exists():
        raise SystemExit(f"[sft] 缺少数据文件: {args.data}")

    ds = JsonlSFTDataset(
        args.data,
        tok,
        max_seq_len=args.max_seq_len,
        max_samples=args.max_samples,
        seed=int(args.seed),
    )
    if args.max_samples is not None and ds.n_seen > len(ds):
        print(
            f"[sft] max-samples={len(ds)} / {ds.n_seen}"
            f"（先打乱再截断，seed={int(args.seed)}）",
            flush=True,
        )
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
                **pack_kw,
            )

        loader = None
        steps_per_epoch = PackedSFTBatchIter(
            ds.samples,
            ds.encode_conversation,
            max_seq_len=args.max_seq_len,
            batch_size=args.batch_size,
            pad_id=pad_id,
            seed=shuffle_seed,
            **pack_kw,
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
    valid_ratio = float(args.valid_ratio or 0.0)
    if int(args.valid_every) > 0 and valid_ratio > 0:
        train_samples, valid_samples = split_samples_by_ratio(
            ds.samples, valid_ratio, int(args.valid_seed)
        )
        ds.samples = train_samples
        if batch_mode == "pack":
            def packed_iter_factory(epoch: int):
                return PackedSFTBatchIter(
                    ds.samples,
                    ds.encode_conversation,
                    max_seq_len=args.max_seq_len,
                    batch_size=args.batch_size,
                    pad_id=pad_id,
                    seed=(shuffle_seed + epoch) % (2**31 - 1),
                    **pack_kw,
                )

            loader = None
            steps_per_epoch = PackedSFTBatchIter(
                ds.samples,
                ds.encode_conversation,
                max_seq_len=args.max_seq_len,
                batch_size=args.batch_size,
                pad_id=pad_id,
                seed=shuffle_seed,
                **pack_kw,
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

        vmax = int(args.valid_max_samples) if int(args.valid_max_samples) > 0 else None
        if vmax is not None:
            valid_samples = valid_samples[:vmax]
        valid_ds = JsonlSFTDataset(
            tokenizer=tok,
            max_seq_len=args.max_seq_len,
            samples=valid_samples,
        )
        valid_loader = DataLoader(
            valid_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=partial(collate_pad, pad_id=pad_id),
            num_workers=0,
        )
        print(
            f"[sft] valid-ratio={valid_ratio} seed={args.valid_seed} "
            f"train={len(ds)} valid={len(valid_ds)} every={args.valid_every} "
            f"max_samples={args.valid_max_samples} (源文件未改写)",
            flush=True,
        )

    plan_steps = steps_per_epoch * max(1, args.epochs)

    opt_kwargs = {"lr": args.lr, "weight_decay": 0.01}
    if is_mm:
        opt = mmt.build_optimizer(model, args.lr, fused=bool(args.fused_adamw) and device.type == "cuda")
        print("[sft] optimizer=Muon(矩阵)+AdamW(norm/标量)+Sinkhorn(embedding/head/Engram)", flush=True)
    elif bool(args.fused_adamw) and device.type == "cuda":
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

    global_step = 0  # this-run counter (LR + progress + stop)
    lifetime_base = 0  # ckpt 累计步；续训加载后写入 ckpt，不进 LR 分母
    best_loss = float("inf")

    if args.init_ckpt and args.resume:
        print("[sft] WARN: 同时给了 --init-ckpt 和 --resume，以 --resume 为准")

    if args.resume:
        resume_path = Path(args.resume)
        blob = torch.load(resume_path, map_location="cpu", weights_only=False)
        _unwrap_model(model).load_state_dict(mmt.clean_state_dict(blob["model"]))
        if "optimizer" in blob:
            try:
                opt.load_state_dict(blob["optimizer"])
            except Exception as e:  # noqa: BLE001
                print(f"[sft] WARN: optimizer 未恢复: {e}")
        lifetime_base = int(blob.get("step", 0))
        global_step = 0
        if blob.get("loss") is not None:
            best_loss = float(blob["loss"])
        print(
            f"[sft] resumed from {resume_path} lifetime_step={lifetime_base} "
            f"best_loss={best_loss}"
        )
    elif args.init_ckpt:
        init_path = Path(args.init_ckpt)
        blob = torch.load(init_path, map_location="cpu", weights_only=False)
        _unwrap_model(model).load_state_dict(mmt.clean_state_dict(blob["model"]), strict=True)
        print(f"[sft] init weights from {init_path} (pretrain step={blob.get('step', '?')})")

    total_steps, step_limit = resolve_step_budget(lifetime_base, args.max_steps, plan_steps)
    if lifetime_base > 0 and step_limit is not None:
        print(
            f"[sft] max-steps={int(args.max_steps)} 为本轮步数，"
            f"进度/LR 从 0→{step_limit}（ckpt lifetime 从 {lifetime_base} 起记）",
            flush=True,
        )

    print(
        f"[sft] batch_mode={batch_mode} samples={len(ds)} steps/epoch≈{steps_per_epoch} "
        f"start_step=0 total_steps={total_steps} warmup={args.warmup_steps} "
        f"lr={args.lr}→{args.min_lr} thresholds={thresholds} "
        f"stable_steps={args.stable_steps}"
        + (f" lifetime_base={lifetime_base}" if lifetime_base else ""),
        flush=True,
    )
    if total_steps < 1:
        print(
            f"[sft] WARN: total_steps={total_steps}，本次不会再有有效训练步",
            flush=True,
        )

    model.train()
    t0 = time.time()
    last_loss: Optional[float] = None

    def save_milestone(step: int, loss_val: float, threshold: float) -> None:
        tag = f"{threshold:g}"
        path = out_dir / f"loss-{tag}.pt"
        ckpt = build_ckpt(model, opt, cfg, lifetime_base + step, vocab, loss_val)
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
            ckpt = build_ckpt(model, opt, cfg, lifetime_base + step, vocab, loss_val)
            save_ckpt(out_dir / "best.pt", ckpt)
            print(f"[sft] best save loss={loss_val:.4f} @ step={step} -> {out_dir / 'best.pt'}", flush=True)

    def maybe_save_step(step: int, loss_val: float) -> None:
        if args.save_every <= 0 or step % args.save_every != 0:
            return
        life = lifetime_base + step
        ckpt = build_ckpt(model, opt, cfg, life, vocab, loss_val)
        step_path = out_dir / f"step-{life}.pt"
        save_ckpt(step_path, ckpt)
        prune_step_ckpts(out_dir, args.keep_last)
        print(f"[sft] step save @ step={step} loss={loss_val:.4f} -> {step_path}", flush=True)

    for epoch in range(args.epochs):
        if batch_mode == "pack":
            assert packed_iter_factory is not None
            epoch_iter = packed_iter_factory(epoch)
        else:
            epoch_iter = loader
        for batch in epoch_iter:
            if step_limit is not None and global_step >= step_limit:
                break
            input_ids, labels, doc_ids = mmt.unpack_batch(batch, device)

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
                if is_mm:
                    return mmt.forward_loss(
                        model,
                        input_ids,
                        labels,
                        doc_ids,
                        aux_coef=args.aux_coef,
                        use_candidate_pool=bool(args.candidate_pool),
                        bounded_replay=bool(args.bounded_replay),
                        window=cfg.window_size,
                    )
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
            if has_moe:
                step_expert_bias(model)
            global_step += 1
            last_loss = float(loss.item())
            if has_moe:
                expert_window_steps += 1

            if global_step % args.log_every == 0 or global_step == 1:
                if is_mm:
                    print(mmt.log_line(epoch, global_step, last_loss, ce_loss, aux_loss, cur_lr), flush=True)
                elif aux_loss is not None:
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
                if has_moe:
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

            if step_limit is not None and global_step >= step_limit:
                break
        if step_limit is not None and global_step >= step_limit:
            break

    if last_loss is None:
        last_loss = best_loss if best_loss < float("inf") else 0.0
    final_ckpt = build_ckpt(model, opt, cfg, lifetime_base + global_step, vocab, last_loss)
    save_ckpt(out_dir / "last.pt", final_ckpt)
    print(f"[sft] final save last.pt @ step={global_step} loss={last_loss:.4f}")

    elapsed = time.time() - t0
    print(
        f"[sft] done steps={global_step} "
        f"lifetime_step={lifetime_base + global_step} elapsed={elapsed:.1f}s "
        f"dir={out_dir} best_loss={best_loss if best_loss < float('inf') else 'n/a'} "
        f"milestones_hit={sorted(hit_thresholds, reverse=True)}"
    )


if __name__ == "__main__":
    main()
