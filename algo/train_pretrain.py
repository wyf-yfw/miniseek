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
import collections
import hashlib
import json
import math
import random
import signal
import time
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from algo import v41_train as v41t
from algo.batching import LengthBucketBatchSampler, PackedTextBatchIter
from algo.model import build_model
from algo.model.v41 import step_expert_bias
from algo.model.attn_backend import configure_attn_backend
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
        "top_k": int(getattr(cfg, "num_experts_per_tok", None) or getattr(cfg, "n_activated_experts", 2)),
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



def reservoir_subset(items, max_samples: Optional[int], seed: int) -> tuple[list, int]:
    """从全量里均匀抽出 max_samples 条。

    分布上等于先打乱再截断。max_samples 为空，或不少于全量时，保持原顺序。
    流式蓄水池只留抽出的那一批，避免为了小子集把整份语料再拷一份。
    """
    if max_samples is None:
        kept = list(items)
        return kept, len(kept)
    k = int(max_samples)
    if k <= 0:
        n = sum(1 for _ in items)
        return [], n
    rng = random.Random(int(seed))
    buf: list = []
    n = 0
    for item in items:
        n += 1
        if len(buf) < k:
            buf.append(item)
        else:
            j = rng.randrange(n)
            if j < k:
                buf[j] = item
    if n > k:
        rng.shuffle(buf)
    return buf, n


class JsonlTextDataset(Dataset):
    def __init__(
        self,
        path: Optional[str] = None,
        tokenizer=None,
        max_seq_len: int = 512,
        max_samples: Optional[int] = None,
        samples: Optional[List[str]] = None,
        seed: int = 0,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        if samples is not None:
            self.samples, self.n_seen = reservoir_subset(samples, max_samples, seed)
        else:
            if not path:
                raise ValueError("JsonlTextDataset 需要 path 或 samples")

            def _texts():
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        obj = json.loads(line)
                        text = obj.get("text") or obj.get("content") or ""
                        if text:
                            yield text

            self.samples, self.n_seen = reservoir_subset(_texts(), max_samples, seed)
        if not self.samples:
            raise RuntimeError(f"数据为空: {path or '(samples)'}")

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


@torch.no_grad()
def evaluate_valid(model: torch.nn.Module, loader, device: torch.device) -> Dict[str, float]:
    """Mean token NLL / ppl on a validation loader (CE only; MoE aux ignored)."""
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for input_ids, labels in loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        out = model(input_ids, labels=labels)
        if isinstance(out, tuple) and len(out) == 3:
            _logits, ce, _aux = out
        else:
            _logits, ce = out
        n = int((labels != -100).sum().item())
        if n <= 0:
            continue
        total_loss += float(ce.item()) * n
        total_tokens += n
    if was_training:
        model.train()
    mean = total_loss / max(1, total_tokens)
    return {
        "loss": mean,
        "ppl": math.exp(min(mean, 20.0)),
        "tokens": total_tokens,
    }



def split_samples_by_ratio(
    samples: List[str],
    ratio: float,
    seed: int,
) -> tuple[List[str], List[str]]:
    """Deterministically split samples into (train, valid) by *ratio*.

    Does not modify *samples* in place beyond reading; returns two new lists that
    share the same string objects (no rewrite of the source jsonl on disk).
    Train ∩ valid is empty; |valid| ≈ round(n * ratio), at least 1 when n>0 and ratio>0.
    """
    n = len(samples)
    if n == 0:
        return [], []
    if ratio <= 0:
        return list(samples), []
    if ratio >= 1:
        raise ValueError(f"valid_ratio must be in (0, 1), got {ratio}")
    rng = random.Random(int(seed))
    order = list(range(n))
    rng.shuffle(order)
    n_valid = int(round(n * float(ratio)))
    n_valid = max(1, min(n - 1, n_valid)) if n >= 2 else n
    valid_idx = set(order[:n_valid])
    train_list = [samples[i] for i in range(n) if i not in valid_idx]
    valid_list = [samples[i] for i in order[:n_valid]]
    return train_list, valid_list


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


_STOP = {"flag": False, "signum": 0}


def handle_stop_signal(signum, _frame) -> None:
    """SIGTERM/SIGINT 不立即退出：打个标记，让训练循环跑完当前步、写完 last.pt 再走。"""
    _STOP["flag"] = True
    _STOP["signum"] = int(signum)
    print(f"[train] 收到信号 {signum}：当前步结束后保存 last.pt 再退出", flush=True)


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
    return {
        "model": v41t.state_for_ckpt(model),
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


def resolve_step_budget(
    start_step: int,
    max_steps: Optional[int],
    plan_steps: int,
) -> Tuple[int, Optional[int]]:
    """本轮 LR / 进度分母，以及本轮停止步（对照从 0 起的 run counter）。

    Max Steps 无论新开还是续训，都只表示**本轮**预算；checkpoint 里的
    历史步数绝不并进余弦分母。``start_step`` 仅为调用方兼容保留，不参与计算。
    返回 (schedule_total, step_limit)；step_limit 与本轮 step（从 0 起）比较。
    """
    _ = start_step  # lifetime must not affect schedule / progress denominator
    plan = max(1, int(plan_steps))
    if max_steps is not None and int(max_steps) > 0:
        n = int(max_steps)
        return n, n
    return plan, None


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
    p.add_argument("--data", type=str, default="data/PT/pretrain_mix_12g_clean.jsonl")
    p.add_argument("--tokenizer", type=str, default=None)
    p.add_argument("--out-dir", type=str, default="tasks/_default_pretrain")
    p.add_argument("--arch", choices=["dense", "tiny", "v41"], default="dense", help="model architecture")
    p.add_argument(
        "--aux-coef",
        type=float,
        default=0.0001,
        help="MoE 序列级均衡损失权重（V4.1 为 0.0001）。专家 bias 另按负载更新，不靠这个系数",
    )
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=None, help="peak LR after warmup（默认 v41=3e-4，其余 5e-5）")
    p.add_argument("--min-lr", type=float, default=None, help="cosine floor LR（默认 v41=lr×0.1，其余 1e-5）")
    p.add_argument(
        "--candidate-pool",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="v41：启用分层稀疏索引候选池（论文在后训练才开，预训练默认关）",
    )
    p.add_argument(
        "--act-ckpt",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="v41：按层 activation checkpointing（显存不够时开）",
    )
    p.add_argument("--warmup-steps", type=int, default=200)
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
        "--save-last-every",
        type=int,
        default=5000,
        help="每 N 步原子写一次 last.pt（0=关闭）。保证中途崩溃/停机能从近期状态续训",
    )
    p.add_argument(
        "--best-min-gap",
        type=int,
        default=300,
        help="两次写 best.pt 的最小步间隔，避免抖动导致反复写 1.7G 大文件",
    )
    p.add_argument(
        "--best-min-improve",
        type=float,
        default=0.005,
        help="新最优要比旧的至少好这个比例（0.005=0.5%%）才写 best.pt",
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
        action=argparse.BooleanOptionalAction,
        default=True,
        help="CUDA 上默认 torch.compile（MoE 可能 graph-break，bench 仍约 +4%%）；"
        "--no-compile 关闭",
    )
    p.add_argument(
        "--batch-mode",
        choices=["random", "bucket", "pack"],
        default="pack",
        help="组批方式：pack=文档拼接（默认，吞吐最高）；"
        "bucket=长度分桶；random=随机 pad。pack 不做跨文档 attention mask",
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
        "--resume",
        type=str,
        default=None,
        help="从 checkpoint 继续（含 optimizer）",
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
    args = p.parse_args()
    is_v41 = args.arch == "v41"
    if args.lr is None:
        args.lr = v41t.V41_DEFAULT_LR if is_v41 else 5e-5
    if args.min_lr is None:
        args.min_lr = args.lr * 0.1 if is_v41 else 1e-5

    thresholds = parse_loss_thresholds(args.save_loss_thresholds)
    hit_thresholds: Set[float] = set()
    # consecutive steps with loss <= each pending threshold
    streak: Dict[float, int] = {th: 0 for th in thresholds}

    set_seed(args.seed)
    for _sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(_sig, handle_stop_signal)
        except (ValueError, OSError):
            pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 权重保持 fp32；CUDA 上用 autocast，避免整模 .to(bfloat16) 弄坏 RoPE complex buffer
    print(f"[train] device={device} weights=fp32 autocast={'bf16' if device.type == 'cuda' else 'off'}")
    attn_name = configure_attn_backend(bool(args.flash_attn))
    if is_v41:
        print("[train] flash_attn=n/a（v41 用按索引 gather 的稀疏注意力，与官方 sparse_attn 同语义）")
    else:
        print(f"[train] flash_attn={'on' if args.flash_attn else 'off'} backend={attn_name}")

    tok = load_tokenizer(args.tokenizer, allow_fallback=True)
    vocab = detect_vocab_size(tok)
    pad_id = int(getattr(tok, "pad_token_id", 0) or 0)

    if is_v41:
        cfg, model = v41t.build_v41(tok, vocab, pad_id, args.max_seq_len, device)
        v41t.unwrap(model).act_ckpt = bool(args.act_ckpt)
        print(v41t.describe(cfg, model), flush=True)
    else:
        if args.arch == "tiny":
            cfg = ModelConfig.tiny()
            cfg = ModelConfig.from_dict({**cfg.to_dict(), "max_seq_len": args.max_seq_len})
        else:
            cfg = ModelConfig(arch=args.arch, max_seq_len=args.max_seq_len)
        cfg = ModelConfig.from_dict({**cfg.to_dict(), "use_flash_attn": bool(args.flash_attn)})
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
    has_moe = is_v41
    print(
        f"[train] arch={args.arch} release={cfg.release_name} "
        f"aux_coef={args.aux_coef if has_moe else 'n/a'}"
    )
    expert_window_steps = 0
    if has_moe:
        _unwrap_model(model).begin_expert_usage()
    # v41：样本级 attention mask 需要文档编号；文档对齐到压缩比，保证压缩组不跨文档
    pack_kw = {"doc_align": v41t.doc_align(cfg), "with_doc_ids": True} if is_v41 else {}

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
        args.data,
        tok,
        max_seq_len=args.max_seq_len,
        max_samples=args.max_samples,
        seed=int(args.seed),
    )
    if args.max_samples is not None and ds.n_seen > len(ds):
        print(
            f"[train] max-samples={len(ds)} / {ds.n_seen}"
            f"（先打乱再截断，seed={int(args.seed)}）",
            flush=True,
        )
    # 打乱顺序按任务派生，避免每次续训都重放同一批开头样本
    salt = run_salt(args)
    shuffle_seed = (int(args.seed) + salt) % (2**31 - 1)
    shuffle_gen = torch.Generator()
    shuffle_gen.manual_seed(shuffle_seed)
    batch_mode = str(args.batch_mode).lower()
    nw = max(0, int(args.num_workers))
    pin = device.type == "cuda"
    packed_iter_factory = None
    if batch_mode == "pack":
        # Online packer; DataLoader not used for train batches
        eos_id = int(getattr(tok, "eos_token_id", 2) or 2)

        def packed_iter_factory(epoch: int):
            return PackedTextBatchIter(
                ds.samples,
                tok,
                max_seq_len=args.max_seq_len,
                batch_size=args.batch_size,
                pad_id=pad_id,
                eos_id=eos_id,
                seed=(shuffle_seed + epoch) % (2**31 - 1),
                **pack_kw,
            )

        loader = None
        steps_per_epoch_est = PackedTextBatchIter(
            ds.samples,
            tok,
            max_seq_len=args.max_seq_len,
            batch_size=args.batch_size,
            pad_id=pad_id,
            eos_id=eos_id,
            seed=shuffle_seed,
            **pack_kw,
        ).estimate_len()
    elif batch_mode == "bucket":
        # Char length ≈ token length for this Chinese BPE; cheap proxy for bucketing
        lengths = [min(args.max_seq_len, max(2, len(t))) for t in ds.samples]
        batch_sampler = LengthBucketBatchSampler(
            lengths, args.batch_size, seed=shuffle_seed, drop_last=False
        )
        loader = DataLoader(
            ds,
            batch_sampler=batch_sampler,
            collate_fn=partial(collate_pad, pad_id=pad_id),
            num_workers=nw,
            pin_memory=pin,
        )
        steps_per_epoch_est = len(batch_sampler)
    else:
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=True,
            generator=shuffle_gen,
            collate_fn=partial(collate_pad, pad_id=pad_id),
            num_workers=nw,
            pin_memory=pin,
        )
        steps_per_epoch_est = max(1, math.ceil(len(ds) / max(1, args.batch_size)))
    print(
        f"[train] batch_mode={batch_mode} compile={bool(args.compile)} "
        f"fused_adamw={bool(args.fused_adamw)}",
        flush=True,
    )

    valid_loader = None
    valid_ratio = float(args.valid_ratio or 0.0)
    if int(args.valid_every) > 0 and valid_ratio > 0:
        train_samples, valid_samples = split_samples_by_ratio(
            ds.samples, valid_ratio, int(args.valid_seed)
        )
        ds.samples = train_samples
        # pack / bucket / random already closed over ds.samples or lengths — rebuild
        # estimates below use updated ds; recreate pack factory closures.
        if batch_mode == "pack":
            eos_id = int(getattr(tok, "eos_token_id", 2) or 2)

            def packed_iter_factory(epoch: int):
                return PackedTextBatchIter(
                    ds.samples,
                    tok,
                    max_seq_len=args.max_seq_len,
                    batch_size=args.batch_size,
                    pad_id=pad_id,
                    eos_id=eos_id,
                    seed=(shuffle_seed + epoch) % (2**31 - 1),
                    **pack_kw,
                )

            steps_per_epoch_est = PackedTextBatchIter(
                ds.samples,
                tok,
                max_seq_len=args.max_seq_len,
                batch_size=args.batch_size,
                pad_id=pad_id,
                eos_id=eos_id,
                seed=shuffle_seed,
                **pack_kw,
            ).estimate_len()
        elif batch_mode == "bucket":
            lengths = [min(args.max_seq_len, max(2, len(t))) for t in ds.samples]
            batch_sampler = LengthBucketBatchSampler(
                lengths, args.batch_size, seed=shuffle_seed, drop_last=False
            )
            loader = DataLoader(
                ds,
                batch_sampler=batch_sampler,
                collate_fn=partial(collate_pad, pad_id=pad_id),
                num_workers=nw,
                pin_memory=pin,
            )
            steps_per_epoch_est = len(batch_sampler)
        else:
            loader = DataLoader(
                ds,
                batch_size=args.batch_size,
                shuffle=True,
                generator=shuffle_gen,
                collate_fn=partial(collate_pad, pad_id=pad_id),
                num_workers=nw,
                pin_memory=pin,
            )
            steps_per_epoch_est = max(1, math.ceil(len(ds) / max(1, args.batch_size)))

        vmax = int(args.valid_max_samples) if int(args.valid_max_samples) > 0 else None
        if vmax is not None:
            valid_samples = valid_samples[:vmax]
        valid_ds = JsonlTextDataset(
            samples=valid_samples,
            tokenizer=tok,
            max_seq_len=args.max_seq_len,
        )
        valid_loader = DataLoader(
            valid_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=partial(collate_pad, pad_id=pad_id),
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )
        print(
            f"[train] valid-ratio={valid_ratio} seed={args.valid_seed} "
            f"train={len(ds)} valid={len(valid_ds)} every={args.valid_every} "
            f"max_samples={args.valid_max_samples} (源文件未改写)",
            flush=True,
        )

    steps_per_epoch = max(1, int(steps_per_epoch_est))
    plan_steps = steps_per_epoch * max(1, args.epochs)

    opt_kwargs = {"lr": args.lr, "weight_decay": 0.01}
    if is_v41:
        opt = v41t.build_optimizer(model, args.lr, fused=bool(args.fused_adamw) and device.type == "cuda")
        print("[train] optimizer=Muon(矩阵)+AdamW(norm/标量)+Sinkhorn(embedding/head/Engram)", flush=True)
    elif bool(args.fused_adamw) and device.type == "cuda":
        try:
            opt = torch.optim.AdamW(model.parameters(), fused=True, **opt_kwargs)
            print("[train] AdamW fused=True", flush=True)
        except (TypeError, RuntimeError) as e:
            opt = torch.optim.AdamW(model.parameters(), **opt_kwargs)
            print(f"[train] WARN: fused AdamW 不可用，回退: {e}", flush=True)
    else:
        opt = torch.optim.AdamW(model.parameters(), **opt_kwargs)
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

    global_step = 0  # this-run counter (LR + progress + stop)
    lifetime_base = 0  # ckpt 累计步；续训加载后写入 ckpt，不进 LR 分母
    best_loss = float("inf")
    best_step = 0
    recent_losses: "collections.deque[float]" = collections.deque(
        maxlen=max(1, int(args.stable_steps))
    )
    if args.resume:
        resume_path = Path(args.resume)
        blob = torch.load(resume_path, map_location="cpu", weights_only=False)
        _unwrap_model(model).load_state_dict(v41t.clean_state_dict(blob["model"]))
        if "optimizer" in blob:
            try:
                opt.load_state_dict(blob["optimizer"])
            except Exception as e:  # noqa: BLE001
                print(f"[train] WARN: optimizer 未恢复: {e}")
        lifetime_base = int(blob.get("step", 0))
        global_step = 0
        if blob.get("loss") is not None:
            best_loss = float(blob["loss"])
            best_step = 0
        print(
            f"[train] resumed from {resume_path} lifetime_step={lifetime_base} "
            f"best_loss={best_loss}"
        )

    total_steps, step_limit = resolve_step_budget(lifetime_base, args.max_steps, plan_steps)
    if lifetime_base > 0 and step_limit is not None:
        print(
            f"[train] max-steps={int(args.max_steps)} 为本轮步数，"
            f"进度/LR 从 0→{step_limit}（ckpt lifetime 从 {lifetime_base} 起记）",
            flush=True,
        )

    print(
        f"[train] samples={len(ds)} steps/epoch≈{steps_per_epoch} "
        f"start_step=0 total_steps={total_steps} "
        f"shuffle_salt={salt} warmup={args.warmup_steps} "
        f"lr={args.lr}→{args.min_lr} thresholds={thresholds} "
        f"stable_steps={args.stable_steps}"
        + (f" lifetime_base={lifetime_base}" if lifetime_base else "")
    )
    if total_steps < 1:
        print(
            f"[train] WARN: total_steps={total_steps}，本次不会再有有效训练步"
        )

    model.train()
    t0 = time.time()
    last_loss: Optional[float] = None

    def save_milestone(step: int, loss_val: float, threshold: float) -> None:
        # filename: loss-10.pt for 10.0, loss-9.5.pt for 9.5
        tag = f"{threshold:g}"
        path = out_dir / f"loss-{tag}.pt"
        ckpt = build_ckpt(model, opt, cfg, lifetime_base + step, vocab, loss_val)
        save_ckpt(path, ckpt)
        print(
            f"[train] milestone save loss<={threshold:g} stable={args.stable_steps} "
            f"@ step={step} loss={loss_val:.4f} -> {path}",
            flush=True,
        )

    def maybe_save_best(step: int, loss_val: float) -> None:
        """最近 stable_steps 步的最小值刷新历史最优时写 best.pt。

        旧逻辑要求“连续 stable_steps 步每一步都低于历史最优”，loss 噪声一大就再也不触发，
        best.pt 会冻在训练早期。改成窗口最小值 + 最小间隔 + 最小改善幅度：
        既能抗噪，也不会因为抖动反复写 1.7G 的大文件。
        """
        nonlocal best_loss, best_step
        if not args.save_best:
            return
        recent_losses.append(loss_val)
        need = max(1, int(args.stable_steps))
        if len(recent_losses) < need:
            return
        window_min = min(recent_losses)
        if window_min >= best_loss * (1.0 - max(0.0, float(args.best_min_improve))):
            return
        if best_step and (step - best_step) < int(args.best_min_gap):
            return
        best_loss = window_min
        best_step = step
        ckpt = build_ckpt(model, opt, cfg, lifetime_base + step, vocab, window_min)
        save_ckpt(out_dir / "best.pt", ckpt)
        print(
            f"[train] best save loss={window_min:.4f} window={need} "
            f"gap={args.best_min_gap} @ step={step} -> {out_dir / 'best.pt'}",
            flush=True,
        )

    def save_last(step: int, loss_val: float) -> None:
        ckpt = build_ckpt(model, opt, cfg, lifetime_base + step, vocab, loss_val)
        save_ckpt(out_dir / "last.pt", ckpt)  # save_ckpt 内部已做 tmp + replace 原子替换

    def maybe_save_last(step: int, loss_val: float) -> None:
        if args.save_last_every <= 0 or step % args.save_last_every != 0:
            return
        save_last(step, loss_val)
        print(f"[train] last save @ step={step} loss={loss_val:.4f}", flush=True)

    def maybe_save_step(step: int, loss_val: float) -> None:
        if args.save_every <= 0 or step % args.save_every != 0:
            return
        life = lifetime_base + step
        ckpt = build_ckpt(model, opt, cfg, life, vocab, loss_val)
        step_path = out_dir / f"step-{life}.pt"
        save_ckpt(step_path, ckpt)
        prune_step_ckpts(out_dir, args.keep_last)
        print(f"[train] step save @ step={step} loss={loss_val:.4f} -> {step_path}", flush=True)

    for epoch in range(args.epochs):
        if batch_mode == "pack":
            assert packed_iter_factory is not None
            epoch_iter = packed_iter_factory(epoch)
        else:
            epoch_iter = loader
        for batch in epoch_iter:
            if _STOP["flag"] or (step_limit is not None and global_step >= step_limit):
                break
            input_ids, labels, doc_ids = v41t.unpack_batch(batch, device)

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
                if is_v41:
                    return v41t.forward_loss(
                        model,
                        input_ids,
                        labels,
                        doc_ids,
                        aux_coef=args.aux_coef,
                        use_candidate_pool=bool(args.candidate_pool),
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
                if is_v41:
                    print(v41t.log_line(epoch, global_step, last_loss, ce_loss, aux_loss, cur_lr), flush=True)
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
            maybe_save_last(global_step, last_loss)

            if _STOP["flag"]:
                break
            if step_limit is not None and global_step >= step_limit:
                break
        if _STOP["flag"]:
            break
        if step_limit is not None and global_step >= step_limit:
            break

    if last_loss is None:
        last_loss = best_loss if best_loss < float("inf") else 0.0
    save_last(global_step, last_loss)
    if _STOP["flag"]:
        print(
            f"[train] 收到信号 {_STOP['signum']}，已在 step={global_step} 保存 last.pt 后退出",
            flush=True,
        )
    print(f"[train] final save last.pt @ step={global_step} loss={last_loss:.4f}")

    elapsed = time.time() - t0
    print(
        f"[train] done steps={global_step} "
        f"lifetime_step={lifetime_base + global_step} elapsed={elapsed:.1f}s "
        f"dir={out_dir} best_loss={best_loss if best_loss < float('inf') else 'n/a'} "
        f"milestones_hit={sorted(hit_thresholds, reverse=True)}"
    )


if __name__ == "__main__":
    main()
