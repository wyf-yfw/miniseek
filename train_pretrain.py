#!/usr/bin/env python3
"""Stage-0 预训练：jsonl 行 {"text": "..."} 的 next-token 预测.

示例::

    python train_pretrain.py --data data/pretrain_t2t_mini.jsonl --max-steps 2
    python train_pretrain.py --data data/pretrain_t2t_mini.jsonl --epochs 1 --batch-size 8
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Iterator, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

from model.config import ModelConfig
from model.dense import DenseLM
from model.tokenizer import detect_vocab_size, load_tokenizer


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
        if len(ids) < 2:
            ids = ids + [getattr(self.tokenizer, "eos_token_id", 2) or 2]
        ids = ids[: self.max_seq_len]
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


def main() -> None:
    p = argparse.ArgumentParser(description="ds41-mini Stage-0 pretrain")
    p.add_argument("--data", type=str, default="data/pretrain_t2t_mini.jsonl")
    p.add_argument("--tokenizer", type=str, default=None)
    p.add_argument("--out-dir", type=str, default="checkpoints/stage0")
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=None, help="smoke: --max-steps 2")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=10)
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"[train] device={device} dtype={dtype}")

    tok = load_tokenizer(args.tokenizer, allow_fallback=True)
    vocab = detect_vocab_size(tok)
    pad_id = int(getattr(tok, "pad_token_id", 0) or 0)

    cfg = ModelConfig(max_seq_len=args.max_seq_len).with_vocab_size(vocab)
    model = DenseLM(cfg).to(device)
    # BF16 权重：仅 CUDA；CPU 保持 fp32
    if dtype == torch.bfloat16:
        model = model.to(dtype=dtype)

    if not Path(args.data).exists():
        # smoke 无数据时造几条
        print(f"[train] WARN: 缺少 {args.data}，使用内置中文占位样本做 smoke")
        Path("data").mkdir(parents=True, exist_ok=True)
        smoke_path = Path("data/_smoke_train.jsonl")
        with open(smoke_path, "w", encoding="utf-8") as f:
            for t in [
                "深度学习是机器学习的一个分支。",
                "大语言模型可以生成自然语言文本。",
                "今天天气很好，适合出去散步。",
                "北京是中国的首都。",
            ]:
                f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
        args.data = str(smoke_path)
        args.max_samples = args.max_samples or 4
        args.max_steps = args.max_steps or 2

    ds = JsonlTextDataset(
        args.data, tok, max_seq_len=args.max_seq_len, max_samples=args.max_samples
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_pad(b, pad_id=pad_id),
        num_workers=0,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    global_step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        for input_ids, labels in loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            opt.zero_grad(set_to_none=True)
            # autocast：CUDA BF16
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
            if global_step % args.log_every == 0 or global_step == 1:
                print(
                    f"epoch={epoch} step={global_step} loss={loss.item():.4f} "
                    f"ppl={math.exp(min(loss.item(), 20)):.2f}"
                )
            if args.max_steps is not None and global_step >= args.max_steps:
                break
        if args.max_steps is not None and global_step >= args.max_steps:
            break

    ckpt = {
        "model": model.state_dict(),
        "config": cfg.to_dict(),
        "step": global_step,
        "vocab_size": vocab,
    }
    ckpt_path = out_dir / "last.pt"
    # 存盘前转回 fp32 便于跨设备加载
    cpu_state = {k: v.detach().float().cpu() for k, v in model.state_dict().items()}
    ckpt["model"] = cpu_state
    torch.save(ckpt, ckpt_path)
    elapsed = time.time() - t0
    print(f"[train] done steps={global_step} elapsed={elapsed:.1f}s saved={ckpt_path}")


if __name__ == "__main__":
    main()
