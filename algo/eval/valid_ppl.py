#!/usr/bin/env python3
"""Valid set perplexity（Eval-Core #1）.

验证集从训练 jsonl 按 --valid-ratio 在内存里划分，不读独立 valid 文件。
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from algo.model.config import ModelConfig
from algo.model.dense import DenseLM
from algo.model.tokenizer import load_tokenizer
from algo.train_pretrain import JsonlTextDataset, collate_pad, split_samples_by_ratio


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="tasks/_default_pretrain/last.pt")
    ap.add_argument(
        "--data",
        type=str,
        default="data/PT/pretrain_mix_12g_clean.jsonl",
        help="训练 jsonl；验证从这里按比例抽",
    )
    ap.add_argument("--valid-ratio", type=float, default=0.01)
    ap.add_argument("--valid-seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--max-samples", type=int, default=None, help="验证条数上限")
    ap.add_argument("--tokenizer", type=str, default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = load_tokenizer(args.tokenizer, allow_fallback=True)
    pad_id = int(getattr(tok, "pad_token_id", 0) or 0)

    if not Path(args.ckpt).exists():
        raise SystemExit(f"缺少 checkpoint: {args.ckpt}")
    if not Path(args.data).exists():
        raise SystemExit(f"缺少训练数据: {args.data}")

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ModelConfig.from_dict(ckpt["config"])
    model = DenseLM(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    ds = JsonlTextDataset(args.data, tok, args.max_seq_len)
    _train, valid_samples = split_samples_by_ratio(
        ds.samples, float(args.valid_ratio), int(args.valid_seed)
    )
    if args.max_samples is not None and int(args.max_samples) > 0:
        valid_samples = valid_samples[: int(args.max_samples)]
    if not valid_samples:
        raise SystemExit("按 valid-ratio 没有抽出验证样本")
    vds = JsonlTextDataset(samples=valid_samples, tokenizer=tok, max_seq_len=args.max_seq_len)
    loader = DataLoader(
        vds, batch_size=args.batch_size, shuffle=False,
        collate_fn=lambda b: collate_pad(b, pad_id=pad_id),
    )

    total_loss, total_tokens = 0.0, 0
    for input_ids, labels in loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        _, loss = model(input_ids, labels=labels)
        n = (labels != -100).sum().item()
        total_loss += loss.item() * n
        total_tokens += n

    mean_nll = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(mean_nll, 20))
    print(json.dumps({
        "valid_nll": mean_nll,
        "valid_ppl": ppl,
        "tokens": total_tokens,
        "valid_samples": len(vds),
        "valid_ratio": args.valid_ratio,
        "valid_seed": args.valid_seed,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
