#!/usr/bin/env python3
"""Valid set perplexity（Eval-Core #1）."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model.config import ModelConfig
from model.dense import DenseLM
from model.tokenizer import load_tokenizer
from train_pretrain import JsonlTextDataset, collate_pad


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="checkpoints/stage0/last.pt")
    ap.add_argument("--valid", type=str, default="data/valid.jsonl")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--tokenizer", type=str, default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = load_tokenizer(args.tokenizer, allow_fallback=True)
    pad_id = int(getattr(tok, "pad_token_id", 0) or 0)

    if not Path(args.ckpt).exists():
        raise SystemExit(f"缺少 checkpoint: {args.ckpt}")
    if not Path(args.valid).exists():
        raise SystemExit(f"缺少 valid: {args.valid}（先跑 scripts/download_data.py）")

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ModelConfig.from_dict(ckpt["config"])
    model = DenseLM(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    ds = JsonlTextDataset(args.valid, tok, args.max_seq_len, max_samples=args.max_samples)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
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
    print(json.dumps({"valid_nll": mean_nll, "valid_ppl": ppl, "tokens": total_tokens}, ensure_ascii=False))


if __name__ == "__main__":
    main()
