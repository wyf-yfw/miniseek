#!/usr/bin/env python3
"""固定 seed 采样生成。"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch

from model.config import ModelConfig
from model.dense import DenseLM
from model.tokenizer import load_tokenizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="checkpoints/stage0/last.pt")
    p.add_argument("--prompt", type=str, default="你好，请介绍一下人工智能。")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tokenizer", type=str, default=None)
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = load_tokenizer(args.tokenizer, allow_fallback=True)

    if Path(args.ckpt).exists():
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        cfg = ModelConfig.from_dict(ckpt["config"])
        model = DenseLM(cfg)
        model.load_state_dict(ckpt["model"], strict=True)
        print(f"[generate] loaded {args.ckpt}")
    else:
        print(f"[generate] WARN: 无 checkpoint，使用随机初始化权重")
        cfg = ModelConfig().with_vocab_size(int(getattr(tok, "vocab_size", 6400)))
        model = DenseLM(cfg)

    model.to(device).eval()
    ids = tok.encode(args.prompt, add_special_tokens=True)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    eos = getattr(tok, "eos_token_id", None)
    out = model.generate(
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        eos_token_id=eos,
    )
    text = tok.decode(out[0].tolist(), skip_special_tokens=True)
    print("---")
    print(text)
    print("---")


if __name__ == "__main__":
    main()
