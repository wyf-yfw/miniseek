#!/usr/bin/env python3
"""固定 seed 采样生成（仅 MiniSeek-V）。"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import random

import torch

from algo.model import load_model_from_blob
from algo.model.tokenizer import load_tokenizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="tasks/_default_pretrain/last.pt")
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

    if not Path(args.ckpt).exists():
        raise SystemExit(f"缺少 checkpoint: {args.ckpt}（main 不再提供随机 Dense 初始化）")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model, cfg = load_model_from_blob(ckpt)
    print(f"[generate] loaded {args.ckpt} arch={getattr(cfg, 'arch', 'miniseek-v')}")

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
