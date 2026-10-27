#!/usr/bin/env python3
"""Gen-Smoke：20 条固定中文 prompt，seed=42（Eval-Core #2）."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from algo.model.config import ModelConfig
from algo.model.dense import DenseLM
from algo.model.tokenizer import load_tokenizer

PROMPTS = [
    "请用一句话介绍人工智能。",
    "北京有哪些著名景点？",
    "解释一下什么是机器学习。",
    "写一句鼓励学习的话。",
    "春天适合做什么运动？",
    "简述太阳系有几大行星。",
    "如何保持健康的作息？",
    "用三句话讲一个小故事。",
    "什么是深度学习？",
    "推荐一本适合入门编程的书。",
    "今天天气很好，我想",
    "中国的传统节日有哪些？",
    "请把下面的话翻译成英文：你好世界。",
    "1+1等于几？请解释。",
    "简述互联网的发展历程。",
    "如何学习一门外语？",
    "写一首关于月亮的短诗。",
    "什么是大语言模型？",
    "请给新手程序员三条建议。",
    "用通俗语言解释量子计算。",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="tasks/_default_pretrain/last.pt")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--out", type=str, default="eval/reports/gen_smoke_stage0.jsonl")
    ap.add_argument("--tokenizer", type=str, default=None)
    args = ap.parse_args()

    assert len(PROMPTS) == 20
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = load_tokenizer(args.tokenizer, allow_fallback=True)

    if Path(args.ckpt).exists():
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        model = DenseLM(ModelConfig.from_dict(ckpt["config"]))
        model.load_state_dict(ckpt["model"])
    else:
        print("[gen_smoke] WARN: 无 ckpt，随机权重")
        model = DenseLM(ModelConfig().with_vocab_size(int(tok.vocab_size)))

    model.to(device).eval()
    eos = getattr(tok, "eos_token_id", None)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for i, prompt in enumerate(PROMPTS):
            set_seed(args.seed + i)  # 每条可复现，整体仍锚定 42
            ids = torch.tensor(
                [tok.encode(prompt, add_special_tokens=True)],
                dtype=torch.long,
                device=device,
            )
            out = model.generate(
                ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                eos_token_id=eos,
            )
            text = tok.decode(out[0].tolist(), skip_special_tokens=True)
            row = {"id": i, "prompt": prompt, "output": text, "seed": args.seed + i}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"[{i:02d}] {prompt} → {text[:80]}…")

    print(f"[gen_smoke] wrote {out_path}")


if __name__ == "__main__":
    main()
