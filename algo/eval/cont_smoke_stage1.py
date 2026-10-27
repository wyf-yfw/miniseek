#!/usr/bin/env python3
"""Continuation-focused gen smoke for stage1 MoE (and optional Dense spot-check)."""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import torch

from algo.model.config import ModelConfig
from algo.model.tokenizer import load_tokenizer

# Explicit continuation stems (not chat Q&A)
PROMPTS = [
    "今天天气很好，我想",
    "北京是中国的",
    "人工智能是",
    "从前有一座山，山上有",
    "学习要靠",
    "春天来了，公园里",
    "太阳系一共有",
    "中国的首都是",
    "机器学习可以用来",
    "夜深了，窗外",
    "一只小猫走在街上，",
    "如果明天放假，我会",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ModelConfig.from_dict(ckpt["config"])
    if getattr(cfg, "arch", "dense") == "moe" or getattr(cfg, "use_moe", False):
        from algo.model.moe import MoELM

        model = MoELM(cfg)
        arch = "moe"
    else:
        from algo.model.dense import DenseLM

        model = DenseLM(cfg)
        arch = "dense"
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device).eval()
    meta = {
        "ckpt": ckpt_path,
        "arch": arch,
        "step": ckpt.get("step"),
        "loss": ckpt.get("loss") or ckpt.get("best_loss"),
    }
    return model, meta


_CJK = re.compile(r"[\u4e00-\u9fff]")
_LAT = re.compile(r"[A-Za-z]")


def metrics(prompt: str, full: str, n_new: int, hit_eos: bool) -> dict:
    cont = full[len(prompt) :] if full.startswith(prompt) else full
    cjk = len(_CJK.findall(cont))
    lat = len(_LAT.findall(cont))
    total_chars = max(len(cont), 1)
    # trigram repeat ratio over chars (rough)
    reps = 0
    if len(cont) >= 9:
        grams = [cont[i : i + 3] for i in range(len(cont) - 2)]
        from collections import Counter

        c = Counter(grams)
        reps = sum(v - 1 for v in c.values() if v > 1) / max(len(grams), 1)
    return {
        "continuation": cont,
        "n_new": n_new,
        "hit_eos": hit_eos,
        "cjk_chars": cjk,
        "latin_chars": lat,
        "cjk_ratio": round(cjk / total_chars, 3),
        "rep3": round(reps, 3),
    }


def run(ckpt: str, out: Path, seed: int, max_new: int, temp: float, top_k: int) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = load_tokenizer(None, allow_fallback=True)
    model, meta = load_model(ckpt, device)
    eos = getattr(tok, "eos_token_id", None)
    rows = []
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for i, prompt in enumerate(PROMPTS):
            set_seed(seed + i)
            ids = torch.tensor(
                [tok.encode(prompt, add_special_tokens=True)],
                dtype=torch.long,
                device=device,
            )
            n0 = ids.size(1)
            out_ids = model.generate(
                ids,
                max_new_tokens=max_new,
                temperature=temp,
                top_k=top_k,
                eos_token_id=eos,
            )
            n_new = int(out_ids.size(1) - n0)
            hit_eos = bool(eos is not None and int(out_ids[0, -1].item()) == int(eos))
            text = tok.decode(out_ids[0].tolist(), skip_special_tokens=True)
            m = metrics(prompt, text, n_new, hit_eos)
            row = {"id": i, "prompt": prompt, "output": text, "seed": seed + i, **m}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.append(row)
            print(f"[{i:02d}] {prompt} → {m['continuation'][:72]!r}…")
    summary = {
        "meta": meta,
        "n": len(rows),
        "avg_n_new": round(sum(r["n_new"] for r in rows) / len(rows), 2),
        "eos_count": sum(1 for r in rows if r["hit_eos"]),
        "avg_cjk_ratio": round(sum(r["cjk_ratio"] for r in rows) / len(rows), 3),
        "avg_rep3": round(sum(r["rep3"] for r in rows) / len(rows), 3),
        "high_rep_ids": [r["id"] for r in rows if r["rep3"] >= 0.15],
        "low_cjk_ids": [r["id"] for r in rows if r["cjk_ratio"] < 0.5],
        "out": str(out),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    args = ap.parse_args()
    run(args.ckpt, Path(args.out), args.seed, args.max_new_tokens, args.temperature, args.top_k)


if __name__ == "__main__":
    main()
