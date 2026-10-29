#!/usr/bin/env python3
"""DSpark 置信度的 STS（Sequential Temperature Scaling）校准。

在验证文本上做投机解码 rollout（每轮验证整块），记录每个草稿位置的原始置信度 logit 与"到该位置为止
是否全部被接受"。然后从左到右逐位置在一维网格上搜温度，使累计乘积 a_k = Π_{i≤k} σ(z_i/T_i)
的 ECE 最小（前面位置的温度固定）。温度写回 dspark.pt 的 sts_temps。

示例::

    python -m algo.dspark_calibrate --dspark tasks/_dspark/dspark.pt --data data/PT/pretrain_mix_12g_clean.jsonl
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import json
import random
from typing import List, Sequence, Tuple

import torch

DEFAULT_GRID = tuple(round(0.25 * i, 2) for i in range(1, 25))  # 0.25 .. 6.0


def ece(probs: torch.Tensor, labels: torch.Tensor, bins: int = 10) -> float:
    probs, labels = probs.float().flatten(), labels.float().flatten()
    edges = torch.linspace(0, 1, bins + 1)
    total, err = probs.numel(), 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        m = (probs >= lo) & ((probs < hi) if i < bins - 1 else (probs <= hi))
        if m.any():
            err += float(m.sum()) / total * abs(float(probs[m].mean()) - float(labels[m].mean()))
    return err


def fit_sts(logits: torch.Tensor, survive: torch.Tensor, grid: Sequence[float] = DEFAULT_GRID) -> List[float]:
    """logits / survive: [M, γ]。返回逐位置温度。"""
    temps: List[float] = []
    prefix = torch.ones(logits.size(0))
    for k in range(logits.size(1)):
        best_e, best_t = float("inf"), 1.0
        for t in grid:
            e = ece(prefix * torch.sigmoid(logits[:, k] / t), survive[:, k])
            if e < best_e:
                best_e, best_t = e, float(t)
        temps.append(best_t)
        prefix = prefix * torch.sigmoid(logits[:, k] / best_t)
    return temps


def cumulative_ece(logits: torch.Tensor, survive: torch.Tensor, temps: Sequence[float]) -> float:
    t = torch.tensor(list(temps), dtype=torch.float32)
    a = torch.cumprod(torch.sigmoid(logits / t), dim=1)
    return ece(a, survive)


@torch.no_grad()
def collect(backbone, drafter, prompts: Sequence[torch.Tensor], *, max_new_tokens: int, temperature: float):
    """每轮验证整块（force_len=γ），收集 (置信度 logit [M,γ], 存活标记 [M,γ])。"""
    from algo.model.miniseek_m.dspark import speculative_stream

    G = drafter.cfg.dspark_block_size
    rows_l: List[torch.Tensor] = []
    rows_s: List[torch.Tensor] = []

    def rec(conf_logits, n_acc, ell):
        rows_l.append(conf_logits[:G])
        rows_s.append((torch.arange(G) < n_acc).float())

    for ids in prompts:
        for _ in speculative_stream(
            backbone, drafter, ids, max_new_tokens=max_new_tokens, temperature=temperature, top_k=None,
            force_len=G, recorder=rec,
        ):
            pass
    if not rows_l:
        raise RuntimeError("没有收集到校准数据")
    return torch.stack(rows_l), torch.stack(rows_s)


def main() -> None:
    from algo.model import load_model_from_blob
    from algo.model.tokenizer import load_tokenizer
    from algo.train_dspark import load_drafter

    p = argparse.ArgumentParser(description="DSpark STS calibration")
    p.add_argument("--dspark", type=str, required=True)
    p.add_argument("--backbone", type=str, default=None, help="默认取 dspark.pt 记录的主干路径")
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--n-prompts", type=int, default=64)
    p.add_argument("--prompt-len", type=int, default=64)
    p.add_argument("--max-new", type=int, default=64)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--tokenizer", type=str, default=None)
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dblob = torch.load(args.dspark, map_location="cpu", weights_only=False)
    backbone, _cfg = load_model_from_blob(torch.load(args.backbone or dblob["backbone"], map_location="cpu", weights_only=False))
    backbone.to(device).eval()
    drafter = load_drafter(dblob).to(device).eval()
    tok = load_tokenizer(args.tokenizer, allow_fallback=True)

    texts = []
    with open(args.data, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                texts.append(json.loads(line).get("text") or "")
    random.shuffle(texts)
    prompts = []
    for t in texts:
        ids = tok.encode(t, add_special_tokens=True)[: args.prompt_len]
        if len(ids) >= 8:
            prompts.append(torch.tensor([ids], device=device))
        if len(prompts) >= args.n_prompts:
            break

    logits, survive = collect(backbone, drafter, prompts, max_new_tokens=args.max_new, temperature=args.temperature)
    before = cumulative_ece(logits, survive, [1.0] * logits.size(1))
    temps = fit_sts(logits, survive)
    after = cumulative_ece(logits, survive, temps)
    drafter.sts_temps.copy_(torch.tensor(temps))
    dblob["dspark"]["sts_temps"] = torch.tensor(temps)
    torch.save(dblob, args.dspark)
    print(f"[sts] blocks={logits.size(0)} temps={temps} ece {before:.4f} -> {after:.4f} -> {args.dspark}")


if __name__ == "__main__":
    main()
