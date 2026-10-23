#!/usr/bin/env python3
"""下载 / 准备 MiniMind 预训练数据，并固定划分 valid.jsonl.

数据源（阶段 0–5 锁死）::

    HF dataset: jingyaogong/minimind_dataset
    Train file: pretrain_t2t_mini.jsonl
    Valid:      从 train 固定抽取 0.5%，seed=42 → data/valid.jsonl

本脚本默认**不**强制下载多 GB 数据：若本地已有文件则只做划分；
若缺少且 ``--download`` 未指定，则打印说明并退出 0（方便 CI / smoke）。

示例::

    python scripts/download_data.py --download   # 需要 huggingface_hub + 网络
    python scripts/download_data.py              # 仅检查 / 划分已有文件
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

HF_DATASET = "jingyaogong/minimind_dataset"
TRAIN_NAME = "pretrain_t2t_mini.jsonl"
VALID_RATIO = 0.005
SEED = 42


def split_valid(train_path: Path, valid_path: Path, ratio: float = VALID_RATIO, seed: int = SEED) -> int:
    lines = []
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)
    if not lines:
        raise RuntimeError(f"空文件: {train_path}")
    rng = random.Random(seed)
    n_valid = max(1, int(len(lines) * ratio))
    indices = sorted(rng.sample(range(len(lines)), n_valid))
    valid_set = set(indices)
    valid_path.parent.mkdir(parents=True, exist_ok=True)
    with open(valid_path, "w", encoding="utf-8") as fv:
        for i in indices:
            fv.write(lines[i] + "\n")
    # 可选：写出剔除 valid 后的 train（默认不覆盖原 train，只写 valid）
    print(
        f"[data] train_lines={len(lines)} valid_lines={n_valid} "
        f"ratio≈{n_valid/len(lines):.4f} seed={seed} → {valid_path}"
    )
    return n_valid


def try_download(dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / TRAIN_NAME
    if out.exists() and out.stat().st_size > 0:
        print(f"[data] already exists: {out}")
        return out
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise SystemExit(
            "需要 huggingface_hub：pip install huggingface_hub\n"
            f"或手动下载 {HF_DATASET} 的 {TRAIN_NAME} 到 {dest_dir}/"
        ) from e
    print(f"[data] downloading {HF_DATASET}/{TRAIN_NAME} …")
    path = hf_hub_download(
        repo_id=HF_DATASET,
        filename=TRAIN_NAME,
        repo_type="dataset",
        local_dir=str(dest_dir),
    )
    # hf 可能放在子目录
    p = Path(path)
    if p.resolve() != out.resolve() and p.exists():
        out.write_bytes(p.read_bytes())
    print(f"[data] saved {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default="data")
    ap.add_argument("--download", action="store_true", help="从 HF 下载（可能较大）")
    ap.add_argument("--valid-ratio", type=float, default=VALID_RATIO)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--force-split", action="store_true", help="即使 valid 已存在也重划")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    train_path = data_dir / TRAIN_NAME
    valid_path = data_dir / "valid.jsonl"

    print(
        f"""
=== ds41-mini 数据说明（阶段 0–5 锁死）===
HF:    {HF_DATASET}
Train: {TRAIN_NAME}
Valid: valid.jsonl  （从 train 抽 {args.valid_ratio*100:.2f}%，seed={args.seed}）
目录:  {data_dir.resolve()}
"""
    )

    if args.download:
        train_path = try_download(data_dir)
    elif not train_path.exists():
        print(
            f"[data] 未找到 {train_path}。\n"
            f"  手动: 从 https://huggingface.co/datasets/{HF_DATASET} 下载 {TRAIN_NAME}\n"
            f"  或:   python scripts/download_data.py --download\n"
            f"跳过下载（smoke / CI 友好）。"
        )
        sys.exit(0)

    if valid_path.exists() and not args.force_split:
        n = sum(1 for _ in open(valid_path, encoding="utf-8"))
        print(f"[data] valid 已存在: {valid_path} ({n} lines)，跳过划分（--force-split 可重做）")
    else:
        split_valid(train_path, valid_path, ratio=args.valid_ratio, seed=args.seed)


if __name__ == "__main__":
    main()
