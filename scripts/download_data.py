#!/usr/bin/env python3
"""MiniMind 数据集下载已退役。

本仓库训练数据已切换为 MiniSeek 自建集，目录与 HF 镜像对齐：

    预训练: data/PT/pretrain_mix_12g_clean.jsonl  → wyf-yfw/MiniSeek-Pretrain
    SFT:    data/SFT/sft_miniseek_v1.jsonl         → wyf-yfw/MiniSeek-SFT

验证集不再单独提供；训练用 ``--valid-ratio`` 从 train 抽样。

兼容软链仍保留在 ``data/pretrain_mix_12g_clean.jsonl`` / ``data/sft_miniseek_v1.jsonl``。

本脚本保留入口以免旧文档/CI 调用报「找不到文件」；任何参数都只打印说明并退出 0。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXPECTED = {
    "pretrain": ROOT / "data" / "PT" / "pretrain_mix_12g_clean.jsonl",
    "sft": ROOT / "data" / "SFT" / "sft_miniseek_v1.jsonl",
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="MiniMind download retired — MiniSeek data/PT + data/SFT"
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="ignored (MiniMind HF download disabled)",
    )
    parser.parse_args()

    print("[data] MiniMind dataset download is retired.")
    print("[data] Use MiniSeek mirrored folders:")
    for label, path in EXPECTED.items():
        ok = path.exists() and path.stat().st_size > 0
        status = "OK" if ok else "MISSING"
        size = f"{path.stat().st_size / (1024**3):.2f}G" if path.exists() else "-"
        try:
            rel = path.relative_to(ROOT)
        except Exception:
            rel = path
        print(f"  [{status}] {label:8s} {rel} ({size})")
    print("[data] HF: wyf-yfw/MiniSeek-Pretrain , wyf-yfw/MiniSeek-SFT")
    print("[data] Validation: --valid-ratio (no separate valid file on Hub)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
