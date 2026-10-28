---
license: other
pretty_name: MiniSeek-SFT
tags:
  - sft
  - miniseek
  - chinese
configs:
  - config_name: default
    data_files:
      - split: train
        path: sft_miniseek_v1.jsonl
---

# MiniSeek-SFT

本地镜像目录：`data/SFT/` → HF [`wyf-yfw/MiniSeek-SFT`](https://huggingface.co/datasets/wyf-yfw/MiniSeek-SFT)。

主文件：`sft_miniseek_v1.jsonl`（~22.9 万条）。来源见 `SOURCES.md`。
