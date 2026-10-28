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
        path: sft_miniseek_v5_10g.jsonl
  - config_name: sft_5g
    data_files:
      - split: train
        path: sft_miniseek_v5_5g.jsonl
---

# MiniSeek-SFT

本地镜像：`data/SFT/` → HF [`wyf-yfw/MiniSeek-SFT`](https://huggingface.co/datasets/wyf-yfw/MiniSeek-SFT)。

## 主文件（默认）

- `sft_miniseek_v5_10g.jsonl` — ≈10.00 GiB，约 596 万条（正式训练默认）
- `sft_miniseek_v5_5g.jsonl` — ≈5.00 GiB，约 311 万条（小跑对照；是 10G 的子集）

格式：`{"conversations":[{"role","content"},...]}`。无独立 valid；训练用 `--valid-ratio`。

配比与来源见 `SOURCES.md` / `SOURCES_v5.md`。v1 已退役删除。
