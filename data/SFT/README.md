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
  - config_name: sft_short256
    data_files:
      - split: train
        path: sft_miniseek_v5_short256.jsonl
---

# MiniSeek-SFT

本地镜像：`data/SFT/` → HF [`wyf-yfw/MiniSeek-SFT`](https://huggingface.co/datasets/wyf-yfw/MiniSeek-SFT)。

## 主文件

| 文件 | 说明 |
|------|------|
| `sft_miniseek_v5_10g.jsonl` | ≈10 GiB，约 596 万条 — **正式训练默认** |
| `sft_miniseek_v5_5g.jsonl` | ≈5 GiB，约 311 万条 — 小跑对照；**是 10G 的子集** |
| `sft_miniseek_v5_short256.jsonl` | 短上下文包 — 从 **10G** 过滤；见下 |

格式：`{"conversations":[{"role","content"},...]}`。无独立 valid；训练用 `--valid-ratio`。

配比与来源见 `SOURCES.md` / `SOURCES_v5.md` / `SOURCES_v5_short256.md`。v1 已退役删除。

## short256（短上下文可行性包）

- **来源**：`sft_miniseek_v5_10g.jsonl`（不是 5G 子集，也不改动 5G/10G 文件）。
- **长度规则**：整段对话所有 `content` 字符数之和 **优先 ≤256**；**软上限 ≤280**（约 +10%）。更长样本 **直接丢弃**（不截断写入）。
- **文件名**：`sft_miniseek_v5_short256.jsonl`（Hub config：`sft_short256`）。
- **条数 / 体积**：2,018,458 条；~0.838 GiB（相对 10G 保留率 33.85%）。其中 ≤256：1,882,126（93.25%）；257–280：136,332（6.75%）。详情见 `SOURCES_v5_short256.md`。
- **与 5G/10G 关系**：独立过滤产物；**不**修改 5G/10G；与 5G 无超集/子集关系。
- **训练建议**：`max_seq_len=256`（或 ≤512）。SFT 编码若仍超长，从 **尾部截断（keep-head）**，优先保留问题/prompt。
