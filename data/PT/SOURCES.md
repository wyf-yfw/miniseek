---

# pretrain_mix_12g_clean — 主训练集（2026-10-28）

生成：对 `pretrain_mix_12g.jsonl` 做 builtin 清洗（job `20261027-223828-3641`）。

| 文件 | 大小 | 行数 |
|------|------|------|
| `pretrain_mix_12g_clean.jsonl` | ~12G | 3,669,988 |

清洗算子：length / repetition / exact_dedup / sensitive。  
**仓库不再提供 validation 文件**；训练用 `--valid-ratio` 从 train 抽样。

本地目录：`data/PT/`（与本 HF 仓库镜像）。真实语料文件位于 `data/clean/20261027-223828-3641/clean.jsonl`，由软链指向，避免重复占盘。

下列保留对旧 5g / 12g 未清洗集的说明。

# pretrain_mix_12g — 余量扩充说明

生成时间：2026-10-27 18:44（Asia/Shanghai）  
机器：wyf `/home/wyf/miniseek`  
目的：在 `pretrain_mix_10g.jsonl` 之上再加约 **2.5–3GB** 高质量余量，清洗后仍尽量剩 ~10GB。  
原则：优先 Fineweb-Edu 中文 / Cosmopedia-v2；**不**追加生网页（未再加 SkyPile / FineWeb2）。

## 产出

| 文件 | 大小 | 行数(估) | 说明 |
|------|------|----------|------|
| `data/pretrain_mix_12g.jsonl` | **12.85 GiB** (13,797,626,296 B) | **3,834,735** | 10g 全量 + 本轮增量 |
| `data/pretrain_mix_10g.jsonl` | 10.15 GiB | 3,362,521 | **未改动** |
| `data/pretrain_mix_5g.jsonl` | 4.94 GiB | 1,834,462 | **未改动** |
| `data/pretrain_mix_12g_stats.json` | — | — | 本轮统计 |

构建：`cp pretrain_mix_10g.jsonl → pretrain_mix_12g.jsonl`，再追加 extras（seed=47）。

## 本轮增量

| 源 | 纳入字节 | 行数 | 说明 |
|----|----------|------|------|
| Fineweb-Edu-Chinese-V2.1 `4_5` 续 shard（131 个） | 1,995,972,539 | 234,753 | 接续 10g 已用的 000000–000035 |
| SmolLM cosmopedia-v2 `train-00000` 截取 | 903,133,512 | 237,461 | 合成教材风英文 |
| **增量合计** | **2,899,106,051** | **472,214** | |
| **= 10g + 增量** | **13,797,626,296** | **3,834,735** | |

## 原始语料

1. [opencsg/Fineweb-Edu-Chinese-V2.1](https://huggingface.co/datasets/opencsg/Fineweb-Edu-Chinese-V2.1) `4_5/` 续下（Clash SOCKS `socks5h://127.0.0.1:7897`）
2. [HuggingFaceTB/smollm-corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) `cosmopedia-v2/train-00000-of-00104.parquet`（按字节预算截取）

## 处理规则

- 行格式：`{"text": "<正文>\n<|im_end|>"}`
- 短文本 `<64` 字符丢弃
- 未覆盖 5g / 10g；未杀训练进程
- 转换后已删本轮 parquet 以腾磁盘

## 磁盘

组装后 `/` 空闲约 **11G**。

