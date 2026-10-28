---
license: other
license_name: mixed-open-sources
license_link: https://huggingface.co/datasets/wyf-yfw/MiniSeek-Pretrain/blob/main/SOURCES.md
language:
  - zh
  - en
size_categories:
  - 1M<n<10M
task_categories:
  - text-generation
pretty_name: MiniSeek-Pretrain
tags:
  - pretrain
  - chinese
  - moe
  - miniseek
  - jsonl
configs:
  - config_name: default
    data_files:
      - split: train
        path: pretrain_mix_12g_clean.jsonl
---

# MiniSeek-Pretrain

中文预训练混合语料，用于 [MiniSeek](https://huggingface.co/collections/wyf-yfw/miniseek-6adc765159d38ab3236c49af) MoE 预训练。  
Chinese pretrain mix for the MiniSeek MoE models.

本仓库是对多个**已发布开源** Hugging Face / 公开语料的再混合（remix），不做网页爬取或自采。来源与许可证细节见 [`SOURCES.md`](SOURCES.md)。

**验证集不再随仓提供**：训练时请用 `--valid-ratio`（例如 `0.01`）从 train 按比例随机划分；不改写源 jsonl。

## Files / 文件

| File | Split | Size (approx) | Lines |
|------|-------|---------------|-------|
| `pretrain_mix_12g_clean.jsonl` | **train（主）** | ~12 GB | 3,669,988 |
| `SOURCES.md` | — | — | 来源说明 |
| `pretrain_mix_12g_clean_stats.json` | — | — | 清洗统计 |

> `valid_mix.jsonl` 已移除。默认配置指向清洗后的 12g。本地镜像目录：`data/PT/`。

## Format / 格式

每行一条 JSON：

```json
{"text": "<正文>\n<|im_end|>"}
```

- UTF-8 JSONL
- 每条 `text` 以 `<|im_end|>` 结尾（训练 EOS）

## Clean / 清洗（12g → clean）

- 输入：`pretrain_mix_12g.jsonl`（3,834,735 行）
- 引擎：builtin（length / repetition / exact_dedup / sensitive）
- 保留：**3,669,988** 行；丢弃 164,747（sensitive 37411，repetition 4456，exact_dedup 122509，length 371）
- 任务 id：`20261027-223828-3641`

## License / 许可

`license: other` — 多源开源语料 remix，请遵循各原始数据集许可，详见 `SOURCES.md`。

## Citation

若使用本混合集，请同时引用上游数据集（见 `SOURCES.md`），并可选引用 MiniSeek 模型页。
