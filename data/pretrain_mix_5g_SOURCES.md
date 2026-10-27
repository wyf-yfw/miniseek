# pretrain_mix_5g — 数据来源说明

生成时间：2026-10-25（Asia/Shanghai）  
机器：wyf `/home/wyf/miniseek`  
原则：**仅使用已发布的开源 HF/GitHub 语料**，不做网页爬取/自采。未使用 MiniMind full（SFT 来源）。

## 产出文件（分层 5% valid / 95% train）

| 文件 | 大小 | 行数 | 格式 |
|------|------|------|------|
| `data/pretrain_mix_5g.jsonl` | **4.94G** (5,307,253,963 B) | 1,834,462 | `{"text":"...\n<\|im_end\|>"}` |
| `data/valid_mix.jsonl` | **253M** (265,371,929 B) | 96,689 | 同上 |
| `data/pretrain_mix_5g_stats.json` | — | — | 分桶与切分统计 |
| `data/pretrain_t2t_mini.jsonl` | 1.2G | — | **保留未改**（旧小集） |
| `data/valid.jsonl` | 6.0M | — | **保留未改** |

备份：`pretrain_mix_5g.jsonl.bak`、`valid_mix.jsonl.bak`（旧版，可删腾盘）。

### 切分规则

- 脚本：`data/scripts/split_mix_5pct.py`
- 对各桶 `data/raw/buckets/{wiki,web,books,code,other}.jsonl` **独立**按行随机抽 **5% → valid / 95% → train**（seed=`42-<bucket>`）
- 合并后分别 shuffle（seed 43 / 44）
- 因 web/other 源内有重复行，额外从 **train 去掉与 valid 内容哈希相同** 的行 2,616 条，保证 holdout 内容不泄漏（全量哈希重叠 = 0）
- 格式不变：每条 `text` 以 `<|im_end|>` 结尾

### 分桶行数占比（train vs valid，应一致）

| 桶 | train 行 | train 占比 | valid 行 | valid 占比 | 桶内 valid% |
|----|----------|------------|----------|------------|-------------|
| wiki | 951,786 | 51.81% | 50,094 | 51.81% | 5.00% |
| web | 516,890 | 28.14% | 27,205 | 28.14% | 5.00% |
| books | 6,570 | 0.36% | 346 | 0.36% | 5.00% |
| code | 9,434 | 0.51% | 497 | 0.51% | 5.00% |
| other | 352,398 | 19.18% | 18,547 | 19.18% | 5.00% |
| **合计** | 1,837,078 → 去重后 1,834,462 | 100% | 96,689 | 100% | ≈5.00% |

wiki+web 行占比（valid）：79.95%  
（books 行少但单条很长，**按字节** books 在 train 中更高；分层按**行**对齐比例。）

## 配比来源（构建混合集时的桶字节，合并前）

| 桶 | 字节 | 占比 | 说明 |
|----|------|------|------|
| wiki | 2,894,286,748 | **51.84%** | 中文维基 |
| web | 2,016,653,214 | **36.12%** | FineWeb2-zh + CNewSum |
| books | 360,385,147 | 6.45% | 诗词 / 文言 / 金庸 |
| code | 75,911,508 | 1.36% | Python 代码 |
| other | 236,223,445 | 4.23% | 中文学术摘要 CSL |
| **wiki+web** | — | **87.96%** | ≥80% 约束 |

## 原始语料（全部来自 Hugging Face，经 hf-mirror）

### 1. Wiki
- [yuhuanstudio/wikipedia-zh](https://huggingface.co/datasets/yuhuanstudio/wikipedia-zh) → `data/raw/buckets/wiki.jsonl`

### 2. Web / News
- [agentlans/fineweb2-chinese](https://huggingface.co/datasets/agentlans/fineweb2-chinese)
- [YeungNLP/firefly-pretrain-dataset](https://huggingface.co/datasets/YeungNLP/firefly-pretrain-dataset) `CNewSum_v2.jsonl`

### 3. Books
- Firefly：`chinese-poem.jsonl` / `classical_chinese.jsonl` / `jinyong.jsonl`

### 4. Code
- [ml6team/the-stack-smol-python](https://huggingface.co/datasets/ml6team/the-stack-smol-python)

### 5. Other
- Firefly：`csl_camera_readly.jsonl`

## 处理规则

- 行格式：`{"text": "<正文>\n<|im_end|>"}`（UTF-8）
- **未**改动 `pretrain_t2t_mini.jsonl`
- 训练尚未启动

## 磁盘

切分后 `/` 约 15G 空闲；若需腾空间可删 `.bak` 与 `data/raw/buckets/_split_*` 临时分片。
