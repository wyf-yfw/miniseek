# MiniSeek SFT v5 short256 短上下文包设计

日期：2026-10-28  
状态：已批准（对话确认：超长丢弃、≤256 字符、从 10G 筛、方案 1 尽量保配比）  
范围：从现有 v5 10G 包筛出短样本新文件；**不改** 5G/10G；**不改** WebUI/训练默认数据集路径。

## 目标

1. 验证小模型可行性时，上下文以 **≤256 字符** 为主，训练可用 `max_seq_len=256`（或最多 512）。
2. 产出独立短上下文 Conversations JSONL，便于快速训完，不必动正式 10G 默认包。

## 已锁定规则

| 项 | 约定 |
|----|------|
| 源 | `data/SFT/sft_miniseek_v5_10g.jsonl` |
| 长度 | 整段对话所有 `content` 字符数之和 **≤256** 才保留 |
| 超长 | **丢弃**（不截断） |
| 产出 | `data/SFT/sft_miniseek_v5_short256.jsonl`（新文件） |
| 5G/10G | **禁止修改** |
| Web 默认 | **不改**（仍指向 10G）；仅在 SOURCES / 本设计注明短包路径 |
| 配比 | 筛完后若行内有能力桶元数据则尽量对齐 v5（对话 43 / 知识 22 / 推理 15 / 创作 12 / 代码 8）；**当前 10G 行仅含 `conversations`，无桶标签** → 保留全部通过长度门的样本，并在 SOURCES 中记录实际语言/轮次/长度分布 |

## 产出物

| 路径 | 说明 |
|------|------|
| `data/SFT/sft_miniseek_v5_short256.jsonl` | 短上下文包（gitignore，仅本机） |
| `data/SFT/SOURCES_v5_short256.md` | 条数、体积、保留率、语言/轮次/长度分布 |
| `scripts/build_sft_v5_short256.py` | 流式过滤脚本（可复跑） |

行格式与 v5 一致：`{"conversations":[{"role","content"},...]}`。

## 流水线

1. 流式读 10G jsonl。
2. 对每行：`L = sum(len(content))`；`L > 256` 丢弃。
3. 通过者原样写入 short256；统计 lang（CJK 启发式）/ single-vs-multi / 长度分桶。
4. 写 `SOURCES_v5_short256.md`；在 `data/SFT/SOURCES.md` 追加短包入口说明（不改默认）。
5. 提交脚本 + 设计 + SOURCES md；大 jsonl 不进 git。

## 非目标

- 不截断长样本、不重抽独立互斥包、不上 HF、不改 WebUI/train 默认路径。
- 不删 5G/10G、不动进行中的训练任务数据文件。
- 不用 Cursor CloudAgent。

## 验收

- 磁盘存在 short256 jsonl + `SOURCES_v5_short256.md`。
- 抽检：每行 content 总字符 ≤256；格式合法。
- 5G/10G 文件 mtime/size 未变。
- 文档写明保留条数、GiB、相对 10G 保留率、分布。
