# MiniSeek SFT v5 short256 短上下文包设计

日期：2026-10-28  
状态：已批准（超长丢弃；从 10G 筛；方案 1；随后修订：软上限 280 + SFT keep-head 截断）  
范围：从现有 v5 10G 包筛出短样本新文件；**不改** 5G/10G；**不改** WebUI/训练默认数据集路径。上传 HF `wyf-yfw/MiniSeek-SFT` 并更新卡片 README。

## 目标

1. 验证小模型可行性时，上下文以 **≤256 字符** 为主，训练可用 `max_seq_len=256`（或最多 512）。
2. 产出独立短上下文 Conversations JSONL，便于快速训完，不必动正式 10G 默认包。

## 已锁定规则

| 项 | 约定 |
|----|------|
| 源 | `data/SFT/sft_miniseek_v5_10g.jsonl` |
| 长度 | **优先** 整段对话所有 `content` 字符数之和 **≤256**；**软上限 ≤280**（约 +10%），软区间样本可保留以免包过薄 |
| 超长 | **>280 丢弃**（不截断） |
| 产出 | `data/SFT/sft_miniseek_v5_short256.jsonl`（新文件） |
| 5G/10G | **禁止修改** |
| Web 默认 | **不改**（仍指向 10G）；SOURCES / Hub README 注明短包路径 |
| 配比 | 10G 行无桶标签 → 保留全部通过长度门的样本，SOURCES 记录实际语言/轮次/长度分布 |
| 训练截断 | SFT 编码超 `max_seq_len` 时 **从尾部截断（keep-head）**，优先保留问题/prompt，丢掉回答尾部；仍尽量保留结尾 eos |

## 产出物

| 路径 | 说明 |
|------|------|
| `data/SFT/sft_miniseek_v5_short256.jsonl` | 短上下文包（gitignore，仅本机 + Hub） |
| `data/SFT/SOURCES_v5_short256.md` | 条数、体积、保留率、≤256 / 257–280 分布 |
| `scripts/build_sft_v5_short256.py` | 流式过滤脚本（可复跑） |
| `algo/batching.py` `truncate_keep_head_with_eos` | SFT 用 keep-head |
| Hub | 同仓 `wyf-yfw/MiniSeek-SFT` 增加 short256 文件与 README 说明 |

行格式与 v5 一致：`{"conversations":[{"role","content"},...]}`。

## 流水线

1. 流式读 10G jsonl。
2. `L = sum(len(content))`；`L > 280` 丢弃；`L ≤ 280` 写入（含 ≤256 与 257–280）。
3. 统计 lang / single-vs-multi / 长度分桶（含 soft-only）。
4. 写 `SOURCES_v5_short256.md`；更新 `data/SFT/README.md`（Hub 卡片）与 `SOURCES.md` 入口。
5. SFT：`JsonlSFTDataset.encode_conversation` 改用 `truncate_keep_head_with_eos`。
6. 上传 HF：short256 jsonl + SOURCES_v5_short256.md + 更新后的 README/SOURCES。
7. 提交脚本 + 设计 + 代码 + SOURCES md；大 jsonl 不进 git。

## 非目标

- 不截断长样本进包、不重抽独立互斥包、不改 WebUI/train 默认数据集路径。
- 不删 5G/10G、不动进行中的训练任务数据文件。
- 不用 Cursor CloudAgent。

## 验收

- 磁盘存在 short256 jsonl + `SOURCES_v5_short256.md`；抽检 content 总字符 ≤280。
- 5G/10G 文件 size/mtime 未变。
- SFT 单测：超长样本 keep-head（保留头、尾强制 eos）。
- Hub 可读 short256 与 README 说明；文档写明保留条数、GiB、相对 10G 保留率。
