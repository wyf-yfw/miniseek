# 对齐 DeepSeek-V4.1-Flash：由简入难阶段计划

> 来源：2026-10-23 Canvas《对齐 DeepSeek-V4.1-Flash：由简入难阶段计划》完整落盘。  
> 仓库：`wyf-yfw/miniseek`。本文件是主规格；实现以本仓库代码与实测为准。

冻结超小规格：`d=512`、8 层；数据与 benchmark 已定；每阶段有层表、参数量、训练数据与能力门槛，逐步对齐 V4.1 Flash 结构。

参考：

- [DeepSeek-V4.1 相关论文 / HTML](https://arxiv.org/html/2609.19969)
- [jingyaogong/minimind_dataset](https://huggingface.co/datasets/jingyaogong/minimind_dataset)
- [C-Eval](https://cevalbenchmark.com/)
- [CMMLU](https://github.com/haonan-li/CMMLU)
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)

---

## 冻结总规格（先按这套实现，再谈放大）

| 项 | 定稿值 |
|----|--------|
| `d_model` | 512 |
| `n_layers` | 8（阶段 4 起：Encoder 4 + Decoder 4） |
| `n_heads` / `head_dim` | 8 / 64 |
| `vocab_size` | **实测 MiniMind = 6400**（见下节）；规划表里曾用 32000 做纸面估算 |
| `max_seq`（训练起步） | 512；以后再拉长 |
| FFN 中间维（Dense） | 2048（=4d） |
| MoE 专家中间维 | 512（=d，M0.1） |
| MoE 结构 | 1 shared + 32 routed，top-2（M0.1） |
| SWA `n_win` | 128 |
| CSA2 Top-K | 128（以后可调到 512） |
| CSA2 压缩比 `m` | 阶段 3–5 先 `m=1`；以后 Encoder 可试 `m=2` |
| LM Head | 与 Embedding 解耦计数（估算按 2×V×d）；实现可 tie |
| Tokenizer | MiniMind 仓库自带 tokenizer（以实际文件为准） |
| 精度 / 硬件 | BF16（CUDA），目标单卡 3090 |

### Tokenizer / 参数量（实测订正）

- MiniMind 自定义 `minimind_tokenizer` **实测词表 = 6400**（不是早期文档里的 ~32k mistral 词表）。
- Stage 0 Dense + vocab=6400 + tie embedding ≈ **28.45M**（本仓库 `DenseLM` init 打印为准）。
- 若强制 vocab=32000（非 MiniMind 默认），纸面 Dense 总参可接近早期估算的 **~58M**；**本仓库以 6400 为准**，便于直接吃 `minimind_dataset`。
- 下文「体量总表」里的 58M / 117M 等，是 **V=32000 纸面估算**；换算到 V=6400 时 Embedding+LM Head 约从 32.8M→6.6M。**M0.1 MoE 实测总参 ≈150.22M**（1+32, h=512）。

---

## 数据集（已定）

**对比原则：** 阶段 0→5 换的是架构，不是数据。训练集、验证集、评测集全程锁死同一份，才看得出「升级架构有没有用」。SFT / 大数据只在架构冻结后另开一条线，不掺进结构对比。

| 锁定项 | 固定内容 | 不许改 |
|--------|----------|--------|
| Tokenizer | MiniMind 自带 tokenizer | 各阶段同一词表 |
> **2026-10-27**：MiniMind 旧集已删除。现行默认改为 `pretrain_mix_5g.jsonl` / `valid_mix.jsonl` / `sft_miniseek_v1.jsonl`。下表保留为阶段 0–5 历史对照记录。

| 预训练集 | `pretrain_t2t_mini.jsonl` 全量 | 阶段 0–5 都用它训 |
| Valid 集 | 从该文件固定 seed 切 0.5%（一次切好存成 `valid.jsonl`） | 以后不许重切 |
| 步数/超参 | 同 batch、同 lr 日程、同 `max_seq_len=512`、同训练步数 | 只改模型结构代码 |
| SFT 集 | `sft_t2t_mini.jsonl` | 仅架构冻结后用；不参与 0–5 结构对比 |

源：`https://huggingface.co/datasets/jingyaogong/minimind_dataset`。完整 10GB/14GB 版留作「冻结后加强」，不进阶段对比表。

---

## Benchmark（各阶段同一套）

每阶段训完必须跑完整套；和上一阶段并排记分，才能看架构收益。

**固定评测包 Eval-Core：** 客观题用 lm-evaluation-harness；主观题用仓库内写死的 20 条 Gen-Smoke（seed=42）。参考锚点：MiniMind-3≈64M 的 C-Eval≈24.9 / CMMLU≈25.4（近随机，看相对变化）。

| # | 项目 | 固定设定 | 阶段 0–5 每阶段都跑 |
|---|------|----------|---------------------|
| 1 | Valid PPL | 同一 `valid.jsonl` | 是 |
| 2 | Gen-Smoke | 同一 20 prompt + seed=42 | 是 |
| 3 | C-Eval valid | `lm_eval ceval-valid`，同 few-shot | 是 |
| 4 | CMMLU | `lm_eval cmmlu`，同设定 | 是 |
| 5 | 结构探针 | MoE 路由 / CSA2 模式 / CED 形状（按阶段有无） | 有则必跑 |
| 6 | 效率 | 同硬件记峰值显存、tokens/s | 是 |

| 阶段 | 对比上一阶段看什么 | 通过线（相对） |
|------|--------------------|----------------|
| 0 | 建立基线分数 | PPL 下降；Smoke≥15/20 不崩；记下 C-Eval/CMMLU 基线 |
| 1 MoE | 质量≈基线 + 路由健康 | PPL 不差于 0 超过 3%；单专家 &lt;70% token |
| 2 SWA | 质量换效率 | PPL 恶化 &lt;5%；显存下降 |
| 3 CSA2 | 稀疏别伤太多 | PPL 恶化 &lt;5%；Reuse 确认跳过 indexer |
| 4 CED | 结构对齐主收益 | PPL/选择题不崩；Prefill 显存或耗时优于同激活 Decoder-only |
| 5 HSI+Replay | 长上下文配方 | 同质量下 KV/延迟更好；探针单测过 |
| SFT 后 | 指令能力（仍用同一 Eval-Core） | C-Eval/CMMLU ≥22% 或明显高于 SFT 前同架构 |

分数一律写入 `eval/reports/stage-N.md`，并维护 `eval/reports/leaderboard.md`（行=阶段，列=PPL/Smoke/C-Eval/CMMLU/显存）。禁止「这阶段换个测试集」。

英文 spot-check（可选）：ARC-Easy / HellaSwag / PIQA。

---

## 参数量怎么算（估算口径，按 V=32000 纸面）

下列为权重参数量近似，忽略 bias / LayerNorm / 路由器以外的小项；与最终 `numel()` 可能差几个百分点。

1. Embedding + LM Head ≈ 2 × V × d = 2 × 32000 × 512 ≈ **32.8M**（V=6400 时 ≈ **6.6M**）
2. Self-Attn（QKV+O）≈ 4 × d × d = 4 × 512² ≈ **1.05M / 层**
3. Dense FFN ≈ 2 × d × 4d = 2 × 512 × 2048 ≈ **2.10M / 层**
4. MoE（M0.1）：每专家 2×d×512；1+32 个专家全计入参数；外加 router ≈ d×32
5. 激活量：每 token 只跑 1 个共享 + top-2 路由专家，不等于总参

---

## 体量总表（V=32000 纸面估算）

单位：百万参数（M）；激活指每 token 大致会碰到的注意力+专家权重。

| 阶段 | 结构变化 | 总参数≈ | 每 token 激活≈ | 备注 |
|------|----------|---------|----------------|------|
| 0 Dense | Full Attn + FFN ×8 | 58M | ≈25M | 8×(1.05+2.10)+32.8；**实测 V=6400 ≈28.45M** |
| 1 MoE | FFN→MoE（1+32,h=512） | ≈150M（V=6400 实测） | ≈激活接近 Dense×3 专家 | 总参涨；激活仍可控 |
| 2 SWA | Attn→窗口 | ≈150M | ≈同 Stage1 激活 | 参数几乎不变，算力降 |
| 3 CSA2 | +三模式/indexer | 118M | ≈27M | indexer 很小，+约 1M 量级 |
| 4 CED | 切 4+4 + 投影 | 118M | Prefill 更低 | 投影矩阵约数×d²，很小 |
| 5 HSI+Replay | 候选池/重放 | 118M | 长上下文更省算 | 逻辑件为主 |
| 6 后置 | Engram 等 | 另计 | 另计 | 默认关闭，不计入主线 |

---

## 阶段 0：Dense 基线

```text
Embedding
   ↓
×8：Full Attn + FFN   (d=512，全注意力)
   ↓
LM Head
```

| 层 | 注意力 | FFN | 其它 |
|----|--------|-----|------|
| Embed | — | — | V×d（V=6400 时 ≈3.3M；tie 后与 Head 共享） |
| L0–L7 | Full causal Attn（≈1.05M） | Dense 4d（≈2.10M） | 2×LN 可忽略 |
| LM Head | — | — | 可与 Embed tie |

- 纸面（V=32000）：合计 ≈ 32.8 + 8×3.15 ≈ **58.0M**
- **实测（本仓库）**：≈ **28.45M**
- 验收：train 一步、generate 不崩；建立 Eval-Core 基线
- **本仓库当前状态：Stage 0 已实现（train / generate / Web UI / checkpoints）**

---

## 阶段 1：MoE

```text
Embedding
   ↓
Full Attn（同阶段 0）  +  MoE（1+32 top-2，h=512）— MiniSeek-M0.1
   ↓
LM Head
```

| 层 | 注意力 | MoE | 参数/层≈ |
|----|--------|-----|----------|
| L0–L7 | Full Attn ≈1.05M | 1 shared + 32 routed，中间维 512，top-2 | ≈专家主导 |

| 项 | 值 |
|----|-----|
| 单专家权重 | 2×512×512 ≈ 0.52M |
| 9 个专家合计 | ≈9.44M |
| router | 512×8 ≈ 0.004M |
| 每 token 激活专家 | shared + 2 routed ≈ 1.57M（仅专家部分） |
| 整模总参（V=32000 纸面） | ≈32.8 + 8×10.49 ≈ **116.7M** |

加 auxiliary loss（负载均衡）。注意力仍全历史。

---

## 阶段 2：SWA

```text
SWA (n_win=128)  +  MoE（同阶段 1）
```

| 层 | 注意力 | 窗口 | MoE |
|----|--------|------|-----|
| L0–L7 | SWA（权重仍≈1.05M） | n_win=128 | 同阶段 1 |

总参仍约 150M（M0.1）；变的是 mask 与 KV cache 长度，不是权重规模。

---

## 阶段 3：简化 CSA2（仍 Decoder-only）

```text
L0 Full（建 KV+Top-K）
L1,L4 Reindex（重选 Top-K）
其余 Reuse（L2,3,5,6,7）
每层另有 SWA + MoE；Top-K=128
```

| 层 | CSA2 模式 | 局部 | MoE | 说明 |
|----|-----------|------|-----|------|
| L0 | Full | SWA 128 | 1+8 top-2 | 建全局 KV + indexer 选 Top-K=128 |
| L1 | Reindex | SWA 128 | 同左 | 复用 KV，重选 Top-K |
| L2 | Reuse | SWA 128 | 同左 | KV+Top-K 都复用 |
| L3 | Reuse | SWA 128 | 同左 | — |
| L4 | Reindex | SWA 128 | 同左 | 第二组刷新 Top-K |
| L5 | Reuse | SWA 128 | 同左 | — |
| L6 | Reuse | SWA 128 | 同左 | — |
| L7 | Reuse | SWA 128 | 同左 | — |

压缩比先 `m=1`。Indexer 按小投影估算，整模约 +1M → ≈118M（纸面）。

---

## 阶段 4：CED 切分（主线对齐）

```text
Encoder L0–L3          ——CED投影——→          Decoder L4–L7
L0 SWA-only；L1 Full；L2–3 Reuse              L4 Full；L5 Reindex；L6–7 Reuse
全局 KV：Encoder 末投影 → Decoder
```

| 层 | 半边 | 模式 | 局部 | MoE |
|----|------|------|------|-----|
| L0 | Encoder | SWA-only（无全局） | 128 | 1+8 top-2 |
| L1 | Encoder | CSA2 Full | 128 | 同左 |
| L2 | Encoder | CSA2 Reuse | 128 | 同左 |
| L3 | Encoder | CSA2 Reuse | 128 | 写完全局 KV |
| CED | 投影 | H_enc→K_dec,V_dec | — | 约 2×d×d ≈ 0.5M |
| L4 | Decoder | CSA2 Full | 128 | 1+8 top-2 |
| L5 | Decoder | CSA2 Reindex | 128 | 同左 |
| L6 | Decoder | CSA2 Reuse | 128 | 同左 |
| L7 | Decoder | CSA2 Reuse | 128 | → LM Head |

| 项 | 值 |
|----|-----|
| 总参数（纸面） | ≈118–119M |
| Prefill 激活叙事 | 主要跑 Encoder 4 层 + 投影（更接近「半边」） |
| Decode 激活叙事 | Encoder 路径更新当前 token + Decoder 4 层 |

---

## 阶段 5–6：HSI / Replay / 后置

```text
Dec Full → Pool≤2048 → Reindex
Replay 末尾 128 token 的 SWA

后置（默认不开）：mHC | Engram | DSpark | FP4/视觉
```

| 项 | 定值 | 对参数量 |
|----|------|----------|
| Candidate Pool | 上限 2048 位置（可调） | 几乎不增权重 |
| Replay | 重放末尾 `n_win=128` | 不增权重 |
| mHC / Engram / DSpark / FP4 / 视觉 | 默认关 | 开启后另开规格表 |

阶段 4 完成即可称结构主线对齐；5 是长上下文配方；6 不挡发布。

---

## 验收清单

- [ ] 代码里打印 `numel`，与上表（按实际 V 换算后）误差 &lt; 5%
- [ ] 每阶段 git tag：`stage-0-dense` … `stage-4-ced`
- [ ] 跑完本阶段门槛：Valid PPL + Gen-Smoke，写入 `eval/reports/`
- [ ] 阶段 1 起打印专家负载直方图
- [ ] 阶段 3 起打印每层模式与 Top-K 是否复用
- [ ] SFT 后跑 C-Eval valid + CMMLU（lm-evaluation-harness）

若要改大：优先 `d=640`（Dense≈80M 纸面，MoE≈172M），层表模式比例保持不变。

---

## 本仓库进度速览

| Stage | 内容 | 状态 |
|-------|------|------|
| 0 | Dense Decoder-only + train/generate/Web UI | **进行中 / 已有权重** |
| 1 | MoE FFN（MiniSeek-M0.1） | 实现中 / `model/moe.py` |
| 2 | SWA | TODO |
| 3 | CSA2 | TODO |
| 4 | CED | TODO |
| 5 | HSI / Replay | TODO |
| 6 | Engram / mHC / DSpark / FP4 / 视觉 | 后置，默认关 |

M0.1 MoE 已实现（`ModelConfig(arch="moe")` + `MoELM`）；SWA/CSA2/CED 仍为钩子，未实现前不要混进对比实验。
