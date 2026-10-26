<div align="center">

# MiniSeek

**由简入难，对齐 DeepSeek-V4.1-Flash 结构思路的超小可训语言模型**

[![GitHub Repo stars](https://img.shields.io/github/stars/wyf-yfw/miniseek?style=social)](https://github.com/wyf-yfw/miniseek/stargazers)
[![GitHub Code License](https://img.shields.io/github/license/wyf-yfw/miniseek)](LICENSE)
[![GitHub last commit](https://img.shields.io/github/last-commit/wyf-yfw/miniseek)](https://github.com/wyf-yfw/miniseek/commits/main)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-blue)](https://github.com/wyf-yfw/miniseek/pulls)
[![Collection](https://img.shields.io/badge/🤗-MiniSeek%20Collection-yellow)](https://huggingface.co/collections/wyf-yfw/miniseek-6adc765159d38ab3236c49af)

中文 | [English](./README_en.md)

</div>

- 从 **Dense 骨架**出发，按阶段逐步接入 MoE / SWA / CSA2 / CED 等结构（见 [`docs/STAGE_PLAN.md`](docs/STAGE_PLAN.md)），用**同一套数据与评测**做公平 A/B。
- Stage 0 主线：`d_model=512`、8 层 Dense Decoder-only，MiniMind 词表 **vocab=6400**，tied 权重约 **28.45M**，单卡 3090 可训。
- 训练数据与分词对齐 [MiniMind](https://github.com/jingyaogong/minimind) 生态（`jingyaogong/minimind_dataset`），核心模型用 **PyTorch 原生**实现。
- 自带 **Web UI**：训练 Task、对话、续写、架构可视化、C-Eval / CMMLU / GSM8K 评测（lm-eval）。
- 权重已开源到 Hugging Face：[`MiniSeek-D0.1-base`](https://huggingface.co/wyf-yfw/MiniSeek-D0.1-base)（预训练）、[`MiniSeek-D0.1`](https://huggingface.co/wyf-yfw/MiniSeek-D0.1)（SFT）。

> 本仓库当前实现重点是 **Stage 0 Dense + 训练/评测/Web 全链路**；后续阶段在规格里已写明，代码侧多为钩子，尚未落地。

---

# 📌 项目介绍

个人 GPU 很难「从零」摸清大模型训练全流程：框架封装太厚，真实模型又太大。MiniSeek 想做一件更窄但可复现的事——**把结构升级拆成可对比的阶段**，每阶段只改架构，数据和 Eval-Core 全程锁死，从而看清「加 MoE / SWA / CSA2 / CED 到底换来了什么」。

#### 本项目当前包含

- Dense Decoder-only（`DenseLM`）完整训练与生成代码（RoPE、RMSNorm、GELU FFN，默认 embedding / LM head tying）
- 预训练脚本 `train_pretrain.py`、生成 `generate.py`、数据下载与 valid 划分
- Eval-Core：Valid PPL、Gen-Smoke、以及 Web/CLI 上的 C-Eval / CMMLU / GSM8K（lm-evaluation-harness）
- FastAPI Web UI（`:7860`）：训练 Task、对话、续写、架构图（简洁 / 标准 / 详细）、评测 Task
- 阶段规格文档 [`docs/STAGE_PLAN.md`](docs/STAGE_PLAN.md) 与冒烟测试

#### 已发布模型

| 模型 | 角色 | Hub 参数量 | Release |
|------|------|------------|---------|
| [MiniSeek-D0.1-base](https://huggingface.co/wyf-yfw/MiniSeek-D0.1-base) | Dense 预训练基座 | **28.45M**（tied） | 2026-10 |
| [MiniSeek-D0.1](https://huggingface.co/wyf-yfw/MiniSeek-D0.1) | SFT / 对话 | **31.73M**（untied emb+head） | 2026-10 |

权重格式：`model.safetensors` + `config.json` + MiniMind tokenizer。自定义 `DenseLM`，暂非 `AutoModelForCausalLM` 即插即用。

---

# 📌 快速开始

## 第 0 步：克隆与依赖

```bash
git clone https://github.com/wyf-yfw/miniseek.git
cd miniseek
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows PowerShell
# .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

CPU 冒烟：

```bash
python -m pytest tests/test_smoke.py -q
```

## Ⅰ 🚀 推理

### 1. 下载开源权重

```bash
# 基座（预训练）
huggingface-cli download wyf-yfw/MiniSeek-D0.1-base --local-dir ./hub/MiniSeek-D0.1-base
# 或对话版
huggingface-cli download wyf-yfw/MiniSeek-D0.1 --local-dir ./hub/MiniSeek-D0.1
```

### 2. CLI 生成（本地 ckpt）

```bash
python generate.py --ckpt checkpoints/stage0/best.pt --seed 42 --prompt "你好，请介绍一下人工智能。"
```

### 3. 从 Hub 加载 DenseLM

```python
import json
from pathlib import Path
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from model.config import ModelConfig
from model.dense import DenseLM
from model.tokenizer import load_tokenizer

repo = snapshot_download("wyf-yfw/MiniSeek-D0.1-base")
cfg = ModelConfig.from_dict(json.loads(Path(repo, "config.json").read_text()))
model = DenseLM(cfg)
state = load_file(Path(repo) / "model.safetensors")
model.load_state_dict(state, strict=False)  # base: lm_head tied → tok_emb
tok = load_tokenizer(str(Path(repo) / "tokenizer"))

ids = tok.encode("你好，你是谁", add_special_tokens=True)
out = model.generate(
    torch.tensor([ids]),
    max_new_tokens=64,
    temperature=0.8,
    top_k=50,
    eos_token_id=tok.eos_token_id,
)
print(tok.decode(out[0].tolist(), skip_special_tokens=True))
```

### 4. Web UI（推荐）

```bash
python -m uvicorn web_ui:app --host 127.0.0.1 --port 7860
```

浏览器打开后可用：

| Tab | 作用 |
|-----|------|
| 训练 | Task 侧栏：配置 → Start → 日志/指标 → Stop；ckpt 在 `tasks/<id>/` |
| 对话 / 续写 | 选 checkpoint 做 chat / completion |
| 架构 | Tiny / Dense / MoE 预设；简洁 · 标准 · 详细图 + 右侧抽屉 |
| 评测 | 与训练同形态的 Bench Task：C-Eval / CMMLU / GSM8K，双进度条 |

CPU 链路冒烟：

```bash
python scripts/smoke_tiny_task.py
```

---

# 📌 数据介绍

## Ⅰ Tokenizer

使用 MiniMind 词表（**vocab=6400**）。推荐：

```bash
huggingface-cli download jingyaogong/MiniMind2 --include "tokenizer*" --local-dir ./tokenizer
# 或 export DS41_TOKENIZER_PATH=/path/to/minimind_tokenizer
```

无本地文件时，代码会回退到 byte-level `FallbackTokenizer`（仅 smoke，不可正式训练）。

## Ⅱ 预训练 / Valid

```bash
python scripts/download_data.py            # 说明 / 划分 valid
python scripts/download_data.py --download # 拉取 pretrain_t2t_mini.jsonl
```

| 锁定项 | 内容 |
|--------|------|
| 源 | [`jingyaogong/minimind_dataset`](https://huggingface.co/datasets/jingyaogong/minimind_dataset) |
| Train | `pretrain_t2t_mini.jsonl` |
| Valid | `valid.jsonl` = train 的 **0.5%**，**seed=42**（阶段 0–5 锁死） |
| SFT | `sft_t2t_mini.jsonl`（架构冻结后的对话线；不参与结构对比） |

对比原则：**换结构，不换数据**。完整 10GB/14GB 版留作冻结后加强，不进阶段表。

---

# 📌 模型

## 结构（Stage 0）

```text
Token Embedding (V=6400 × d=512)
        ↓
×8 Transformer Block
  RMSNorm → Causal Attn (RoPE) → Residual
  RMSNorm → Dense FFN (intermediate=2048) → Residual
        ↓
RMSNorm → LM Head（基座默认与 Embedding tying）
```

| 项 | 定稿值 |
|----|--------|
| `d_model` / `n_layers` / `n_heads` / `head_dim` | 512 / 8 / 8 / 64 |
| Dense FFN | 2048 (=4d) |
| `max_seq_len`（训练起步） | 512 |
| 精度 / 硬件目标 | BF16 · 单卡 3090 |

## 阶段路线

| Stage | 结构 | 本仓 |
|-------|------|------|
| 0 | Dense Decoder-only | ✅ |
| 1 | + MoE（1 shared + 8 routed, top-2） | TODO |
| 2 | + SWA | TODO |
| 3 | + CSA2 | TODO |
| 4 | + CED | TODO |
| 5 | + HSI / Replay | TODO |

细节与参数量表：[`docs/STAGE_PLAN.md`](docs/STAGE_PLAN.md)。

命名（DeepSeek 风格，Hub 一仓一权重）：

- `MiniSeek-D{x.y}-base` — Dense 预训练
- `MiniSeek-D{x.y}` — Dense SFT / chat
- `MiniSeek-M{x.y}-*` — 后续 MoE 线

---

# 📌 训练

```bash
# 正式预训练
python train_pretrain.py --data data/pretrain_t2t_mini.jsonl --out-dir checkpoints/stage0

# 短跑 / 调参
python train_pretrain.py --data data/pretrain_t2t_mini.jsonl --out-dir checkpoints/stage0 \
  --max-steps 5000 --warmup-steps 200 --lr 1e-4 \
  --save-loss-thresholds 20,15,12,10,9,8,7 --stable-steps 200
```

### Checkpoint 策略

- **里程碑** `loss-*.pt`：`--save-loss-thresholds` 某一档需连续 `--stable-steps`（默认 200）步 `loss<=阈值` 才写入
- **best.pt**：历史最低 train loss
- **last.pt**：跑结束写一次
- **`--save-every`**：默认关；需要按步备份再开

也可在 Web UI「训练」Tab 里用 Task 形态启停，产物在 `tasks/<id>/`。

---

# 📌 评估

**Eval-Core**（各阶段同一套）：

| # | 项目 | 说明 |
|---|------|------|
| 1 | Valid PPL | 同一 `valid.jsonl` |
| 2 | Gen-Smoke | 20 条固定 prompt，seed=42 |
| 3 | C-Eval valid | lm-eval |
| 4 | CMMLU | lm-eval（需 `trust_remote_code`；`datasets>=2.14,<4.0`） |
| 5 | GSM8K | lm-eval generate_until |
| 6 | 效率 | 同硬件记峰值显存 / tokens/s |

CLI 示例：

```bash
python eval/valid_ppl.py --ckpt checkpoints/stage0/last.pt --valid data/valid.jsonl
python eval/gen_smoke.py --ckpt checkpoints/stage0/last.pt --seed 42
# 或 Web「评测」Tab / eval/run_benchmark.py
```

报告模板：`eval/reports/stage-0.md`、`eval/reports/leaderboard.md`。更多见 [`eval/README.md`](eval/README.md)。

锚点参考：MiniMind-3（~64M）C-Eval≈24.9 / CMMLU≈25.4——Stage 0 体量更小，重点看**相对变化**而非绝对分数。

---

# 📌 目录

```text
model/              # config · DenseLM · tokenizer
train_pretrain.py   # 预训练入口
generate.py         # CLI 生成
web_ui.py + web/    # FastAPI UI · Task/Bench/架构
eval/               # PPL · Gen-Smoke · lm-eval · reports
scripts/            # 数据下载 · smoke
docs/STAGE_PLAN.md  # 阶段规格（主文档）
tests/              # 冒烟与单测
```

---

# 📌 致谢

- 数据与词表生态：[jingyaogong/minimind](https://github.com/jingyaogong/minimind)、[`minimind_dataset`](https://huggingface.co/datasets/jingyaogong/minimind_dataset)
- 结构对齐参考：[DeepSeek-V4.1 相关论文](https://arxiv.org/html/2609.19969)
- 客观评测：[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)、C-Eval、CMMLU

欢迎 Issue / PR。

---

# 🎓 引用

若本项目对你有帮助，欢迎引用仓库或在论文/博客中注明链接：

```bibtex
@misc{miniseek2026,
  title        = {MiniSeek: Tiny Trainable LMs Toward DeepSeek-V4.1-Flash Structures},
  author       = {wyf-yfw},
  year         = {2026},
  howpublished = {\url{https://github.com/wyf-yfw/miniseek}}
}
```

---

# ⚖️ 开源协议

[MIT](LICENSE)
