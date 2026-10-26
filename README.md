# miniseek

**中文为主 / English summary below.**

对齐 DeepSeek-V4.1-Flash 思路的超小可训语言模型（MiniMind 风格数据与分词器），单卡 3090 可跑。  
**当前仓库 = Stage 0：Dense Decoder-only 骨架**（后续 MoE / SWA / CSA2 / CED 仅钩子，未实现）。

## English (short)

Ultra-small trainable LM scaffold aligned with DeepSeek-V4.1-Flash ideas. Stage 0 = dense decoder-only (d=512, L=8, H=8, FFN=2048, MiniMind tokenizer vocab=**6400**, ~26–32M params). Later stages (MoE/SWA/CSA2/CED) are TODOs only. Data: `jingyaogong/minimind_dataset`. Eval-Core locked for fair A/B across stages 0–5.

## 目标与阶段路线

详见 [`docs/STAGE_PLAN.md`](docs/STAGE_PLAN.md)。

| Stage | 结构 | 本仓 |
|-------|------|------|
| 0 | Dense Decoder-only | ✅ |
| 1 | + MoE（混合专家） | TODO |
| 2 | + SWA（滑动窗口注意力） | TODO |
| 3 | + CSA2 | TODO |
| 4 | + CED（因果编码器-解码器） | TODO |
| 5 | + HSI / Replay | TODO |

### 冻结规格（Stage 0）

- `d_model=512`, `n_layers=8`, `n_heads=8`, `head_dim=64`
- Dense FFN `intermediate=2048`
- RoPE + RMSNorm + GELU FFN，词嵌入与 lm_head 默认 weight tying
- Tokenizer：**MiniMind 实测 vocab=6400**（不是旧 mistral ~32k）
- 参数量：init 时打印；6400 词表下约 **26–32M**（早期 ~58M 估算对应 ~32k 词表）

## Web UI（Task 驱动）

```bash
.venv/bin/python -m uvicorn web_ui:app --host 127.0.0.1 --port 7860
```

打开浏览器后：左侧 **新建 Task** → 选 **Tiny** → Start。每个 task 的配置/日志/指标/ckpt 在 `tasks/<id>/`（已 gitignore）。同时最多一个 running。对话页默认列出当前 task 的 checkpoint。

CPU 冒烟整条链路：

```bash
.venv/bin/python scripts/smoke_tiny_task.py
```

## 快速开始

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# CPU smoke
python -m pytest tests/test_smoke.py -q

# 预训练 smoke（无数据时自动写占位 jsonl）
python train_pretrain.py --max-steps 2

# 生成（固定 seed）
python generate.py --seed 42 --prompt "你好，请介绍一下人工智能。"
```

## 数据下载

```bash
# 只打印说明 / 对已有 train 做 valid 划分
python scripts/download_data.py

# 真正从 HF 拉 pretrain_t2t_mini.jsonl（体积可能很大，按需）
python scripts/download_data.py --download
```

- HF：[`jingyaogong/minimind_dataset`](https://huggingface.co/datasets/jingyaogong/minimind_dataset)
- Train：`pretrain_t2t_mini.jsonl`
- Valid：`valid.jsonl` = train 的 **0.5%**，**seed=42**（阶段 0–5 锁死）
- 阶段间保持同一超参与步数，便于公平对比

### Tokenizer

```bash
# 推荐：从 MiniMind2 拉 tokenizer 文件到 ./tokenizer
huggingface-cli download jingyaogong/MiniMind2 --include "tokenizer*" --local-dir ./tokenizer
# 或 export DS41_TOKENIZER_PATH=/path/to/minimind_tokenizer
```

无本地文件时，代码会用 byte-level `FallbackTokenizer`（仅 smoke，不可正式训练）。

## 训练 / 生成 / 评测

```bash
# 正式训（默认：loss 连续 200 步低于阈值才写 loss-*.pt；结束写 last.pt；刷新最低写 best.pt）
python train_pretrain.py --data data/pretrain_t2t_mini.jsonl --out-dir checkpoints/stage0

# 短跑 / 调参示例
python train_pretrain.py --data data/pretrain_t2t_mini.jsonl --out-dir checkpoints/stage0 \
  --max-steps 5000 --warmup-steps 200 --lr 1e-4 \
  --save-loss-thresholds 20,15,12,10,9,8,7 --stable-steps 200

python generate.py --ckpt checkpoints/stage0/last.pt --seed 42

python eval/valid_ppl.py --ckpt checkpoints/stage0/last.pt --valid data/valid.jsonl
python eval/gen_smoke.py --ckpt checkpoints/stage0/last.pt --seed 42
```

### Checkpoint 策略

- **里程碑**：`--save-loss-thresholds`（默认 `20,15,...,3`）——某一档需 **连续** `--stable-steps`（默认 200）步 `loss<=阈值` 才写 `loss-10.pt` 等；中途回升则计数清零。
- **best.pt**：新历史最低 train loss 时更新（`--no-save-best` 可关）。
- **last.pt**：跑结束写一次。
- **`--save-every`**：默认 `0`（关）；需要按步备份时可开。

C-Eval / CMMLU：见 [`eval/README.md`](eval/README.md)（lm-evaluation-harness，Stage 0 预留）。

结果模板：`eval/reports/stage-0.md`、`eval/reports/leaderboard.md`。

## 目录

```
model/           # config + DenseLM + tokenizer
train_pretrain.py
generate.py
scripts/download_data.py
eval/            # valid_ppl, gen_smoke, reports
tests/test_smoke.py
docs/STAGE_PLAN.md
```

## License

MIT
