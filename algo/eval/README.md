# Eval-Core（阶段 0–5 锁死同一套）

每阶段训练结束后，**必须**跑完下列指标，并写入 `eval/reports/stage-N.md` + 更新 `leaderboard.md`。

| # | 指标 | 脚本 / 工具 | 备注 |
|---|------|-------------|------|
| 1 | Valid PPL | `python eval/valid_ppl.py --ckpt ...` | 固定 `data/valid_mix.jsonl` |
| 2 | Gen-Smoke | `python eval/gen_smoke.py --seed 42` | 20 条中文 prompt |
| 3 | C-Eval valid | lm-evaluation-harness | 见下方 |
| 4 | CMMLU | lm-evaluation-harness | 见下方 |
| 5 | Efficiency | 训练日志 / `nvidia-smi` | peak VRAM、tokens/s |

超参与训练步数在阶段 0–5 保持一致，便于 Dense → MoE → SWA → CSA2 → CED 公平 A/B。

## C-Eval / CMMLU / GSM8K

安装 [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)：

```bash
pip install lm-eval
# 需要先把本仓库 DenseLM 包成 HuggingFace-compatible 或写自定义 model wrapper
# Stage 0 骨架仅预留接口，正式 wrapper 在有可用 ckpt 后补齐。
```

示例（占位，待 HF wrapper）：

```bash
lm_eval --model hf \
  --model_args pretrained=./export/stage0_hf \
  --tasks ceval-valid,cmmlu \
  --batch_size 4 \
  --output_path eval/reports/lm_eval_stage0
```

## Structure probes（Stage 1+）

- MoE（混合专家）：路由负载、专家激活占比
- SWA / CSA2：有效感受野、稀疏命中率
- CED（因果编码器-解码器）：encoder/decoder 梯度与表示探针


## Web / CLI（一期默认套件）

默认任务：`ceval-valid`, `cmmlu`, `gsm8k`（Agent / MATH 二期）。

```bash
pip install 'lm-eval>=0.4.5'
python -m eval.run_benchmark \
  --ckpt tasks/<id>/checkpoints/best.pt \
  --tasks ceval-valid,cmmlu,gsm8k \
  --out eval/runs/manual-1 \
  --device cuda \
  --limit 8
```

Web「评测」Tab 调用同一 CLI；跑次落在 `eval/runs/<id>/`（`run.json` + `results.json` + `run.log`）。
