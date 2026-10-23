# DeepSeek-V4.1-Flash 对齐：超小可训模型

## 目标

分阶段从 Dense → MoE（混合专家）→ SWA → CSA2 → CED（因果编码器-解码器）→ HSI/Replay，
最终结构对齐 DeepSeek-V4.1-Flash（超小体量，单 3090）。

## 冻结规格

- d_model=512, n_layers=8, n_heads=8, head_dim=64
- Dense FFN intermediate=2048
- MoE（后续）: 1 shared + 8 routed, top-2, intermediate=1024
- SWA n_win=128; CSA2 Top-K=128, m=1 first
- Stage 4: Encoder L0–L3 + Decoder L4–L7 + CED projection
- BF16（CUDA）, MiniMind tokenizer

## Tokenizer / 参数量

- **实测** MiniMind 自定义 `minimind_tokenizer` 词表 = **6400**
  （旧文档中的 ~32000 来自曾用过的 mistral_tokenizer，已废弃）
- Stage 0 Dense + vocab=6400 + tie embedding ≈ **~26–32M**（init 时打印精确值）
- 若强制 vocab=32000（非 MiniMind 默认），参数量可接近早期估算的 ~58M；
  本仓库以 MiniMind 实测词表为准，便于直接吃 `minimind_dataset`

## 数据（阶段 0–5 锁死对比）

- HF: `jingyaogong/minimind_dataset`
- Train: `pretrain_t2t_mini.jsonl`
- Valid: fixed 0.5% split `valid.jsonl`（seed=42，一次划定）
- Same hparams/steps across stages 0–5
- SFT `sft_t2t_mini` 仅在架构冻结后

## Eval-Core（每阶段必跑同一套）

1. Valid PPL
2. Gen-Smoke 20 prompts seed=42
3. C-Eval valid (lm-eval)
4. CMMLU (lm-eval)
5. Structure probes + peak VRAM / tokens/s

记录：`eval/reports/stage-N.md` + `leaderboard.md`

## 阶段路线

| Stage | 内容 | 状态 |
|-------|------|------|
| 0 | Dense Decoder-only scaffold + train/generate/eval stubs | **本仓库** |
| 1 | MoE FFN | TODO |
| 2 | SWA | TODO |
| 3 | CSA2 | TODO |
| 4 | CED | TODO |
| 5 | HSI / Replay | TODO |

## 本仓库 = Stage 0

Scaffold Dense Decoder-only train/generate，README，数据集下载说明，eval harness 占位。
MoE/SWA/CSA2/CED 仅在 `ModelConfig` 与 block 注释中留扩展钩子，**不实现**。
