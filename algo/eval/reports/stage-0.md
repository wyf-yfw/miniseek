# Stage 0 Report — Dense Decoder-only

## 配置

- d_model=512, n_layers=8, n_heads=8, head_dim=64
- FFN intermediate=2048
- vocab_size=（填 MiniMind tokenizer 实测，默认 6400）
- params=（训练日志 / `DenseLM` init 打印）
- dtype=BF16 (CUDA) / fp32 (CPU)
- data=`pretrain_t2t_mini.jsonl` + 固定 `valid.jsonl` seed=42
- steps / lr / batch=（与后续阶段保持一致后填入）

## Eval-Core

| 指标 | 结果 | 备注 |
|------|------|------|
| Valid PPL | | `eval/valid_ppl.py` |
| Gen-Smoke seed=42 | | `eval/reports/gen_smoke_stage0.jsonl` |
| C-Eval valid | — | Stage 0 可后补 |
| CMMLU | — | Stage 0 可后补 |
| Peak VRAM | | |
| tokens/s | | |

## 结论

- （训练稳定性、loss 曲线、明显失败 case）

## 下一步

- Stage 1：MoE（1 shared + 8 routed, top-2, intermediate=1024）
