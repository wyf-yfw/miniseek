# 训练 train/valid 双曲线设计

日期：2026-10-27
状态：已批准（对话确认）

## 目标

训练过程中同时显示 **train loss** 与 **valid loss/ppl** 曲线，便于判断平台 vs 过拟合。不做独立「验证」页；不绑训后单点工具（可后续另做）。

## 形态

- 训练任务表单：`valid_path`（默认 `data/valid_mix.jsonl`，可改）、`valid_every`（每隔 N step 跑一次 valid；默认与现有 plot 节奏对齐，建议 500，UI 可改）。
- 任务详情损失图：**两条线** train / valid。
- metrics：`loss[]` 保持；新增 `valid_loss[]`（每项含 `step`, `loss`, `ppl`, 可选 `tokens`）。

## 实现要点

1. **`algo/train_pretrain.py`**（SFT 若同构则同步）：CLI `--valid` / `--valid-every` / `--valid-max-samples`；每 N step 在 `torch.no_grad()` 下对 valid 子集算 mean NLL → ppl；打日志并 append 到 metrics 文件（与现有 train metrics 写入路径一致，经 task `out_dir`/runner 约定）。
2. **Dense 与 MoE**：用 `build_model` 加载，禁止写死 DenseLM。
3. **`api/task_runner.py`**：把表单 hparams 传进训练命令。
4. **`frontend/index.html`**：表单字段 + 损失图双序列（颜色区分，图例 train/valid）。
5. **设备**：valid 与训练同进程同卡短跑，不另起进程；训练占满时接受短暂停顿。

## 非目标

- 侧栏独立验证页 / 训后多 ckpt 曲线
- 解码惩罚
- 改动现有三服务拓扑

## 验收

- 新开 tiny/短跑任务，metrics 出现 `valid_loss` 点，UI 可见双线。
- MoE ckpt 路径可跑 valid（build_model）。
- `valid_every=0` 或空路径时行为与现在一致（只 train）。
- pytest 覆盖 metrics 写入与 CLI 传参（可 mock 小 valid）。
