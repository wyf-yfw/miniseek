# MiniSeek 训练页实时专家负载设计

Date: 2026-10-25  
Status: Approved  
Repo: `wyf-yfw/miniseek`  
Machine: wyf（本地实现；不用 Cloud Agent）  
Depends on: `docs/superpowers/specs/2026-10-25-miniseek-m0.1-moe-design.md`；已有 `MoELM.begin_expert_usage` / `end_expert_usage` / `probe_expert_load` 与「专家负载」验证探针页。

## Goal

训练 **MiniSeek-M0.1（MoE）** 时，在 **训练页** 与 loss 同屏看到 **32 路路由专家** 在当前 log 窗口内的 top-k 分配占比，并随 **现有训练日志刷新节奏** 更新（与 `--log-every` 同步），而不是另开仅离线探针的页签才可见。

## Non-goals

- 不改 MoE 架构（仍 1 shared + 32 routed、top-2、expert intermediate=512）。
- 不改 AUX 默认：`λ=0.01`（`--aux-coef`）。
- 不每 step 单独落盘/推 SSE 负载曲线（只在写 log 时汇总并推送）。
- 不替换或删除「专家负载」页的 **验证集离线探针**（`POST /api/moe/expert-load` 保留）。
- 不在本设计内做 SFT-MoE；若日后 SFT 接 MoE，可复用同一 log schema。
- 不强制中断当前正在跑的训练进程；旧进程无实时字段，需 **重启或断点续训** 后才有 live 负载。

## UX placement

| 位置 | 行为 |
|------|------|
| 训练页 · Loss 卡片下方（同列） | 新增「专家负载（实时）」：与验证探针同风格的 32 柱条图 + 一行摘要（max/min/熵可选） |
| 训练页 · Dense | 整块隐藏或固定文案 N/A（不占柱图） |
| 「专家负载」页签 | 不变：手动选 ckpt + valid 探针 |

刷新：每条训练 log 窗口结束时刷一次柱图（与 log 同步）；loss 曲线仍按现有 `plot_every` 降采样，二者节奏独立。

## Data flow

```text
train_pretrain.py (arch=moe)
  ├─ 窗口开始: MoELM.begin_expert_usage()
  ├─ 每 step forward: DeepSeekMoE 内 _record_usage 累加 top-k 槽位计数（已有）
  ├─ 写 log 时 (step==1 或 step % log_every == 0):
  │     counts = end_expert_usage()   # shape (n_routed,), 跨层合计
  │     fractions = counts / sum(counts)
  │     stdout: 原有 step/loss/ce/aux/ppl/lr 行（不变）
  │     stdout: expert_load=<JSON>     # 新增一行
  │     begin_expert_usage()           # 下一窗口
  └─ Dense: 不调用 usage API，不打印 expert_load 行

train_web.log / 进程 stdout
  └─ web_ui.TrainManager._emit_log
        ├─ 仍解析 STEP_RE → status + 可选 loss SSE
        └─ 若行匹配 expert_load=JSON → 校验后 fanout SSE event "expert_load"

浏览器 EventSource /api/logs/stream
  └─ 训练页监听 "expert_load" → 更新柱图；"reset" 时清空
```

`torch.compile` 时通过 `getattr(model, "_orig_mod", model)` 取底层 `MoELM` 再调 `begin/end_expert_usage`。

## Log schema

独立一行（便于解析、不破坏现有 `STEP_RE`）：

```text
expert_load={"step":1234,"window_steps":10,"n_routed":32,"top_k":2,"n_layers":8,"n_shared":1,"counts":[...32 ints...],"fractions":[...32 floats...],"total_assignments":N,"max_frac":0.05,"min_frac":0.02,"max_expert":7,"min_expert":3}
```

| 字段 | 含义 |
|------|------|
| `step` | 本窗口结束时的 global_step |
| `window_steps` | 本窗口内优化步数（通常 ≈ `log_every`；首条 log 可能为 1） |
| `n_routed` / `top_k` / `n_layers` / `n_shared` | 与 cfg 一致 |
| `counts` | 各路由专家 top-k 槽位分配次数，**跨层求和**（与 `end_expert_usage` / 离线探针同口径） |
| `fractions` | `counts[i] / sum(counts)`；总和为 0 时全 0 |
| `total_assignments` | `sum(counts)` |
| `max_frac` / `min_frac` / `max_expert` / `min_expert` | 便于 UI 摘要与 hot/cold 高亮 |

约定：

- 前缀必须为字面量 `expert_load=`，后接单行 JSON（UTF-8，无换行）。
- shared 专家不计入（与探针页文案一致）。
- Dense 训练：无此行。
- 熵（bits）由 UI 或 web_ui 可选计算；log 内不强制带 `entropy_bits`，避免训练端多依赖。

## Web UI / SSE

1. `web_ui.py`：识别 `expert_load=` 行 → `json.loads` → 基本校验（`n_routed` 与 `len(fractions)` 一致）→ `_fanout(..., ("expert_load", payload))`；缓存 `latest_expert_load` 供订阅瞬间的 snapshot（若现有 hello/status 已带 loss 列表，则一并带上最近一次 expert_load，没有则为 `null`）。
2. `web/index.html` 训练页：
   - Loss canvas 下增加 `#live-expert-load` 区域（标题、caption、bars 容器）。
   - `trainArch === "moe"` 或收到首条 `expert_load` 时显示；Dense 隐藏。
   - 渲染复用探针页柱条样式（可抽小函数，避免两套视觉）。
   - `reset` / 清空日志时清空实时柱图。
3. 静态页与 `web_ui` 解析逻辑更新后，**重启 web_ui 进程**即可；**不必**停当前训练。当前已在跑的 `train_pretrain.py` 不会冒出 `expert_load=` 行，页面保持空/提示「需续训或重启训练后生效」。

## Dense N/A

- 训练脚本：不 begin/end，不打印。
- UI：架构选择为 Dense 时隐藏实时块；若误收到行则忽略或显示 N/A。
- 离线探针页对 Dense ckpt 仍返回现有 `ok: false` + N/A message。

## Restart / resume note

| 组件 | 是否要重启 |
|------|------------|
| `web_ui.py` + `web/index.html` | 要（否则无解析/无柱图） |
| 正在跑的 MoE train | **不强制停**；要看到实时负载需之后 **停掉再开或 `--resume`** 加载含本改动的 `train_pretrain.py` |
| AUX / 超参 | 不变；续训继续传 `--aux-coef 0.01` |

## Testing smoke

1. **单元**：`begin_expert_usage` → 若干 `forward` → `end_expert_usage`，`sum(counts) == n_layers * B * T * top_k`（与现有 `test_probe_expert_load_counts_sum` 同口径）；可选断言一行 `expert_load=` JSON 可被 `json.loads` 且长度 32。
2. **训练烟测**：`python train_pretrain.py --arch miniseek-moe --max-steps 2 --log-every 1 ...`，stdout 在 step 行后出现 `expert_load={`，且 `len(fractions)==32`。
3. **Dense 烟测**：`--arch dense --max-steps 2`，stdout **无** `expert_load=`。
4. **UI**：启动/重启 `web_ui.py`，训练页 MoE 跑烟测或 attach 带新日志的进程，Loss 下柱图随 log 更新；切 Dense 表单时实时块隐藏；「专家负载」页探针仍可用。
5. **回归**：现有 MoE/step 日志的 loss SSE 解析不被 `expert_load` 行打断（STEP_RE 仍只匹配 step 行）。

## Implementation touchpoints

| 文件 | 改动 |
|------|------|
| `train_pretrain.py` | MoE 窗口 begin/end；log 时打印 `expert_load=`；compile 取 `_orig_mod` |
| `web_ui.py` | 解析该行 → SSE `expert_load`；可选 snapshot 字段 |
| `web/index.html` | 训练页实时柱图 + SSE 监听 + Dense 隐藏 |
| `tests/test_moe.py`（或 train 小测） | 窗口计数 / JSON 行烟测 |
| 本 spec | 已批准；实现后勿改口径除非另开设计 |

## Acceptance

- MoE 训练 log 按 `log_every` 冒出合法 `expert_load=` 行；Dense 没有。
- 训练页 Loss 旁/下实时柱图与 log 同步；Dense N/A/隐藏。
- 验证探针页与 AUX=0.01 行为保持。
- 不依赖 Cloud Agent；改动仅在 wyf 本地仓库提交。
