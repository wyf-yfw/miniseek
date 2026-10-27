# 训练执行页实时指标（显存 / 已跑时长 / step/s）

Date: 2026-10-27  
Status: Approved（用户选 SSE）  
Repo: `wyf-yfw/miniseek`  
Machine: wyf（本地实现；不用 Cloud Agent）  
Depends on: Task 驱动 Web UI（`/api/tasks/{id}/logs` SSE、`status-box`、`metrics.json`）

## Goal

训练 **执行页** 在现有状态区实时显示三样指标，刷新/关页再开仍可看：

- **显存**（VRAM）
- **已跑时长**（elapsed）
- **速度**（step/s）

实现方式：扩展现有 SSE `GET /api/tasks/{id}/logs` 的 `status` 事件，不新开轮询、不改训练脚本日志。

## Non-goals

- 不改 `train_pretrain.py` / SFT 脚本的 stdout 格式或额外指标行。
- 不新增前端定时 `fetch('/api/status')` 或其他轮询。
- 不做进程隔离 / 防误杀（训练已可后台跑；本需求 **仅展示**）。
- 不改 Loss 曲线、专家负载、checkpoint 逻辑。
- 不要求 `metrics.json` 的 loss 点带 wall-clock 时间戳（本轮不扩 schema）。

## Context（已确认）

- 训练是 Web UI 拉起的子进程，**不依赖页面常开**；刷新不会调 `/stop`。
- 执行页 `status-box` 目前只有 id/status/arch/pid/时间戳/error；显存只在顶栏环境 tip（`_gpu_text()`）；没有 elapsed / step/s。
- SSE 循环：`asyncio.sleep(0.4)`，无新日志时 `idle % 10 == 0` 发 `: ping` + `status` → **约每 4s** 一次，与本设计推送节奏一致。

## UX

在执行页 `#status-box` 现有多行元数据 **下方**（或末行）增加一行可读指标，例如：

```text
显存 NVIDIA … | 显存 12.3/24.0 GB | 已跑 1h23m45s | step 83349 · 1.24 step/s
```

行为：

| 状态 | 行为 |
|------|------|
| `running` | 随 SSE `status` 更新（约 4s） |
| 非 running（stopped / finished / failed） | 最后一次 `status` 定格；不再追逐墙钟 |
| 无 loss 点 / 刚启动 | `step` 显示 `—` 或 `0`；`step/s` 显示 `—` 直至可估 |
| 无 GPU | `gpu` 文案为现有 `_gpu_text()` 的「无 GPU」 |

时长格式：优先 `HhMmSs`（如 `1h05m03s`）；不足 1h 可省略小时（`12m03s` / `45s`）。前端或后端格式化均可，推荐 **后端给 `elapsed_sec`，前端格式化**，便于测试。

## Data flow

```text
GET /api/tasks/{task_id}/logs  (已有 EventSource)
  ├─ 初始 / idle ping(~4s):
  │     yield status {
  │       id, status, pid,           # 已有
  │       gpu,                       # 新增：_gpu_text()
  │       started_at,                # 新增：task.json
  │       elapsed_sec,               # 新增：见下
  │       step,                      # 新增：metrics.loss 最新点
  │       step_s                     # 新增：见下
  │     }
  └─ 浏览器 status 监听
        └─ 更新 status-box 指标行；running 时刷新，结束后定格
```

不新增 SSE event 名；只扩 `status` payload。`metrics` / `log` 事件不变。

## Backend 字段定义

在 `web_ui.py` 的 `api_task_logs` 里，构造 `status` 的两处（初始 yield 与 idle ping）共用同一 helper，例如 `_task_live_status(task_id) -> dict`。

| 字段 | 类型 | 含义 |
|------|------|------|
| `gpu` | `str` | 直接调用现有 `_gpu_text()`（设备名 + 已用/总量 GB，或「无 GPU」） |
| `started_at` | `str \| null` | `task.json` 的 `started_at`（ISO8601） |
| `elapsed_sec` | `float \| null` | running：`now - started_at`；已结束：`ended_at - started_at`（缺一端则为 null） |
| `step` | `int \| null` | `ts.read_metrics(task_id)["loss"]` 最后一点的 `step`；无点则 null |
| `step_s` | `float \| null` | 见下；无法估计时 null |

### `step_s` 估法

当前 `metrics.json` loss 点 **无时间戳**（仅 `step/loss/ppl/lr`），故：

1. **优先（滑动）**：SSE 生成器内记住上次发出 status 时的 `(step, monotonic_time)`；当本次 `step` 更大时，`step_s = Δstep / Δt`。
2. **回退**：若尚无上次快照或 `Δt` 过小，且 `elapsed_sec > 0` 且 `step` 已知：`step_s = step / elapsed_sec`（从 `started_at` 起的平均速度）。
3. 否则 `step_s = null`。

不把 wall time 写入 `metrics.json`（本轮非目标）。

### 推送节奏

沿用现有 idle 分支：`sleep(0.4)` × `idle % 10 == 0` ≈ **4s**。有新日志时仍会推 `metrics`；`status` 不必在每条 log 上强推，idle ping 足够；若实现方便，也可在读到新 metrics 后顺带发一次 enriched `status`（可选，非必须）。

## Frontend

文件：`web/index.html`（`renderStatusBox` / `connectTaskLogs` 的 `status` 监听）。

- 解析 `status` 事件里的 `gpu` / `elapsed_sec` / `step` / `step_s`，拼指标行写入 `#status-box`（可追加一行，勿抹掉现有 id/status/pid 块）。
- `status` 监听里现有 `selectTask` + `refreshTasks` 可保留；指标行优先用 SSE payload 直接画，避免多一次 GET 才刷新。
- 任务结束后保留最后一帧数字；不要用客户端 `Date.now()` 继续累加 elapsed。
- `step_s` 展示约两位小数；null 时显示 `—`。

## Out of scope reminder

训练进程后台运行、与 UI 生命周期解耦，**已是现状**；本次只补展示，不改 Start/Stop/子进程策略。

## Acceptance

1. 开跑任一 task，执行页约 4s 内出现显存文案、已跑时长递增、在有 loss 点后出现 step 与 step/s。
2. 手机刷新或关掉再打开同一 running task：指标继续更新（SSE 重连即可）。
3. Stop / 自然结束后：时长与 step/s **定格**，与 `ended_at` 一致口径。
4. `train_pretrain` 输出与 `metrics.json` schema 无变更；网络面板无新增周期性 XHR 轮询（仅 EventSource）。

## Implementation notes（实现时，非本提交）

- 改动面：`web_ui.py`（status payload + helper）、`web/index.html`（status-box 一行）。
- 单测/自测：用 running task 看 SSE；无 GPU 机器应显示「无 GPU」且不抛错。
