# MiniSeek Web UI：Task 驱动重构 + Tiny 冒烟档

日期：2026-10-26  
状态：已对齐需求，待实现

## 目标

把训练从「全局单例 + 扫进程接入」改成 **task 驱动**：

- 每次启动一次训练对应一个 task
- task 元数据与产物全隔离，打开 task 即可看到状态/日志/指标/ckpt
- **不再**手动扫描 `/proc`、不再「接入选中进程」
- 同时最多 **一个** running task（可有多个历史）
- 新增架构档 **Tiny**：与 Dense 同结构（`DenseLM`）、极小超参，CPU 可快速跑通 创建→训练→ckpt→对话

## 非目标（本轮不做）

- 多卡 / 多 task 并行训练
- 把对话/续写本身也做成 task
- 迁移旧的 `train_web.json` / `train_web.log` / 散落 `checkpoints/*` 进 tasks（旧产物仍可读路径加载；新训练一律进 task 目录）
- MoE Tiny 变体
- 缩小 vocab / 另训词表

## 目录布局（真相源）

根目录：`tasks/`（相对仓库根，与 `web_ui.py` 同级）。

```
tasks/
  <task_id>/
    task.json          # 元数据 + 启动配置快照 + pid/status
    train.log          # 该次训练 stdout/stderr
    metrics.json       # loss 点、专家负载等（与 log 分离，便于画图）
    checkpoints/       # --out-dir 固定指向此处
      last.pt
      best.pt
      ...
```

`task_id`：时间戳 + 短随机，例如 `20261026-114500-a3f2`（列表展示用 `name` 字段，可读）。

### `task.json` 字段

| 字段 | 说明 |
|------|------|
| `id` | 目录名 |
| `name` | 展示名（创建时可改，默认 `arch-mode-时间`） |
| `status` | `draft` \| `running` \| `stopped` \| `finished` \| `failed` |
| `arch` | `dense` \| `moe` \| `tiny` |
| `mode` | `pretrain` \| `sft`（Tiny 默认 `pretrain`） |
| `pid` | running 时的训练进程 pid；否则 `null` |
| `created_at` / `started_at` / `ended_at` | ISO8601（本地时区写入即可） |
| `hparams` | 启动参数快照：data_path、epochs、batch、lr、plot_every、resume、init_ckpt、device、compile、thresholds 等 |
| `out_dir` | 恒为该 task 下 `checkpoints/` 的绝对或相对路径 |
| `exit_code` | 进程结束后的返回码（若有） |
| `error` | 失败摘要（可选） |

`metrics.json`：`{ "loss": [{"step", "loss", ...}], "expert_load": ... }`，由现有 log tail / 解析逻辑写入**当前 task** 文件，禁止写全局文件。

## 生命周期与并发

```
创建 → draft
  start → running   （若已有任一 task status=running → 拒绝）
  stop  → stopped
  进程自然退出 exit 0 → finished
  非 0 / 崩溃 → failed
```

- **去掉** `list_processes` / `attach` 及 UI「接入选中进程」
- 启动时：创建或更新目录 → 写 `task.json`（含 pid）→ `Popen`，stdout 重定向到该 task 的 `train.log`
- 打开 task / 拉 status：只根据 `task.json.pid` +「是否仍是本仓库训练脚本」校验是否存活；死了则把 `running` 纠成 `finished`/`failed`
- Web UI 进程重启：不扫全机；遍历 `tasks/*/task.json`，对 `running` 做上述 pid 校验后展示

## API

替换/收缩原 `/api/train/*` 进程扫描接口，改为：

| 方法 | 路径 | 行为 |
|------|------|------|
| GET | `/api/tasks` | 列表（id、name、status、arch、updated） |
| POST | `/api/tasks` | 创建 draft（body：name、arch、mode、hparams） |
| GET | `/api/tasks/{id}` | 详情（task.json + 摘要） |
| PATCH | `/api/tasks/{id}` | draft 可改 name/hparams；非 draft 仅允许改 name |
| POST | `/api/tasks/{id}/start` | draft/stopped/failed → running；已有 running 则 409 |
| POST | `/api/tasks/{id}/stop` | SIGTERM→必要时 SIGKILL；status=stopped |
| DELETE | `/api/tasks/{id}` | running 禁止；其余删整个目录 |
| GET | `/api/tasks/{id}/logs` | 日志（支持 tail/stream，等价原 SSE） |
| GET | `/api/tasks/{id}/metrics` | metrics.json |
| GET | `/api/tasks/{id}/checkpoints` | 该 task `checkpoints/` 下文件列表 |

保留对话/加载接口，但「列出 ckpt」默认范围改为**当前选中 task**；仍允许用户粘贴绝对路径加载旧 ckpt（兼容，不强制迁移）。

`/api/defaults`：为 `dense` / `moe` / `tiny` 提供默认 hparams；Tiny 的 `out_dir` 由服务端在创建 task 时强制设为 task 内路径，前端不手填混用目录。

## UI

- **左侧栏**：task 列表（状态色点 + name + arch）+「新建」
- **右侧**：选中 task 详情
  - draft：可编辑配置表单 → Start
  - running：Stop、实时 loss、日志流、进度/ETA（沿用现有解析）
  - 结束后：只读配置、历史 loss、该 task 的 ckpt 列表
- **顶栏**：训练 / 对话（续写若已有则保留）；对话加载 ckpt **默认**来自当前选中 task，不绑死（可选其它路径）

去掉进程下拉与「接入」按钮。

## Tiny 档

- `arch=tiny`，模型类 **同一套 `DenseLM`**
- 固定超参（写入 config 预设或 defaults，不可在 UI 改结构维；batch/steps/lr 可改）：
  - `d_model=64`
  - `n_layers=2`
  - `n_heads=1`，`head_dim=64`（满足 `d_model = n_heads * head_dim`）
  - `intermediate_size=256`（约 4×d）
  - `vocab_size=6400`（沿用现有 tokenizer，不缩词表）
- 默认训练：小 batch、少 steps（例如 batch=4、max_steps 或 epochs 很小），CPU 几秒内能出若干 step 与 ckpt
- Web 架构选择：Dense / MoE / **Tiny** 三档；选 Tiny 时默认填上述结构 + 冒烟友好训练默认值
- 验收：新建 Tiny task → Start → 出现 loss 点与 `checkpoints/*.pt` → 对话页加载该 ckpt 能生成（质量不要求）

## 实现边界

- 优先改 `web_ui.py` + `web/index.html`；`ModelConfig` / `build_model` 增加 tiny 预设或显式传参
- `train_pretrain.py` 已接受 config 字段则尽量不改训练脚本；若脚本写死 d=512，则通过已有 CLI/config 注入或最小补丁
- 删除或停用：`train_web.json`、`train_web.log` 全局读写；`TrainingManager.list_processes` / `attach`
- 单测/烟测：创建 task 目录结构；拒绝第二 running；Tiny 前向 + 存盘 load 一轮（可 CPU）

## 风险与约定

- Windows（wangyifan6）上原 `/proc` 逻辑本就不通用；改 pid 文件后需用跨平台存活检测（`os.kill(pid, 0)` + cmdline 校验若可得）
- 用户强杀训练进程：下次读 status 时纠偏为 failed/stopped
- 磁盘：每个 task 自带 ckpt，注意清理；DELETE 只删目录，二次确认在 UI

## 成功标准

1. 新建并启动训练只产生 `tasks/<id>/` 下文件，无全局 train_web.* 依赖
2. UI 无进程扫描/接入；打开历史 task 能看到当时配置、日志、loss、ckpt
3. 同时第二个 Start 被拒绝并提示
4. Tiny 在 CPU 上可跑通完整链路
