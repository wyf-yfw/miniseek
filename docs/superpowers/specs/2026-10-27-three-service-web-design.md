# MiniSeek Web：三服务拆分（frontend / API / worker）

Date: 2026-10-27  
Status: Approved（grill 对齐：方案 A；训练+评测进 worker；本机三 systemd user unit；Funnel 或公网机反代均可指前端）  
Repo: `wyf-yfw/miniseek`  
Machine: wyf（实现与部署目标）；Windows 开发机可暂留单体  
Depends on: Task 驱动 Web UI；Benchmark tab；train live metrics SSE

## Goal

**重启前端或 API 不得杀死正在跑的训练 / 评测进程。**

把现在「一个 uvicorn（`miniseek-webui`，`:7860`）里同时托管静态页 + FastAPI + `Popen` 训练/评测」拆成三个本机服务：

| 服务 | 端口 | 职责 |
|------|------|------|
| **frontend** | `:7860`（对外） | 托管静态 `web/`；把 `/api/*` 反代到 API |
| **api** | `:8000`（仅本机） | 现有 FastAPI 路由（defaults / tasks / bench / fs / arch / chat…）；**不再** `Popen` 训练或评测；start/stop **转发**给 worker |
| **worker** | `:8001`（仅本机） | HTTP 控制面；真正 `Popen` 训练与评测；写 `task.json` / metrics / logs（及 bench 对等落盘）；Stop 只经 worker 或经 API→worker |

对外入口仍是 `:7860`（Tailscale Funnel，或另机公网 IP + nginx/Caddy 反代到内网 `:7860`）。浏览器从不直连 `:8000` / `:8001`。

## Non-goals

- 不改训练算法、超参默认、loss 曲线或 live metrics 语义。
- 不强制拆除 Funnel；公网 VPS 反代是**可选**替代入口，与本机三服务正交。
- 不在本轮做多 task / 多 bench 并行（仍至多一个 running train、一个 running bench，规则与现网一致）。
- 不把 Windows（wangyifan6）立刻拆成三服务；Windows 可继续单体 uvicorn，待需要再对齐。
- 不引入远程队列（Redis / Celery / k8s）；worker 与 API 同机、共享仓库与 `tasks/` 磁盘。
- 不改 `tasks/<id>/`、`eval/runs/<id>/` 目录与 `task.json` / run 元数据 schema（除为 worker 对账可能增加的可选字段外，见下；默认沿用现有字段即可）。

## Context（现状痛点）

- `miniseek-webui.service`：`python3 -m uvicorn web_ui:app --host 0.0.0.0 --port 7860`。
- `web/task_runner.py`、`web/bench_runner.py` 在 **同一进程** 内 `subprocess.Popen(...)`，**未** `start_new_session=True` / 新进程组；内存里 `_owned_procs` 持有句柄。
- `systemctl --user restart miniseek-webui`（或杀 uvicorn）会 SIGTERM 整棵进程树，训练/评测子进程一并退出——这正是用户要消掉的行为。
- 状态真相源已是磁盘（`tasks/*/task.json`、`metrics.json`、`train.log`；bench 在 `eval/runs/`），但启停生命周期绑在 Web UI 进程上。

## Architecture

```text
                    ┌─────────────────────────────────────┐
  Browser / Funnel  │  frontend :7860                      │
  or public VPS ───►│  static web/                         │
                    │  reverse-proxy /api/* ──► api :8000  │
                    └─────────────────────────────────────┘
                                              │
                                              │ HTTP (localhost)
                                              ▼
                                    ┌─────────────────────┐
                                    │  api :8000          │
                                    │  FastAPI (web_ui)   │
                                    │  CRUD / SSE / chat  │
                                    │  start/stop → proxy │
                                    └─────────┬───────────┘
                                              │ HTTP (localhost)
                                              ▼
                                    ┌─────────────────────┐
                                    │  worker :8001       │
                                    │  start/stop/health  │
                                    │  Popen train+bench  │
                                    │  new session / pgid │
                                    │  disk = source of   │
                                    │  truth              │
                                    └─────────────────────┘
                                              │
                                              ▼
                                    tasks/<id>/…  eval/runs/<id>/…
```

### frontend（`miniseek-web`）

- 监听 `0.0.0.0:7860`（或本机约定地址；对外暴露面仅此端口）。
- 静态根目录：仓库 `web/`（至少 `index.html` 及同目录前端资源）；根路径 `/` 与静态资源不经 API。
- `/api/*` → `http://127.0.0.1:8000/api/*`（保留路径与查询串；流式/SSE 必须支持长连接与禁用错误缓冲，例如关闭 proxy buffering）。
- 实现选型（实现阶段二选一即可，本设计不锁死）：轻量 Caddy、nginx，或极薄的静态+反代进程。要求：**无业务逻辑**、重启不影响 GPU 进程。
- Funnel / 公网反代继续指 `:7860`，无需改公网拓扑即可先切本机三服务。

### api（`miniseek-api`）

- 监听 `127.0.0.1:8000`（**禁止**对公网直开，除非另有鉴权——本设计默认仅本机）。
- 从现有 `web_ui.py` 剥离：保留 defaults、tasks CRUD、logs/metrics SSE、checkpoints、bench CRUD/日志、fs、arch、chat/generate 等读路径与对话推理。
- **删除或旁路** API 进程内对 `start_task` / `stop_task` / `bench_start_run` / `bench_stop_run` 的本地 `Popen` 调用。
- `POST /api/tasks/{id}/start|stop`、`POST /api/bench/.../start|stop`（及现有等价路径）：校验元数据后 **HTTP 转发**到 worker（例如 `POST http://127.0.0.1:8001/v1/tasks/{id}/start`）。失败时向浏览器返回 worker 的错误码/信息。
- 状态展示：读磁盘 + `pid` 存活检测（现有 `reconcile_task` / `reconcile_run` 逻辑迁到 API 侧只读对账，或调用 worker 的 status；**不得**依赖 API 内存中的 `_owned_procs`）。
- SSE（`/api/tasks/{id}/logs` 等）仍由 API 读 `train.log` / `metrics.json` 推送；不要求经 worker 转流。
- 对话/加载 ckpt：仍在 API 进程内做推理即可（短请求）；不经 worker。若日后要把推理也隔离，另开 spec。

### worker（`miniseek-worker`）

- 监听 `127.0.0.1:8001`。
- 职责边界：
  - `POST .../tasks/{id}/start`：组装命令（复用/迁入 `web.task_runner.build_train_cmd`）、`Popen`、写 `task.json`（status/pid/started_at…）、启动 log tail → `metrics.json`（现有 tail 逻辑随 runner 迁入 worker）。
  - `POST .../tasks/{id}/stop`：对训练进程组发 SIGTERM，超时 SIGKILL；更新 `stopped`。
  - Bench 对等：`start` / `stop`，落盘 `eval/runs/<id>/`。
  - `GET .../health`：存活探测（供 systemd / API 探活）。
- **进程隔离（硬性）**：`Popen(..., start_new_session=True)`（POSIX 新会话 / 新进程组），使训练/评测不落在 frontend/api 的服务树里；Stop 与 worker 关机钩子均对 **进程组** 发信号（例如 `os.killpg(pgid, SIGTERM)` → 超时 `SIGKILL`），避免只杀包装进程、留下孤儿。
- `_owned_procs` 仅存在于 **worker** 进程内存。worker **优雅退出**（systemd stop/restart）必须遍历托管 pid/pgid 并 stop，再写盘 `stopped`/`failed`。若 worker **崩溃**（无钩子），磁盘上可能仍留 `running` + 活 pid：API/下次 worker 启动时 reconcile——活着则由新 worker 接管 tail（或按实现选择补停）；本设计优先「接管并恢复 tail」，与「systemd stop 必停训」不冲突。
- 启停权威在 worker；API 不持有 Popen。
- 并发规则不变：已有 running train → 新 start 409/拒绝；bench 同理；train 忙时 bench 设备选择逻辑保持现有语义。

## Data flow

1. 用户点 Start → 浏览器 `POST /api/tasks/{id}/start` → frontend 反代 → API 校验 draft/stopped/failed、无其它 running → API `POST` worker start → worker `Popen` + 写盘 → API 返回更新后的 task。
2. 执行页 SSE：浏览器 `EventSource /api/tasks/{id}/logs` → API 读磁盘推 `log` / `metrics` / `status`（含 live 指标字段）；**不经过** worker。
3. 用户点 Stop → `POST /api/.../stop` → API → worker stop → 仅杀训练/评测进程组。
4. `systemctl --user restart miniseek-web`：仅静态+反代闪断；API、worker、GPU 进程均在。
5. `systemctl --user restart miniseek-api`：SSE/页面 API 短暂不可用；worker 与训练 pid 仍在；API 起来后读 `task.json` 对账为 `running`。
6. `systemctl --user stop|restart miniseek-worker`：worker 在退出前 **显式** 向其托管的训练/评测进程组发 SIGTERM（超时 SIGKILL），与点 Stop 同等效果；然后才退出。这样「停 worker」与「点 Stop」都会停训，符合验收。`start_new_session` 用于与 API/frontend cgroup 隔离，避免只靠 systemd 误杀范围不清；**不得**依赖「孤儿续跑」作为正常路径。
7. 正常结束路径：用户/API→worker Stop、worker 服务 stop/restart 时的关机钩子、或训练/评测自然退出。**restart frontend / api 不在此列。**

## systemd（wyf）

三个 user unit（路径示例；WorkingDirectory=`/home/wyf/miniseek`）：

| Unit | 替换关系 | ExecStart（示意） |
|------|----------|-------------------|
| `miniseek-web.service` | **取代** `miniseek-webui` | 静态+反代，`:7860` → 静态 `web/`，`/api`→`127.0.0.1:8000` |
| `miniseek-api.service` | 新建 | `uvicorn` 加载 API app（可由 `web_ui:app` 拆出或同模块改监听），`--host 127.0.0.1 --port 8000` |
| `miniseek-worker.service` | 新建 | worker HTTP 入口，`--host 127.0.0.1 --port 8001` |

迁移步骤（实现时）：

1. 落地代码与三份 unit 文件（可放仓库 `deploy/` 或文档记录安装命令）。
2. `systemctl --user disable --now miniseek-webui`；enable/start 三个新 unit。
3. 确认 Funnel / 公网反代仍指向 `:7860`。
4. 旧 unit 文件可保留注释「已退役」，避免误启双开 `:7860`。

依赖建议：`miniseek-web` After/Wants `miniseek-api`；`miniseek-api` After/Wants `miniseek-worker`（web 无 worker 也能起静态，但 Start 会失败——可接受）。三者均为 `Restart=on-failure`。

## Module boundary（实现指引，非本提交改码）

| 现有 | 迁往 |
|------|------|
| `web/task_runner.py` 的 spawn/stop/tail/`_owned_procs` | worker 进程（可仍为同模块，由 worker 入口 import） |
| `web/bench_runner.py` 同上 | worker |
| `web_ui.py` 路由 + SSE + chat | api |
| `web/index.html` 等静态 | frontend 直接托管；**前端 JS 仍请求相对路径 `/api/...`**，不写死 `:8000` |
| `tasks_store` / `bench_store` | API 与 worker **共享**读盘；写 pid/status 的权威路径以 worker 的 start/stop/tail 为准，API 侧 reconcile 只修陈旧 `running` |

Worker HTTP 契约（稳定字段即可，路径前缀实现时可定为 `/v1`）：

| 方法 | 路径 | 行为 |
|------|------|------|
| GET | `/health` | 200 + `{ok: true}` |
| POST | `/v1/tasks/{id}/start` | 启动训练；冲突 409 |
| POST | `/v1/tasks/{id}/stop` | 停止训练 |
| POST | `/v1/bench/{id}/start` | 启动评测（body 可含 `training_busy` 或由 worker 自查） |
| POST | `/v1/bench/{id}/stop` | 停止评测 |

API 对外路径保持现有浏览器契约（`/api/tasks/...`、`/api/bench/...`），避免改前端。

## External access

- **默认**：维持 Funnel → wyf `:7860`。
- **可选**：另有公网服务器时，其上 nginx/Caddy `proxy_pass` 到内网 wyf `:7860`（VPN/专线/SSH tunnel 等达内网）；**不必**再经 Funnel。本机仍是三服务；公网机只做 TLS 与反代。
- `:8000` / `:8001` 不暴露到公网。

## Windows

开发机可继续：`uvicorn web_ui:app --port 7860` 单体（静态+API+Popen）。与 wyf 三服务并行存在时，文档/README 注明「生产（wyf）三服务；Windows 可选单体」。本 spec 不要求 Windows systemd。

## Acceptance

1. 启动一次 pretrain（或 Tiny 冒烟）至 `status=running` 且 pid 可见。
2. `systemctl --user restart miniseek-web` → 训练 pid **仍存活**，稍后页面可再连上 SSE。
3. `systemctl --user restart miniseek-api` → 训练 pid **仍存活**；API 恢复后 task 仍显示 running，日志/指标继续更新。
4. 调用 Stop（UI→API→worker）或 `systemctl --user stop|restart miniseek-worker`（走关机钩子）后，训练进程退出，`task.json` 不再为 `running`。**重启 `miniseek-web` / `miniseek-api` 不得触发上述停训。**
5. Bench 同样：restart web/api 不杀评测；Stop 或 stop worker 才停评测。
6. Funnel 或公网反代访问 `:7860` 仍能完成创建→Start→看 loss→Stop。
7. 无第二套对外端口被误开到 `0.0.0.0:8000/8001`。

## Risks

- Worker **崩溃**（非优雅 stop）后：训练可能仍在跑而 tail 已死 → metrics 暂停；实现须在 worker startup 对仍存活的 `running` pid 重挂 tail（或统一补停，若选补停须在实现 PR 写明；本 spec 默认重挂）。
- systemd `KillMode`：优雅路径以 worker 关机钩子为准；勿依赖「cgroup 连坐」作为唯一停训手段（`start_new_session` 后子进程可能已不在同一 cgroup）。实现时用 stop/restart worker 实测进程是否退出。
- API 与 worker 同时写 `task.json`：约定 start/stop/exit 只由 worker 写 status/pid；API 仅在发现 pid 已死时做失败纠偏（或委托 worker reconcile 接口）。避免双写打架。
- 前端反代缓冲导致 SSE 卡住：反代须关 buffering、拉长超时。

## Success criteria（总结）

- wyf 上三个 user service 替代 `miniseek-webui`；对外仍 `:7860`。
- UI/API 启停与训练/评测生命周期解耦。
- 浏览器契约与 task/bench 磁盘布局保持兼容。
