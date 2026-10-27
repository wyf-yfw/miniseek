# MiniSeek 三目录重组设计

日期：2026-10-27
状态：已批准（对话确认）

## 目标

把仓库按 **frontend / api / algo** 分门别类，便于浏览与部署对齐；不引入 monorepo 多包，不留旧入口兼容壳。

## 范围补充（对话追加）

- **删除**根目录 `checkpoints/`（Web Task 权重只在 `tasks/<id>/`；旧实验目录不再保留）

## 非目标

- 不搬 `data/`、`tasks/`、`tokenizer/`、`deploy/`
- 不做完整 `packages/*` monorepo
- 不单独给 Windows 留 monolith 结构（同一仓库布局）

## 目标树

```
miniseek/
  frontend/           # 静态页 + 反向代理
    index.html
    frontend_server.py
    __init__.py
  api/                # 控制面 API + worker
    app.py            # 现 web_ui.py
    worker_app.py
    worker_client.py
    task_runner.py
    bench_runner.py
    tasks_store.py
    bench_store.py
    arch_inspect.py
    __init__.py
  algo/               # 模型与训练/评测
    model/            # 现 model/
    train_pretrain.py
    train_sft.py
    generate.py
    eval/             # 现 eval/
    __init__.py
  data/
  tasks/
  tokenizer/
  deploy/
  tests/
  docs/
  scripts/
  release/
  README.md / README_en.md
  pyproject.toml
  requirements.txt
```

根目录删除/搬走：`web_ui.py`、`web/`、`model/`、`train_*.py`、`generate.py`、`eval/`（迁入上表）。

## 归属规则

| 目标 | 来源 |
|------|------|
| `frontend/` | `web/index.html`、`web/frontend_server.py` |
| `api/` | `web_ui.py`→`api/app.py`；`web/` 其余（除前端两文件） |
| `algo/` | `model/`、`train_pretrain.py`、`train_sft.py`、`generate.py`、`eval/` |
| 根不动 | `data/`、`tasks/`、`tokenizer/`、`deploy/`、`tests/`、`docs/`、`scripts/`、`release/` |
| 删除 | 根目录 `checkpoints/`（整目录） |

## 入口与 systemd

WorkingDirectory 仍为仓库根 `/home/wyf/miniseek`。

| 服务 | 命令 |
|------|------|
| web :7860 | `python3 -m uvicorn frontend.frontend_server:app --host 0.0.0.0 --port 7860` |
| api :8000 | `python3 -m uvicorn api.app:app --host 127.0.0.1 --port 8000` |
| worker :8001 | `python3 -m uvicorn api.worker_app:app --host 127.0.0.1 --port 8001` |

训练 CLI：`python -m algo.train_pretrain` / `python -m algo.train_sft`（相对根路径的 `data/` 不变；Web Task 输出在 `tasks/<id>/`）。

不保留 `web_ui:app`、`web.*` 兼容壳；文档与 `deploy/*.service` 同步改。

## Import 与打包

- `from model…` → `from algo.model…`
- `from web…` → `from api…`
- 前端静态目录解析指向 `frontend/`（相对包或相对仓库根，实现时统一一处）
- runner 启动训练子进程的模块路径改为 `algo.train_*`
- `pyproject.toml` `packages.find` 包含 `algo*`、`api*`、`frontend*`（替代仅 `model*`）
- `pytest`：`testpaths = tests`，`pythonpath = ["."]` 保留

## 验收

1. `pytest` 全绿
2. 三服务可启动；`http://127.0.0.1:7860`（或现有公网/Funnel）页面正常
3. 重启 web/api 不影响正在跑的 train；Stop/worker 可停训
4. README / deploy README 路径与命令已更新

## 风险

- 大文件移动易漏改字符串路径（尤其 `task_runner` 拼 CLI、HTML 静态路径）
- 进行中任务目录不受影响，但若子进程命令写死旧模块名需一并改
