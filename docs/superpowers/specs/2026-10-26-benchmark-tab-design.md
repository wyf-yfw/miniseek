# MiniSeek Web UI：评测（Benchmark）Tab

日期：2026-10-26  
状态：已对齐需求，待实现

## 目标

在 Web UI 增加顶层 **「评测」** Tab：选 checkpoint → 跑有分数的客观评测 → 展示准确率与历史跑次。

一期默认套件（锁定）：

| 任务 | lm-eval 名 | 测什么 | 计分 |
|------|------------|--------|------|
| C-Eval valid | `ceval-valid` | 中文多学科选择题 | accuracy |
| CMMLU | `cmmlu` | 中文多任务理解选择题 | accuracy |
| GSM8K | `gsm8k` | 英文小学～初中应用题（数字答案） | exact / flexible-extract accuracy |

与 Eval-Core（`eval/README.md`）对齐：C-Eval / CMMLU 从「预留」落到可跑；GSM8K 作为一期数学项补充进默认套件。

## 非目标（本轮不做）

- Agent 能力测（AgentBench / ToolBench / BFCL）— **二期**
- MATH 竞赛集、英文通识（MMLU 等）— 可后续勾选扩展，一期 UI 不默认勾选
- 把 Valid PPL / Gen-Smoke 并进本 Tab（脚本仍留在 `eval/`，不迁 UI）
- 多评测并行；与训练抢 GPU 时自动排队以外的复杂调度
- 自动写 `leaderboard.md` / `stage-N.md`（跑次结果落盘即可，人工或后续脚本汇总）
- 改模型结构或训练循环

## 用户流程

1. 打开 **评测** Tab  
2. 选 checkpoint（Browse / 粘贴路径；可默认填当前对话已加载 ckpt 或当前 task 的 `last.pt`）  
3. 套件勾选：默认全选 `ceval-valid` + `cmmlu` + `gsm8k`；可少选  
4. 可选：`limit`（每任务最多 N 题，冒烟用；空=全量）  
5. 点 **开始评测** → 后台跑 → 状态/日志流 → 结束后表格展示各任务 accuracy + 汇总  
6. 历史跑次列表可点开看结果；支持 **停止** 正在跑的一次

## 架构

```
UI (web/index.html 评测 Tab)
    │
    ▼
FastAPI (web_ui.py)  /api/bench/*
    │
    ├─ web/bench_store.py     # eval/runs/<id>/ 元数据
    └─ web/bench_runner.py    # Popen → eval/run_benchmark.py
            │
            ▼
      eval/run_benchmark.py   # CLI：组 lm-eval、写 results.json
            │
            ├─ eval/miniseek_lm.py   # lm-eval 自定义 model（读 .pt / HF 目录）
            └─ lm-evaluation-harness (依赖)
```

### 跑次目录

```
eval/runs/
  <run_id>/
    run.json       # 元数据：ckpt、tasks、status、pid、timestamps、error
    run.log        # stdout/stderr
    results.json   # 各 task 分数（结束后写）
```

`run_id`：与 training task 相同风格，`YYYYMMDD-HHMMSS-xxxx`。

### `run.json` 字段

| 字段 | 说明 |
|------|------|
| `id` | 目录名 |
| `status` | `queued` \| `running` \| `finished` \| `failed` \| `stopped` |
| `ckpt_path` | 绝对或仓库相对路径 |
| `tasks` | `["ceval-valid","cmmlu","gsm8k"]` 子集 |
| `limit` | `null` 或正整数 |
| `pid` | 子进程；非 running 为 `null` |
| `created_at` / `started_at` / `ended_at` | ISO8601 |
| `exit_code` / `error` | 可选 |
| `scores` | 结束后可内嵌摘要；完整见 `results.json` |

### 并发

- 同时最多 **一个** `running` 评测（与训练 task 独立计数，但：若已有训练 `running` 且要用 CUDA，评测默认 CPU 或拒绝 GPU——实现上：**评测子进程默认跟随 `cuda` if available and no training running，否则 `cpu`**，并在 `run.json` 记录 `device`）。
- 再点开始 → 409。

## API

| 方法 | 路径 | 行为 |
|------|------|------|
| GET | `/api/bench/runs` | 列表（新在前）：id、status、ckpt 名、tasks、摘要 scores、时间 |
| POST | `/api/bench/runs` | 创建并启动；body：`ckpt_path`, `tasks[]`, `limit?`；校验路径存在、tasks ⊆ 白名单 |
| GET | `/api/bench/runs/{id}` | `run.json` + 若有则 `scores` |
| GET | `/api/bench/runs/{id}/logs` | 文本日志（支持 `?tail=`） |
| GET | `/api/bench/runs/{id}/results` | `results.json`（未完成 404） |
| POST | `/api/bench/runs/{id}/stop` | SIGTERM→必要时 SIGKILL；status=`stopped` |

白名单 tasks：`ceval-valid`, `cmmlu`, `gsm8k`（一期写死；扩展时改常量）。

## lm-eval 接入

- 依赖：`lm-eval`（写入 `requirements.txt`；版本下限在实现时 pin 可解析的稳定版，如 `lm-eval>=0.4.5`）。
- **不**要求先导出标准 `transformers` AutoModel：实现 `eval/miniseek_lm.py`，注册为 lm-eval 自定义 model（`register_model`），内部复用与 `InferenceEngine.load_model` 相同的 blob 加载逻辑（`.pt` 训练包或 HF 目录/`pytorch_model.bin`）。
- CLI：`python -m eval.run_benchmark --ckpt PATH --tasks ceval-valid,cmmlu,gsm8k [--limit N] --out eval/runs/<id> --device cpu|cuda`
- 输出：解析 lm-eval 结果写入 `results.json`：

```json
{
  "ckpt_path": "...",
  "tasks": ["ceval-valid", "cmmlu", "gsm8k"],
  "metrics": {
    "ceval-valid": {"acc": 0.25, "acc_stderr": 0.01},
    "cmmlu": {"acc": 0.24, "acc_stderr": 0.01},
    "gsm8k": {"acc": 0.02, "acc_stderr": 0.01}
  },
  "raw_path": "..."
}
```

- few-shot：跟随各 task lm-eval 默认（不在一期 UI 暴露）；在 `run.json` / 日志打印实际 num_fewshot，保证可复现。
- 小模型预期：C-Eval/CMMLU 近随机（~25%）；GSM8K 可能极低——UI 不恐吓，只展示数字。

## UI

- 顶栏 tabs：`训练 | 对话 | 续写 | 架构 | 评测`（五列；`page-tabs` 加宽）。
- 评测页布局：
  - 上：ckpt 路径 + Browse（`/api/pick-file`）+ 任务勾选 + limit + 开始/停止
  - 中：当前/选中跑次状态 + 分数字典表（任务 | acc | stderr）
  - 下：历史列表；点选切换详情；日志折叠区
- 文案中文；状态色点与训练 task 一致风格。

## 验收

1. 安装 `lm-eval` 后，CLI 对 Tiny 或 D0.1 ckpt 以 `--limit 2` 跑通三任务（或至少 ceval-valid），写出非空 `results.json`。  
2. Web：选 ckpt → 开始 → status 变 running → finished，表格有 acc。  
3. 再开一次 → 409；停止中途 → `stopped`。  
4. 单元测试：store CRUD、白名单校验、results 解析 fixture；不强制 CI 上下载全量 C-Eval（可用 mock / limit）。  

## 实现边界

- 主工作区：`/workspace/miniseek`；验证后同步 `D:\miniseek` 并重启 `:7860`（用户机 wangyifan6）。  
- 不改训练 TaskStore 语义；评测独立 `eval/runs/`。  
- Agent / MATH 仅在设计备注保留，代码不预埋 UI 开关（YAGNI）。
