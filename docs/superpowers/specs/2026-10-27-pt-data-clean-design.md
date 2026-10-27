# Pretrain 数据清洗页（Data-Juicer 引擎）

日期：2026-10-27  
状态：已批准（对话确认：接 Data-Juicer；范围 / 页面 / 后端三节均 OK）  
Repo：`wyf-yfw/miniseek`  
Machine：wyf

## 目标

在 MiniSeek Web 增加侧栏页 **「数据清洗」**，用 **Data-Juicer** 清洗 **Pretrain** jsonl（`{"text":...}`），产出可直接喂训练的 clean 集 + 丢弃集 + 报告。后台经 worker 跑，**不占用 / 不打断 GPU 训练进程**。

## 非目标

- 本轮不做 SFT 对话格式清洗（后续可复用任务壳）
- 不做完整 Data-Juicer Studio / 全量算子 UI
- 不做模糊去重（MinHash）与模型打分（PPL / 质量分类器）；仅常用启发式 + 精确去重
- 不改三服务拓扑；不改训练 task 生命周期
- 清洗进程不申请 CUDA；训练 running 时仍可启清洗

## 范围（已确认）

| 项 | 约定 |
|----|------|
| 入口 | 顶栏 page-tabs 新增「数据清洗」（与训练/评测并列） |
| 数据 | 仅 PT：输入 `data/*.jsonl`（默认候选 `data/pretrain_mix_5g.jsonl`） |
| 引擎 | Data-Juicer（`dj-process` / Python API）；环境未装时允许**同算子语义**的内置回退，装好后自动走 DJ |
| 输出根 | `data/clean/<job_id>/`：`clean.jsonl` + `dropped.jsonl` + `report.json` + `job.json` + `clean.log` + `recipe.yaml` |
| 形态 | 与评测类似：可后台跑、可停；至多一个 running 清洗 job |

## 页面（已确认）

- 选输入 jsonl（浏览 `/api/fs/list`）
- 勾选算子 + 阈值：
  - **长度** `text_length_filter`：`min_len` / `max_len`（字符）
  - **语言** `language_id_score_filter`：默认 `zh`，`min_score`
  - **重复** `character_repetition_filter`（或等价）：重复字符比例阈值
  - **精确去重** `document_deduplicator`：整篇 MD5
  - **敏感词** `flagged_words_filter`（或词表路径）：命中则丢
- **抽样预览**：对前 N 条（默认 200）同步跑勾选算子，返回 keep/drop 样例与粗统计，不落完整 clean
- **开始清洗** → 进度 / 日志；结束后展示路径与 keep/drop 计数

## 后端（已确认）

```text
Browser → frontend :7860 → API :8000
  POST /api/clean/jobs            创建 draft
  PATCH /api/clean/jobs/{id}      改配置
  POST /api/clean/jobs/{id}/start → worker POST /v1/clean/{id}/start
  POST /api/clean/jobs/{id}/stop  → worker stop
  POST /api/clean/preview         同步抽样预览（API 进程内短跑，可限时）
  GET  logs / report / list / delete
Worker :8001
  Popen scripts/pt_clean_job.py（CPU；start_new_session）
  写 data/clean/<id>/…
```

- **禁止**在 API/frontend 进程内长跑全量清洗；全量只经 worker。
- **禁止**清洗逻辑 `torch.cuda` / 抢训同卡；与 train/bench 的 GPU 策略解耦。
- 训练 running **不**阻止清洗 start（与 bench「训练忙则 CPU」不同——清洗本就 CPU）。

### `job.json` 字段（要点）

| 字段 | 说明 |
|------|------|
| `id` / `name` / `status` | 同 task：`draft` \| `running` \| `stopped` \| `finished` \| `failed` |
| `input_path` | 源 jsonl |
| `operators` | 勾选与参数快照 |
| `pid` / 时间戳 / `exit_code` / `error` | 同评测 |
| `output_dir` | `data/clean/<id>/` |
| `stats` | 结束后：`input_rows` / `kept` / `dropped` / 每算子丢弃数（能拿到则写） |

### 算子默认（可 UI 改）

- 长度：`min_len=50`，`max_len=100000`
- 语言：`lang=zh`，`min_score=0.5`（无 DJ 语言模型时回退用 CJK 占比启发式，并在 report 注明）
- 字符重复：`rep_len=10`，`max_ratio=0.5`（对齐 DJ 常见默认量级）
- 精确去重：开启；`lowercase=false`
- 敏感词：内置小词表 + 可选自定义文件；默认开启

### Data-Juicer 配方

Worker 在 job 目录写 `recipe.yaml`（`dataset_path` / `export_path` / `process:`），优先：

```bash
dj-process --config data/clean/<id>/recipe.yaml
```

或 `.venv-dj/bin/dj-process`。导出为 keep 集；`dropped.jsonl` 由输入∖输出（按行 MD5 / 稳定 id）差分得到；`report.json` 汇总计数与引擎名（`data-juicer` \| `builtin`）。

## 风险与约定

- 全量 `pretrain_mix_5g`（约 180 万行）耗时长、磁盘翻倍级；UI 提示输出目录与预估；支持 Stop。
- DJ 依赖重：第一版允许 builtin 回退，避免阻塞页面上线；文档与 report 标明实际引擎。
- 不改动正在训练读取的源文件；只写 `data/clean/`。

## 验收

1. 顶栏出现「数据清洗」；可选 jsonl、勾算子、抽样预览有 keep/drop 样例。
2. Start 后 worker 有 pid；`data/clean/<id>/clean.jsonl` + `dropped.jsonl` + `report.json` 生成；日志可看。
3. 训练 running 时启清洗成功；Stop 可停清洗且不影响训练 pid。
4. 未装 DJ 时 builtin 能跑通小文件冒烟；装 DJ 后 report.engine=`data-juicer`。
