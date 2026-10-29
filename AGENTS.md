# AGENTS.md — MiniSeek

给各类 AI / 编码助手的项目级约定。改代码前先读本文件与根目录 `README.md`。

## 项目是什么

- **目标**：做一套**层实现对齐 DeepSeek-V4.1-Flash**（arXiv:2609.19969）的超小可训语言模型。`d`、层数、专家个数、top-k、中间维、总参可以小于论文；**层内算子不能换成另一种算法**；不清楚的以论文和官方推理代码（HF `deepseek-ai/DeepSeek-V4.1-Flash/inference`）为准。
- **主线 MiniSeek-F0.1**（`--arch v41`，代码 `algo/model/v41/`，spec `docs/superpowers/specs/2026-10-29-v41-flash-full-arch-design.md`）：CED 编码 4 + 解码 4，层模式 `SWA,SWA,Full,Reuse | Full,Reuse,Reindex,Reuse`（编码 m=2、解码 m=1，L4 建候选池）；Single-Pass mHC（4 流、Sinkhorn 20）；MoE 1 共享 + 64 路由 top-6、中间维 176（sqrt-softplus 打分，分数 + bias 选专家，选中分数归一后 ×1.5 加权，clamp SwiGLU 阈值 10，bias 每步 ±0.001，序列级 aux 1e-4）；Engram@L1·L3；DSpark 3 block 另训。主干 + Engram 186,933,744（每 token 激活约 33M），DSpark 19,256,850；预算是主干 + Engram ≤200M。全程 BF16（不做论文的 FP4/FP8 KV），暂无视觉。
- M0.1 `MoELM` 已移除：旧 M0.1 checkpoint 不能再训练或推理（架构页仍可按 state_dict 画）。Dense / Tiny（`DenseLM`，MiniSeek-D0.1）保留。MiniMind 仅作小模型参考（tokenizer、README 版式），已弃用其数据集。
- **作者署名**：`wyf-yfw`（勿写真实姓名）。
- **仓库**：https://github.com/wyf-yfw/miniseek（公仓）。大语料与权重不进 Git。

## 机器与目录

| 角色 | 机器 | 路径 | 用途 |
|------|------|------|------|
| 训练 / 正式 Web | `wyf`（Linux） | `/home/wyf/miniseek` | GPU 训练、三服务部署、真实数据 |
| 开发 | Windows `wangyifan6` | `D:\miniseek` | 改代码、看 UI；大数据多为空桩/软链，勿当训练源 |
| 公网入口 | VPS `60.205.250.186` | — | 反代 Web；**仅运维本机配置，禁止提交公仓** |

数据布局：

- `data/PT/`、`data/SFT/`：与 HF `wyf-yfw/MiniSeek-Pretrain`、`wyf-yfw/MiniSeek-SFT` 镜像（README / SOURCES / stats 可进库）。
- 大 `.jsonl`、checkpoint、`tasks/`：**gitignore，勿提交**。
- 训练默认 PT 清洗集：`data/clean/.../clean.jsonl`（常由 `data/PT/pretrain_mix_12g_clean.jsonl` 软链）；**SFT 默认**仍 `data/SFT/sft_miniseek_v5_10g.jsonl`（另有 `sft_miniseek_v5_5g.jsonl` 小跑；5G ⊂ 10G）。
- **可行性 SFT 推荐**：`data/SFT/sft_miniseek_v5_short256.jsonl` + `max_seq_len=256`，`--arch v41` 从最近 PT ckpt 初始化；超长 keep-head 截断。Web/CLI 默认路径**不**自动改成 short256，需显式选文件。
- **SFT v5**：见 `docs/superpowers/specs/2026-10-28-sft-v5-mix-design.md` 与 `data/SFT/SOURCES_v5.md`。`sft_miniseek_v5_10g.jsonl` ≈10.00 GiB（~596 万条）；`sft_miniseek_v5_5g.jsonl` ≈5.00 GiB（子集）；`sft_miniseek_v5_short256.jsonl` ≈0.84 GiB（~202 万条，从 10G 按 ≤256/软≤280 丢长样本；见 `SOURCES_v5_short256.md` 与 short256 design spec）。v1 已删；raw 缓存 `_raw_sft_v5/` 已清。

## 代码结构（简）

- `algo/`：`model/v41/`（MiniSeek-F：attention / mhc / moe / engram / model / cache / dspark）、`model/dense.py`（DenseLM）、`optim/`（Muon / Sinkhorn 动量 / 参数分组），入口 `train_pretrain` / `train_sft` / `train_dspark` / `dspark_calibrate` / `generate`。
- `tests/v41_ref/`：官方推理代码移植；`tests/test_v41_*.py` 对照它做逐算子等价测试。
- `api/`：FastAPI；`worker_app` 跑训练/评测（与 Web/API 进程隔离）。
- `frontend/`：Web UI（训练 / 对话 / 续写 / 架构 / 评测 / 数据清洗 / 专家负载）。
- `images/`：README 配图；`miniseek-f0.1-arch.png` 截自架构页 F0.1 预设（隐藏图注、视口 820 宽），图改了要重截。
- `deploy/`：用户级 systemd 单元（web / api / worker）。**不含**反向 SSH 隧道文件。
- `tokenizer/`：词表与相关资产。

## Git 提交流程

1. **在 wyf 仓库根**改、测、提交（Windows 可改，合并前以 wyf 为准 pull/push）。
2. `git status` 确认无 jsonl / `.pt` / `tasks/` / venv / 隧道配置。
3. `git commit` → `git push origin main`。
4. Windows：`git pull` 同步代码即可。

勿 force-push `main`，除非用户明确要求并接受改写历史。

## 部署（仅 wyf）

三服务用户 systemd（重启 web/api **不得**杀掉 worker 上的训练）：

| Unit | 端口 | 作用 |
|------|------|------|
| `miniseek-worker` | `127.0.0.1:8001` | 训练 / 评测 |
| `miniseek-api` | `127.0.0.1:8000` | API |
| `miniseek-web` | `0.0.0.0:7860` | 静态 + `/api` 反代 |

Web UI：`index.html` 每约 1s 轮询 `/__dev/mtime`，mtime 变则自动刷新；`miniseek-web` 用 uvicorn `--reload --reload-dir frontend`，改 `frontend/` 会热重启 web（勿连带 restart worker）。

单元模板：`deploy/miniseek-*.service` → 安装到 `~/.config/systemd/user/`，再 `daemon-reload` + `enable --now`。公网：`http://60.205.250.186:7860` → 隧道到本机 7860。

反向 SSH 单元只存在于本机 `~/.config/systemd/user/miniseek-tunnel.service`（已在 `.gitignore`）。**永远不要**把 VPS IP/端口映射、密钥、隧道 unit 提交进仓库。

Windows 本地可：`MINISEEK_MONOLITH=1 uvicorn api.app:app --host 0.0.0.0 --port 7860`。

## 训练约定

- **不要 kill** 正在跑的 `moe-pretrain-*` / `train_pretrain` 等进程，除非用户明确要求停训。
- PT 续训：从 **预训练** `last.pt` 续，不要误从 SFT ckpt 续 PT。
- Max Steps：新开与续训都是**本轮**步数上限；续训进度条/ETA/LR（warmup+cosine）从 0 起算，**不把** ckpt 旧步数并进分母。ckpt 内仍记 lifetime_step 供下次续训。0 或空只按 epochs。
- `--max-samples`：先从全量合格样本按 `--seed` 均匀抽取再截断，不用文件前缀。不传则全量。验证集仍在这批样本上按 `--valid-ratio` 抽。`--valid-max-samples` 只限制已经抽出的验证子集。
- SFT：新开才需要 Init Ckpt；填了 Resume 则不必再填 Init。
- **v41 默认**（CLI 与 Web 一致）：PT `max_seq_len=1024`、lr 3e-4（min = lr×0.1）、`torch.compile` 关；优化器分组：矩阵走 Muon，embed / head / Engram 表走 Sinkhorn 动量（Engram lr×5），norm 与标量走 AdamW。pack 时文档按 2 对齐并带 `doc_ids` 做样本级注意力掩码。PT 候选池默认关；SFT 默认开候选池 + Decoder Bounded Replay 模拟（`--no-candidate-pool` / `--no-bounded-replay` 可关）。显存紧加 `--act-ckpt`。
- **可行性 SFT 流程（README 里那套）**：数据 `sft_miniseek_v5_short256.jsonl`；`max_seq_len=256`；`--arch v41` + 最近 PT `last.pt`/`best.pt`。超 `max_seq_len` 时 SFT **keep-head**（从尾截，优先留问题）。正式长训用 10G + `max_seq_len=1024`。
- **DSpark**：主干训完（建议 SFT 后）再 `python -m algo.train_dspark --backbone <ckpt> --out-dir <主干 ckpt 所在目录>`（主干冻结），然后 `python -m algo.dspark_calibrate --dspark <dir>/dspark.pt --data <jsonl>` 写回 STS 温度。`dspark.pt` 与主干 ckpt 同目录时，Web 对话/续写自动走投机解码；没有时 F0.1 对话走前缀缓存。
- 验证集：没有独立 valid 文件，也没有 `--valid` 路径。Pretrain / SFT / `valid_ppl` 都只用 `--valid-ratio` 从训练集内存抽样（Pretrain 默认约 `0.01`），不改源 jsonl。`hparams.valid_path` 忽略。
- **PT pack（默认 `batch_mode=pack`）**：整篇文档先 tokenize，文档间加 eos 拼成 token 流，再按训练配置的 `max_seq_len`（256/512/1024…）做**非重叠**切块组 batch；**不再**在 pack 前把单篇截到 `max_seq_len`（长文尾巴进后续块，不是滑动窗口）。random/bucket 仍按样本截断；SFT 超长仍是 keep-head，与此无关。
- **FlashAttention**（仅 Dense / Tiny）：PT / SFT 默认开（`--flash-attn` / UI「FlashAttention」）。优先 `flash_attn` 包；未安装则用 torch SDPA flash（bf16/fp16 + CUDA）。`--no-flash-attn` 可关。日志有 `flash_attn=on backend=...`。v41 用自带的按索引 gather 稀疏注意力，日志为 `flash_attn=n/a`。
- Checkpoint：默认滚动 `last.pt`（`--save-last-every=5000`）+ 覆盖 `best.pt`；勿默认按 loss 阈值狂存，除非用户显式开 `--save-loss-thresholds`。
- 清洗：走 Web「数据清洗」；PT 默认倾向保留中英混合（语言过滤常关），中间产物尽量不落盘。

## HF 数据

- Pretrain：`wyf-yfw/MiniSeek-Pretrain`（主文件 `pretrain_mix_12g_clean.jsonl`）。
- SFT：`wyf-yfw/MiniSeek-SFT`（主文件 `sft_miniseek_v5_10g.jsonl`；另有 `sft_miniseek_v5_5g.jsonl`、**`sft_miniseek_v5_short256.jsonl` / Hub config `sft_short256`**；Hub 仅 train，无独立 valid）。
- Hub 上只保留 train 主文件 + 卡片文档；勿再上传独立 valid；Valid-ratio 仅本地。
- 上传用本机 `hf` / `huggingface_hub`，**不要**把大文件塞进 Git。

## 给 AI 的硬规则

1. 公仓安全：无私钥、无隧道、无机器内网细节进 commit；文档里若必须提 VPS，仅运维说明且提醒勿提交配置文件。
2. 改部署/重启服务前确认 worker 与训练隔离仍在；优先热更 API/前端，勿连带停训。
3. 层实现对齐 DeepSeek-V4.1-Flash。宽度和专家数量可以缩小，算子不换。MoE 用 sqrt-softplus 打分 + bias 选专家、选中分数归一 ×route_scale 加权、clamp SwiGLU。Tiny / Dense 是普通因果注意力 + Pre-Norm，不要画成 SWA / CSA2 / CED / mHC；只有 F0.1 画 V4.1 结构。不把 MiniMind 当学习目标。
4. 用户未要求时不要新建无关大文件、不要擅自删正在训练引用的数据路径。
5. 中文沟通可默认；代码/commit 信息保持清晰英文或中英均可，与现有风格一致。

## 快速自检

```bash
# wyf
cd /home/wyf/miniseek
git status -sb
systemctl --user status miniseek-worker miniseek-api miniseek-web --no-pager
ss -ltnp | grep -E ':7860|:8000|:8001'
pgrep -af 'train_pretrain|train_sft' | head
```

## 当前状态（滚动更新）

- **PT 训练**：wyf 上的 `moe-pretrain-3` 是 M0.1 旧代码进程（resume 自 PT `last.pt`，数据 12g clean）。新代码已删 M0.1，它的 ckpt 新代码不能加载；它跑完或用户决定停训之前，wyf 上先别 `git pull` / 重启 worker（运行中的进程若延迟 import 已删改的模块会出错）。F0.1 尚未开训。
- **数据**：`data/PT` + `data/SFT` 布局；HF `MiniSeek-Pretrain` 主文件 `pretrain_mix_12g_clean.jsonl`；SFT **默认与 HF 已切 v5 10G**（`sft_miniseek_v5_10g.jsonl`；5G 对照包保留；见 `SOURCES_v5.md`）。**可行性快跑推荐 short256**（已推 HF `sft_short256`；见 `SOURCES_v5_short256.md`）。训练用 `--valid-ratio`，Hub 无独立 valid。v1 与 `_raw_sft_v5` 已清。
- **代码**：公仓已含 MiniSeek-F0.1（v41）全架构、Muon / Sinkhorn 优化器、KV 缓存与前缀缓存、DSpark 训练 / 校准 / 投机解码，以及 PT/SFT 布局、valid-ratio、`.gitignore`；M0.1 `MoELM` 已删；`deploy/miniseek-tunnel.service` 不上库。
- **README**：按 MiniMind 版式直接介绍仓库（F0.1 全架构复现 + 训练 / 推理流程），**不写**逐阶段推进到 DeepSeek 的叙事；阶段规划只留在 `docs/STAGE_PLAN.md`（历史记录）。
- **架构页**：默认预设 **V4.1 (F0.1)**（可训练，`meta.trainable`，Fig.3 版式）：编码 / 解码分区，CED / Indexer / Reindex / Engram 连线，色块左侧标层号（L0–L1、L2…），解码器顶部画输出端（mHC 合流 → Final RMSNorm → LM Head）与 DSpark；点色块打开对应层的真实 block，抽屉说明按 v41 算子写（`archV41ModuleInfo`），层公式是 mHC 不是 Pre-Norm。另有 V4.1-Flash 论文参考预设（`source=reference`，不可训练，嵌套 ×3(Full+Reuse×5) / ×4(Reindex+Reuse×3)）。两者都勿退回扁平 layer 卡片。Tiny / Dense 用自下而上括号图按 `DenseLM` 画（注意力在下、FFN 在上、右侧 ×层数、tied 用虚线），不画 SWA / CSA2 / CED / mHC。
- **UI**：顶栏 7 个 page-tabs（含专家负载）单行 `flex`，禁止折成两行。
- **UI**：续写页「停止」与对话页一致，常显；仅生成中可点。
- **Loss 图**：取点间隔可在 UI 调节（从 train.log 按 `plot_every` 重采样），不绑死训练启动时的 plot_every。
- **推理停止**：对话/续写点停止会 `POST /api/infer/stop`，打断服务端 `generate_stream` 并释放 `_gen_lock`（不能只 abort 前端 fetch）。

## 维护约定（固定动作）

本文件是**唯一**跟着 Git 走的项目级 AI 记忆，不另装记忆插件。

以下情况发生后，**同一次改动里**更新本文件并随代码 commit/push：

1. 和用户达成的设计 / 训练 / 数据 / 部署约定  
2. 当前项目阶段推进（例如新跑通某训练、数据上 HF、目录重组）  
3. 硬规则变化（安全、勿杀训、默认参数等）

写短、可执行、与现状一致；过时条目直接删或改，不要堆历史流水账（流水账用 git log）。

