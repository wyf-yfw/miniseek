# AGENTS.md — MiniSeek

给各类 AI / 编码助手的项目级约定。改代码前先读本文件与根目录 `README.md`。

## 项目是什么

- **目标**：DeepSeek 结构对齐的超小可训语言模型（MoE 约 150M），**不是**跟 MiniMind 学架构；MiniMind 仅可作小模型参考（如 tokenizer），已弃用其数据集。
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
- 训练默认 PT 清洗集：`data/clean/.../clean.jsonl`（常由 `data/PT/pretrain_mix_12g_clean.jsonl` 软链）；SFT 默认 `data/SFT/sft_miniseek_v1.jsonl`。

## 代码结构（简）

- `algo/`：模型与 `train_pretrain` / `train_sft` 等训练入口。
- `api/`：FastAPI；`worker_app` 跑训练/评测（与 Web/API 进程隔离）。
- `frontend/`：Web UI（训练 / 对话 / 续写 / 架构 / 评测 / 数据清洗）。
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

单元模板：`deploy/miniseek-*.service` → 安装到 `~/.config/systemd/user/`，再 `daemon-reload` + `enable --now`。公网：`http://60.205.250.186:7860` → 隧道到本机 7860。

反向 SSH 单元只存在于本机 `~/.config/systemd/user/miniseek-tunnel.service`（已在 `.gitignore`）。**永远不要**把 VPS IP/端口映射、密钥、隧道 unit 提交进仓库。

Windows 本地可：`MINISEEK_MONOLITH=1 uvicorn api.app:app --host 0.0.0.0 --port 7860`。

## 训练约定

- **不要 kill** 正在跑的 `moe-pretrain-*` / `train_pretrain` 等进程，除非用户明确要求停训。
- PT 续训：从 **预训练** `last.pt` 续，不要误从 SFT ckpt 续 PT。
- SFT：新开才需要 Init Ckpt；填了 Resume 则不必再填 Init。
- 验证集：**不要**再往 HF/仓库塞独立 valid 文件；用训练参数 `--valid-ratio`（默认约 `0.01`）从 train **内存抽样**，不改源 jsonl。
- Checkpoint：默认滚动 `last.pt`（`--save-last-every=5000`）+ 覆盖 `best.pt`；勿默认按 loss 阈值狂存，除非用户显式开 `--save-loss-thresholds`。
- 清洗：走 Web「数据清洗」；PT 默认倾向保留中英混合（语言过滤常关），中间产物尽量不落盘。

## HF 数据

- Pretrain：`wyf-yfw/MiniSeek-Pretrain`（主文件 `pretrain_mix_12g_clean.jsonl`）。
- SFT：`wyf-yfw/MiniSeek-SFT`。
- Hub 上只保留 train 主文件 + 卡片文档；旧 5g / 独立 valid 可删，勿再上传 valid。
- 上传用本机 `hf` / `huggingface_hub`，**不要**把大文件塞进 Git。

## 给 AI 的硬规则

1. 公仓安全：无私钥、无隧道、无机器内网细节进 commit；文档里若必须提 VPS，仅运维说明且提醒勿提交配置文件。
2. 改部署/重启服务前确认 worker 与训练隔离仍在；优先热更 API/前端，勿连带停训。
3. 架构叙事对齐 DeepSeek，不把 MiniMind 当学习目标。
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
