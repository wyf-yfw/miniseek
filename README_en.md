<div align="center">

# MiniSeek

**Tiny, trainable LMs that climb toward DeepSeek-V4.1-Flash-style structures—one stage at a time**

[![GitHub Repo stars](https://img.shields.io/github/stars/wyf-yfw/miniseek?style=social)](https://github.com/wyf-yfw/miniseek/stargazers)
[![GitHub Code License](https://img.shields.io/github/license/wyf-yfw/miniseek)](LICENSE)
[![GitHub last commit](https://img.shields.io/github/last-commit/wyf-yfw/miniseek)](https://github.com/wyf-yfw/miniseek/commits/main)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-blue)](https://github.com/wyf-yfw/miniseek/pulls)
[![Collection](https://img.shields.io/badge/🤗-MiniSeek%20Collection-yellow)](https://huggingface.co/collections/wyf-yfw/miniseek-6adc765159d38ab3236c49af)

[中文](./README.md) | English

</div>

- Start from a **Dense** backbone, then add MoE / SWA / CSA2 / CED stage by stage ([`docs/STAGE_PLAN.md`](docs/STAGE_PLAN.md)), with **locked data + Eval-Core** for fair A/B.
- Stage 0: `d_model=512`, 8-layer dense decoder-only, MiniMind tokenizer **vocab=6400**, ~**28.45M** tied params—trainable on a single RTX 3090.
- Data/tokenizer aligned with the [MiniMind](https://github.com/jingyaogong/minimind) stack (`jingyaogong/minimind_dataset`); core model in **native PyTorch**.
- Built-in **Web UI**: training Tasks, chat, continuation, architecture viz, and C-Eval / CMMLU / GSM8K (lm-eval).
- Open weights: [`MiniSeek-D0.1-base`](https://huggingface.co/wyf-yfw/MiniSeek-D0.1-base) (pretrain), [`MiniSeek-D0.1`](https://huggingface.co/wyf-yfw/MiniSeek-D0.1) (SFT).

> This repo’s shipping focus is **Stage 0 Dense + train / eval / Web**. Later stages are specified; code hooks exist, full implementations are still TODO.

---

# 📌 Overview

Training large models from scratch is hard on personal GPUs, and high-level frameworks often hide the interesting parts. MiniSeek takes a narrower bet: **upgrade the architecture in discrete stages**, keep data and benchmarks frozen, and measure what each structural change actually buys.

#### What’s in the box today

- Dense decoder-only (`DenseLM`) train + generate (RoPE, RMSNorm, GELU FFN; embedding/LM-head tying by default on the base)
- `train_pretrain.py`, `generate.py`, data download + fixed valid split
- Eval-Core: Valid PPL, Gen-Smoke, plus C-Eval / CMMLU / GSM8K via Web/CLI (lm-evaluation-harness)
- FastAPI Web UI (`:7860`): training Tasks, chat, continuation, architecture diagrams, bench Tasks
- Stage plan [`docs/STAGE_PLAN.md`](docs/STAGE_PLAN.md) and smoke tests

#### Released models

| Model | Role | Hub parameter count | Release |
|-------|------|---------------------|---------|
| [MiniSeek-D0.1-base](https://huggingface.co/wyf-yfw/MiniSeek-D0.1-base) | Dense pretrained base | **28.45M** (tied) | 2026-10 |
| [MiniSeek-D0.1](https://huggingface.co/wyf-yfw/MiniSeek-D0.1) | SFT / chat | **31.73M** (untied emb+head) | 2026-10 |

Format: `model.safetensors` + `config.json` + MiniMind tokenizer. Custom `DenseLM`—not drop-in `AutoModelForCausalLM` yet.

---

# 📌 Quick start

## Step 0 — clone & deps

```bash
git clone https://github.com/wyf-yfw/miniseek.git
cd miniseek
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows PowerShell
# .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

CPU smoke:

```bash
python -m pytest tests/test_smoke.py -q
```

## Ⅰ 🚀 Inference

### 1. Download Hub weights

```bash
huggingface-cli download wyf-yfw/MiniSeek-D0.1-base --local-dir ./hub/MiniSeek-D0.1-base
# or chat
huggingface-cli download wyf-yfw/MiniSeek-D0.1 --local-dir ./hub/MiniSeek-D0.1
```

### 2. CLI generate (local ckpt)

```bash
python generate.py --ckpt checkpoints/stage0/best.pt --seed 42 --prompt "Hello, introduce AI briefly."
```

### 3. Load DenseLM from the Hub

```python
import json
from pathlib import Path
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from model.config import ModelConfig
from model.dense import DenseLM
from model.tokenizer import load_tokenizer

repo = snapshot_download("wyf-yfw/MiniSeek-D0.1-base")
cfg = ModelConfig.from_dict(json.loads(Path(repo, "config.json").read_text()))
model = DenseLM(cfg)
state = load_file(Path(repo) / "model.safetensors")
model.load_state_dict(state, strict=False)  # base: lm_head tied → tok_emb
tok = load_tokenizer(str(Path(repo) / "tokenizer"))

ids = tok.encode("你好，你是谁", add_special_tokens=True)
out = model.generate(
    torch.tensor([ids]),
    max_new_tokens=64,
    temperature=0.8,
    top_k=50,
    eos_token_id=tok.eos_token_id,
)
print(tok.decode(out[0].tolist(), skip_special_tokens=True))
```

### 4. Web UI (recommended)

```bash
python -m uvicorn web_ui:app --host 127.0.0.1 --port 7860
```

| Tab | What it does |
|-----|----------------|
| Train | Task sidebar: configure → Start → logs/metrics → Stop; artifacts under `tasks/<id>/` |
| Chat / Continue | Pick a checkpoint for chat or completion |
| Architecture | Tiny / Dense / MoE presets; simple · standard · detailed diagrams + drawer |
| Bench | Same Task UX as training: C-Eval / CMMLU / GSM8K with dual progress |

```bash
python scripts/smoke_tiny_task.py
```

---

# 📌 Data

## Ⅰ Tokenizer

MiniMind vocab (**6400**). Recommended:

```bash
huggingface-cli download jingyaogong/MiniMind2 --include "tokenizer*" --local-dir ./tokenizer
# or export DS41_TOKENIZER_PATH=/path/to/minimind_tokenizer
```

Without local files the code falls back to a byte-level `FallbackTokenizer` (smoke only—not for real training).

## Ⅱ Pretrain / valid

```bash
python scripts/download_data.py            # docs / split valid
python scripts/download_data.py --download # fetch pretrain_t2t_mini.jsonl
```

| Locked item | Value |
|-------------|--------|
| Source | [`jingyaogong/minimind_dataset`](https://huggingface.co/datasets/jingyaogong/minimind_dataset) |
| Train | `pretrain_t2t_mini.jsonl` |
| Valid | `valid.jsonl` = **0.5%** of train, **seed=42** (frozen for stages 0–5) |
| SFT | `sft_t2t_mini.jsonl` (chat line after architecture freeze; not for structure A/B) |

Rule: **change architecture, not data**. Full 10GB/14GB dumps are for post-freeze scaling, not the stage table.

---

# 📌 Model

## Stage 0 layout

```text
Token Embedding (V=6400 × d=512)
        ↓
×8 Transformer Block
  RMSNorm → Causal Attn (RoPE) → Residual
  RMSNorm → Dense FFN (intermediate=2048) → Residual
        ↓
RMSNorm → LM Head (tied with Embedding on the base release)
```

| Spec | Value |
|------|--------|
| `d_model` / `n_layers` / `n_heads` / `head_dim` | 512 / 8 / 8 / 64 |
| Dense FFN | 2048 (=4d) |
| `max_seq_len` (train start) | 512 |
| Precision / target HW | BF16 · single 3090 |

## Stage roadmap

| Stage | Structure | Status |
|-------|-----------|--------|
| 0 | Dense decoder-only | ✅ |
| 1 | + MoE (1 shared + 8 routed, top-2) | TODO |
| 2 | + SWA | TODO |
| 3 | + CSA2 | TODO |
| 4 | + CED | TODO |
| 5 | + HSI / Replay | TODO |

Full tables: [`docs/STAGE_PLAN.md`](docs/STAGE_PLAN.md).

Naming (one Hub repo per checkpoint):

- `MiniSeek-D{x.y}-base` — Dense pretrain
- `MiniSeek-D{x.y}` — Dense SFT / chat
- `MiniSeek-M{x.y}-*` — MoE line (later)

---

# 📌 Training

```bash
python train_pretrain.py --data data/pretrain_t2t_mini.jsonl --out-dir checkpoints/stage0

# short run (no loss-*.pt by default; pass --save-loss-thresholds only if needed)
python train_pretrain.py --data data/pretrain_t2t_mini.jsonl --out-dir checkpoints/stage0 \
  --max-steps 5000 --warmup-steps 200 --lr 1e-4
```

### Checkpoint policy

- **Default**: only `best.pt` (overwrite when improved) and `last.pt` (once at end) — avoids filling the disk
- **Milestones** `loss-*.pt`: opt-in via `--save-loss-thresholds` (writes after `--stable-steps` consecutive `loss<=threshold`); default empty = off
- **`--save-every`**: default `0` (off); enable only if you need step backups

Or drive the same loop from the Web **Train** tab (`tasks/<id>/`).

---

# 📌 Evaluation

**Eval-Core** (same suite every stage):

| # | Item | Notes |
|---|------|--------|
| 1 | Valid PPL | shared `valid.jsonl` |
| 2 | Gen-Smoke | 20 fixed prompts, seed=42 |
| 3 | C-Eval valid | lm-eval |
| 4 | CMMLU | lm-eval (`trust_remote_code`; pin `datasets>=2.14,<4.0`) |
| 5 | GSM8K | lm-eval `generate_until` |
| 6 | Efficiency | peak VRAM / tokens/s on fixed HW |

```bash
python eval/valid_ppl.py --ckpt checkpoints/stage0/last.pt --valid data/valid.jsonl
python eval/gen_smoke.py --ckpt checkpoints/stage0/last.pt --seed 42
# or Web Bench tab / eval/run_benchmark.py
```

Templates: `eval/reports/stage-0.md`, `eval/reports/leaderboard.md`. See [`eval/README.md`](eval/README.md).

Anchor: MiniMind-3 (~64M) C-Eval≈24.9 / CMMLU≈25.4—Stage 0 is smaller; track **deltas**, not absolute SOTA.

---

# 📌 Layout

```text
model/              # config · DenseLM · tokenizer
train_pretrain.py   # pretrain entry
generate.py         # CLI generate
web_ui.py + web/    # FastAPI UI · Task/Bench/arch
eval/               # PPL · Gen-Smoke · lm-eval · reports
scripts/            # data download · smoke
docs/STAGE_PLAN.md  # stage spec (source of truth)
tests/              # smoke / unit tests
```

---

# 📌 Acknowledgements

- Data & tokenizer: [jingyaogong/minimind](https://github.com/jingyaogong/minimind), [`minimind_dataset`](https://huggingface.co/datasets/jingyaogong/minimind_dataset)
- Structure reference: [DeepSeek-V4.1 paper HTML](https://arxiv.org/html/2609.19969)
- Benchmarks: [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), C-Eval, CMMLU

Issues and PRs welcome.

---

# 🎓 Citation

```bibtex
@misc{miniseek2026,
  title        = {MiniSeek: Tiny Trainable LMs Toward DeepSeek-V4.1-Flash Structures},
  author       = {wyf-yfw},
  year         = {2026},
  howpublished = {\url{https://github.com/wyf-yfw/miniseek}}
}
```

---

# ⚖️ License

[MIT](LICENSE)
