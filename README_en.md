<div align="center">

# MiniSeek

**A tiny DeepSeek-V4.1-Flash, full architecture, trained from scratch on one RTX 3090**

[![GitHub Repo stars](https://img.shields.io/github/stars/wyf-yfw/miniseek?style=social)](https://github.com/wyf-yfw/miniseek/stargazers)
[![GitHub Code License](https://img.shields.io/github/license/wyf-yfw/miniseek)](LICENSE)
[![GitHub last commit](https://img.shields.io/github/last-commit/wyf-yfw/miniseek)](https://github.com/wyf-yfw/miniseek/commits/main)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-blue)](https://github.com/wyf-yfw/miniseek/pulls)
[![Collection](https://img.shields.io/badge/🤗-MiniSeek%20Collection-yellow)](https://huggingface.co/collections/wyf-yfw/miniseek-6adc765159d38ab3236c49af)

[中文](./README.md) | English

</div>

* This project trains **MiniSeek-V1** from scratch on a single RTX 3090 in native PyTorch: a tiny language model whose layers match [DeepSeek-V4.1-Flash](https://arxiv.org/abs/2609.19969). Backbone + Engram total **187M** parameters, with about **33M** active per token.
* Depth, width and expert count are scaled down to fit a 3090; no operator inside a layer is swapped for something else: CED causal encoder–decoder, SWA + CSA2 (Full / Reuse / Reindex) compressed sparse attention with the hierarchical sparse indexer, Single-Pass mHC multi-stream residuals, sqrt-softplus routed MoE, Engram n-gram memory, and the DSpark speculative-decoding drafter.
* The training recipe follows the paper too: grouped Muon (hybrid Newton–Schulz) + AdamW + Sinkhorn momentum, sample-level attention masking, the indexer KL loss, and auxiliary-loss-free expert-bias balancing.
* Inference implements KV caches, CED prefill, a prefix cache with bounded replay, and confidence-scheduled DSpark speculative decoding whose output distribution matches the backbone.
* Core operators (RoPE, sparse attention, mHC, routing, Engram hashing, …) are checked against ports of DeepSeek's official inference code, and end-to-end tests confirm that cached generation matches a full forward pass and that speculative decoding matches plain decoding.
* Data cleaning, pretraining, SFT, evaluation and chat all run from the built-in Web UI; the architecture page lets you open any layer to see its operators, formulas and parameters.

> Note: there are only three deliberate differences from the paper: KV and indexer tensors stay in BF16 (the paper uses FP4 for the main KV and FP8 for the sliding-window KV, which the 3090 does not support); there is no vision branch yet; and the model is smaller.

---

<div align="center">

![MiniSeek-V1 architecture](./images/miniseek-v1-arch.png)

</div>

---

# 📌 Overview

DeepSeek-V4.1-Flash adds a whole set of new parts to the Transformer: a global KV computed once by the encoder and reused by every decoder layer (CED), compressed sparse attention that alternates between three modes, four residual streams with mHC, hashed Engram memory, and DSpark, which drafts several tokens at once. The paper and the official code target hundreds of GPUs, FP4 / FP8 kernels and hundreds of billions of parameters, which makes them hard to read, run or modify on a personal GPU.

MiniSeek shrinks the same architecture to a size a 3090 can train. Depth, width and expert count are smaller, but every operator follows the paper and the official code instead of a common stand-in. On one machine you can walk the whole path (data cleaning → pretraining → SFT → drafter training → inference with speculative decoding → evaluation) and use the code and tests to see what each part does.

#### 🎉 What's included

- All MiniSeek-V model code (`algo/model/miniseek_v/`): SWA / CSA2 attention, compressor, indexer and candidate pool, Single-Pass mHC, MoE, Engram, the CED backbone, KV and prefix caches, DSpark.
- Optimizers (`algo/optim/`): Muon, Sinkhorn momentum, and the V4 rules that split parameters into Muon / AdamW / Sinkhorn groups.
- Training scripts: pretraining `algo/train_pretrain.py`, SFT `algo/train_sft.py`, DSpark `algo/train_dspark.py` and its confidence calibration `algo/dspark_calibrate.py`, with resume, packed batches with sample-level attention masks, and a validation split drawn from the training set.
- Ports of the official inference code (`tests/miniseek_v_ref/`) and per-operator equivalence tests (`tests/test_v41_*.py`).
- Web UI (FastAPI, `:7860`): training Tasks, chat, continuation, architecture diagrams, evaluation (C-Eval / CMMLU / GSM8K), data cleaning, expert load.
- A small dense line, `DenseLM` (MiniSeek-D0.1), with released weights.

#### 🎉 Released models

| Model | Params | Notes | Release |
|-------|--------|-------|---------|
| MiniSeek-V1 | 187M-A33M (+19M DSpark) | Full DeepSeek-V4.1-Flash architecture | see HF / release |
| [MiniSeek-M0.1](https://huggingface.co/wyf-yfw/MiniSeek-M0.1) | simple MoE baseline | plain MoE (not full Flash) | released |
| [MiniSeek-D0.1](https://huggingface.co/wyf-yfw/MiniSeek-D0.1) | 31.73M | Dense SFT / chat | 2026-10 |
| [MiniSeek-D0.1-base](https://huggingface.co/wyf-yfw/MiniSeek-D0.1-base) | 28.45M | Dense pretrained base | 2026-10 |

---

# 📌 Quick start

<details>
<summary>My hardware and software (for reference)</summary>

* GPU: NVIDIA GeForce RTX 3090 (24GB) × 1
* Training on Linux, development on Windows
* PyTorch ≥ 2.1; see [requirements.txt](./requirements.txt) for the rest

</details>

## Step 0

```bash
git clone https://github.com/wyf-yfw/miniseek.git
cd miniseek
python -m venv .venv && source .venv/bin/activate   # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Tokenizer: MiniMind (vocab=6400)
hf download jingyaogong/MiniMind2 --include "tokenizer*" --local-dir ./tokenizer
```

> With an older `huggingface_hub` that has no `hf` command, use `huggingface-cli download` instead. You can also point `DS41_TOKENIZER_PATH` at an existing tokenizer directory.

Run the tests on CPU first (they include a tiny V1 pretrain → SFT smoke run):

```bash
python -m pytest tests -q
```

## Ⅰ 🚀 Inference

### 1' Download a model

MiniSeek-V1 will be published on Hugging Face once training finishes. For now you can try the dense model:

```bash
hf download wyf-yfw/MiniSeek-D0.1 --local-dir ./hub/MiniSeek-D0.1
```

### 2' CLI generation

```bash
# any checkpoint trained with this repo (V1 or D0.1)
python -m algo.generate --ckpt tasks/f01-sft/last.pt --prompt "Hello, please introduce yourself."
```

<details>
<summary>Load MiniSeek-D0.1 from Hugging Face (Python)</summary>

```python
import json
from pathlib import Path
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from algo.model.config import ModelConfig
from algo.model.dense import DenseLM
from algo.model.tokenizer import load_tokenizer

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

</details>

### 3' Web UI

```bash
MINISEEK_MONOLITH=1 python -m uvicorn api.app:app --host 127.0.0.1 --port 7860
```

`MINISEEK_MONOLITH=1` runs training, evaluation and the web app in one process (the default on Windows). For a long-running setup, use the three systemd units in `deploy/`: training lives in the worker, so restarting the web app or the API never interrupts a run.

| Page | What it does |
|------|--------------|
| Train | Create Tasks, pick M / Dense / Tiny, start and stop runs, follow logs and loss curves; artifacts go to `tasks/<id>/` |
| Chat / Continue | Load a checkpoint to chat or continue text; V1 multi-turn chat reuses the prefix cache, and a `dspark.pt` next to the checkpoint switches to DSpark speculative decoding |
| Architecture | The V1 diagram (open any layer for operators, formulas, parameters), the DeepSeek-V4.1-Flash paper reference, Dense / Tiny |
| Bench | C-Eval / CMMLU / GSM8K (lm-eval) |
| Data cleaning | Length / language / repetition / dedup / sensitive-word filters for pretraining jsonl |
| Expert load | Runs a routing probe on training data and shows how often each of the 64 routed experts is picked |

## Ⅱ 🛠️ Training

### 1' Download data

```bash
hf download wyf-yfw/MiniSeek-Pretrain pretrain_mix_12g_clean.jsonl --repo-type dataset --local-dir data/PT
hf download wyf-yfw/MiniSeek-SFT sft_miniseek_v5_short256.jsonl --repo-type dataset --local-dir data/SFT
```

See "Data" below for what is in each file.

### 2' Train

> Run these from the repo root, or pick M in the Web "Train" page and fill in the same values.

#### 2.1 Pretraining

```bash
python -m algo.train_pretrain --arch miniseek-v \
  --data data/PT/pretrain_mix_12g_clean.jsonl \
  --out-dir tasks/f01-pretrain \
  --batch-size 8 --valid-ratio 0.01
```

With `--arch miniseek-v` the defaults are `max_seq_len=1024`, a peak learning rate of `3e-4` with cosine decay to `3e-5`, BF16 autocast, and packed batches with sample-level attention masks. Parameters are grouped by the V4 rules: matrices use Muon, the embedding / LM head / Engram tables use Sinkhorn momentum, and RMSNorm weights and scalars use AdamW. The candidate pool is off during pretraining (`--candidate-pool` turns it on). If memory is tight, add `--act-ckpt` for per-layer recomputation or lower `--batch-size`.

The run writes `tasks/f01-pretrain/last.pt` (rolled every 5000 steps) and `best.pt`.

#### 2.2 SFT

```bash
python -m algo.train_sft --arch miniseek-v \
  --init-ckpt tasks/f01-pretrain/last.pt \
  --data data/SFT/sft_miniseek_v5_short256.jsonl \
  --max-seq-len 256 \
  --out-dir tasks/f01-sft
```

SFT turns on the candidate pool by default (the paper enables the hierarchical sparse indexer only in post-training) together with the decoder bounded-replay simulation. Conversations longer than `max_seq_len` are cut from the end so the question is kept. `sft_miniseek_v5_short256.jsonl` is a quick way to check that the model can chat; for full long-context SFT use `sft_miniseek_v5_10g.jsonl` with `--max-seq-len 1024`.

#### 2.3 DSpark drafter (optional)

```bash
python -m algo.train_dspark --backbone tasks/f01-sft/last.pt --out-dir tasks/f01-sft
python -m algo.dspark_calibrate --dspark tasks/f01-sft/dspark.pt --data data/PT/pretrain_mix_12g_clean.jsonl
```

The backbone stays frozen and only the drafter trains, with loss `0.1·CE + 0.9·TV + 1.0·BCE`; calibration writes per-position temperatures back into `dspark.pt`. Keep `dspark.pt` in the same directory as the backbone checkpoint and the web app enables speculative decoding when it loads the backbone.

#### 2.4 Try the model

```bash
python -m algo.generate --ckpt tasks/f01-sft/last.pt --prompt "Hello, please introduce yourself."
python -m algo.eval.valid_ppl --ckpt tasks/f01-pretrain/last.pt --data data/PT/pretrain_mix_12g_clean.jsonl --valid-ratio 0.01
```

<details>
<summary>Other notes</summary>

- Resume: `--resume <ckpt>` (includes optimizer state). `--max-steps` caps the current run; on resume, progress and the learning-rate schedule restart from 0.
- Checkpoints: only the rolling `last.pt` and the best `best.pt` are kept by default; pass `--save-loss-thresholds` if you want milestones.
- Validation: there is no separate file; `--valid-ratio` samples from the training set without touching the source jsonl.
- `torch.compile`: off by default for v41 (MoE dispatch and sparse indexing cause frequent graph breaks), on by default for Dense.
- Dense line: replace `--arch miniseek-v` with `--arch dense` (default `max_seq_len=512`).

</details>

---

# 📌 Data

## Ⅰ Tokenizer

We use the [MiniMind](https://github.com/jingyaogong/minimind) vocabulary (vocab=6400). With a small vocabulary the embedding and the LM head take only 3.3M parameters each, leaving the budget to attention and experts. Without a local tokenizer the code falls back to a byte-level `FallbackTokenizer`, which is only good for smoke tests.

## Ⅱ Pretraining / SFT data

| Use | Path | Notes |
|-----|------|-------|
| Pretraining | `data/PT/pretrain_mix_12g_clean.jsonl` | One `{"text": ...}` per line, mixed Chinese and English, cleaned from a 12G mix |
| SFT (full) | `data/SFT/sft_miniseek_v5_10g.jsonl` | About 10 GiB / 5.96M conversations; a 5G subset is `sft_miniseek_v5_5g.jsonl` |
| SFT (quick check) | `data/SFT/sft_miniseek_v5_short256.jsonl` | Short conversations filtered from the 10G file, about 0.84 GiB / 2.02M rows |
| Validation | none | Training and `valid_ppl` sample it from the training set with `--valid-ratio` |

Hub: [`MiniSeek-Pretrain`](https://huggingface.co/datasets/wyf-yfw/MiniSeek-Pretrain), [`MiniSeek-SFT`](https://huggingface.co/datasets/wyf-yfw/MiniSeek-SFT) (with config `sft_short256`). Sources and mixing ratios are in `data/PT/SOURCES.md`, `data/SFT/SOURCES_v5.md` and `data/SFT/SOURCES_v5_short256.md`.

---

# 📌 Model

## Structure

MiniSeek-V1 has 8 layers: the first 4 form the causal encoder and the last 4 the decoder. Every operator matches DeepSeek-V4.1-Flash.

* **Attention**: Q goes through a low-rank path (512→128→16×128); K and V are one shared 128-dim latent used by all heads; RoPE covers only the last 32 dims and is rotated back on the output; softmax has a learnable per-head sink; the output uses a grouped low-rank projection with 2 groups. Every layer has its own 128-token sliding window.
* **CSA2 modes**: a Full layer compresses the sequence into a global KV (m=2 pairwise pooling in the encoder, m=1 in the decoder) and the indexer picks the top-64 entries; a Reuse layer reuses the source layer's global KV and top-k; a Reindex layer reuses the global KV and re-selects top-k inside the candidate pool.
* **CED**: the encoder output is projected into the global KV only at the first decoder layer (L4), and all later decoder layers reuse it. At inference, a prefix-cache hit needs only a bounded replay of the last 128 tokens.
* **Hierarchical sparse indexer**: L4 takes the max indexer score over blocks of 8 positions and keeps 32 blocks as the candidate pool; L6 (Reindex) re-selects only inside the pool. Off for pretraining, on for SFT.
* **Single-Pass mHC**: four residual streams; each sublayer reads and writes them through pre / post / comb coefficients, with comb normalized by 20 Sinkhorn iterations.
* **MoE**: 1 shared expert plus 64 routed experts with top-6; experts are clamp SwiGLU (inner dim 176). Scores are sqrt(softplus); selection uses score + bias, weights use the raw scores normalized and scaled by 1.5; the bias moves ±0.001 by load, plus a 1e-4 sequence-level balance loss.
* **Engram**: a 2–4-gram hashed memory before L1 and L3; each residual stream writes through its own gate.
* **DSpark**: a 3-block drafter reads the backbone's L5–L7 hidden states, drafts 5 tokens at once, stops early by confidence, and the backbone verifies the block in one pass.

Layer layout (every layer is an attention sublayer plus an MoE sublayer, with mHC residuals):

| Layer | Side | Attention mode | Compression m | Notes |
|-------|------|----------------|---------------|-------|
| L0 | encoder | SWA | — | sliding window only |
| L1 | encoder | SWA | — | Engram |
| L2 | encoder | CSA2 Full | 2 | produces global KV, indexer K and top-k |
| L3 | encoder | CSA2 Reuse | 2 | Engram |
| L4 | decoder | CSA2 Full | 1 | CED: projects the global KV from the encoder output; builds the candidate pool |
| L5 | decoder | CSA2 Reuse | 1 | |
| L6 | decoder | CSA2 Reindex | 1 | re-selects top-k inside the candidate pool |
| L7 | decoder | CSA2 Reuse | 1 | |

Full hyperparameters are in [`algo/model/miniseek_v/config.py`](algo/model/miniseek_v/config.py); the design write-up is [`docs/superpowers/specs/2026-10-29-v41-flash-full-arch-design.md`](docs/superpowers/specs/2026-10-29-v41-flash-full-arch-design.md).

## Configurations

| Model Name | params | vocab | max_seq | d_model | n_layers | heads × head_dim | MoE | note |
|------------|--------|-------|---------|---------|----------|------------------|-----|------|
| MiniSeek-V1 | 187M-A33M | 6400 | 1024 | 512 | 8 (4+4) | 16 × 128 (shared KV) | 1+64, top-6, h=176 | full DeepSeek-V4.1-Flash architecture |
| MiniSeek-D0.1 | 28M | 6400 | 512 | 512 | 8 | 8 × 64 | — | dense |

187M = 158M backbone + 29M Engram tables; A33M counts the backbone parameters each token actually uses (only the 6 selected routed experts, no Engram tables). The DSpark drafter adds another 19M.

## Differences from DeepSeek-V4.1-Flash

| Item | DeepSeek-V4.1-Flash | MiniSeek-V1 |
|------|---------------------|---------------|
| Layers | 40 (20 encoder + 20 decoder) | 8 (4 encoder + 4 decoder) |
| MoE | 1 shared + 384 routed, top-6 | 1 shared + 64 routed, top-6 |
| Parameters | 552B backbone + 196B Engram | 158M backbone + 29M Engram |
| Indexer top-k / candidate pool | 512 / 2048 blocks × 8 | 64 / 32 blocks × 8 |
| KV precision | FP4 main KV, FP8 window KV | BF16 throughout |
| Vision | yes | not yet |

---

# 📌 Evaluation

Every model runs the same Eval-Core:

| # | Item | Notes |
|---|------|-------|
| 1 | Valid PPL | sampled from the training jsonl with `--valid-ratio` |
| 2 | Gen-Smoke | 20 fixed prompts, seed=42 |
| 3 | C-Eval valid | lm-eval |
| 4 | CMMLU | lm-eval (needs `trust_remote_code`; `datasets>=2.14,<4.0`) |
| 5 | GSM8K | lm-eval generate_until |
| 6 | Efficiency | peak memory and tokens/s on the same hardware |

```bash
python -m algo.eval.valid_ppl --ckpt tasks/f01-pretrain/last.pt --data data/PT/pretrain_mix_12g_clean.jsonl --valid-ratio 0.01
python -m algo.eval.gen_smoke --ckpt tasks/f01-sft/last.pt --seed 42
```

Run C-Eval / CMMLU / GSM8K from the Web "Bench" page or with `algo/eval/run_benchmark.py` (see [`algo/eval/README.md`](algo/eval/README.md)). V1 results will be added here after training; dense-line results are in `algo/eval/reports/`.

---

# 📌 Layout

```text
algo/
├── model/miniseek_v/          # MiniSeek-V: config · rope · attention · mhc · moe · engram · model · cache · dspark
├── model/dense.py      # MiniSeek-D: DenseLM
├── optim/              # Muon · Sinkhorn momentum · parameter groups
├── train_pretrain.py   # pretraining
├── train_sft.py        # SFT
├── train_dspark.py     # DSpark drafter
├── dspark_calibrate.py # DSpark confidence calibration (STS)
├── generate.py         # CLI generation
└── eval/               # Valid PPL · Gen-Smoke · lm-eval
api/                    # FastAPI: training Tasks · inference · architecture · bench · worker
frontend/               # Web UI
deploy/                 # systemd units (web / api / worker)
scripts/                # data checks · cleaning · smoke
tests/                  # tests; tests/miniseek_v_ref/ ports the official inference code
docs/                   # design docs and implementation plans
images/                 # README images
tasks/                  # training outputs (not in Git)
```

---

# 📌 Acknowledgements

- Architecture and reference implementation: the [DeepSeek-V4.1-Flash paper](https://arxiv.org/abs/2609.19969) and the official inference code in [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) (MIT)
- Tokenizer and README layout: [jingyaogong/minimind](https://github.com/jingyaogong/minimind)
- Benchmarks: [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), C-Eval, CMMLU

Issues and PRs are welcome.

---

# 🎓 Citation

If this project helps you, please cite the repository or link to it:

```bibtex
@misc{miniseek2026,
  title        = {MiniSeek: A Tiny Trainable Reproduction of the DeepSeek-V4.1-Flash Architecture},
  author       = {wyf-yfw},
  year         = {2026},
  howpublished = {\url{https://github.com/wyf-yfw/miniseek}}
}
```

---

# ⚖️ License

[MIT](LICENSE)
