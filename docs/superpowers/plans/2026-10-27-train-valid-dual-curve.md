# Train/Valid Dual-Curve Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** During MiniSeek training, periodically evaluate on a validation jsonl and show train + valid loss curves on the task chart.

**Architecture:** `train_pretrain.py` runs in-process valid every N steps and prints a dedicated log line; `task_runner` tails that line into `metrics.json` under `valid_loss`; the Web UI draws a second series. Same GPU/process as training. Default valid file `data/valid_mix.jsonl`, `valid_every=500`.

**Tech Stack:** Python/PyTorch training, FastAPI task runner, canvas chart in `frontend/index.html`.

## Global Constraints

- Default `valid_path`: `data/valid_mix.jsonl`; form-overridable.
- Default `valid_every`: `500` (tiny arch may use smaller, e.g. 20).
- `valid_every=0` or empty/missing valid file → train-only (current behavior).
- Use `build_model` for MoE/Dense; do not hardcode DenseLM for valid.
- No separate 验证 sidebar page in this plan.
- Never use CloudAgent; edit on wyf; deploy with `./scripts/deploy.sh`.
- Do not restart tunnel; do not kill a running MoE train for deploy unless user asks.

## File map

| File | Role |
|------|------|
| `algo/train_pretrain.py` | CLI + valid loop + log line |
| `algo/train_sft.py` | Same flags if SFT shares loop pattern (parity) |
| `api/task_runner.py` | Pass CLI flags; parse valid log → metrics |
| `api/app.py` / defaults | Default hparams for UI |
| `frontend/index.html` | Form fields + dual series chart |
| `tests/test_valid_metrics.py` | Parser + CLI flag tests |

**Log line contract (must match parser):**

```
valid step=500 loss=3.2100 ppl=24.78 tokens=102400
```

Regex suggestion: `valid\s+step=(\d+)\s+loss=([0-9.eE+-]+)\s+ppl=([0-9.eE+-]+)`

**metrics.json shape:**

```json
{
  "loss": [{"step": 500, "loss": 2.9, "ppl": 18.1, "lr": 5e-5}],
  "valid_loss": [{"step": 500, "loss": 3.21, "ppl": 24.78, "tokens": 102400}],
  "expert_load": null
}
```

---

### Task 1: Valid eval helper + train_pretrain CLI/loop

**Files:**
- Modify: `algo/train_pretrain.py`
- Test: `tests/test_valid_metrics.py`

**Interfaces:**
- Consumes: existing `JsonlTextDataset`, `collate_pad`, `build_model`
- Produces:
  - `evaluate_valid(model, loader, device) -> dict` with keys `loss: float`, `ppl: float`, `tokens: int`
  - CLI: `--valid` (str, default ""), `--valid-every` (int, default 0 meaning off unless Web sets 500), `--valid-max-samples` (int, default 512)
  - Prints `valid step=... loss=... ppl=... tokens=...` when due

- [ ] **Step 1: Write failing unit test for evaluate_valid on tiny random model**

```python
# tests/test_valid_metrics.py
import torch
from algo.model import build_model
from algo.model.config import ModelConfig
from algo.train_pretrain import evaluate_valid

def test_evaluate_valid_returns_finite():
    cfg = ModelConfig(arch="tiny", n_layers=2, d_model=64, n_heads=4, vocab_size=128, max_seq_len=32)
    # adjust fields to match real ModelConfig tiny constructor used in repo
    model = build_model(cfg).eval()
    # one fake batch: input_ids [B,T], labels [B,T]
    B, T = 2, 16
    x = torch.randint(0, 128, (B, T))
    y = x.clone()
    class Loader:
        def __iter__(self):
            yield x, y
    out = evaluate_valid(model, Loader(), torch.device("cpu"))
    assert out["tokens"] > 0
    assert out["loss"] == out["loss"]  # not NaN
    assert out["ppl"] >= 1.0
```

(Adapt `ModelConfig(...)` to whatever the repo's tiny config factory is — prefer calling the same helper tests already use in `test_config_tiny.py`.)

- [ ] **Step 2: Run test — expect fail (missing evaluate_valid)**

```bash
cd /home/wyf/miniseek && python -m pytest tests/test_valid_metrics.py::test_evaluate_valid_returns_finite -v
```

- [ ] **Step 3: Implement `evaluate_valid` and wire loop**

```python
@torch.no_grad()
def evaluate_valid(model, loader, device) -> dict:
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for input_ids, labels in loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        out = model(input_ids, labels=labels)  # match train forward API
        # extract scalar CE loss and token count the same way train does
        ...
    mean = total_loss / max(1, total_tokens)
    return {"loss": mean, "ppl": math.exp(min(mean, 20)), "tokens": total_tokens}
```

Add argparse flags; after optimizer step / logging block, if `args.valid_every > 0` and `global_step % args.valid_every == 0` and valid path exists:

```python
stats = evaluate_valid(model, valid_loader, device)
print(
    f"valid step={global_step} loss={stats['loss']:.4f} "
    f"ppl={stats['ppl']:.2f} tokens={stats['tokens']}",
    flush=True,
)
model.train()
```

Build `valid_loader` once at start if path non-empty.

- [ ] **Step 4: Re-run test — PASS**

- [ ] **Step 5: Commit**

```bash
git add algo/train_pretrain.py tests/test_valid_metrics.py
git commit -m "feat(train): periodic valid eval log lines for dual-curve"
```

---

### Task 2: task_runner CLI flags + metrics parser

**Files:**
- Modify: `api/task_runner.py` (`build_train_cmd`, `_parse_and_append_metrics`, metrics reset)
- Modify: `api/app.py` defaults if present
- Test: `tests/test_valid_metrics.py`

**Interfaces:**
- Consumes: log line contract from Task 1
- Produces: `metrics["valid_loss"]` list; hparams `valid_path`, `valid_every`, `valid_max_samples`

- [ ] **Step 1: Failing test for parser**

```python
def test_parse_valid_line_appends(tmp_path, monkeypatch):
    from api import tasks_store as ts
    from api import task_runner as tr
    # point TASKS_ROOT at tmp_path via monkeypatch like other tests
    tid = "t-valid"
    # create task dir + empty metrics
    ts.write_metrics(tid, {"loss": [], "valid_loss": [], "expert_load": None})
    state = {"loss_bucket": 0, "valid_bucket": 0}
    tr._parse_and_append_metrics(
        tid,
        "valid step=500 loss=3.2100 ppl=24.78 tokens=102400",
        plot_every=500,
        state=state,
    )
    m = ts.read_metrics(tid)
    assert m["valid_loss"][-1]["step"] == 500
    assert abs(m["valid_loss"][-1]["loss"] - 3.21) < 1e-6
```

- [ ] **Step 2: Run — FAIL until parser handles valid lines**

- [ ] **Step 3: Implement**

In `build_train_cmd`, after other flags:

```python
valid_path = str(_hp(hp, "valid_path", PROJECT_ROOT / "data" / "valid_mix.jsonl") or "").strip()
valid_every = int(_hp(hp, "valid_every", 20 if arch == "tiny" else 500))
valid_max_samples = int(_hp(hp, "valid_max_samples", 512))
if valid_path and valid_every > 0:
    cmd.extend([
        "--valid", valid_path,
        "--valid-every", str(valid_every),
        "--valid-max-samples", str(valid_max_samples),
    ])
```

In `_parse_and_append_metrics`, before or after STEP_RE:

```python
VALID_RE = re.compile(r"valid\s+step=(\d+)\s+loss=([0-9.eE+-]+)\s+ppl=([0-9.eE+-]+)")
# optional tokens= group
vm = VALID_RE.search(line)
if vm:
    step = int(vm.group(1))
    metrics = ts.read_metrics(task_id)
    points = list(metrics.get("valid_loss") or [])
    points.append({"step": step, "loss": float(vm.group(2)), "ppl": float(vm.group(3))})
    metrics["valid_loss"] = points[-5000:]
    ts.write_metrics(task_id, metrics)
    return
```

On start reset: `ts.write_metrics(task_id, {"loss": [], "valid_loss": [], "expert_load": None})`.

- [ ] **Step 4: Tests PASS**

- [ ] **Step 5: Commit**

```bash
git add api/task_runner.py api/app.py tests/test_valid_metrics.py
git commit -m "feat(api): parse valid log into metrics.valid_loss"
```

---

### Task 3: Frontend form + dual-series chart

**Files:**
- Modify: `frontend/index.html` (hparams form near plot-every; `drawLoss`; `applyMetrics`; collect/apply hp)

**Interfaces:**
- Consumes: `metrics.loss`, `metrics.valid_loss`
- Produces: UI fields `valid-path`, `valid-every`, `valid-max-samples`; chart with train (blue) + valid (orange)

- [ ] **Step 1: Add form inputs** next to plot-every:

```html
<label>valid 路径 <input id="valid-path" value="data/valid_mix.jsonl" /></label>
<label>valid every <input id="valid-every" type="number" value="500" /></label>
<label>valid max samples <input id="valid-max-samples" type="number" value="512" /></label>
```

Wire into `collectHparams` / `applyHparams` as `valid_path`, `valid_every`, `valid_max_samples`.

- [ ] **Step 2: Update `applyMetrics`**

```javascript
let lossPoints = [];
let validPoints = [];
function applyMetrics(m) {
  lossPoints = (m && m.loss) || [];
  validPoints = (m && m.valid_loss) || [];
  drawLoss();
  ...
}
```

- [ ] **Step 3: Update `drawLoss`** to include both series in y-scale domain; stroke train as now; stroke valid with `#ea580c`; legend in caption: train latest + valid latest.

- [ ] **Step 4: Manual smoke** (after deploy): open UI, confirm fields exist; optional tiny run later in Task 4.

- [ ] **Step 5: Commit**

```bash
git add frontend/index.html
git commit -m "feat(web): train/valid dual loss curve and form fields"
```

---

### Task 4: SFT parity (if train_sft shares metrics) + deploy

**Files:**
- Modify: `algo/train_sft.py` only if it emits the same step log pattern and is launched by Web SFT mode
- Deploy: `./scripts/deploy.sh` on wyf

- [ ] **Step 1:** If SFT training is used from UI, mirror `--valid*` flags and `evaluate_valid` call; else skip with note in commit body.

- [ ] **Step 2: Full pytest**

```bash
cd /home/wyf/miniseek && python -m pytest -q
```

Expected: all previous + new tests PASS.

- [ ] **Step 3: Push and deploy**

```bash
git push origin HEAD
./scripts/deploy.sh   # or --restart if already up to date
```

- [ ] **Step 4: Optional tiny smoke** with `valid_every=20`, `valid_max_samples=64`, confirm `metrics.json` has `valid_loss` and UI shows second line. Do **not** stop user's MoE run.

- [ ] **Step 5: Sync Windows** `D:\miniseek` `git pull --ff-only`.

---

## Spec coverage

| Spec item | Task |
|-----------|------|
| valid_path default valid_mix | 2–3 |
| valid_every default 500 | 2–3 |
| dual curve train/valid | 3 |
| metrics valid_loss | 1–2 |
| build_model MoE | 1 |
| same process/GPU | 1 |
| valid_every=0 train-only | 1–2 |
| no standalone 验证 page | — |
