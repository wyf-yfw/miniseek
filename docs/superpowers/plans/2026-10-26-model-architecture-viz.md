# Model Architecture Viz Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a top-level「架构」tab that inspects a MiniSeek `.pt` / HF export and renders an interactive layer diagram with a detail drawer.

**Architecture:** Pure inspect logic in `web/arch_inspect.py` (load blob on CPU, aggregate `numel` by key prefix, return JSON blocks). `web_ui.py` only exposes `POST /api/arch/inspect`. Frontend is vanilla JS/CSS in `web/index.html`: fourth tab, left picker + summary, center stack diagram, right drawer.

**Tech Stack:** Python 3, FastAPI, PyTorch CPU `torch.load`, existing MiniSeek `ModelConfig` / Dense key layout, pytest, native HTML/CSS/JS (no new frontend deps).

## Global Constraints

- Spec: `docs/superpowers/specs/2026-10-26-model-architecture-viz-design.md`
- No GPU load; `map_location="cpu"`; do not instantiate full model for inspect (state_dict + config only)
- Same path normalization as `InferenceEngine.load_model`: training `{config,model}` blob, HF dir / `pytorch_model.bin` + `config.json`
- Tabs order: `训练 | 对话 | 续写 | 架构`; `page-tabs` four columns, wider max-width
- No Netron / ONNX / 3D / activation viz in v1
- MoE: if `arch=moe` or keys contain expert prefixes, expose router / shared / routed in layer `parts`; Dense path must stay correct when MoE files absent (current `model/moe.py` is still a stub — use synthetic MoE state_dict in unit tests)
- Primary worktree: `/workspace/miniseek`; sync to `D:\miniseek` on wangyifan6 after UI tasks when verifying live
- Sample Dense ckpt on Windows: `D:\miniseek\checkpoints\hf\MiniSeek-D0.1-base.pt` (~28.45M, 8 layers)

## File map

| File | Responsibility |
|------|----------------|
| `web/arch_inspect.py` | NEW: `load_ckpt_blob`, `inspect_blob` / `inspect_path` → API-shaped dict |
| `web_ui.py` | `POST /api/arch/inspect` + thin pydantic body |
| `web/index.html` | 架构 tab UI, CSS, fetch+render diagram/drawer |
| `tests/test_arch_inspect.py` | Unit tests: tiny Dense, tie, missing path, synthetic MoE parts |
| `tests/test_arch_api.py` | FastAPI client: 404 + happy path on synthetic `.pt` |

---

### Task 1: `arch_inspect` pure module + Dense unit tests

**Files:**
- Create: `web/arch_inspect.py`
- Create: `tests/test_arch_inspect.py`

**Interfaces:**
- Produces:
  - `load_ckpt_blob(path: str | Path) -> dict` with keys `model` (state_dict), `config` (dict), optional `step`/`loss`/`source`
  - `inspect_blob(blob: dict, *, path: str = "") -> dict` matching spec JSON (`path`, `config`, `meta`, `total_params`, `tie_word_embeddings`, `blocks`, `tail`)
  - `inspect_path(path: str | Path) -> dict` = load + inspect; raises `FileNotFoundError` if missing; raises `ValueError` for unreadable/malformed

Dense key prefixes (from `DenseLM`):
- `tok_emb.` → embed
- `layers.{i}.attn` / `layers.{i}.attn_norm` → attn part (+ norms counted in layer total; put norm params into attn or a silent bucket included in layer `params`)
- `layers.{i}.ffn` / `layers.{i}.ffn_norm` → ffn part
- `norm.` → final norm
- `lm_head.` → lm_head; if `tie_word_embeddings` and `lm_head.weight` is same storage as `tok_emb.weight` OR config says tie, set `tied: true` and **do not double-count** lm_head into `total_params` (count embed once; lm_head `params` still reported as shape numel with `tied: true`)

Param counting rule for `total_params`: sum unique parameter storages. Practical approach for state_dict tensors: if tie and both keys present, count `tok_emb.weight` once and skip adding `lm_head.weight` numel to total (still list lm_head block with `params=numel`, `tied=true`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_arch_inspect.py
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from model.config import ModelConfig
from model.dense import DenseLM, count_parameters
from web.arch_inspect import inspect_blob, inspect_path, load_ckpt_blob


def _tiny_blob():
    cfg = ModelConfig.tiny()
    model = DenseLM(cfg)
    return {
        "config": cfg.to_dict(),
        "model": model.state_dict(),
        "step": 12,
        "loss": 2.5,
    }


def test_inspect_tiny_dense_layers_and_params():
    blob = _tiny_blob()
    out = inspect_blob(blob, path="tiny.pt")
    assert out["config"]["arch"] == "tiny"
    assert out["config"]["n_layers"] == 2
    assert out["config"]["d_model"] == 64
    assert out["tie_word_embeddings"] is True
    layers = [b for b in out["blocks"] if b["kind"] == "layer"]
    assert len(layers) == 2
    assert layers[0]["id"] == "layer-0"
    part_ids = {p["id"] for p in layers[0]["parts"]}
    assert "attn" in part_ids and "ffn" in part_ids
    # tied: total matches nn.Module unique params
    expected = count_parameters(DenseLM(ModelConfig.tiny()))
    assert out["total_params"] == expected
    lm = next(x for x in out["tail"] if x["id"] == "lm_head")
    assert lm["tied"] is True


def test_inspect_path_missing():
    with pytest.raises(FileNotFoundError):
        inspect_path("/no/such/ckpt.pt")


def test_load_and_inspect_roundtrip(tmp_path: Path):
    blob = _tiny_blob()
    p = tmp_path / "t.pt"
    torch.save(blob, p)
    loaded = load_ckpt_blob(p)
    assert "model" in loaded and "config" in loaded
    out = inspect_path(p)
    assert out["meta"]["step"] == 12
    assert out["blocks"][0]["kind"] == "embedding"


def test_synthetic_moe_parts():
    """MoE implementation may be stub; inspect must still classify expert keys."""
    cfg = ModelConfig(
        arch="moe",
        d_model=64,
        n_layers=1,
        n_heads=1,
        head_dim=64,
        intermediate_size=256,
        vocab_size=100,
        max_seq_len=32,
        n_shared_experts=1,
        n_routed_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=128,
        tie_word_embeddings=True,
    )
    # Minimal fake weights matching intended MoE naming
    sd = {
        "tok_emb.weight": torch.zeros(100, 64),
        "layers.0.attn_norm.weight": torch.ones(64),
        "layers.0.attn.qkv.weight": torch.zeros(192, 64),
        "layers.0.attn.o_proj.weight": torch.zeros(64, 64),
        "layers.0.ffn_norm.weight": torch.ones(64),
        "layers.0.ffn.gate.weight": torch.zeros(4, 64),
        "layers.0.ffn.shared_experts.0.fc1.weight": torch.zeros(128, 64),
        "layers.0.ffn.shared_experts.0.fc2.weight": torch.zeros(64, 128),
        "layers.0.ffn.experts.0.fc1.weight": torch.zeros(128, 64),
        "layers.0.ffn.experts.0.fc2.weight": torch.zeros(64, 128),
        "layers.0.ffn.experts.1.fc1.weight": torch.zeros(128, 64),
        "layers.0.ffn.experts.1.fc2.weight": torch.zeros(64, 128),
        "layers.0.ffn.experts.2.fc1.weight": torch.zeros(128, 64),
        "layers.0.ffn.experts.2.fc2.weight": torch.zeros(64, 128),
        "layers.0.ffn.experts.3.fc1.weight": torch.zeros(128, 64),
        "layers.0.ffn.experts.3.fc2.weight": torch.zeros(64, 128),
        "norm.weight": torch.ones(64),
        "lm_head.weight": torch.zeros(100, 64),  # untied storage in this fake
    }
    # Force tie accounting via config even if tensors differ
    blob = {"config": cfg.to_dict(), "model": sd, "step": 1, "loss": 1.0}
    out = inspect_blob(blob)
    layer = next(b for b in out["blocks"] if b["kind"] == "layer")
    ids = {p["id"] for p in layer["parts"]}
    assert "attn" in ids
    assert "router" in ids or "gate" in ids
    assert "shared_experts" in ids
    assert "routed_experts" in ids
    routed = next(p for p in layer["parts"] if p["id"] == "routed_experts")
    assert routed["detail"]["n_routed_experts"] == 4
    assert routed["detail"]["num_experts_per_tok"] == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /workspace/miniseek && .venv/bin/python -m pytest tests/test_arch_inspect.py -v`
Expected: FAIL (`ModuleNotFoundError: web.arch_inspect` or import error)

- [ ] **Step 3: Implement `web/arch_inspect.py`**

```python
# web/arch_inspect.py
"""Inspect MiniSeek checkpoints into architecture-viz JSON (CPU only)."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from model.config import ModelConfig

_LAYER_RE = re.compile(r"^layers\.(\d+)\.(.+)$")


def load_ckpt_blob(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))

    weight_path: Optional[Path] = None
    cfg_path: Optional[Path] = None
    if path.is_dir() and (path / "pytorch_model.bin").exists():
        weight_path = path / "pytorch_model.bin"
        cfg_path = path / "config.json"
    elif path.name == "pytorch_model.bin":
        weight_path = path
        cfg_path = path.parent / "config.json"

    if weight_path is not None:
        sd = torch.load(weight_path, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "model" in sd:
            blob = sd
            blob.setdefault("source", "hf-blob")
        else:
            cfg_raw: Dict[str, Any] = {}
            if cfg_path and cfg_path.exists():
                cfg_raw = json.loads(cfg_path.read_text(encoding="utf-8"))
            allowed = set(ModelConfig.__dataclass_fields__)  # type: ignore[attr-defined]
            cfg_dict = {k: v for k, v in cfg_raw.items() if k in allowed}
            blob = {
                "model": sd,
                "config": cfg_dict,
                "step": cfg_raw.get("train_step", cfg_raw.get("step")),
                "loss": cfg_raw.get("train_loss", cfg_raw.get("loss")),
                "source": "hf",
            }
        return blob

    blob = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(blob, dict) or "model" not in blob:
        raise ValueError(f"无法识别的 checkpoint 格式: {path}")
    blob.setdefault("source", "training")
    return blob


def _numel(t: torch.Tensor) -> int:
    return int(t.numel())


def _sum_prefix(sd: Dict[str, torch.Tensor], prefix: str) -> int:
    return sum(_numel(v) for k, v in sd.items() if k.startswith(prefix))


def inspect_blob(blob: Dict[str, Any], *, path: str = "") -> Dict[str, Any]:
    if "model" not in blob:
        raise ValueError("checkpoint 缺少 model 权重")
    sd: Dict[str, torch.Tensor] = blob["model"]
    if not isinstance(sd, dict):
        raise ValueError("model 不是 state_dict")

    cfg_raw = blob.get("config") or {}
    if not isinstance(cfg_raw, dict):
        cfg_raw = {}
    try:
        cfg = ModelConfig.from_dict(cfg_raw)
        cfg_out = cfg.to_dict()
    except Exception:
        cfg = None
        cfg_out = dict(cfg_raw)

    tie = bool(cfg_out.get("tie_word_embeddings", True))
    # Detect shared storage if both present
    if "tok_emb.weight" in sd and "lm_head.weight" in sd:
        try:
            if sd["tok_emb.weight"].data_ptr() == sd["lm_head.weight"].data_ptr():
                tie = True
        except Exception:
            pass

    n_layers = int(cfg_out.get("n_layers") or 0)
    if not n_layers:
        idxs = []
        for k in sd:
            m = _LAYER_RE.match(k)
            if m:
                idxs.append(int(m.group(1)))
        n_layers = (max(idxs) + 1) if idxs else 0

    blocks: List[Dict[str, Any]] = []
    embed_params = _sum_prefix(sd, "tok_emb.")
    blocks.append(
        {
            "id": "embed",
            "kind": "embedding",
            "label": "Token Embedding",
            "params": embed_params,
        }
    )

    is_moe = bool(
        (cfg_out.get("arch") == "moe")
        or cfg_out.get("use_moe")
        or any(".ffn.experts." in k or ".ffn.shared_experts." in k for k in sd)
    )

    for i in range(n_layers):
        pref = f"layers.{i}."
        layer_keys = {k: v for k, v in sd.items() if k.startswith(pref)}
        attn_params = sum(
            _numel(v)
            for k, v in layer_keys.items()
            if k.startswith(pref + "attn")  # attn. and attn_norm.
        )
        parts: List[Dict[str, Any]] = [
            {
                "id": "attn",
                "label": "Attention",
                "params": attn_params,
                "detail": {
                    "n_heads": cfg_out.get("n_heads"),
                    "head_dim": cfg_out.get("head_dim"),
                },
            }
        ]
        if is_moe:
            gate_params = _sum_prefix(sd, pref + "ffn.gate.")
            shared_params = _sum_prefix(sd, pref + "ffn.shared_experts.")
            routed_params = _sum_prefix(sd, pref + "ffn.experts.")
            ffn_norm = _sum_prefix(sd, pref + "ffn_norm.")
            # fold ffn_norm into router bucket display or attn-adjacent; add to layer total via sum
            parts.append(
                {
                    "id": "router",
                    "label": "Router / Gate",
                    "params": gate_params + ffn_norm,
                    "detail": {
                        "num_experts_per_tok": cfg_out.get("num_experts_per_tok"),
                    },
                }
            )
            parts.append(
                {
                    "id": "shared_experts",
                    "label": "Shared Experts",
                    "params": shared_params,
                    "detail": {
                        "n_shared_experts": cfg_out.get("n_shared_experts"),
                        "moe_intermediate_size": cfg_out.get("moe_intermediate_size"),
                    },
                }
            )
            n_routed = int(cfg_out.get("n_routed_experts") or 0)
            per = (routed_params // n_routed) if n_routed else routed_params
            parts.append(
                {
                    "id": "routed_experts",
                    "label": "Routed Experts",
                    "params": routed_params,
                    "detail": {
                        "n_routed_experts": n_routed,
                        "num_experts_per_tok": cfg_out.get("num_experts_per_tok"),
                        "params_per_expert": per,
                        "moe_intermediate_size": cfg_out.get("moe_intermediate_size"),
                    },
                }
            )
        else:
            ffn_params = sum(
                _numel(v)
                for k, v in layer_keys.items()
                if k.startswith(pref + "ffn")  # ffn. and ffn_norm.
            )
            parts.append(
                {
                    "id": "ffn",
                    "label": "FFN",
                    "params": ffn_params,
                    "detail": {
                        "intermediate_size": cfg_out.get("intermediate_size"),
                    },
                }
            )
        layer_params = sum(_numel(v) for v in layer_keys.values())
        blocks.append(
            {
                "id": f"layer-{i}",
                "kind": "layer",
                "index": i,
                "label": f"Layer {i}",
                "params": layer_params,
                "parts": parts,
            }
        )

    norm_params = _sum_prefix(sd, "norm.")
    lm_params = _sum_prefix(sd, "lm_head.")
    tail = [
        {
            "id": "norm",
            "kind": "norm",
            "label": "Final RMSNorm",
            "params": norm_params,
        },
        {
            "id": "lm_head",
            "kind": "lm_head",
            "label": "LM Head",
            "params": lm_params,
            "tied": tie,
        },
    ]

    total = 0
    seen_ptrs = set()
    for k, v in sd.items():
        if tie and k.startswith("lm_head."):
            continue
        try:
            ptr = v.data_ptr()
            if ptr in seen_ptrs:
                continue
            seen_ptrs.add(ptr)
        except Exception:
            pass
        total += _numel(v)

    meta = {
        "step": blob.get("step"),
        "loss": blob.get("loss"),
        "source": blob.get("source"),
    }
    return {
        "path": path,
        "config": cfg_out,
        "meta": meta,
        "total_params": total,
        "tie_word_embeddings": tie,
        "blocks": blocks,
        "tail": tail,
    }


def inspect_path(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    blob = load_ckpt_blob(path)
    return inspect_blob(blob, path=str(path))
```

Ensure `web/` is a package (`web/__init__.py` already exists via tasks_store imports).

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /workspace/miniseek && .venv/bin/python -m pytest tests/test_arch_inspect.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /workspace/miniseek
git add web/arch_inspect.py tests/test_arch_inspect.py
git commit -m "feat(arch): add checkpoint inspect helper for architecture viz"
```

---

### Task 2: FastAPI `POST /api/arch/inspect`

**Files:**
- Modify: `web_ui.py` (imports + route near infer endpoints)
- Create: `tests/test_arch_api.py`

**Interfaces:**
- Consumes: `inspect_path` from Task 1
- Produces: HTTP JSON; 404 if path missing; 400 if ValueError

- [ ] **Step 1: Write the failing API test**

```python
# tests/test_arch_api.py
from __future__ import annotations

from pathlib import Path

import torch
from fastapi.testclient import TestClient

from model.config import ModelConfig
from model.dense import DenseLM
from web_ui import app


def test_arch_inspect_404():
    client = TestClient(app)
    r = client.post("/api/arch/inspect", json={"ckpt_path": "/no/such.pt"})
    assert r.status_code == 404


def test_arch_inspect_tiny(tmp_path: Path):
    cfg = ModelConfig.tiny()
    model = DenseLM(cfg)
    p = tmp_path / "tiny.pt"
    torch.save({"config": cfg.to_dict(), "model": model.state_dict(), "step": 3, "loss": 1.1}, p)
    client = TestClient(app)
    r = client.post("/api/arch/inspect", json={"ckpt_path": str(p)})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["config"]["n_layers"] == 2
    assert len([b for b in body["blocks"] if b["kind"] == "layer"]) == 2
    assert body["total_params"] > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /workspace/miniseek && .venv/bin/python -m pytest tests/test_arch_api.py -v`
Expected: FAIL (404 on route missing → actually 404/405 for unknown path; TestClient may return 404 for missing route)

- [ ] **Step 3: Wire the route**

Near other pydantic models, reuse or add:

```python
class ArchInspectRequest(BaseModel):
    ckpt_path: str
```

Near infer routes:

```python
from web.arch_inspect import inspect_path as arch_inspect_path

@app.post("/api/arch/inspect")
def api_arch_inspect(req: ArchInspectRequest) -> dict:
    path = (req.ckpt_path or "").strip()
    if not path:
        raise HTTPException(status_code=400, detail="ckpt_path 不能为空")
    try:
        return arch_inspect_path(path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"找不到 checkpoint: {path}")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"解析失败: {exc}") from exc
```

- [ ] **Step 4: Run API tests**

Run: `cd /workspace/miniseek && .venv/bin/python -m pytest tests/test_arch_api.py tests/test_arch_inspect.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add web_ui.py tests/test_arch_api.py
git commit -m "feat(arch): add POST /api/arch/inspect endpoint"
```

---

### Task 3:「架构」tab UI — layout, diagram, drawer

**Files:**
- Modify: `web/index.html`

**Interfaces:**
- Consumes: `POST /api/arch/inspect`, `/api/pick-file`, `/api/checkpoints`
- Produces: interactive panel `#arch`

- [ ] **Step 1: Widen tabs and add fourth button + empty panel**

Change `.page-tabs`:
- `grid-template-columns: repeat(4, minmax(0, 1fr));`
- `max-width: 560px;` (was 420)

In `<nav class="page-tabs">` after 续写:

```html
<button type="button" role="tab" aria-selected="false" data-tab="arch">架构</button>
```

After `#continue` section, before `</main>`:

```html
<section id="arch" class="panel">
  <div class="arch-shell">
    <aside class="arch-side">
      <h2>Checkpoint</h2>
      <label><span class="field-title">路径</span>
        <span class="field-hint">训练产出的 .pt，或 HF 目录 / pytorch_model.bin。</span>
        <div class="path-row">
          <input id="arch-ckpt-manual" type="text" placeholder="选择或粘贴 .pt 路径" />
          <button type="button" class="btn-browse" data-for="arch-ckpt-manual" id="btn-pick-arch-ckpt">打开选择</button>
        </div>
      </label>
      <label>最近/扫描到的列表
        <select id="arch-ckpt-select"></select>
      </label>
      <div class="row">
        <button type="button" id="btn-arch-refresh-ckpt">刷新列表</button>
        <button type="button" id="btn-arch-inspect">解析</button>
      </div>
      <p id="arch-msg" class="arch-msg"></p>
      <div id="arch-summary" class="arch-summary" style="display:none"></div>
    </aside>
    <div class="arch-main">
      <div class="arch-toolbar">
        <div class="title">结构图</div>
        <div id="arch-status" class="live"><span class="dot" id="arch-dot"></span><span id="arch-status-text">未解析</span></div>
      </div>
      <div class="arch-canvas-wrap">
        <div id="arch-canvas" class="arch-canvas">
          <span class="chat-empty" id="arch-empty">选择 checkpoint 后点「解析」。</span>
        </div>
        <aside id="arch-drawer" class="arch-drawer" hidden>
          <div class="arch-drawer-head">
            <strong id="arch-drawer-title">层详情</strong>
            <button type="button" id="btn-arch-drawer-close" aria-label="关闭">×</button>
          </div>
          <div id="arch-drawer-body"></div>
        </aside>
      </div>
    </div>
  </div>
</section>
```

- [ ] **Step 2: Add CSS** (inside existing `<style>`)

```css
.arch-shell { display: grid; grid-template-columns: 280px 1fr; gap: 16px; min-height: 560px; }
.arch-side { background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 14px; }
.arch-main { background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 14px; display: flex; flex-direction: column; min-width: 0; }
.arch-toolbar { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
.arch-canvas-wrap { position: relative; flex: 1; display: grid; grid-template-columns: 1fr auto; gap: 0; min-height: 420px; }
.arch-canvas { overflow: auto; padding: 8px 4px 24px; display: flex; flex-direction: column; align-items: center; gap: 8px; }
.arch-block {
  width: min(420px, 100%);
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 10px 14px;
  background: #f8fafc;
  cursor: default;
  text-align: center;
}
.arch-block.layer { cursor: pointer; }
.arch-block.layer:hover, .arch-block.layer.selected {
  border-color: var(--accent);
  box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.15);
}
.arch-block .lbl { font-weight: 600; }
.arch-block .sub { color: var(--muted); font-size: 12px; margin-top: 4px; }
.arch-connector { width: 2px; height: 12px; background: var(--line); }
.arch-drawer {
  width: 280px;
  border-left: 1px solid var(--line);
  padding: 12px;
  background: #fbfbfc;
  overflow: auto;
}
.arch-drawer[hidden] { display: none; }
.arch-drawer-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
.arch-msg { color: var(--muted); font-size: 13px; min-height: 1.2em; }
.arch-msg.err { color: var(--fail); }
.arch-summary { font-size: 13px; line-height: 1.5; color: var(--ink); margin-top: 10px; padding-top: 10px; border-top: 1px solid var(--line); }
.arch-part { border: 1px solid var(--line); border-radius: 8px; padding: 8px 10px; margin: 8px 0; background: #fff; }
@media (max-width: 900px) {
  .arch-shell { grid-template-columns: 1fr; }
  .arch-canvas-wrap { grid-template-columns: 1fr; }
  .arch-drawer { width: auto; border-left: 0; border-top: 1px solid var(--line); }
}
```

- [ ] **Step 3: Wire PATH_PICK + JS**

Add to `PATH_PICK`:

```javascript
"arch-ckpt-manual": { title: "选择架构 checkpoint", filetypes: "Checkpoint|*.pt;All|*.*" },
```

Append script section (after 续写 block is fine):

```javascript
// --- 架构可视化 ---
let archData = null;
let archSelectedLayer = null;

function fmtParams(n) {
  if (n == null) return "-";
  if (n >= 1e6) return (n / 1e6).toFixed(2) + "M";
  if (n >= 1e3) return (n / 1e3).toFixed(1) + "K";
  return String(n);
}

function setArchMsg(text, isErr) {
  const el = $("arch-msg");
  el.textContent = text || "";
  el.classList.toggle("err", !!isErr);
}

async function refreshArchCkpts() {
  const sel = $("arch-ckpt-select");
  sel.innerHTML = "";
  try {
    const res = await fetch("/api/checkpoints");
    const data = await res.json();
    const items = data.checkpoints || data || [];
    const list = Array.isArray(items) ? items : [];
    list.forEach((c) => {
      const path = typeof c === "string" ? c : (c.path || c.name || "");
      if (!path) return;
      const opt = document.createElement("option");
      opt.value = path;
      opt.textContent = path;
      sel.appendChild(opt);
    });
  } catch (e) {
    setArchMsg("刷新列表失败: " + e, true);
  }
}

function closeArchDrawer() {
  archSelectedLayer = null;
  $("arch-drawer").hidden = true;
  document.querySelectorAll(".arch-block.layer").forEach((el) => el.classList.remove("selected"));
}

function openArchDrawer(block) {
  archSelectedLayer = block.id;
  const drawer = $("arch-drawer");
  drawer.hidden = false;
  $("arch-drawer-title").textContent = block.label || block.id;
  const body = $("arch-drawer-body");
  const parts = block.parts || [];
  body.innerHTML = `<p class="field-hint">本层参数 ${fmtParams(block.params)}</p>` +
    parts.map((p) => {
      const detail = p.detail ? Object.entries(p.detail).filter(([,v]) => v != null)
        .map(([k,v]) => `${k}: ${v}`).join(" · ") : "";
      return `<div class="arch-part"><div class="lbl">${p.label}</div>
        <div class="sub">${fmtParams(p.params)}${detail ? " · " + detail : ""}</div></div>`;
    }).join("");
  document.querySelectorAll(".arch-block.layer").forEach((el) => {
    el.classList.toggle("selected", el.dataset.id === block.id);
  });
}

function renderArch(data) {
  archData = data;
  const canvas = $("arch-canvas");
  canvas.innerHTML = "";
  const cfg = data.config || {};
  const meta = data.meta || {};
  $("arch-summary").style.display = "block";
  $("arch-summary").innerHTML =
    `<div><b>${cfg.arch || "-"}</b> · d=${cfg.d_model || "-"} · L=${cfg.n_layers || "-"} · heads=${cfg.n_heads || "-"}</div>` +
    `<div>总参数 ${fmtParams(data.total_params)}${data.tie_word_embeddings ? " · tie embeddings" : ""}</div>` +
    `<div class="field-hint">step=${meta.step ?? "-"} · loss=${meta.loss ?? "-"}</div>`;

  const nodes = [...(data.blocks || []), ...(data.tail || [])];
  nodes.forEach((b, idx) => {
    if (idx) {
      const c = document.createElement("div");
      c.className = "arch-connector";
      canvas.appendChild(c);
    }
    const el = document.createElement("div");
    el.className = "arch-block" + (b.kind === "layer" ? " layer" : "");
    el.dataset.id = b.id;
    el.innerHTML = `<div class="lbl">${b.label}</div><div class="sub">${fmtParams(b.params)}${b.tied ? " · tied" : ""}</div>`;
    if (b.kind === "layer") {
      el.addEventListener("click", (ev) => {
        ev.stopPropagation();
        openArchDrawer(b);
      });
    }
    canvas.appendChild(el);
  });
  $("arch-status-text").textContent = "已解析";
  $("arch-dot").className = "dot finished";
  closeArchDrawer();
}

$("btn-arch-refresh-ckpt").addEventListener("click", refreshArchCkpts);
$("arch-ckpt-select").addEventListener("change", () => {
  if ($("arch-ckpt-select").value) $("arch-ckpt-manual").value = $("arch-ckpt-select").value;
});
$("btn-arch-drawer-close").addEventListener("click", closeArchDrawer);
$("arch-canvas").addEventListener("click", () => closeArchDrawer());

$("btn-arch-inspect").addEventListener("click", async () => {
  const path = $("arch-ckpt-manual").value.trim() || $("arch-ckpt-select").value.trim();
  if (!path) {
    setArchMsg("请先选择或粘贴 checkpoint 路径", true);
    return;
  }
  setArchMsg("解析中…");
  $("arch-status-text").textContent = "解析中…";
  $("arch-dot").className = "dot running";
  $("btn-arch-inspect").disabled = true;
  try {
    const res = await fetch("/api/arch/inspect", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ckpt_path: path }),
    });
    if (!res.ok) throw new Error(await readError(res));
    const data = await res.json();
    renderArch(data);
    setArchMsg("解析完成");
  } catch (e) {
    setArchMsg(String(e.message || e), true);
    $("arch-status-text").textContent = "失败";
    $("arch-dot").className = "dot failed";
  } finally {
    $("btn-arch-inspect").disabled = false;
  }
});

// Prefetch list when switching to arch tab
document.querySelectorAll(".page-tabs button").forEach((btn) => {
  btn.addEventListener("click", () => {
    if (btn.dataset.tab === "arch") refreshArchCkpts();
  });
});
```

Note: existing tab listener already shows `#arch` via `$(btn.dataset.tab)`. Do not duplicate the whole tab switcher — only add the prefetch listener (or fold into the existing click handler carefully).

- [ ] **Step 4: Manual smoke on box**

Run unit tests again. Optionally:

```bash
cd /workspace/miniseek && .venv/bin/python -c "
from web.arch_inspect import inspect_path
# if HF pt exists on box:
from pathlib import Path
p=Path('checkpoints/hf/MiniSeek-D0.1-base.pt')
if p.exists():
  o=inspect_path(p)
  print(o['config'].get('n_layers'), o['total_params'], len([b for b in o['blocks'] if b['kind']=='layer']))
else:
  print('skip real ckpt')
"
```

Expected for D0.1-base: `n_layers=8`, `total_params` ≈ 28451328.

- [ ] **Step 5: Commit**

```bash
git add web/index.html
git commit -m "feat(arch): add 架构 tab with diagram and layer drawer"
```

---

### Task 4: Sync to wangyifan6 + live smoke

**Files:**
- Sync: `web/arch_inspect.py`, `web_ui.py`, `web/index.html`, tests → `D:\miniseek`
- No new code beyond sync/restart

- [ ] **Step 1: Copy changed files to `D:\miniseek`**

Use CopyFromBox or scp-equivalent via Shell `machineId=ea181723-6900-45bd-a649-0c19c5dc5d77` after placing files, or write files directly on the machine. Ensure `web/arch_inspect.py` exists beside `tasks_store.py`.

- [ ] **Step 2: Restart Web UI on :7860**

Stop old uvicorn if needed; start with the project venv python (same as prior sessions). Confirm process listens on 7860.

- [ ] **Step 3: Smoke inspect API against real ckpt**

```powershell
cd D:\miniseek
.\.venv\Scripts\python -c "from web.arch_inspect import inspect_path; o=inspect_path(r'D:\miniseek\checkpoints\hf\MiniSeek-D0.1-base.pt'); print(o['config']['n_layers'], o['total_params'], o['tie_word_embeddings'])"
```

Expected: `8` and ~`28451328`, `True`.

Browser: open「架构」→ pick that `.pt` → 解析 → see 8 layers; click Layer 0 → drawer shows Attention / FFN.

- [ ] **Step 4: Commit on the machine if that clone is the git remote source of truth** (or push from box if remotes match). Prefer one canonical commit already done on box; machine sync is file copy.

- [ ] **Step 5: Mark plan tasks done in the plan checkboxes when executing.**

---

## Self-review

1. **Spec coverage:** API inspect ✓; pick-file reuse ✓; SVG/HTML stack ✓; drawer ✓; tabs order + widen ✓; arch_inspect module ✓; tests tiny + MoE synthetic ✓; loading state ✓; errors inline ✓; no Netron/3D ✓.
2. **Placeholders:** none intentional; MoE key names (`ffn.gate` / `shared_experts` / `experts`) documented for future `MoELM` — if real MoE training uses different prefixes later, update `_sum_prefix` paths in Task 1 only.
3. **Type consistency:** `inspect_path` / `inspect_blob` / `load_ckpt_blob` names stable across tasks; request field `ckpt_path` matches infer load.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-10-26-model-architecture-viz.md`. Two execution options:

**1. Subagent-Driven (recommended)** - dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
