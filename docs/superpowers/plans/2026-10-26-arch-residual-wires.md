# Arch Residual Bypass Wires Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** In the 架构 detailed view only, draw textbook side-bypass residual wires + ⊕ nodes; remove Residual formula cards; keep diagram blocks label-only (params/details in the right drawer).

**Architecture:** Backend design JSON drops `residual_*` parts and adds per-layer `residuals` metadata. Frontend detailed renderer inserts DOM ⊕ nodes on the main path, measures anchors, overlays SVG polylines for orange bypass wires labeled `x`. Compact/standard views unchanged.

**Tech Stack:** Python (`web/arch_inspect.py`), vanilla JS/HTML/CSS (`web/index.html`), pytest.

## Global Constraints

- Bypass wires **only** in `archViewMode === "detailed"`.
- No Residual formula cards in parts.
- Diagram blocks show **label only** (no `.sub` params/detail on canvas); drawer keeps full detail.
- Param totals unchanged vs DenseLM / design formulas.
- Spec: `docs/superpowers/specs/2026-10-26-arch-residual-wires-design.md`

## File map

| File | Role |
| --- | --- |
| `web/arch_inspect.py` | Remove residual parts; add `residuals` + `add_*` markers via part order |
| `tests/test_arch_design.py` | Assert no residual_* ; assert residuals present |
| `web/index.html` | Detailed render: label-only, ⊕ nodes, SVG bypass; CSS gutter |

---

### Task 1: Design JSON — residuals metadata, drop formula cards

**Files:**
- Modify: `web/arch_inspect.py` (`inspect_from_config`)
- Test: `tests/test_arch_design.py`

**Interfaces:**
- Produces: each layer dict has `residuals: [{"id":"attn","from":"layer_in","to":"add_attn","label":"x"}, {"id":"ffn","from":"post_attn","to":"add_ffn","label":"x"}]` and parts include zero-param markers `{id:"add_attn", label:"⊕", params:0, kind:"add"}` and `add_ffn` (no formula text). Frontend may also synthesize ⊕ without these parts; prefer explicit `add_*` parts for stable `data-node`.

- [ ] **Step 1: Update failing expectations in tests**

Replace residual assertions in `test_design_tiny_matches_denselm` and add residuals checks:

```python
def test_design_tiny_matches_denselm():
    out = inspect_design("tiny")
    assert out["config"]["arch"] == "tiny"
    assert out["config"]["n_layers"] == 2
    assert out["total_params"] == count_parameters(DenseLM(ModelConfig.tiny()))
    layer = next(b for b in out["blocks"] if b["kind"] == "layer")
    ids = {p["id"] for p in layer["parts"]}
    assert "attn" in ids and "ffn" in ids
    assert "residual_attn" not in ids and "residual_ffn" not in ids
    assert "add_attn" in ids and "add_ffn" in ids
    assert layer["residuals"] == [
        {"id": "attn", "from": "layer_in", "to": "add_attn", "label": "x"},
        {"id": "ffn", "from": "post_attn", "to": "add_ffn", "label": "x"},
    ]

def test_design_moe_has_experts():
    out = inspect_design("moe")
    assert out["config"]["arch"] == "moe"
    layer = next(b for b in out["blocks"] if b["kind"] == "layer")
    ids = {p["id"] for p in layer["parts"]}
    assert "router" in ids and "shared_experts" in ids and "routed_experts" in ids
    assert "residual_attn" not in ids
    assert "add_attn" in ids and "add_ffn" in ids
    assert out["total_params"] > inspect_design("dense")["total_params"]
```

- [ ] **Step 2: Run tests — expect FAIL**

`cd /workspace/miniseek && .venv/bin/python -m pytest tests/test_arch_design.py -v` (or system python with PYTHONPATH=.)

- [ ] **Step 3: Implement `inspect_from_config` changes**

In the layer loop:
1. After `attn` part, insert `{"id":"add_attn","label":"⊕","params":0,"kind":"add"}` instead of `residual_attn`.
2. After FFN/MoE parts, insert `{"id":"add_ffn","label":"⊕","params":0,"kind":"add"}` instead of `residual_ffn`.
3. Set `residuals` on the layer dict as in the test.
4. `layer_params` unchanged (add nodes params 0).

- [ ] **Step 4: Run tests — expect PASS**

- [ ] **Step 5: Commit**

`git commit -m "feat(arch): residual metadata + add nodes, drop formula cards"`

---

### Task 2: Detailed UI — label-only blocks, ⊕, SVG bypass wires

**Files:**
- Modify: `web/index.html` (CSS ~354+, `makeBlock`, `renderArch` detailed branch)

**Interfaces:**
- Consumes: layer.`residuals`, parts with `id` `add_attn`/`add_ffn` or `kind==="add"`
- Produces: detailed view with orange bypass SVG; drawer still shows params+detail

- [ ] **Step 1: CSS**

```css
.arch-layer-group {
  position: relative;
  padding-right: 56px; /* gutter for bypass */
}
.arch-block.part.add-node {
  width: 36px;
  height: 36px;
  border-radius: 50%;
  padding: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  border-color: #3dba7a;
  background: #13261c;
  color: #3dba7a;
  font-weight: 700;
  margin: 6px auto;
}
.arch-block.part.add-node .sub { display: none; }
.arch-residual-svg {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
  overflow: visible;
}
.arch-layer-group .arch-block.part .sub { display: none; } /* label-only on diagram */
.arch-layer-head .sub { display: none; } /* Layer i only on head */
```

- [ ] **Step 2: `makeBlock` — support add-node class; optional hideSub**

When `b.kind === "add"` or id starts with `add_`, add class `add-node` and only set `.lbl` to `⊕`.

- [ ] **Step 3: Detailed render path**

For each layer part in order (skip any leftover residual_*):
- Render part blocks with `data-node="${part.id}"`
- After all parts appended, call `drawLayerResiduals(group, b.residuals || defaultResiduals)`

`drawLayerResiduals(group, residuals)`:
1. Remove old `svg.arch-residual-svg`
2. Create SVG sized to group clientWidth/Height
3. For each residual, resolve from-anchor and to-anchor:
   - `layer_in`: top-center of first main part (`attn_norm`) — wire starts at right edge mid of that part's **top** (entry): use right-middle of `attn_norm` for attn residual start, OR top of group content
   - Spec A: start at right of layer entry = right-middle of first block before Norm... Use **right edge mid of `attn_norm`'s previous sibling or the point just above attn_norm**. Simplest matching mock A: from right-middle of `attn_norm` (or a `data-node=layer_in` spacer). Prefer: insert invisible/zero-height anchor `data-node="layer_in"` before attn_norm; `post_attn` anchor after add_attn.
4. Draw path `M x1 y1 H xRight V y2 H x2` orange stroke #f0a030 width 2.5, arrow at end; text `x` midway on vertical segment.

Default residuals if missing: same two edges as backend.

- [ ] **Step 4: Drawer**

Keep showing params + detail for parts; skip parts with `kind==="add"` or id `add_*` in drawer list (or show briefly as structure). Prefer skip add nodes in drawer.

Layer head click still opens drawer; head text only `Layer i` (no params line on canvas — params in drawer first line already).

- [ ] **Step 5: Resize**

`window.addEventListener("resize", () => { if (archData && archViewMode==="detailed") renderArch(archData); })` debounced lightly — or re-call draw on each render only (re-render on resize).

- [ ] **Step 6: Manual smoke**

Serve UI; open 架构 → 详细 → Dense: two orange bypasses per layer, no formula cards, no n_heads under Attention; click layer → drawer has details.

- [ ] **Step 7: Commit**

`git commit -m "feat(arch): detailed view bypass residual wires + label-only blocks"`

---

### Task 3: Sync Windows + restart :7860 + push

**Files:** sync `D:\miniseek` from box commits

- [ ] Bundle/tarball commits → wangyifan6 Downloads → merge → push origin main → restart web_ui.py → smoke GET `/` and `/api/arch/design?arch=dense`

---

## Spec coverage

| Spec item | Task |
| --- | --- |
| A side bypass | 2 |
| Detailed only | 2 |
| Drop formula cards | 1+2 |
| Label-only / drawer details | 2 |
| residuals JSON | 1 |
| Param totals unchanged | 1 tests |
| Sync/restart | 3 |
