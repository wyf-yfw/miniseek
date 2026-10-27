# Train Live Metrics (VRAM / elapsed / step/s) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend existing task SSE `status` events so the train execution page shows live VRAM, elapsed time, and step/s without new polling or train-script changes.

**Architecture:** Add `_live_train_status(task_id)` in `web_ui.py` that reads task.json + metrics.json and `_gpu_text()`. SSE `api_task_logs` emits this enriched status on connect, on idle ping (~4s), and at most once per ~4s when metrics are flushed during busy log writing. Frontend formats `elapsed_sec` and paints one metrics line under the existing status-box metadata from SSE payload.

**Tech Stack:** FastAPI SSE, Python 3, vanilla JS in `web/index.html`, existing `tasks_store` / `reconcile_task`.

## Global Constraints

- Do not change `train_pretrain.py` / SFT stdout or `metrics.json` schema.
- Do not add frontend XHR polling for `/api/status`.
- Do not change Loss chart, expert load, Start/Stop, or child-process isolation in this plan.
- Coding only on wyf; never Cursor CloudAgent.

---

### Task 1: Backend `_live_train_status` + SSE enrichment

**Files:**
- Modify: `web_ui.py` (near `_gpu_text` / `api_task_logs`)
- Test: inline `python3 -c` against a finished task dir

**Interfaces:**
- Consumes: `ts.load_task`, `ts.read_metrics`, `reconcile_task`, `_gpu_text`, `datetime.fromisoformat`
- Produces: `_live_train_status(task_id: str, *, prev_step: int | None = None, prev_mono: float | None = None) -> dict` with keys `id`, `status`, `pid`, `gpu`, `started_at`, `ended_at`, `elapsed_sec`, `step`, `step_s`, and optionally `_mono` / reused sliding inputs via return or side channel in the SSE generator

- [ ] **Step 1: Add helper**

```python
from datetime import datetime

def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        return None


def _live_train_status(
    task_id: str,
    *,
    prev_step: Optional[int] = None,
    prev_mono: Optional[float] = None,
) -> Dict[str, Any]:
    task = reconcile_task(ts.load_task(task_id))
    metrics = ts.read_metrics(task_id)
    loss = metrics.get("loss") or []
    step = int(loss[-1]["step"]) if loss else None
    started = _parse_iso(task.get("started_at"))
    ended = _parse_iso(task.get("ended_at"))
    now = datetime.now()
    if started is None:
        elapsed_sec = 0
    elif ended is not None:
        elapsed_sec = max(0, int((ended - started).total_seconds()))
    else:
        elapsed_sec = max(0, int((now - started).total_seconds()))

    step_s: Optional[float] = None
    mono = time.monotonic()
    if step is not None and prev_step is not None and prev_mono is not None and step > prev_step:
        dt = mono - prev_mono
        if dt >= 0.5:
            step_s = (step - prev_step) / dt
    if step_s is None and step is not None and elapsed_sec > 0:
        if len(loss) >= 2:
            first = int(loss[0]["step"])
            step_s = (step - first) / max(elapsed_sec, 1)
            if step_s < 0:
                step_s = step / max(elapsed_sec, 1)
        else:
            step_s = step / max(elapsed_sec, 1)

    return {
        "id": task["id"],
        "status": task.get("status"),
        "pid": task.get("pid"),
        "gpu": _gpu_text(),
        "started_at": task.get("started_at"),
        "ended_at": task.get("ended_at"),
        "elapsed_sec": elapsed_sec,
        "step": step,
        "step_s": None if step_s is None else round(float(step_s), 4),
        "_mono": mono,
        "_step_snap": step,
    }
```

- [ ] **Step 2: Wire `api_task_logs`**

Replace bare status dicts with helper. Keep sliding `prev_step` / `prev_mono` in the generator. Emit enriched status:
1. After initial metrics
2. On idle `% 10 == 0` (~4s)
3. When new log chunk yields metrics, if `time.monotonic() - last_status_ts >= 4.0`

Strip `_mono` / `_step_snap` before `_sse` (or build a public payload without them).

```python
def _public_status(raw: dict) -> dict:
    return {k: v for k, v in raw.items() if not k.startswith("_")}
```

- [ ] **Step 3: Smoke helper against finished task**

```bash
cd /home/wyf/miniseek && python3 - <<'PY'
import web_ui
s = web_ui._live_train_status("20261026-220926-0c52")
assert "gpu" in s and "elapsed_sec" in s
assert s["step"] is not None
print(s)
PY
```

Expected: `elapsed_sec` ≈ 1169 (ended−started), `step` 83200, `step_s` not null, `gpu` string.

- [ ] **Step 4: Commit backend+frontend together in Task 3** (or interim if preferred)

---

### Task 2: Frontend status-box live line

**Files:**
- Modify: `web/index.html` (`renderStatusBox`, `connectTaskLogs` status listener)

**Interfaces:**
- Consumes: SSE `status` payload fields from Task 1
- Produces: `formatElapsed(sec)`, `renderLiveMetrics(s)` updating `#status-box` without wiping id/status/pid block

- [ ] **Step 1: Add formatters + live line painter**

```javascript
function formatElapsed(sec) {
  if (sec == null || sec < 0 || Number.isNaN(sec)) return "—";
  sec = Math.floor(sec);
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = sec % 60;
  if (h > 0) return `${h}h${String(m).padStart(2, "0")}m${String(s).padStart(2, "0")}s`;
  if (m > 0) return `${m}m${String(s).padStart(2, "0")}s`;
  return `${s}s`;
}

function liveMetricsLine(s) {
  const parts = [];
  if (s.gpu) parts.push(`显存: ${s.gpu}`);
  if (s.elapsed_sec != null) parts.push(`已跑: ${formatElapsed(s.elapsed_sec)}`);
  const stepTxt = s.step == null ? "—" : String(s.step);
  const spd = s.step_s == null ? "—" : `${Number(s.step_s).toFixed(2)} step/s`;
  parts.push(`step ${stepTxt} · ${spd}`);
  return parts.join(" | ");
}

function applyLiveStatus(s) {
  const box = $("status-box");
  const lines = (box.textContent || "").split("\n").filter((ln) => !ln.startsWith("显存:"));
  // Prefer keeping metadata from last renderStatusBox; append/replace metrics line
  const base = lines.filter(Boolean);
  // If box only has "—", skip
  if (!base.length || (base.length === 1 && base[0] === "—")) {
    box.textContent = liveMetricsLine(s);
    return;
  }
  box.textContent = base.join("\n") + "\n" + liveMetricsLine(s);
}
```

- [ ] **Step 2: Update SSE `status` listener**

On `status`: call `applyLiveStatus(s)`. If `s.status` changed away from `running` (or differs from last known), call `selectTask(taskId, { keepForm: true, silent: true })` then re-apply live line so ended_at sticks; still `refreshTasks()`. Avoid full thrash every 4s while status stays `running` — only refresh task list lightly / skip `selectTask` when still running.

```javascript
eventSource.addEventListener("status", async (ev) => {
  try {
    const s = JSON.parse(ev.data);
    if (selectedId !== taskId) return;
    const wasRunning = true; // track lastStatus if needed
    applyLiveStatus(s);
    if (s.status && s.status !== "running") {
      await selectTask(taskId, { keepForm: true, silent: true });
      applyLiveStatus(s);
    }
    refreshTasks();
  } catch (_) {}
});
```

- [ ] **Step 3: Keep `renderStatusBox` metadata; do not put live line only there** — after `renderStatusBox(t)`, if last SSE live payload cached (`window.__lastLiveStatus`), re-apply so selectTask does not wipe the line.

---

### Task 3: Restart, smoke, commit

**Files:** none new

- [ ] **Step 1: Confirm no training**

```bash
curl -s http://127.0.0.1:7860/api/status
```

If `running: true`, **skip restart** and report to parent. Else:

```bash
systemctl --user restart miniseek-webui.service
systemctl --user is-active miniseek-webui.service
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:7860/
```

Expected: `active`, `200`.

- [ ] **Step 2: Import smoke**

```bash
cd /home/wyf/miniseek && python3 -c "from web_ui import _live_train_status,_gpu_text; print(_gpu_text()); print(_live_train_status('20261026-220926-0c52'))"
```

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/plans/2026-10-27-train-live-metrics.md web_ui.py web/index.html
git commit -m "$(cat <<'MSG'
feat(web): live train metrics via SSE (VRAM/elapsed/step_s)

MSG
)"
```

(Plan-only commit first as `docs: plan train live metrics SSE`, then this feat commit after impl.)

---

## Spec coverage checklist

| Spec item | Task |
|-----------|------|
| `gpu` / `started_at` / `elapsed_sec` / `step` / `step_s` on status | Task 1 |
| ~4s idle ping + optional busy flush throttle | Task 1 |
| status-box line; frontend formats elapsed | Task 2 |
| No train script / metrics schema / new polling | Global + Tasks 1–2 |
| Restart + acceptance smoke | Task 3 |
