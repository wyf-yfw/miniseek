from __future__ import annotations

import json
import shutil
import secrets
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from eval.bench_results import ALLOWED_TASKS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_ROOT = PROJECT_ROOT / "eval" / "runs"


def new_run_id() -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{secrets.token_hex(2)}"


def run_dir(run_id: str) -> Path:
    return RUNS_ROOT / run_id


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_name(name: str) -> str:
    return (name or "").strip()


def _default_name(ckpt_path: str, run_id: str) -> str:
    stem = Path(ckpt_path or "").stem or Path(ckpt_path or "").name
    if stem:
        return stem
    return run_id


def create_run(
    *,
    ckpt_path: str = "",
    tasks: Optional[List[str]] = None,
    limit: Optional[int] = None,
    device: str = "cpu",
    name: str = "",
) -> Dict[str, Any]:
    cleaned = _clean_tasks(tasks if tasks is not None else ["ceval-valid", "cmmlu", "gsm8k"])
    rid = new_run_id()
    d = run_dir(rid)
    d.mkdir(parents=True, exist_ok=False)
    now = datetime.now().isoformat(timespec="seconds")
    display = normalize_name(name) or _default_name(ckpt_path, rid)
    data = {
        "id": rid,
        "name": display,
        "status": "draft",
        "ckpt_path": (ckpt_path or "").strip(),
        "tasks": cleaned,
        "limit": limit,
        "device": device,
        "pid": None,
        "created_at": now,
        "started_at": None,
        "ended_at": None,
        "exit_code": None,
        "error": None,
        "scores": None,
    }
    _write_json(d / "run.json", data)
    (d / "run.log").write_text("", encoding="utf-8")
    write_progress(rid, {
        "suite_done": 0,
        "suite_total": len(cleaned),
        "current_task": None,
        "sample_done": 0,
        "sample_total": 0,
    })
    return data


def _clean_tasks(tasks: List[str]) -> List[str]:
    cleaned: List[str] = []
    for t in tasks:
        t = (t or "").strip()
        if not t:
            continue
        if t not in ALLOWED_TASKS:
            raise ValueError(f"unsupported task: {t}")
        if t not in cleaned:
            cleaned.append(t)
    if not cleaned:
        raise ValueError("tasks must be non-empty")
    return cleaned


def save_run(data: Dict[str, Any]) -> None:
    rid = data["id"]
    _write_json(run_dir(rid) / "run.json", data)


def load_run(run_id: str) -> Dict[str, Any]:
    path = run_dir(run_id) / "run.json"
    if not path.is_file():
        raise FileNotFoundError(run_id)
    data = _read_json(path)
    # Compat: old runs without name / draft
    if "name" not in data or not data.get("name"):
        data["name"] = _default_name(data.get("ckpt_path") or "", data.get("id") or run_id)
    if data.get("status") == "queued":
        data["status"] = "draft"
    return data


def patch_run(
    run_id: str,
    *,
    name: Optional[str] = None,
    ckpt_path: Optional[str] = None,
    tasks: Optional[List[str]] = None,
    limit: Any = ...,
) -> Dict[str, Any]:
    data = load_run(run_id)
    st = data.get("status")
    if st == "running":
        raise RuntimeError("running task 不能改配置")
    editable = st in {"draft", "stopped", "failed", "finished"}
    if not editable:
        raise RuntimeError(f"status={st} 不可改")
    if name is not None:
        data["name"] = normalize_name(name) or data.get("name") or run_id
    if ckpt_path is not None:
        data["ckpt_path"] = ckpt_path.strip()
    if tasks is not None:
        data["tasks"] = _clean_tasks(tasks)
    if limit is not ...:
        if limit is None or limit == "":
            data["limit"] = None
        else:
            data["limit"] = int(limit)
    # After finished/stopped/failed edit, treat as draft again for Start
    if st in {"stopped", "failed", "finished"}:
        if any(x is not None for x in (ckpt_path, tasks)) or limit is not ...:
            data["status"] = "draft"
            data["pid"] = None
            data["started_at"] = None
            data["ended_at"] = None
            data["exit_code"] = None
            data["error"] = None
            data["scores"] = None
    save_run(data)
    return load_run(run_id)


def list_runs() -> List[Dict[str, Any]]:
    if not RUNS_ROOT.is_dir():
        return []
    rows = []
    for p in RUNS_ROOT.iterdir():
        jp = p / "run.json"
        if jp.is_file():
            try:
                rows.append(load_run(p.name))
            except Exception:  # noqa: BLE001
                continue
    rows.sort(key=lambda r: r.get("created_at") or "", reverse=True)
    return rows


def find_running_run_id() -> Optional[str]:
    for row in list_runs():
        if row.get("status") == "running":
            return row["id"]
    return None




def delete_run(run_id: str) -> None:
    d = run_dir(run_id)
    if not d.exists():
        raise FileNotFoundError(run_id)
    shutil.rmtree(d)

def write_progress(run_id: str, progress: Dict[str, Any]) -> None:
    _write_json(run_dir(run_id) / "progress.json", progress)


def read_progress(run_id: str) -> Dict[str, Any]:
    path = run_dir(run_id) / "progress.json"
    if not path.is_file():
        data = load_run(run_id)
        return {
            "suite_done": 0,
            "suite_total": len(data.get("tasks") or []),
            "current_task": None,
            "sample_done": 0,
            "sample_total": 0,
        }
    return _read_json(path)
