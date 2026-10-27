from __future__ import annotations

import json
import secrets
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TASKS_ROOT = PROJECT_ROOT / "tasks"


def new_task_id() -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{secrets.token_hex(2)}"


def task_dir(task_id: str) -> Path:
    return TASKS_ROOT / task_id


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))



def normalize_name(name: str) -> str:
    return (name or "").strip()


def name_taken(name: str, *, exclude_id: Optional[str] = None) -> bool:
    key = normalize_name(name).casefold()
    if not key:
        return False
    for row in list_tasks():
        if exclude_id and row.get("id") == exclude_id:
            continue
        if normalize_name(str(row.get("name") or "")).casefold() == key:
            return True
    return False


def allocate_unique_name(base: str) -> str:
    """Return base, or base-2, base-3, ... until free."""
    base = normalize_name(base) or "task"
    if not name_taken(base):
        return base
    n = 2
    while name_taken(f"{base}-{n}"):
        n += 1
    return f"{base}-{n}"


def create_task(*, name: str, arch: str, mode: str, hparams: Dict[str, Any], uniquify: bool = False) -> Dict[str, Any]:
    tid = new_task_id()
    d = task_dir(tid)
    ckpt = d / "checkpoints"
    ckpt.mkdir(parents=True, exist_ok=False)
    now = datetime.now().isoformat(timespec="seconds")
    display = normalize_name(name) or f"{arch}-{mode}"
    if uniquify:
        display = allocate_unique_name(display)
    elif name_taken(display):
        raise ValueError(f"已存在同名 Task：{display}")
    data = {
        "id": tid,
        "name": display,
        "status": "draft",
        "arch": arch,
        "mode": mode,
        "pid": None,
        "created_at": now,
        "started_at": None,
        "ended_at": None,
        "hparams": hparams or {},
        "out_dir": str(ckpt),
        "exit_code": None,
        "error": None,
    }
    _write_json(d / "task.json", data)
    _write_json(d / "metrics.json", {"loss": [], "valid_loss": [], "expert_load": None})
    (d / "train.log").write_text("", encoding="utf-8")
    return data


def load_task(task_id: str) -> Dict[str, Any]:
    path = task_dir(task_id) / "task.json"
    if not path.exists():
        raise FileNotFoundError(task_id)
    return _read_json(path)


def save_task(data: Dict[str, Any]) -> None:
    tid = data["id"]
    _write_json(task_dir(tid) / "task.json", data)


def list_tasks() -> List[Dict[str, Any]]:
    if not TASKS_ROOT.exists():
        return []
    rows = []
    for p in TASKS_ROOT.iterdir():
        jf = p / "task.json"
        if jf.is_file():
            t = _read_json(jf)
            rows.append({
                "id": t["id"],
                "name": t.get("name"),
                "status": t.get("status"),
                "arch": t.get("arch"),
                "mode": t.get("mode"),
                "created_at": t.get("created_at"),
                "started_at": t.get("started_at"),
                "ended_at": t.get("ended_at"),
            })
    rows.sort(key=lambda r: r.get("created_at") or "", reverse=True)
    return rows


def delete_task(task_id: str) -> None:
    d = task_dir(task_id)
    if not d.exists():
        raise FileNotFoundError(task_id)
    shutil.rmtree(d)


def write_metrics(task_id: str, metrics: Dict[str, Any]) -> None:
    _write_json(task_dir(task_id) / "metrics.json", metrics)


def read_metrics(task_id: str) -> Dict[str, Any]:
    path = task_dir(task_id) / "metrics.json"
    if not path.exists():
        return {"loss": [], "valid_loss": [], "expert_load": None}
    return _read_json(path)


def find_running_task_id() -> Optional[str]:
    for row in list_tasks():
        if row.get("status") == "running":
            return row["id"]
    return None
