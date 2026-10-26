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


def create_task(*, name: str, arch: str, mode: str, hparams: Dict[str, Any]) -> Dict[str, Any]:
    tid = new_task_id()
    d = task_dir(tid)
    ckpt = d / "checkpoints"
    ckpt.mkdir(parents=True, exist_ok=False)
    now = datetime.now().isoformat(timespec="seconds")
    data = {
        "id": tid,
        "name": name or f"{arch}-{mode}-{tid}",
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
    _write_json(d / "metrics.json", {"loss": [], "expert_load": None})
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
        return {"loss": [], "expert_load": None}
    return _read_json(path)


def find_running_task_id() -> Optional[str]:
    for row in list_tasks():
        if row.get("status") == "running":
            return row["id"]
    return None
