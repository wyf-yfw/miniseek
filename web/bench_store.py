from __future__ import annotations

import json
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


def create_run(
    *,
    ckpt_path: str,
    tasks: List[str],
    limit: Optional[int],
    device: str,
) -> Dict[str, Any]:
    cleaned: List[str] = []
    for t in tasks:
        t = (t or "").strip()
        if t not in ALLOWED_TASKS:
            raise ValueError(f"unsupported task: {t}")
        if t not in cleaned:
            cleaned.append(t)
    if not cleaned:
        raise ValueError("tasks must be non-empty")
    rid = new_run_id()
    d = run_dir(rid)
    d.mkdir(parents=True, exist_ok=False)
    now = datetime.now().isoformat(timespec="seconds")
    data = {
        "id": rid,
        "status": "queued",
        "ckpt_path": ckpt_path,
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
    return data


def save_run(data: Dict[str, Any]) -> None:
    rid = data["id"]
    _write_json(run_dir(rid) / "run.json", data)


def load_run(run_id: str) -> Dict[str, Any]:
    path = run_dir(run_id) / "run.json"
    if not path.is_file():
        raise FileNotFoundError(run_id)
    return _read_json(path)


def list_runs() -> List[Dict[str, Any]]:
    if not RUNS_ROOT.is_dir():
        return []
    rows = []
    for p in RUNS_ROOT.iterdir():
        jp = p / "run.json"
        if jp.is_file():
            rows.append(_read_json(jp))
    rows.sort(key=lambda r: r.get("created_at") or "", reverse=True)
    return rows
