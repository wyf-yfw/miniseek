"""PT data-clean jobs under data/clean/<id>/ (disk-safe defaults)."""
from __future__ import annotations

import json
import secrets
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEAN_ROOT = PROJECT_ROOT / "data" / "clean"

# Pretrain wash defaults for data/pretrain_mix_12g.jsonl (CN Edu + EN Cosmopedia mix).
# language OFF: zh-only filter would wipe substantial English Cosmopedia.
DEFAULT_OPERATORS: Dict[str, Any] = {
    "length": {"enabled": True, "min_len": 50, "max_len": 100000},
    "language": {"enabled": False, "lang": "zh", "min_score": 0.5},
    "repetition": {"enabled": True, "rep_len": 10, "max_ratio": 0.5},
    "exact_dedup": {"enabled": True, "lowercase": True},
    "sensitive": {"enabled": True, "wordlist_path": ""},
}

# Disk-safe: streaming builtin by default (no HF datasets cache pile-up).
# Set engine_pref to "data-juicer" only when DJ is explicitly wanted.
DEFAULT_ENGINE_PREF = "builtin"
DEFAULT_WRITE_DROPPED = False
DISK_HINT = (
    "默认 builtin 流式清洗（峰值约 输入+clean.jsonl）。"
    "启动前校验可用空间 ≥ 约 4× 输入（≥8GiB 输入另有 50GiB 下限）。"
    "不默认写 dropped.jsonl；DJ 缓存仅落在 job/.dj_work 并在结束后删除。"
)


def new_job_id() -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{secrets.token_hex(2)}"


def job_dir(job_id: str) -> Path:
    return CLEAN_ROOT / job_id


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_name(name: str) -> str:
    return (name or "").strip()


def _default_name(input_path: str, job_id: str) -> str:
    stem = Path(input_path or "").stem
    return stem or job_id


def _merge_operators(raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {k: dict(v) for k, v in DEFAULT_OPERATORS.items()}
    if not raw:
        return out
    for key, defaults in DEFAULT_OPERATORS.items():
        src = raw.get(key)
        if not isinstance(src, dict):
            continue
        merged = dict(defaults)
        for sk, sv in src.items():
            if sk in merged or sk == "enabled":
                merged[sk] = sv
        # coerce
        if "enabled" in merged:
            merged["enabled"] = bool(merged["enabled"])
        for num_key in ("min_len", "max_len", "rep_len"):
            if num_key in merged and merged[num_key] is not None:
                merged[num_key] = int(merged[num_key])
        for float_key in ("min_score", "max_ratio"):
            if float_key in merged and merged[float_key] is not None:
                merged[float_key] = float(merged[float_key])
        if "lowercase" in merged:
            merged["lowercase"] = bool(merged["lowercase"])
        if "lang" in merged:
            merged["lang"] = str(merged["lang"] or "zh").strip() or "zh"
        if "wordlist_path" in merged:
            merged["wordlist_path"] = str(merged["wordlist_path"] or "").strip()
        out[key] = merged
    return out


def create_job(
    *,
    name: str = "",
    input_path: str = "",
    operators: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    jid = new_job_id()
    d = job_dir(jid)
    d.mkdir(parents=True, exist_ok=False)
    now = datetime.now().isoformat(timespec="seconds")
    inp = (input_path or "").strip()
    display = normalize_name(name) or _default_name(inp, jid)
    data = {
        "id": jid,
        "name": display,
        "status": "draft",
        "input_path": inp,
        "operators": _merge_operators(operators),
        "engine_pref": DEFAULT_ENGINE_PREF,
        "write_dropped": DEFAULT_WRITE_DROPPED,
        "pid": None,
        "created_at": now,
        "started_at": None,
        "ended_at": None,
        "exit_code": None,
        "error": None,
        "output_dir": str(d),
        "stats": None,
        "engine": None,
    }
    _write_json(d / "job.json", data)
    (d / "clean.log").write_text("", encoding="utf-8")
    return data


def save_job(data: Dict[str, Any]) -> None:
    jid = data["id"]
    _write_json(job_dir(jid) / "job.json", data)


def load_job(job_id: str) -> Dict[str, Any]:
    path = job_dir(job_id) / "job.json"
    if not path.is_file():
        raise FileNotFoundError(job_id)
    return _read_json(path)


def patch_job(
    job_id: str,
    *,
    name: Optional[str] = None,
    input_path: Optional[str] = None,
    operators: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    data = load_job(job_id)
    st = data.get("status")
    if st == "running":
        raise RuntimeError("running job 不能改配置")
    editable = st in {"draft", "stopped", "failed", "finished"}
    if not editable:
        raise RuntimeError(f"status={st} 不可改")
    if name is not None:
        data["name"] = normalize_name(name) or data.get("name") or job_id
    if input_path is not None:
        data["input_path"] = input_path.strip()
    if operators is not None:
        data["operators"] = _merge_operators(operators)
    if st in {"stopped", "failed", "finished"}:
        if any(x is not None for x in (input_path, operators)):
            data["status"] = "draft"
            data["pid"] = None
            data["started_at"] = None
            data["ended_at"] = None
            data["exit_code"] = None
            data["error"] = None
            data["stats"] = None
            data["engine"] = None
    save_job(data)
    return load_job(job_id)


def list_jobs() -> List[Dict[str, Any]]:
    if not CLEAN_ROOT.is_dir():
        return []
    rows = []
    for p in CLEAN_ROOT.iterdir():
        if not p.is_dir():
            continue
        jp = p / "job.json"
        if jp.is_file():
            try:
                rows.append(load_job(p.name))
            except Exception:  # noqa: BLE001
                continue
    rows.sort(key=lambda r: r.get("created_at") or "", reverse=True)
    return rows


def find_running_job_id() -> Optional[str]:
    for row in list_jobs():
        if row.get("status") == "running":
            return row["id"]
    return None


def delete_job(job_id: str) -> None:
    d = job_dir(job_id)
    if not d.exists():
        raise FileNotFoundError(job_id)
    shutil.rmtree(d)


def read_report(job_id: str) -> Optional[Dict[str, Any]]:
    path = job_dir(job_id) / "report.json"
    if not path.is_file():
        return None
    return _read_json(path)


def read_progress(job_id: str) -> Optional[Dict[str, Any]]:
    path = job_dir(job_id) / "progress.json"
    if not path.is_file():
        return None
    try:
        return _read_json(path)
    except Exception:  # noqa: BLE001
        return None
