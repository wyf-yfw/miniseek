#!/usr/bin/env python3
"""Run one Pretrain jsonl clean job → data/clean/<id>/{clean,report,log}.

Disk-safe defaults:
  - Prefer builtin streaming (peak ≈ input + clean.jsonl; no HF datasets cache).
  - Optional Data-Juicer with use_cache=false and all temp/cache under job/.dj_work,
    wiped during/after the run.
  - Preflight refuses start when free space < ~4× input (floor 50GiB for ≥8GiB inputs).
  - dropped.jsonl off by default (counts still in report.json).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SENSITIVE = [
    "加微信", "加我微信", "代孕", "色情", "赌博", "博彩", "六合彩",
    "免费领现金", "刷单赚钱", "日赚", "兼职刷单",
]

DISK_RATIO = 4
DISK_FLOOR_LARGE = 50 * (1 << 30)  # 50 GiB
LARGE_INPUT = 8 * (1 << 30)  # apply 50G floor when input ≥ 8 GiB
KEEP_AFTER_CLEAN = {
    "job.json",
    "clean.jsonl",
    "report.json",
    "clean.log",
    "recipe.yaml",
    "dropped.jsonl",
    "progress.json",
}


class DiskSpaceError(RuntimeError):
    """Raised when free disk is below the clean preflight requirement."""


def _log(msg: str) -> None:
    print(msg, flush=True)


def _load_job(job_dir: Path) -> Dict[str, Any]:
    return json.loads((job_dir / "job.json").read_text(encoding="utf-8"))


def _save_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _save_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def write_progress(job_dir: Path, payload: Dict[str, Any]) -> None:
    """Periodic progress for UI (atomic). Safe to call often."""
    data = dict(payload)
    data.setdefault("updated_at", time.strftime("%Y-%m-%dT%H:%M:%S"))
    try:
        _save_json_atomic(job_dir / "progress.json", data)
    except OSError as exc:
        _log(f"[progress] write failed: {exc}")



def _text_of(obj: Dict[str, Any]) -> str:
    t = obj.get("text")
    if t is None:
        return ""
    return t if isinstance(t, str) else str(t)


def _md5_text(text: str, *, lowercase: bool) -> str:
    s = text.lower() if lowercase else text
    return hashlib.md5(s.encode("utf-8", errors="replace")).hexdigest()


def _cjk_ratio(text: str) -> float:
    if not text:
        return 0.0
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return cjk / max(1, len(text))


def _char_repetition_ratio(text: str, rep_len: int) -> float:
    """Fraction of chars covered by most frequent rep_len-gram (DJ-ish heuristic)."""
    if rep_len <= 0 or len(text) < rep_len:
        return 0.0
    counts: Counter[str] = Counter()
    n = len(text) - rep_len + 1
    for i in range(n):
        counts[text[i : i + rep_len]] += 1
    if not counts:
        return 0.0
    top = counts.most_common(1)[0][1]
    covered = top * rep_len
    return min(1.0, covered / max(1, len(text)))


def _load_wordlist(path: str) -> List[str]:
    words = list(DEFAULT_SENSITIVE)
    p = (path or "").strip()
    if p:
        fp = Path(p)
        if fp.is_file():
            for line in fp.read_text(encoding="utf-8", errors="replace").splitlines():
                w = line.strip()
                if w and not w.startswith("#"):
                    words.append(w)
    words = sorted(set(words), key=len, reverse=True)
    return words


def _fmt_bytes(n: int) -> str:
    n = max(0, int(n))
    for unit, div in (("GiB", 1 << 30), ("MiB", 1 << 20), ("KiB", 1 << 10)):
        if n >= div:
            return f"{n / div:.1f}{unit}"
    return f"{n}B"


def required_free_bytes(input_bytes: int) -> int:
    """Free space needed before starting a clean (~4× input; ≥50GiB floor for big inputs)."""
    size = max(0, int(input_bytes))
    need = size * DISK_RATIO
    if size >= LARGE_INPUT:
        need = max(need, DISK_FLOOR_LARGE)
    return need


def free_bytes_for(path: Path) -> int:
    target = path if path.exists() else path.parent
    return int(shutil.disk_usage(str(target)).free)


def preflight_disk(input_path: Path, *, anchor: Path) -> Dict[str, int]:
    """Refuse when free space is too low. Returns sizes for logging/UI."""
    if not input_path.is_file():
        raise FileNotFoundError(str(input_path))
    size = int(input_path.stat().st_size)
    need = required_free_bytes(size)
    free = free_bytes_for(anchor)
    info = {"input_bytes": size, "need_bytes": need, "free_bytes": free}
    if free < need:
        raise DiskSpaceError(
            "磁盘空间不足，已拒绝启动清洗："
            f"输入 {_fmt_bytes(size)}，建议至少预留 {_fmt_bytes(need)} "
            f"（约 {DISK_RATIO}× 输入"
            + (f"，大文件下限 {_fmt_bytes(DISK_FLOOR_LARGE)}" if size >= LARGE_INPUT else "")
            + f"），当前可用 {_fmt_bytes(free)}。"
            "请先腾出空间，或改用更小输入。"
        )
    return info


def dj_work_dir(job_dir: Path) -> Path:
    return job_dir / ".dj_work"


def cleanup_job_intermediates(job_dir: Path) -> List[str]:
    """Delete DJ/HF temp dirs and stray files; keep clean.jsonl + report + log (+ optional dropped)."""
    removed: List[str] = []
    if not job_dir.is_dir():
        return removed
    work = dj_work_dir(job_dir)
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
        removed.append(str(work.name))
    for p in list(job_dir.iterdir()):
        name = p.name
        if name in KEEP_AFTER_CLEAN or name.startswith("."):
            # still remove hidden work dirs except we already handled .dj_work
            if name.startswith(".") and name != ".dj_work" and p.is_dir():
                # leave unknown dotfiles alone
                continue
            continue
        try:
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            else:
                p.unlink(missing_ok=True)
            removed.append(name)
        except OSError as exc:
            _log(f"[cleanup] skip {name}: {exc}")
    return removed


def resolve_engine_pref(job: Dict[str, Any]) -> str:
    raw = (
        job.get("engine_pref")
        or os.environ.get("MINISEEK_CLEAN_ENGINE")
        or "builtin"
    )
    pref = str(raw).strip().lower()
    if pref in {"dj", "data-juicer", "data_juicer", "juicer"}:
        return "data-juicer"
    return "builtin"


def write_dropped_enabled(job: Dict[str, Any]) -> bool:
    return bool(job.get("write_dropped", False))


def _find_dj_process() -> Optional[List[str]]:
    for cand in (
        PROJECT_ROOT / ".venv-dj" / "bin" / "dj-process",
        Path(shutil.which("dj-process") or ""),
    ):
        if cand and Path(cand).is_file():
            return [str(cand)]
    try:
        import data_juicer  # noqa: F401
        return [sys.executable, "-m", "data_juicer.tools.process_data"]
    except Exception:  # noqa: BLE001
        pass
    which_py = shutil.which("dj-process")
    if which_py:
        return [which_py]
    return None


def _write_recipe(job_dir: Path, job: Dict[str, Any], export_path: Path) -> Path:
    ops = job.get("operators") or {}
    process: List[Dict[str, Any]] = []
    length = ops.get("length") or {}
    if length.get("enabled", True):
        process.append(
            {
                "text_length_filter": {
                    "min_len": int(length.get("min_len", 50)),
                    "max_len": int(length.get("max_len", 100000)),
                }
            }
        )
    language = ops.get("language") or {}
    if language.get("enabled", True):
        process.append(
            {
                "language_id_score_filter": {
                    "lang": str(language.get("lang") or "zh"),
                    "min_score": float(language.get("min_score", 0.5)),
                }
            }
        )
    repetition = ops.get("repetition") or {}
    if repetition.get("enabled", True):
        process.append(
            {
                "character_repetition_filter": {
                    "rep_len": int(repetition.get("rep_len", 10)),
                    "max_ratio": float(repetition.get("max_ratio", 0.5)),
                }
            }
        )
    sensitive = ops.get("sensitive") or {}
    if sensitive.get("enabled", True):
        process.append({"flagged_words_filter": {"lang": "zh", "max_ratio": 0.0}})
    exact = ops.get("exact_dedup") or {}
    if exact.get("enabled", True):
        process.append(
            {
                "document_deduplicator": {
                    "lowercase": bool(exact.get("lowercase", False)),
                    "ignore_non_character": False,
                }
            }
        )

    work = dj_work_dir(job_dir)
    tmp = work / "tmp"
    hf_ds = work / "hf_datasets"
    for d in (work, tmp, hf_ds):
        d.mkdir(parents=True, exist_ok=True)

    recipe = {
        "project_name": f"miniseek-clean-{job['id']}",
        "dataset_path": str(Path(job["input_path"]).resolve()),
        "export_path": str(export_path.resolve()),
        "np": 1,
        "text_keys": "text",
        "use_cache": False,
        "keep_stats_in_res_ds": False,
        "open_tracer": False,
        "use_checkpoint": False,
        "export_shard_size": 0,
        "export_in_parallel": False,
        "work_dir": str(work.resolve()),
        "temp_dir": str(tmp.resolve()),
        "ds_cache_dir": str(hf_ds.resolve()),
        "process": process,
    }
    path = job_dir / "recipe.yaml"
    lines = [
        f"project_name: {recipe['project_name']}",
        f"dataset_path: {recipe['dataset_path']}",
        f"export_path: {recipe['export_path']}",
        f"np: {recipe['np']}",
        "text_keys: text",
        "use_cache: false",
        "keep_stats_in_res_ds: false",
        "open_tracer: false",
        "use_checkpoint: false",
        "export_shard_size: 0",
        "export_in_parallel: false",
        f"work_dir: {recipe['work_dir']}",
        f"temp_dir: {recipe['temp_dir']}",
        f"ds_cache_dir: {recipe['ds_cache_dir']}",
        "process:",
    ]
    for item in process:
        for name, params in item.items():
            lines.append(f"  - {name}:")
            for pk, pv in params.items():
                if isinstance(pv, bool):
                    lines.append(f"      {pk}: {'true' if pv else 'false'}")
                elif isinstance(pv, str):
                    lines.append(f"      {pk}: {pv}")
                else:
                    lines.append(f"      {pk}: {pv}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _dj_env(job_dir: Path) -> Dict[str, str]:
    work = dj_work_dir(job_dir)
    hf_ds = work / "hf_datasets"
    hf_home = work / "hf_home"
    tmp = work / "tmp"
    for d in (work, hf_ds, hf_home, tmp):
        d.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.pop("all_proxy", None)
    env.pop("ALL_PROXY", None)
    env["PYTHONUNBUFFERED"] = "1"
    env["HF_DATASETS_CACHE"] = str(hf_ds.resolve())
    env["HF_HOME"] = str(hf_home.resolve())
    env["HF_DATASETS_OFFLINE"] = env.get("HF_DATASETS_OFFLINE", "1")
    env["TMPDIR"] = str(tmp.resolve())
    env["TMP"] = str(tmp.resolve())
    env["TEMP"] = str(tmp.resolve())
    # discourage datasets from writing under ~/.cache
    env["XDG_CACHE_HOME"] = str((work / "xdg_cache").resolve())
    (work / "xdg_cache").mkdir(parents=True, exist_ok=True)
    return env


def _count_jsonl_rows(path: Path) -> int:
    n = 0
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.strip():
                n += 1
    return n


def try_data_juicer(job_dir: Path, job: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    cmd_base = _find_dj_process()
    if not cmd_base:
        _log("[engine] Data-Juicer not found; will use builtin")
        return None
    cleanup_job_intermediates(job_dir)  # wipe prior .dj_work before start
    export_path = job_dir / "clean.jsonl"
    if export_path.is_file():
        export_path.unlink()
    recipe = _write_recipe(job_dir, job, export_path)
    cmd = list(cmd_base) + ["--config", str(recipe)]
    _log(f"[engine] data-juicer (disk-safe): {' '.join(cmd)}")
    t0 = time.time()
    env = _dj_env(job_dir)
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
    except FileNotFoundError as exc:
        _log(f"[engine] dj launch failed: {exc}")
        cleanup_job_intermediates(job_dir)
        return None
    if proc.stdout:
        print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n", flush=True)
    if proc.stderr:
        print(proc.stderr, end="" if proc.stderr.endswith("\n") else "\n", flush=True)
    if proc.returncode != 0 or not export_path.is_file():
        _log(f"[engine] dj-process failed rc={proc.returncode}; fallback builtin")
        cleanup_job_intermediates(job_dir)
        if export_path.is_file():
            try:
                export_path.unlink()
            except OSError:
                pass
        return None

    kept = _count_jsonl_rows(export_path)
    write_dropped = write_dropped_enabled(job)
    exact = (job.get("operators") or {}).get("exact_dedup") or {}
    lowercase = bool(exact.get("lowercase", False))
    input_rows = 0
    dropped = 0
    dropped_path = job_dir / "dropped.jsonl"

    if write_dropped:
        kept_hashes = set()
        with export_path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                kept_hashes.add(_md5_text(_text_of(obj), lowercase=lowercase))
        with Path(job["input_path"]).open("r", encoding="utf-8", errors="replace") as fin, dropped_path.open(
            "w", encoding="utf-8"
        ) as fout:
            for line in fin:
                raw = line.strip()
                if not raw:
                    continue
                input_rows += 1
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError:
                    fout.write(raw + "\n")
                    dropped += 1
                    continue
                h = _md5_text(_text_of(obj), lowercase=lowercase)
                if h not in kept_hashes:
                    fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    dropped += 1
    else:
        if dropped_path.is_file():
            try:
                dropped_path.unlink()
            except OSError:
                pass
        input_rows = _count_jsonl_rows(Path(job["input_path"]))
        dropped = max(0, input_rows - kept)

    # Always wipe DJ intermediates after export is finalized.
    removed = cleanup_job_intermediates(job_dir)
    if removed:
        _log(f"[cleanup] removed intermediates: {', '.join(removed)}")

    report = {
        "engine": "data-juicer",
        "input_rows": input_rows,
        "kept": kept,
        "dropped": dropped,
        "seconds": round(time.time() - t0, 3),
        "per_operator": {},
        "write_dropped": write_dropped,
        "note": (
            "disk-safe DJ: use_cache=false, caches under .dj_work then wiped; "
            + (
                "dropped via hash diff"
                if write_dropped
                else "dropped.jsonl skipped (set write_dropped=true to keep)"
            )
        ),
    }
    _save_json(job_dir / "report.json", report)
    return report


def apply_filters(
    text: str,
    ops: Dict[str, Any],
    *,
    sensitive_words: List[str],
    seen: set,
) -> Tuple[bool, Optional[str]]:
    """Return (keep, drop_reason)."""
    length = ops.get("length") or {}
    if length.get("enabled", True):
        n = len(text)
        if n < int(length.get("min_len", 50)) or n > int(length.get("max_len", 100000)):
            return False, "length"
    language = ops.get("language") or {}
    if language.get("enabled", True):
        lang = str(language.get("lang") or "zh").lower()
        score = float(language.get("min_score", 0.5))
        if lang.startswith("zh"):
            if _cjk_ratio(text) < score:
                return False, "language"
        elif lang.startswith("en"):
            letters = sum(1 for ch in text if ("a" <= ch.lower() <= "z"))
            if (letters / max(1, len(text))) < score:
                return False, "language"
    repetition = ops.get("repetition") or {}
    if repetition.get("enabled", True):
        ratio = _char_repetition_ratio(text, int(repetition.get("rep_len", 10)))
        if ratio > float(repetition.get("max_ratio", 0.5)):
            return False, "repetition"
    sensitive = ops.get("sensitive") or {}
    if sensitive.get("enabled", True) and sensitive_words:
        for w in sensitive_words:
            if w and w in text:
                return False, "sensitive"
    exact = ops.get("exact_dedup") or {}
    if exact.get("enabled", True):
        h = _md5_text(text, lowercase=bool(exact.get("lowercase", False)))
        if h in seen:
            return False, "exact_dedup"
        seen.add(h)
    return True, None


def run_builtin(job_dir: Path, job: Dict[str, Any]) -> Dict[str, Any]:
    ops = job.get("operators") or {}
    sensitive_words = _load_wordlist(str((ops.get("sensitive") or {}).get("wordlist_path") or ""))
    inp = Path(job["input_path"])
    clean_path = job_dir / "clean.jsonl"
    dropped_path = job_dir / "dropped.jsonl"
    write_dropped = write_dropped_enabled(job)
    per: Counter[str] = Counter()
    seen: set = set()
    input_rows = 0
    kept = 0
    dropped = 0
    t0 = time.time()
    total_bytes = int(inp.stat().st_size) if inp.is_file() else 0
    last_prog_t = t0
    last_prog_rows = 0
    PROGRESS_EVERY_ROWS = 10000
    PROGRESS_EVERY_SEC = 2.0

    def _emit_progress(*, force: bool = False) -> None:
        nonlocal last_prog_t, last_prog_rows
        now = time.time()
        if not force:
            if (input_rows - last_prog_rows) < PROGRESS_EVERY_ROWS and (now - last_prog_t) < PROGRESS_EVERY_SEC:
                return
        elapsed = max(0.001, now - t0)
        rate = input_rows / elapsed
        # prefer byte position for pct; fall back to rows if tell unavailable
        bytes_read = int(getattr(_emit_progress, "_bytes_read", 0) or 0)
        pct = None
        eta_sec = None
        if total_bytes > 0 and bytes_read >= 0:
            pct = round(min(100.0, 100.0 * bytes_read / total_bytes), 2)
            if bytes_read > 0 and bytes_read < total_bytes:
                eta_sec = round((total_bytes - bytes_read) * (elapsed / bytes_read), 1)
        write_progress(
            job_dir,
            {
                "engine": "builtin",
                "phase": "streaming",
                "bytes_read": bytes_read,
                "total_bytes": total_bytes,
                "input_rows": input_rows,
                "kept": kept,
                "dropped": dropped,
                "pct": pct,
                "elapsed_sec": round(elapsed, 1),
                "rate_rows_s": round(rate, 1),
                "eta_sec": eta_sec,
                "per_operator": dict(per),
            },
        )
        last_prog_t = now
        last_prog_rows = input_rows

    _log("[engine] builtin streaming operators (no HF cache)")
    write_progress(
        job_dir,
        {
            "engine": "builtin",
            "phase": "starting",
            "bytes_read": 0,
            "total_bytes": total_bytes,
            "input_rows": 0,
            "kept": 0,
            "dropped": 0,
            "pct": 0.0,
            "elapsed_sec": 0.0,
            "rate_rows_s": 0.0,
            "eta_sec": None,
        },
    )
    if dropped_path.is_file() and not write_dropped:
        try:
            dropped_path.unlink()
        except OSError:
            pass
    fdrop = dropped_path.open("w", encoding="utf-8") if write_dropped else None
    try:
        with inp.open("r", encoding="utf-8", errors="replace") as fin, clean_path.open(
            "w", encoding="utf-8"
        ) as fkeep:
            for line in fin:
                try:
                    _emit_progress._bytes_read = fin.tell()
                except OSError:
                    _emit_progress._bytes_read = int(getattr(_emit_progress, "_bytes_read", 0)) + len(
                        line.encode("utf-8", errors="replace")
                    )
                raw = line.strip()
                if not raw:
                    _emit_progress()
                    continue
                input_rows += 1
                if input_rows % 50000 == 0:
                    _log(f"progress rows={input_rows} kept={kept} dropped={dropped}")
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError:
                    if fdrop is not None:
                        fdrop.write(raw + "\n")
                    dropped += 1
                    per["json"] += 1
                    _emit_progress()
                    continue
                text = _text_of(obj)
                ok, reason = apply_filters(text, ops, sensitive_words=sensitive_words, seen=seen)
                if ok:
                    fkeep.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    kept += 1
                else:
                    if fdrop is not None:
                        fdrop.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    dropped += 1
                    per[reason or "other"] += 1
                _emit_progress()
            try:
                _emit_progress._bytes_read = total_bytes
            except Exception:
                pass
            _emit_progress(force=True)
    finally:
        if fdrop is not None:
            fdrop.close()
    report = {
        "engine": "builtin",
        "input_rows": input_rows,
        "kept": kept,
        "dropped": dropped,
        "seconds": round(time.time() - t0, 3),
        "per_operator": dict(per),
        "write_dropped": write_dropped,
        "note": (
            "streaming builtin; peak disk ≈ input + clean.jsonl"
            + ("" if write_dropped else "; dropped.jsonl skipped")
            + "; install/use Data-Juicer via engine_pref=data-juicer if needed"
        ),
    }
    _save_json(job_dir / "report.json", report)
    write_progress(
        job_dir,
        {
            "engine": "builtin",
            "phase": "done",
            "bytes_read": total_bytes,
            "total_bytes": total_bytes,
            "input_rows": input_rows,
            "kept": kept,
            "dropped": dropped,
            "pct": 100.0,
            "elapsed_sec": report["seconds"],
            "rate_rows_s": round(input_rows / max(0.001, report["seconds"]), 1),
            "eta_sec": 0.0,
            "per_operator": dict(per),
        },
    )
    _log(f"done kept={kept} dropped={dropped} seconds={report['seconds']}")
    return report



def preview_samples(
    input_path: str,
    operators: Dict[str, Any],
    *,
    limit: int = 200,
) -> Dict[str, Any]:
    ops = operators or {}
    sensitive_words = _load_wordlist(str((ops.get("sensitive") or {}).get("wordlist_path") or ""))
    seen: set = set()
    kept_ex: List[Dict[str, Any]] = []
    drop_ex: List[Dict[str, Any]] = []
    per: Counter[str] = Counter()
    input_rows = 0
    kept = 0
    dropped = 0
    path = Path(input_path)
    with path.open("r", encoding="utf-8", errors="replace") as fin:
        for line in fin:
            raw = line.strip()
            if not raw:
                continue
            input_rows += 1
            if input_rows > limit:
                break
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                dropped += 1
                per["json"] += 1
                if len(drop_ex) < 5:
                    drop_ex.append({"reason": "json", "text": raw[:200]})
                continue
            text = _text_of(obj)
            ok, reason = apply_filters(text, ops, sensitive_words=sensitive_words, seen=seen)
            if ok:
                kept += 1
                if len(kept_ex) < 5:
                    kept_ex.append({"text": text[:240]})
            else:
                dropped += 1
                per[reason or "other"] += 1
                if len(drop_ex) < 5:
                    drop_ex.append({"reason": reason, "text": text[:240]})
    return {
        "engine": "builtin-preview",
        "scanned": input_rows,
        "kept": kept,
        "dropped": dropped,
        "per_operator": dict(per),
        "kept_examples": kept_ex,
        "dropped_examples": drop_ex,
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-dir", required=True)
    args = ap.parse_args(argv)
    job_dir = Path(args.job_dir)
    job = _load_job(job_dir)
    inp = Path(job.get("input_path") or "")
    if not inp.is_file():
        _log(f"input not found: {inp}")
        return 2
    try:
        info = preflight_disk(inp, anchor=job_dir)
        _log(
            f"[preflight] input={_fmt_bytes(info['input_bytes'])} "
            f"need≥{_fmt_bytes(info['need_bytes'])} free={_fmt_bytes(info['free_bytes'])}"
        )
    except DiskSpaceError as exc:
        _log(str(exc))
        job["status"] = "failed"
        job["error"] = str(exc)
        _save_json(job_dir / "job.json", job)
        return 3

    pref = resolve_engine_pref(job)
    _log(f"job={job['id']} input={inp} engine_pref={pref}")
    report: Optional[Dict[str, Any]] = None
    if pref == "data-juicer":
        report = try_data_juicer(job_dir, job)
    if report is None:
        # Ensure no leftover DJ caches before streaming builtin.
        cleanup_job_intermediates(job_dir)
        report = run_builtin(job_dir, job)
    else:
        cleanup_job_intermediates(job_dir)

    try:
        job["engine"] = report.get("engine")
        job["stats"] = {
            "input_rows": report.get("input_rows"),
            "kept": report.get("kept"),
            "dropped": report.get("dropped"),
            "per_operator": report.get("per_operator"),
            "seconds": report.get("seconds"),
        }
        _save_json(job_dir / "job.json", job)
    except Exception as exc:  # noqa: BLE001
        _log(f"warn: could not update job.json: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
