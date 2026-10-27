#!/usr/bin/env python3
"""Run one Pretrain jsonl clean job → data/clean/<id>/{clean,dropped,report}.

Prefers Data-Juicer (dj-process / data_juicer). Falls back to builtin operators
with the same names/semantics when DJ is unavailable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Minimal default sensitive lexicon (Chinese + common spam); extend via wordlist_path.
DEFAULT_SENSITIVE = [
    "加微信", "加我微信", "代孕", "色情", "赌博", "博彩", "六合彩",
    "免费领现金", "刷单赚钱", "日赚", "兼职刷单",
]


def _log(msg: str) -> None:
    print(msg, flush=True)


def _load_job(job_dir: Path) -> Dict[str, Any]:
    return json.loads((job_dir / "job.json").read_text(encoding="utf-8"))


def _save_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


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
    # longer first for naive contains
    words = sorted(set(words), key=len, reverse=True)
    return words


def _find_dj_process() -> Optional[List[str]]:
    for cand in (
        PROJECT_ROOT / ".venv-dj" / "bin" / "dj-process",
        Path(shutil.which("dj-process") or ""),
    ):
        if cand and Path(cand).is_file():
            return [str(cand)]
    # module form
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
        # DJ flagged_words_filter expects lang + use_words_aug etc.; keep simple
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
    recipe = {
        "project_name": f"miniseek-clean-{job['id']}",
        "dataset_path": str(Path(job["input_path"]).resolve()),
        "export_path": str(export_path.resolve()),
        "np": 1,
        "text_keys": "text",
        "process": process,
    }
    path = job_dir / "recipe.yaml"
    # minimal YAML without PyYAML dependency
    lines = [
        f"project_name: {recipe['project_name']}",
        f"dataset_path: {recipe['dataset_path']}",
        f"export_path: {recipe['export_path']}",
        f"np: {recipe['np']}",
        "text_keys: text",
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


def try_data_juicer(job_dir: Path, job: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    cmd_base = _find_dj_process()
    if not cmd_base:
        _log("[engine] Data-Juicer not found; will use builtin")
        return None
    export_path = job_dir / "clean.jsonl"
    recipe = _write_recipe(job_dir, job, export_path)
    cmd = list(cmd_base) + ["--config", str(recipe)]
    _log(f"[engine] data-juicer: {' '.join(cmd)}")
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError as exc:
        _log(f"[engine] dj launch failed: {exc}")
        return None
    if proc.stdout:
        print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n", flush=True)
    if proc.stderr:
        print(proc.stderr, end="" if proc.stderr.endswith("\n") else "\n", flush=True)
    if proc.returncode != 0 or not export_path.is_file():
        _log(f"[engine] dj-process failed rc={proc.returncode}; fallback builtin")
        return None
    # Build dropped by set difference on md5
    exact = (job.get("operators") or {}).get("exact_dedup") or {}
    lowercase = bool(exact.get("lowercase", False))
    kept_hashes = set()
    kept = 0
    with export_path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            kept += 1
            kept_hashes.add(_md5_text(_text_of(obj), lowercase=lowercase))
    dropped_path = job_dir / "dropped.jsonl"
    input_rows = 0
    dropped = 0
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
    report = {
        "engine": "data-juicer",
        "input_rows": input_rows,
        "kept": kept,
        "dropped": dropped,
        "seconds": round(time.time() - t0, 3),
        "per_operator": {},
        "note": "per-operator counts unavailable from dj-process export; dropped via hash diff",
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
            # crude: letters ratio
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
    per: Counter[str] = Counter()
    seen: set = set()
    input_rows = 0
    kept = 0
    dropped = 0
    t0 = time.time()
    _log("[engine] builtin operators")
    with inp.open("r", encoding="utf-8", errors="replace") as fin, clean_path.open(
        "w", encoding="utf-8"
    ) as fkeep, dropped_path.open("w", encoding="utf-8") as fdrop:
        for line in fin:
            raw = line.strip()
            if not raw:
                continue
            input_rows += 1
            if input_rows % 50000 == 0:
                _log(f"progress rows={input_rows} kept={kept} dropped={dropped}")
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                fdrop.write(raw + "\n")
                dropped += 1
                per["json"] += 1
                continue
            text = _text_of(obj)
            ok, reason = apply_filters(text, ops, sensitive_words=sensitive_words, seen=seen)
            if ok:
                fkeep.write(json.dumps(obj, ensure_ascii=False) + "\n")
                kept += 1
            else:
                fdrop.write(json.dumps(obj, ensure_ascii=False) + "\n")
                dropped += 1
                per[reason or "other"] += 1
    report = {
        "engine": "builtin",
        "input_rows": input_rows,
        "kept": kept,
        "dropped": dropped,
        "seconds": round(time.time() - t0, 3),
        "per_operator": dict(per),
        "note": "CJK-ratio language heuristic; install Data-Juicer for language_id_score_filter",
    }
    _save_json(job_dir / "report.json", report)
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
    _log(f"job={job['id']} input={inp}")
    report = try_data_juicer(job_dir, job)
    if report is None:
        report = run_builtin(job_dir, job)
    # stamp engine onto job.json (best-effort)
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
