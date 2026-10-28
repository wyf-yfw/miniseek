#!/usr/bin/env python3
"""Filter MiniSeek SFT v5 10G pack to short-context conversations.

Primary prefer ≤ preferred_chars (256). Soft max soft_max_chars (280):
keep all rows with content-char sum ≤ soft_max (drop longer; no truncate).
Does not modify 5g/10g packs. Writes SOURCES_v5_short256.md.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC = ROOT / "data" / "SFT" / "sft_miniseek_v5_10g.jsonl"
DEFAULT_OUT = ROOT / "data" / "SFT" / "sft_miniseek_v5_short256.jsonl"
DEFAULT_SOURCES = ROOT / "data" / "SFT" / "SOURCES_v5_short256.md"
GiB = 1024**3
PREFERRED = 256
SOFT_MAX = 280


def content_len(obj: dict) -> int:
    convos = obj.get("conversations") or []
    return sum(len(t.get("content") or "") for t in convos)


def detect_lang(text: str) -> str:
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return "zh" if cjk >= max(3, len(text) // 20) else "en"


def is_multi(convos: list) -> bool:
    return sum(1 for t in convos if t.get("role") == "user") >= 2


def length_bucket(n: int, preferred: int, soft: int) -> str:
    if n <= 64:
        return "<=64"
    if n <= 128:
        return "65-128"
    if n <= 192:
        return "129-192"
    if n <= preferred:
        return f"193-{preferred}"
    return f"{preferred + 1}-{soft}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    ap.add_argument("--preferred-chars", type=int, default=PREFERRED)
    ap.add_argument("--soft-max-chars", type=int, default=SOFT_MAX)
    args = ap.parse_args()
    if args.soft_max_chars < args.preferred_chars:
        print("soft-max must be >= preferred", file=sys.stderr)
        return 1

    if not args.src.is_file():
        print(f"missing source: {args.src}", file=sys.stderr)
        return 1

    src_size = args.src.stat().st_size
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(args.out.suffix + ".tmp")

    total = 0
    kept = 0
    kept_pref = 0
    kept_soft_only = 0
    bad = 0
    lang_c: Counter[str] = Counter()
    turn_c: Counter[str] = Counter()
    len_c: Counter[str] = Counter()
    kept_chars = 0

    print(
        f"src={args.src} size_gib={src_size / GiB:.2f} "
        f"preferred<={args.preferred_chars} soft_max<={args.soft_max_chars}",
        flush=True,
    )
    with args.src.open("r", encoding="utf-8") as fin, tmp.open(
        "w", encoding="utf-8", newline="\n"
    ) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            convos = obj.get("conversations")
            if not isinstance(convos, list) or not convos:
                bad += 1
                continue
            L = content_len(obj)
            if L > args.soft_max_chars:
                continue
            out_obj = {
                "conversations": [
                    {"role": t.get("role", ""), "content": t.get("content") or ""}
                    for t in convos
                    if isinstance(t, dict)
                ]
            }
            if not out_obj["conversations"]:
                bad += 1
                continue
            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            kept += 1
            kept_chars += L
            if L <= args.preferred_chars:
                kept_pref += 1
            else:
                kept_soft_only += 1
            text = "".join(t["content"] for t in out_obj["conversations"])
            lang_c[detect_lang(text)] += 1
            turn_c["multi" if is_multi(out_obj["conversations"]) else "single"] += 1
            len_c[length_bucket(L, args.preferred_chars, args.soft_max_chars)] += 1
            if kept % 200000 == 0:
                print(f"  progress total={total} kept={kept}", flush=True)

    os.replace(tmp, args.out)
    out_size = args.out.stat().st_size
    keep_rate = (kept / total) if total else 0.0

    def frac(c: Counter, k: str) -> float:
        return (c[k] / kept) if kept else 0.0

    md = f"""# sft_miniseek_v5_short256

- Built: from `sft_miniseek_v5_10g.jsonl`
- Length rule: **prefer** total conversation content chars ≤ {args.preferred_chars}; **soft max** ≤ {args.soft_max_chars} (≈10% over preferred). Longer rows **dropped** (no truncate).
- Capability rebalance: **not applied** — 10G rows have no per-sample bucket tags; distribution is the natural short-sample mix
- Format: Conversations JSONL `{{"conversations":[{{"role","content"}},...]}}`
- Relation: independent filter of 10G; does **not** modify 5G/10G; not a 5G subset
- Training tip: use `max_seq_len=256` (or ≤512). SFT tokenization truncates overflow from the **end** (keep prompt/question head).

## Pack (`sft_miniseek_v5_short256.jsonl`)

- rows_kept: {kept}
- rows_preferred_le_{args.preferred_chars}: {kept_pref} ({(100 * kept_pref / kept) if kept else 0:.2f}% of kept)
- rows_soft_only_{args.preferred_chars + 1}_{args.soft_max_chars}: {kept_soft_only} ({(100 * kept_soft_only / kept) if kept else 0:.2f}% of kept)
- rows_source_10g: {total}
- rows_bad_json_or_empty: {bad}
- keep_rate: {keep_rate:.4f} ({100 * keep_rate:.2f}% of 10G)
- size_bytes: {out_size} (~{out_size / GiB:.3f} GiB)
- source_size_bytes: {src_size} (~{src_size / GiB:.3f} GiB)
- mean_content_chars_kept: {(kept_chars / kept) if kept else 0:.1f}

### Length buckets (kept only)

- {dict(sorted(len_c.items()))}

### Language (heuristic CJK)

- zh: {lang_c.get('zh', 0)} ({100 * frac(lang_c, 'zh'):.2f}%)
- en: {lang_c.get('en', 0)} ({100 * frac(lang_c, 'en'):.2f}%)

### Turns

- single: {turn_c.get('single', 0)} ({100 * frac(turn_c, 'single'):.2f}%)
- multi: {turn_c.get('multi', 0)} ({100 * frac(turn_c, 'multi'):.2f}%)

## Notes

- v5 capability targets are **not** re-enforced here.
- WebUI / train defaults still point at 10G unless you select this file.
- 5G and 10G packs were not modified.
"""
    args.sources.write_text(md, encoding="utf-8")
    print(
        f"done kept={kept}/{total} ({100 * keep_rate:.2f}%) "
        f"pref={kept_pref} soft_only={kept_soft_only} "
        f"out_gib={out_size / GiB:.3f} -> {args.out}",
        flush=True,
    )
    print(f"sources -> {args.sources}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
