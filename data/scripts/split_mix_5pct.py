#!/usr/bin/env python3
"""Stratified 5% valid / 95% train split from bucket jsonls. Seed=42."""
from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import time
from pathlib import Path

SEED = 42
VALID_RATIO = 0.05
DATA = Path("/home/wyf/miniseek/data")
BUCKET_DIR = DATA / "raw/buckets"
TRAIN_PARTS = BUCKET_DIR / "_split_train"
VALID_PARTS = BUCKET_DIR / "_split_valid"
OUT_TRAIN = DATA / "pretrain_mix_5g.jsonl"
OUT_VALID = DATA / "valid_mix.jsonl"
STATS_OUT = DATA / "pretrain_mix_5g_stats.json"
BUCKETS = ["wiki", "web", "books", "code", "other"]


def backup(path: Path) -> None:
    if not path.exists():
        return
    bak = Path(str(path) + ".bak")
    if bak.exists():
        bak.unlink()
    path.rename(bak)
    print(f"backed up {path.name} -> {bak.name}", flush=True)


def count_lines(path: Path) -> int:
    n = 0
    with path.open("rb") as f:
        for _ in f:
            n += 1
    return n


def split_bucket(name: str) -> dict:
    src = BUCKET_DIR / f"{name}.jsonl"
    n = count_lines(src)
    k = int(round(n * VALID_RATIO))
    if n == 0:
        raise SystemExit(f"empty bucket {name}")
    if k < 1 and n >= 1:
        k = 1
    if k >= n:
        k = max(1, n // 20)  # still leave train if tiny
    rng = random.Random(f"{SEED}-{name}")
    valid_idxs = set(rng.sample(range(n), k))

    t_path = TRAIN_PARTS / f"{name}.jsonl"
    v_path = VALID_PARTS / f"{name}.jsonl"
    n_train = n_valid = 0
    train_bytes = valid_bytes = 0
    with src.open("r", encoding="utf-8") as fin, t_path.open(
        "w", encoding="utf-8"
    ) as ft, v_path.open("w", encoding="utf-8") as fv:
        for i, line in enumerate(fin):
            if i in valid_idxs:
                fv.write(line)
                n_valid += 1
                valid_bytes += len(line.encode("utf-8"))
            else:
                ft.write(line)
                n_train += 1
                train_bytes += len(line.encode("utf-8"))
    info = {
        "bucket": name,
        "total_lines": n,
        "train_lines": n_train,
        "valid_lines": n_valid,
        "valid_ratio_lines": n_valid / n,
        "train_bytes": train_bytes,
        "valid_bytes": valid_bytes,
        "train_part": str(t_path),
        "valid_part": str(v_path),
    }
    print(
        f"[{name}] total={n} train={n_train} valid={n_valid} "
        f"({100*n_valid/n:.2f}%)",
        flush=True,
    )
    return info


def shuffle_concat(parts: list[Path], out: Path, seed: int) -> tuple[int, int]:
    """Load all lines from parts, shuffle, write. Returns (n_lines, n_bytes)."""
    lines: list[str] = []
    for p in parts:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    lines.append(line if line.endswith("\n") else line + "\n")
    rng = random.Random(seed)
    rng.shuffle(lines)
    nbytes = 0
    with out.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line)
            nbytes += len(line.encode("utf-8"))
    return len(lines), nbytes


def first_hash(path: Path, n: int = 200) -> set[str]:
    out = set()
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            out.add(hashlib.md5(line.encode("utf-8")).hexdigest())
    return out


def sample_hashes(path: Path, k: int, seed: int) -> set[str]:
    """Reservoir-ish: take every Nth after seed offset for cheap overlap check."""
    n = count_lines(path)
    if n == 0:
        return set()
    rng = random.Random(seed)
    idxs = set(rng.sample(range(n), min(k, n)))
    out = set()
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i in idxs:
                out.add(hashlib.md5(line.encode("utf-8")).hexdigest())
    return out


def main() -> None:
    t0 = time.time()
    TRAIN_PARTS.mkdir(parents=True, exist_ok=True)
    VALID_PARTS.mkdir(parents=True, exist_ok=True)

    # free space check
    st = os.statvfs(str(DATA))
    free_g = st.f_bavail * st.f_frsize / (1024**3)
    print(f"free disk before: {free_g:.1f}G", flush=True)
    if free_g < 8:
        raise SystemExit(f"not enough free disk: {free_g:.1f}G")

    backup(OUT_TRAIN)
    backup(OUT_VALID)

    bucket_stats = []
    for name in BUCKETS:
        bucket_stats.append(split_bucket(name))

    train_parts = [TRAIN_PARTS / f"{b}.jsonl" for b in BUCKETS]
    valid_parts = [VALID_PARTS / f"{b}.jsonl" for b in BUCKETS]

    print("shuffling train...", flush=True)
    n_train, b_train = shuffle_concat(train_parts, OUT_TRAIN, SEED + 1)
    print(f"train written: {n_train} lines, {b_train} bytes", flush=True)

    print("shuffling valid...", flush=True)
    n_valid, b_valid = shuffle_concat(valid_parts, OUT_VALID, SEED + 2)
    print(f"valid written: {n_valid} lines, {b_valid} bytes", flush=True)

    # overlap check on 2000 random hashes from each
    print("overlap check...", flush=True)
    h_train = sample_hashes(OUT_TRAIN, 3000, SEED + 3)
    h_valid = sample_hashes(OUT_VALID, 3000, SEED + 4)
    # also check all valid hashes against train sample + scan? 
    # Better: all valid line hashes vs train sample of 50k
    h_valid_all = set()
    with OUT_VALID.open("r", encoding="utf-8") as f:
        for line in f:
            h_valid_all.add(hashlib.md5(line.encode("utf-8")).hexdigest())
    h_train_big = sample_hashes(OUT_TRAIN, 50000, SEED + 5)
    overlap = h_valid_all & h_train_big
    # also first-200 cross
    overlap_small = first_hash(OUT_TRAIN, 500) & first_hash(OUT_VALID, 500)

    # EOS / format spot check
    bad = 0
    with OUT_VALID.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 20:
                break
            obj = json.loads(line)
            if "text" not in obj or not obj["text"].endswith("<|im_end|>"):
                bad += 1
    with OUT_TRAIN.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 20:
                break
            obj = json.loads(line)
            if "text" not in obj or not obj["text"].endswith("<|im_end|>"):
                bad += 1

    total_lines = n_train + n_valid
    summary = {
        "seed": SEED,
        "valid_ratio_target": VALID_RATIO,
        "train_lines": n_train,
        "valid_lines": n_valid,
        "train_bytes": b_train,
        "valid_bytes": b_valid,
        "valid_line_pct": n_valid / total_lines,
        "buckets": bucket_stats,
        "overlap_valid_all_vs_train_50k_sample": len(overlap),
        "overlap_first500": len(overlap_small),
        "format_bad_in_40_spot": bad,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    # per-bucket share in train vs valid (by lines)
    for b in bucket_stats:
        b["train_share_of_train"] = b["train_lines"] / n_train
        b["valid_share_of_valid"] = b["valid_lines"] / n_valid

    STATS_OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"DONE in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
