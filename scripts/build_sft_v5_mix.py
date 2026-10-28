#!/usr/bin/env python3
"""Build MiniSeek SFT v5 5G/10G Conversations JSONL mixes (route A).

Memory-safe: pools on disk; stratified sample via line-offset index.
Does not delete v1, change WebUI defaults, or upload HF.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import struct
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(it, **kwargs):
        return it

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "_raw_sft_v5"
OUT_DIR = ROOT / "data" / "SFT"
POOL = RAW / "pool"
SEED = 20261028
GiB = 1024**3

BUCKETS = ("dialogue", "knowledge", "reasoning", "creative", "code")
BUCKET_RATIO = {
    "dialogue": 0.43,
    "knowledge": 0.22,
    "reasoning": 0.15,
    "creative": 0.12,
    "code": 0.08,
}
TYPE_TO_BUCKET = {
    31: "code", 33: "code", 35: "code",
    7: "reasoning", 15: "reasoning", 38: "reasoning", 45: "reasoning", 49: "reasoning",
    10: "creative", 11: "creative", 22: "creative", 32: "creative", 40: "creative", 42: "creative",
    0: "knowledge", 1: "knowledge", 2: "knowledge", 4: "knowledge", 16: "knowledge",
    17: "knowledge", 23: "knowledge", 27: "knowledge", 36: "knowledge",
}
MAGPIE_CREATIVE = {
    "Creative writing", "Role playing", "Brainstorming",
    "Creative Writing", "Role Playing",
}
THINK_RE = re.compile(
    r"<think>[\s\S]*?</think>|<thinking>[\s\S]*?</thinking>",
    re.IGNORECASE,
)
BUCKET_ID = {b: i for i, b in enumerate(BUCKETS)}
LANG_ID = {"zh": 0, "en": 1}
SOURCE_ID = {
    "magpie-cn": 0,
    "fineweb-pure-qa": 1,
    "congliu-r1-final": 2,
    "deepctrl-zh": 3,
    "deepctrl-en": 4,
}
ID_SOURCE = {v: k for k, v in SOURCE_ID.items()}
ID_BUCKET = {v: k for k, v in BUCKET_ID.items()}
ID_LANG = {v: k for k, v in LANG_ID.items()}

# Compact index record: file_id u16, pad u16, offset u64, out_bytes u32, hash_prefix u32,
# bucket u8, lang u8, multi u8, source u8
IDX_FMT = "<HHI I I BBBB"  # wait — use explicit
# Better: <H x x Q I I B B B B  = 2+2+8+4+4+4 = 24 bytes
IDX_STRUCT = struct.Struct("<H2xQIIBBBB")  # 24 bytes
assert IDX_STRUCT.size == 24


def ensure_dirs() -> None:
    for p in (RAW, OUT_DIR, POOL, RAW / "deepctrl", RAW / "magpie", RAW / "fineweb_qa", RAW / "r1"):
        p.mkdir(parents=True, exist_ok=True)


def free_gb(path: Path = ROOT) -> float:
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / GiB


def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def hash_convo(convos: List[dict]) -> str:
    parts = []
    for t in convos:
        parts.append(t.get("role", ""))
        parts.append(norm_text(t.get("content", "")))
    return hashlib.sha1("\x1f".join(parts).encode("utf-8")).hexdigest()


def hash_prefix(h: str) -> int:
    return int(h[:8], 16)


def detect_lang(text: str, hint: Optional[str] = None) -> str:
    if hint:
        h = str(hint).lower()
        if "zh" in h or "chinese" in h or h in {"cn", "zh-cn", "zh_cn"}:
            return "zh"
        if "en" in h or "english" in h:
            return "en"
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return "zh" if cjk >= max(3, len(text) // 20) else "en"


def is_multi(convos: List[dict]) -> bool:
    return sum(1 for t in convos if t.get("role") == "user") >= 2


def strip_think(text: str) -> str:
    return THINK_RE.sub("", text or "").strip()



def as_text(v) -> str:
    """Coerce dataset fields that may be str|list|None into a stripped string."""
    if v is None:
        return ""
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, (list, tuple)):
        parts = []
        for x in v:
            if x is None:
                continue
            if isinstance(x, str):
                s = x.strip()
            elif isinstance(x, dict):
                s = str(x.get("text") or x.get("content") or x.get("value") or "").strip()
            else:
                s = str(x).strip()
            if s:
                parts.append(s)
        return "\n".join(parts).strip()
    return str(v).strip()


def user_content(instruction: str, inp: str) -> str:
    instruction = (instruction or "").strip()
    inp = (inp or "").strip()
    if instruction and inp:
        return f"{instruction}\n{inp}"
    return instruction or inp


def deepctrl_to_convo(obj: dict) -> Optional[List[dict]]:
    history = obj.get("history") or []
    turns: List[dict] = []
    if isinstance(history, list):
        for pair in history:
            if not isinstance(pair, (list, tuple)) or len(pair) < 2:
                continue
            u, a = as_text(pair[0]), as_text(pair[1])
            if u and a:
                turns.append({"role": "user", "content": u})
                turns.append({"role": "assistant", "content": a})
    u = user_content(as_text(obj.get("instruction") or ""), as_text(obj.get("input") or ""))
    a = as_text(obj.get("output") or "")
    if not u or not a:
        return None
    turns.append({"role": "user", "content": u})
    turns.append({"role": "assistant", "content": a})
    return turns


def pack_out(convos: List[dict]) -> Tuple[str, int, str]:
    line = json.dumps({"conversations": convos}, ensure_ascii=False) + "\n"
    h = hash_convo(convos)
    return line, len(line.encode("utf-8")), h


def append_tagged(f, idx_f, file_id: int, convos, bucket, lang, multi, source) -> None:
    line, nbytes, h = pack_out(convos)
    if nbytes < 40:
        return
    offset = f.tell()
    f.write(line)
    # Also write a sidecar meta line? Index is enough for sampling; but we need meta in index.
    # Store ONLY the conversations line in pool file; meta lives in .idx
    idx_f.write(
        IDX_STRUCT.pack(
            file_id,
            offset,
            nbytes,
            hash_prefix(h),
            BUCKET_ID[bucket],
            LANG_ID[lang],
            1 if multi else 0,
            SOURCE_ID[source],
        )
    )


# Pool file registry
POOL_FILES = {
    0: POOL / "magpie.jsonl",
    1: POOL / "fineweb.jsonl",
    2: POOL / "r1.jsonl",
    3: POOL / "deepctrl.jsonl",
}
IDX_PATH = POOL / "all.idx"



def _disk_reservoir_write(tmp: Path, max_n: int, rng: random.Random, producer):
    """producer yields output jsonl lines; returns list of live offsets and seen count."""
    offsets: List[int] = []
    seen = 0
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open('w', encoding='utf-8') as fh:
        for out_line in producer:
            seen += 1
            if len(offsets) < max_n:
                offsets.append(fh.tell())
                fh.write(out_line)
            else:
                j = rng.randrange(seen)
                if j < max_n:
                    offsets[j] = fh.tell()
                    fh.write(out_line)
    return offsets, seen


def convert_magpie() -> None:
    out = POOL_FILES[0]
    idx = POOL / "magpie.idx"
    if out.exists() and out.stat().st_size > 10_000_000 and idx.exists():
        print(f"[skip] magpie {out.stat().st_size/GiB:.2f}G")
        return
    import pyarrow.parquet as pq

    n = 0
    with out.open("w", encoding="utf-8") as f, idx.open("wb") as ix:
        for fp in sorted((RAW / "magpie").rglob("*.parquet")):
            table = pq.read_table(fp)
            cols = {name: table.column(name) for name in table.column_names}
            for i in tqdm(range(table.num_rows), desc=fp.name):
                obj = {name: cols[name][i].as_py() for name in cols}
                conv = obj.get("conversations")
                turns: List[dict] = []
                if conv:
                    for t in conv:
                        role = t.get("from") or t.get("role")
                        content = (t.get("value") or t.get("content") or "").strip()
                        if role in ("human", "user") and content:
                            turns.append({"role": "user", "content": content})
                        elif role in ("gpt", "assistant") and content:
                            turns.append({"role": "assistant", "content": content})
                else:
                    ins = (obj.get("instruction") or "").strip()
                    resp = (obj.get("response") or "").strip()
                    if ins and resp:
                        turns = [{"role": "user", "content": ins}, {"role": "assistant", "content": resp}]
                if len(turns) < 2:
                    continue
                cat = obj.get("task_category") or ""
                bucket = "creative" if cat in MAGPIE_CREATIVE else "dialogue"
                lang = detect_lang(turns[0]["content"], obj.get("language"))
                append_tagged(f, ix, 0, turns, bucket, lang, is_multi(turns), "magpie-cn")
                n += 1
    print(f"[magpie] {n} rows")


def convert_fineweb(max_rows: int = 1_200_000) -> None:
    out = POOL_FILES[1]
    idx = POOL / "fineweb.idx"
    if out.exists() and out.stat().st_size > 50_000_000 and idx.exists():
        print(f"[skip] fineweb {out.stat().st_size/GiB:.2f}G")
        return
    files = sorted((RAW / "fineweb_qa").rglob("*.jsonl"))
    if not files:
        raise FileNotFoundError("Fineweb cleaned missing")
    rng = random.Random(SEED)
    tmp = POOL / "fineweb_res.jsonl"

    def producer():
        for fp in files:
            with fp.open("r", encoding="utf-8", errors="ignore") as inf:
                for line in tqdm(inf, desc=fp.name):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    ins = as_text(obj.get("instruction") or obj.get("question") or "")
                    extra = as_text(obj.get("input") or "")
                    if ins and extra:
                        ins = user_content(ins, extra)
                    elif not ins:
                        ins = extra
                    outp = as_text(obj.get("output") or obj.get("answer") or obj.get("response") or "")
                    if not ins or not outp:
                        continue
                    turns = [{"role": "user", "content": ins}, {"role": "assistant", "content": outp}]
                    out_line, nbytes, _h = pack_out(turns)
                    if nbytes < 40:
                        continue
                    yield out_line

    offsets, seen = _disk_reservoir_write(tmp, max_rows, rng, producer())
    n = 0
    with out.open("w", encoding="utf-8") as f, idx.open("wb") as ix, tmp.open("r", encoding="utf-8") as shard:
        for off in tqdm(offsets, desc="write-fineweb"):
            shard.seek(off)
            line = shard.readline()
            try:
                convos = json.loads(line)["conversations"]
            except Exception:
                continue
            append_tagged(f, ix, 1, convos, "knowledge", "zh", False, "fineweb-pure-qa")
            n += 1
    tmp.unlink(missing_ok=True)
    print(f"[fineweb] kept {n}/{seen}")


def convert_r1() -> None:
    out = POOL_FILES[2]
    idx = POOL / "r1.idx"
    if out.exists() and out.stat().st_size > 1_000_000 and idx.exists():
        print(f"[skip] r1 {out.stat().st_size/GiB:.2f}G")
        return
    src = RAW / "r1" / "distill_r1_110k.jsonl"
    if not src.exists():
        alts = list((RAW / "r1").rglob("*.jsonl"))
        if not alts:
            raise FileNotFoundError("R1 jsonl missing")
        src = alts[0]
    n = 0
    with src.open("r", encoding="utf-8", errors="ignore") as inf, out.open("w", encoding="utf-8") as f, idx.open("wb") as ix:
        for line in tqdm(inf, desc="r1"):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            ins = as_text(obj.get("input") or obj.get("instruction") or obj.get("question") or "")
            # final only: content (NOT reasoning_content)
            ans = strip_think(as_text(obj.get("content") or obj.get("answer") or obj.get("final_answer") or ""))
            if not ins or not ans:
                continue
            if len(ans) > 6000:
                ans = ans[:6000]
            turns = [{"role": "user", "content": ins}, {"role": "assistant", "content": ans}]
            append_tagged(f, ix, 2, turns, "reasoning", "zh", False, "congliu-r1-final")
            n += 1
    print(f"[r1] {n} rows")


class DiskReservoir:
    """Append-only file + offset list reservoir (low RAM)."""

    def __init__(self, path: Path, max_n: int, rng: random.Random, meta: tuple):
        self.path = path
        self.max_n = max_n
        self.rng = rng
        self.meta = meta  # bucket, lang, multi, source
        self.offsets: List[int] = []
        self.seen = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("w", encoding="utf-8")

    def consider_line(self, line: str) -> None:
        self.seen += 1
        if len(self.offsets) < self.max_n:
            off = self._fh.tell()
            self._fh.write(line)
            self.offsets.append(off)
            return
        j = self.rng.randrange(self.seen)
        if j < self.max_n:
            # tombstone-replace: append new, retarget slot (old bytes wasted until compact)
            off = self._fh.tell()
            self._fh.write(line)
            self.offsets[j] = off

    def close(self) -> None:
        self._fh.close()


DEEPCTRL_EXPECTED = {
    "zh": 17_193_006_547,
    "en": 12_221_139_899,
}
DEEPCTRL_URLS = {
    "zh": "https://www.modelscope.cn/datasets/deepctrl/deepctrl-sft-data/resolve/master/sft_data_zh.jsonl",
    "en": "https://www.modelscope.cn/datasets/deepctrl/deepctrl-sft-data/resolve/master/sft_data_en.jsonl",
}


def deepctrl_raw_path(lang: str) -> Path:
    return RAW / "deepctrl" / f"sft_data_{lang}.jsonl"


def deepctrl_complete(lang: str) -> bool:
    p = deepctrl_raw_path(lang)
    if not p.exists():
        return False
    sz = p.stat().st_size
    exp = DEEPCTRL_EXPECTED[lang]
    # allow tiny under-read (CDN) but require >= 99%
    return sz >= int(exp * 0.99)


def download_deepctrl(force: bool = False) -> None:
    """Download full deepctrl zh+en jsonl onto disk (~27GiB)."""
    dest_dir = RAW / "deepctrl"
    dest_dir.mkdir(parents=True, exist_ok=True)
    for lang, url in DEEPCTRL_URLS.items():
        dest = deepctrl_raw_path(lang)
        if deepctrl_complete(lang) and not force:
            print(f"[skip] deepctrl raw {lang} {dest.stat().st_size/GiB:.2f}G")
            continue
        print(f"[deepctrl] download {lang} → {dest}")
        cmd = [
            "curl", "-L", "--fail", "--retry", "20", "--retry-delay", "5",
            "--retry-all-errors", "-C", "-", "-o", str(dest), url,
        ]
        rc = subprocess.call(cmd)
        if rc != 0:
            raise RuntimeError(f"curl deepctrl-{lang} failed rc={rc}")
        if not deepctrl_complete(lang):
            raise RuntimeError(
                f"deepctrl-{lang} incomplete size={dest.stat().st_size} expected≈{DEEPCTRL_EXPECTED[lang]}"
            )
        print(f"[deepctrl] {lang} done {dest.stat().st_size/GiB:.2f}G")


def convert_deepctrl(pool_budget_gib: float = 14.0) -> None:
    """Convert on-disk deepctrl into pool via large stratified reservoir (full raw kept)."""
    out = POOL_FILES[3]
    idx = POOL / "deepctrl.idx"
    # require a substantial pool; partials redo
    if out.exists() and out.stat().st_size > int(8 * GiB) and idx.exists() and idx.stat().st_size > 10_000_000:
        print(f"[skip] deepctrl pool {out.stat().st_size/GiB:.2f}G")
        return
    for lang in ("zh", "en"):
        if not deepctrl_complete(lang):
            raise FileNotFoundError(f"deepctrl raw missing/incomplete: {deepctrl_raw_path(lang)}")

    rng = random.Random(SEED)
    keep_budget = int(pool_budget_gib * GiB)
    avg = 2000
    total_slots = max(500_000, min(8_000_000, keep_budget // avg))
    print(f"[deepctrl] local reservoir slots≈{total_slots} (~{total_slots*avg/GiB:.1f}G budget)")

    shard_dir = POOL / "deepctrl_shards"
    if shard_dir.exists():
        import shutil
        shutil.rmtree(shard_dir)
    shard_dir.mkdir(parents=True)

    reservoirs: Dict[Tuple[str, str, bool], DiskReservoir] = {}

    def res_for(bucket: str, lang: str, multi: bool, source: str) -> DiskReservoir:
        key = (bucket, lang, multi)
        if key not in reservoirs:
            br = BUCKET_RATIO[bucket]
            lr = 0.80 if lang == "zh" else 0.20
            # oversample multi in dialogue; still allow multi elsewhere
            if bucket == "dialogue":
                tr = 0.45 if multi else 0.55
            else:
                tr = 0.20 if multi else 0.80
            share = br * lr * tr
            # boost scarce creative/code slots
            boost = 1.6 if bucket in ("creative", "code") else 1.15
            slots = max(800, int(total_slots * share * boost))
            path = shard_dir / f"{bucket}_{lang}_{'m' if multi else 's'}.jsonl"
            reservoirs[key] = DiskReservoir(path, slots, rng, (bucket, lang, multi, source))
        return reservoirs[key]

    for lang_key in ("zh", "en"):
        src = deepctrl_raw_path(lang_key)
        print(f"[deepctrl] reservoir-scan {lang_key} from {src}")
        n_lines = bad = 0
        with src.open("r", encoding="utf-8", errors="ignore") as inf:
            for line in tqdm(inf, desc=f"deepctrl-{lang_key}"):
                n_lines += 1
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    bad += 1
                    continue
                turns = deepctrl_to_convo(obj)
                if not turns:
                    continue
                try:
                    typ_i = int(obj.get("type"))
                except Exception:
                    typ_i = -1
                bucket = TYPE_TO_BUCKET.get(typ_i, "dialogue")
                lang = "en" if lang_key == "en" else detect_lang(turns[0]["content"], obj.get("language"))
                multi = is_multi(turns)
                source = f"deepctrl-{lang_key}"
                out_line, nbytes, _h = pack_out(turns)
                if nbytes < 40:
                    continue
                res_for(bucket, lang, multi, source).consider_line(out_line)
                if n_lines % 1_000_000 == 0:
                    kept = sum(len(r.offsets) for r in reservoirs.values())
                    print(f"  … {lang_key} lines={n_lines} live={kept} bad={bad}")
        print(f"[deepctrl] {lang_key} lines={n_lines} bad={bad}")

    for r in reservoirs.values():
        r.close()

    n = 0
    with out.open("w", encoding="utf-8") as f, idx.open("wb") as ix:
        for res in reservoirs.values():
            bucket, lang, multi, source = res.meta
            with res.path.open("r", encoding="utf-8") as shard:
                for off in res.offsets:
                    shard.seek(off)
                    line = shard.readline()
                    if not line:
                        continue
                    try:
                        convos = json.loads(line)["conversations"]
                    except Exception:
                        continue
                    append_tagged(f, ix, 3, convos, bucket, lang, multi, source)
                    n += 1
    stats = {
        "converted_live": n,
        "shards": {
            f"{b}/{l}/{'m' if m else 's'}": {"seen": r.seen, "kept": len(r.offsets)}
            for (b, l, m), r in reservoirs.items()
        },
        "out_bytes": out.stat().st_size,
    }
    (POOL / "deepctrl_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    import shutil
    shutil.rmtree(shard_dir, ignore_errors=True)
    print(f"[deepctrl] wrote {n} rows ({out.stat().st_size/GiB:.2f}G)")



def merge_indexes() -> List[Path]:
    """Concat per-source idx into all.idx; return pool file paths by file_id."""
    parts = [
        (0, POOL / "magpie.idx"),
        (1, POOL / "fineweb.idx"),
        (2, POOL / "r1.idx"),
        (3, POOL / "deepctrl.idx"),
    ]
    with IDX_PATH.open("wb") as out:
        for _fid, p in parts:
            if p.exists():
                out.write(p.read_bytes())
    return [POOL_FILES[i] for i in range(4)]


def load_index() -> List[tuple]:
    data = IDX_PATH.read_bytes()
    n = len(data) // IDX_STRUCT.size
    rows = []
    for i in range(n):
        rows.append(IDX_STRUCT.unpack_from(data, i * IDX_STRUCT.size))
    return rows


def stratified_select(index_rows: List[tuple], target_bytes: int, seed: int) -> Tuple[List[int], dict]:
    """Return list of index positions into index_rows."""
    rng = random.Random(seed)
    # dedupe by hash_prefix (approx) within same out_bytes — collision ok rare
    # Better: keep first of hash_prefix
    seen_h = set()
    filtered = []
    for i, rec in enumerate(index_rows):
        hp = rec[3]
        if hp in seen_h:
            continue
        seen_h.add(hp)
        filtered.append(i)
    # group filtered positions by stratum
    by: Dict[Tuple[int, int, int], List[int]] = defaultdict(list)
    for i in filtered:
        file_id, offset, nbytes, hp, b, lang, multi, src = index_rows[i]
        by[(b, lang, multi)].append(i)
    for k in by:
        rng.shuffle(by[k])

    selected: List[int] = []
    selected_bytes = 0
    counts_b = [0] * 5
    counts_l = [0, 0]
    counts_t = [0, 0]  # single, multi
    source_counts: Dict[str, int] = defaultdict(int)
    ptr = {k: 0 for k in by}

    def n_sel() -> int:
        return len(selected)

    stalled = 0
    keys = list(by.keys())
    while selected_bytes < target_bytes and stalled < 100000:
        n = max(1, n_sel())
        # rank strata by deficit
        ranked = []
        for b, lang, multi in keys:
            need_b = BUCKET_RATIO[ID_BUCKET[b]] - (counts_b[b] / n)
            need_l = (0.80 if lang == 0 else 0.20) - (counts_l[lang] / n)
            need_t = (0.25 if multi else 0.75) - (counts_t[multi] / n)
            ranked.append((need_b * 2.0 + need_l * 1.5 + need_t * 1.2, (b, lang, multi)))
        ranked.sort(reverse=True)
        progressed = False
        for _score, key in ranked:
            lst = by[key]
            i = ptr[key]
            while i < len(lst):
                pos = lst[i]
                i += 1
                ptr[key] = i
                file_id, offset, nbytes, hp, b, lang, multi, src = index_rows[pos]
                nn = n_sel() + 1
                if nn > 3000:
                    if (counts_b[b] + 1) / nn > BUCKET_RATIO[ID_BUCKET[b]] + 0.05:
                        continue
                    if (counts_l[lang] + 1) / nn > ((0.80 if lang == 0 else 0.20) + 0.05):
                        continue
                    if (counts_t[multi] + 1) / nn > ((0.25 if multi else 0.75) + 0.06):
                        continue
                    if ID_SOURCE[src] == "magpie-cn" and (source_counts["magpie-cn"] + 1) / nn > 0.14:
                        continue
                selected.append(pos)
                selected_bytes += nbytes
                counts_b[b] += 1
                counts_l[lang] += 1
                counts_t[multi] += 1
                source_counts[ID_SOURCE[src]] += 1
                progressed = True
                break
            if progressed:
                break
        if progressed:
            stalled = 0
        else:
            stalled += 1

    n = max(1, len(selected))
    report = {
        "n": len(selected),
        "bytes": selected_bytes,
        "gb": selected_bytes / GiB,
        "bucket": {ID_BUCKET[i]: counts_b[i] / n for i in range(5)},
        "bucket_n": {ID_BUCKET[i]: counts_b[i] for i in range(5)},
        "lang": {"zh": counts_l[0] / n, "en": counts_l[1] / n},
        "turn": {"single": counts_t[0] / n, "multi": counts_t[1] / n},
        "source_n": dict(source_counts),
    }
    return selected, report


def write_from_index(out_path: Path, index_rows: List[tuple], positions: List[int], pool_paths: List[Path], seed: int) -> None:
    rng = random.Random(seed)
    order = positions[:]
    rng.shuffle(order)
    # group by file for sequential-ish reads
    handles = {i: p.open("r", encoding="utf-8") for i, p in enumerate(pool_paths) if p.exists()}
    try:
        with out_path.open("w", encoding="utf-8") as out:
            for pos in tqdm(order, desc=out_path.name):
                file_id, offset, nbytes, hp, b, lang, multi, src = index_rows[pos]
                fh = handles[file_id]
                fh.seek(offset)
                line = fh.readline()
                if not line:
                    continue
                # line already conversations-only JSONL
                out.write(line if line.endswith("\n") else line + "\n")
    finally:
        for fh in handles.values():
            fh.close()


def write_sources(path: Path, rep5: dict, rep10: dict, extras: dict) -> None:
    text = f"""# sft_miniseek_v5

- Seed: `{SEED}`
- Format: Conversations JSONL `{{"conversations":[{{"role","content"}},...]}}`
- Capability targets: dialogue43 / knowledge22 / reasoning15 / creative12 / code8
- Turns: single75 / multi25; Language: zh80 / en20
- Route A: deepctrl **full on-disk** (ModelScope zh+en) + Fineweb Pure QA (`sft/cleaned`) + Magpie-Qwen2-Pro-200K-Chinese + Congliu R1 **final only** (`content`, not `reasoning_content`)
- 10G is a superset of 5G (same seed family); v1 not deleted; no HF upload in this step

## 10G pack (`sft_miniseek_v5_10g.jsonl`)

- rows: {rep10['n']}
- size_bytes: {rep10['bytes']} (~{rep10['gb']:.2f} GiB)
- bucket: {json.dumps({k: round(v, 4) for k, v in rep10['bucket'].items()}, ensure_ascii=False)}
- lang: {json.dumps({k: round(v, 4) for k, v in rep10['lang'].items()}, ensure_ascii=False)}
- turn: {json.dumps({k: round(v, 4) for k, v in rep10['turn'].items()}, ensure_ascii=False)}
- sources: {json.dumps(rep10['source_n'], ensure_ascii=False)}

## 5G pack (`sft_miniseek_v5_5g.jsonl`) — subset of 10G

- rows: {rep5['n']}
- size_bytes: {rep5['bytes']} (~{rep5['gb']:.2f} GiB)
- bucket: {json.dumps({k: round(v, 4) for k, v in rep5['bucket'].items()}, ensure_ascii=False)}
- lang: {json.dumps({k: round(v, 4) for k, v in rep5['lang'].items()}, ensure_ascii=False)}
- turn: {json.dumps({k: round(v, 4) for k, v in rep5['turn'].items()}, ensure_ascii=False)}
- sources: {json.dumps(rep5['source_n'], ensure_ascii=False)}

## Notes

- free_gb_before: {extras.get('free_gb_before')}
- free_gb_after: {extras.get('free_gb_after')}
- deepctrl full zh+en retained under `data/_raw_sft_v5/deepctrl/`; pool built via large stratified reservoir from local files
- Fineweb Pure QA path: `opencsg/Fineweb-Edu-Chinese-V2.2` `sft/cleaned/*.jsonl`
"""
    path.write_text(text, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-10g", type=float, default=10.0)
    ap.add_argument("--target-5g", type=float, default=5.0)
    ap.add_argument("--skip-deepctrl", action="store_true")
    ap.add_argument("--force-download", action="store_true")
    ap.add_argument("--skip-download-check", action="store_true")
    args = ap.parse_args()

    ensure_dirs()
    free0 = free_gb()
    print(f"[disk] free {free0:.1f}G")
    if free0 < 25 and not args.skip_download_check:
        print("[disk] <25G free — abort")
        return 2

    if not list((RAW / "magpie").rglob("*.parquet")):
        print("Magpie missing"); return 1
    if not list((RAW / "fineweb_qa").rglob("*.jsonl")):
        print("Fineweb missing"); return 1
    if not list((RAW / "r1").rglob("*.jsonl")):
        print("R1 missing"); return 1

    convert_magpie()
    convert_fineweb()
    convert_r1()
    target10 = int(args.target_10g * GiB)
    target5 = int(args.target_5g * GiB)
    if not args.skip_deepctrl:
        download_deepctrl(force=args.force_download)
        convert_deepctrl()

    merge_indexes()
    index_rows = load_index()
    print(f"[mix] index entries={len(index_rows)}")
    if not index_rows:
        print("empty index"); return 1

    sel10, rep10 = stratified_select(index_rows, target10, SEED)
    print("[mix] 10G", json.dumps(rep10, ensure_ascii=False))
    # 5G ⊂ 10G: run select on the selected-10 subset of index
    sub_index = [index_rows[i] for i in sel10]
    # map sub positions back: stratified_select returns indices into sub_index
    sel5_local, rep5 = stratified_select(sub_index, target5, SEED + 1)
    # top-up if short
    have = set(sel5_local)
    b = rep5["bytes"]
    for j in range(len(sub_index)):
        if b >= target5:
            break
        if j in have:
            continue
        have.add(j)
        sel5_local.append(j)
        b += sub_index[j][2]
    # remap to global index positions
    sel5 = [sel10[j] for j in sel5_local]
    # recount rep5 from global
    counts_b = [0] * 5
    counts_l = [0, 0]
    counts_t = [0, 0]
    source_counts: Dict[str, int] = defaultdict(int)
    byt = 0
    for pos in sel5:
        file_id, offset, nbytes, hp, bkt, lang, multi, src = index_rows[pos]
        counts_b[bkt] += 1
        counts_l[lang] += 1
        counts_t[multi] += 1
        source_counts[ID_SOURCE[src]] += 1
        byt += nbytes
    n = max(1, len(sel5))
    rep5 = {
        "n": len(sel5),
        "bytes": byt,
        "gb": byt / GiB,
        "bucket": {ID_BUCKET[i]: counts_b[i] / n for i in range(5)},
        "bucket_n": {ID_BUCKET[i]: counts_b[i] for i in range(5)},
        "lang": {"zh": counts_l[0] / n, "en": counts_l[1] / n},
        "turn": {"single": counts_t[0] / n, "multi": counts_t[1] / n},
        "source_n": dict(source_counts),
    }

    pool_paths = [POOL_FILES[i] for i in range(4)]
    out10 = OUT_DIR / "sft_miniseek_v5_10g.jsonl"
    out5 = OUT_DIR / "sft_miniseek_v5_5g.jsonl"
    write_from_index(out10, index_rows, sel10, pool_paths, SEED)
    write_from_index(out5, index_rows, sel5, pool_paths, SEED)
    free1 = free_gb()
    write_sources(OUT_DIR / "SOURCES_v5.md", rep5, rep10, {
        "free_gb_before": round(free0, 2),
        "free_gb_after": round(free1, 2),
    })
    print(f"[done] {out5} {out5.stat().st_size/GiB:.2f}G rows~{rep5['n']}")
    print(f"[done] {out10} {out10.stat().st_size/GiB:.2f}G rows~{rep10['n']}")
    print(f"[disk] free {free1:.1f}G")
    return 0


if __name__ == "__main__":
    sys.exit(main())
