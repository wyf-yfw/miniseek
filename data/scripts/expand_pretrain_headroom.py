#!/usr/bin/env python3
"""Append ~2.5–3GB HQ headroom → pretrain_mix_12g.jsonl (keep 10g intact)."""
from __future__ import annotations

import json
import os
import shutil
import time
import traceback
from pathlib import Path

from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

ROOT = Path("/home/wyf/miniseek/data")
RAW_ZH = ROOT / "raw" / "fineweb_edu_zh_more"
RAW_COSMO = ROOT / "raw" / "cosmopedia_v2"
BUCKET = ROOT / "raw" / "buckets_headroom"
LOG = ROOT / "raw" / "expand_tmp" / "headroom.log"
OUT_12G = ROOT / "pretrain_mix_12g.jsonl"
BASE_10G = ROOT / "pretrain_mix_10g.jsonl"
STATS = ROOT / "pretrain_mix_12g_stats.json"
SOURCES = ROOT / "pretrain_mix_12g_SOURCES.md"

IM_END = "\n<|im_end|>"
MIN_CHARS = 64
SEED = 47
TARGET_EXTRA = int(2.7 * 1024**3)
ZH_TARGET = int(1.85 * 1024**3)
MIN_FREE_BYTES = int(3.5 * 1024**3)


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def free_bytes(path: Path = Path("/")) -> int:
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize


def disk_ok(need: int = 0) -> None:
    fb = free_bytes()
    if fb < MIN_FREE_BYTES + need:
        raise SystemExit(f"disk too low: free={fb/1024**3:.1f}G need_margin+{need/1024**3:.1f}G")


def write_jsonl_text(fp, text: str) -> int:
    text = (text or "").strip()
    if len(text) < MIN_CHARS:
        return 0
    if not text.endswith("<|im_end|>"):
        text = text + IM_END
    line = json.dumps({"text": text}, ensure_ascii=False) + "\n"
    fp.write(line)
    return len(line.encode("utf-8"))


def convert_edu_zh_parquet(src: Path, out_fp) -> tuple[int, int]:
    table = pq.read_table(src)
    cols = set(table.column_names)
    text_col = "text" if "text" in cols else ("content" if "content" in cols else None)
    if text_col is None:
        log(f"skip no text col {src} cols={table.column_names}")
        return 0, 0
    nbytes = nlines = 0
    for t in table[text_col].to_pylist():
        if not isinstance(t, str):
            continue
        b = write_jsonl_text(out_fp, t)
        if b:
            nbytes += b
            nlines += 1
    return nbytes, nlines


def convert_cosmo_parquet(src: Path, out_fp, byte_budget: int) -> tuple[int, int]:
    pf = pq.ParquetFile(src)
    cols = set(pf.schema_arrow.names)
    text_col = "text" if "text" in cols else ("content" if "content" in cols else None)
    if text_col is None:
        log(f"cosmo columns {pf.schema_arrow.names}")
        return 0, 0
    nbytes = nlines = 0
    for rg in range(pf.num_row_groups):
        if nbytes >= byte_budget:
            break
        table = pf.read_row_group(rg, columns=[text_col])
        for t in table[text_col].to_pylist():
            if nbytes >= byte_budget:
                break
            if not isinstance(t, str):
                continue
            b = write_jsonl_text(out_fp, t)
            if b:
                nbytes += b
                nlines += 1
    return nbytes, nlines


def download_zh_one(i: int) -> Path:
    RAW_ZH.mkdir(parents=True, exist_ok=True)
    dest = RAW_ZH / f"{i:06d}.parquet"
    if dest.exists() and dest.stat().st_size > 1000:
        return dest
    nested = RAW_ZH / "4_5" / f"{i:06d}.parquet"
    if nested.exists() and nested.stat().st_size > 1000:
        shutil.move(str(nested), str(dest))
        return dest
    name = f"4_5/{i:06d}.parquet"
    log(f"download zh {name}")
    p = Path(
        hf_hub_download(
            repo_id="opencsg/Fineweb-Edu-Chinese-V2.1",
            filename=name,
            repo_type="dataset",
            local_dir=str(RAW_ZH),
        )
    )
    if p.resolve() != dest.resolve():
        if dest.exists():
            dest.unlink()
        shutil.move(str(p), str(dest))
    return dest


def download_cosmo_one() -> Path:
    RAW_COSMO.mkdir(parents=True, exist_ok=True)
    name = "cosmopedia-v2/train-00000-of-00104.parquet"
    dest = RAW_COSMO / "train-00000-of-00104.parquet"
    if dest.exists() and dest.stat().st_size > 1_000_000:
        return dest
    nested = RAW_COSMO / name
    if nested.exists() and nested.stat().st_size > 1_000_000:
        shutil.move(str(nested), str(dest))
        return dest
    log(f"download {name}")
    p = Path(
        hf_hub_download(
            repo_id="HuggingFaceTB/smollm-corpus",
            filename=name,
            repo_type="dataset",
            local_dir=str(RAW_COSMO),
        )
    )
    if p.resolve() != dest.resolve():
        if dest.exists():
            dest.unlink()
        shutil.move(str(p), str(dest))
    return dest


def main():
    t0 = time.time()
    # fresh log section
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"\n==== restart {time.strftime('%Y-%m-%d %H:%M:%S')} ====\n")
    BUCKET.mkdir(parents=True, exist_ok=True)
    disk_ok()
    if not BASE_10G.exists():
        raise SystemExit("missing 10g base")

    zh_out = BUCKET / "web_edu_zh_more.jsonl"
    zh_bytes = zh_lines = 0
    zh_shards_used = []
    # resume if partial jsonl exists
    if zh_out.exists() and zh_out.stat().st_size > 0:
        zh_bytes = zh_out.stat().st_size
        # approximate lines
        with open(zh_out, "rb") as f:
            zh_lines = sum(1 for _ in f)
        log(f"resume zh jsonl at {zh_bytes/1024**3:.2f} GiB / {zh_lines} lines")
        mode = "a"
    else:
        mode = "w"

    with open(zh_out, mode, encoding="utf-8") as fp:
        i = 36
        while zh_bytes < ZH_TARGET and i < 400:
            disk_ok()
            try:
                p = download_zh_one(i)
            except Exception as e:
                log(f"download fail {i}: {e}")
                traceback.print_exc()
                i += 1
                continue
            try:
                b, n = convert_edu_zh_parquet(p, fp)
                fp.flush()
                zh_bytes += b
                zh_lines += n
                zh_shards_used.append(p.name)
                log(f"converted {p.name}: +{b/1e6:.1f}MB lines={n} total={zh_bytes/1024**3:.2f}GiB")
            except Exception as e:
                log(f"convert fail {p}: {e}")
                traceback.print_exc()
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass
            # cleanup empty nested
            nested_dir = RAW_ZH / "4_5"
            if nested_dir.exists():
                for leftover in nested_dir.glob("*.parquet"):
                    try:
                        leftover.unlink()
                    except Exception:
                        pass
            i += 1
            if i % 20 == 0:
                log(f"zh progress i={i} {zh_bytes/1024**3:.2f} GiB free={free_bytes()/1024**3:.1f}G")

    cosmo_out = BUCKET / "cosmo_v2_en.jsonl"
    cosmo_bytes = cosmo_lines = 0
    cosmo_budget = max(int(0.7 * 1024**3), TARGET_EXTRA - zh_bytes)
    cosmo_budget = min(cosmo_budget, int(1.2 * 1024**3))
    if free_bytes() > int(5.5 * 1024**3):
        try:
            cosmo_path = download_cosmo_one()
            log(f"cosmo parquet size={cosmo_path.stat().st_size/1024**3:.2f}GiB budget={cosmo_budget/1024**3:.2f}")
            with open(cosmo_out, "w", encoding="utf-8") as fp:
                cosmo_bytes, cosmo_lines = convert_cosmo_parquet(cosmo_path, fp, cosmo_budget)
            log(f"cosmo done {cosmo_bytes/1024**3:.2f} GiB / {cosmo_lines} lines")
            try:
                cosmo_path.unlink(missing_ok=True)
            except Exception:
                pass
        except Exception as e:
            log(f"cosmo fail: {e}")
            traceback.print_exc()
    else:
        log(f"skip cosmo low disk free={free_bytes()/1024**3:.1f}G")

    appended_plan = zh_bytes + cosmo_bytes
    log(f"extra planned {appended_plan/1024**3:.2f} GiB")

    # build 12g
    need = BASE_10G.stat().st_size + 50_000_000
    disk_ok(need=0)
    if free_bytes() < need + MIN_FREE_BYTES:
        # not enough to full-copy; append in-place onto a hardlinked? 
        # fallback: rename strategy — copy with sparse? Just require space.
        raise SystemExit(f"not enough space to copy 10g→12g free={free_bytes()/1024**3:.1f}G need~{need/1024**3:.1f}G")

    if OUT_12G.exists():
        OUT_12G.unlink()
    log(f"copy {BASE_10G.name} -> {OUT_12G.name}")
    shutil.copy2(BASE_10G, OUT_12G)
    appended_bytes = appended_lines = 0
    with open(OUT_12G, "a", encoding="utf-8") as out:
        for src in (zh_out, cosmo_out):
            if not src.exists() or src.stat().st_size == 0:
                continue
            with open(src, "r", encoding="utf-8") as inp:
                for line in inp:
                    out.write(line)
                    appended_bytes += len(line.encode("utf-8"))
                    appended_lines += 1
    final_bytes = OUT_12G.stat().st_size
    base_lines = 3362521
    final_lines = base_lines + appended_lines

    stats = {
        "seed": SEED,
        "base_file": str(BASE_10G),
        "base_bytes": BASE_10G.stat().st_size,
        "base_lines": base_lines,
        "target_extra_bytes": TARGET_EXTRA,
        "sources": {
            "web_edu_zh_more.jsonl": {
                "bytes": zh_bytes,
                "lines": zh_lines,
                "shards": zh_shards_used,
                "shard_count": len(zh_shards_used),
                "repo": "opencsg/Fineweb-Edu-Chinese-V2.1",
                "split": "4_5",
            },
            "cosmo_v2_en.jsonl": {
                "bytes": cosmo_bytes,
                "lines": cosmo_lines,
                "repo": "HuggingFaceTB/smollm-corpus",
                "file": "cosmopedia-v2/train-00000-of-00104.parquet",
            },
        },
        "extra_bytes_written": appended_bytes,
        "extra_lines": appended_lines,
        "final_file": str(OUT_12G),
        "final_bytes": final_bytes,
        "final_giB": round(final_bytes / 1024**3, 2),
        "final_lines_est": final_lines,
        "elapsed_sec": round(time.time() - t0, 1),
        "disk_free_after": free_bytes(),
    }
    STATS.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    sources_md = f"""# pretrain_mix_12g — 余量扩充说明

生成时间：{time.strftime('%Y-%m-%d %H:%M')}（Asia/Shanghai）  
机器：wyf `/home/wyf/miniseek`  
目的：在 `pretrain_mix_10g.jsonl` 之上再加约 **2.5–3GB** 高质量余量，清洗后仍尽量剩 ~10GB。  
原则：优先 Fineweb-Edu 中文 / Cosmopedia-v2；**不**追加生网页（未再加 SkyPile / FineWeb2）。

## 产出

| 文件 | 大小 | 行数(估) | 说明 |
|------|------|----------|------|
| `data/pretrain_mix_12g.jsonl` | **{stats['final_giB']} GiB** ({final_bytes:,} B) | **{final_lines:,}** | 10g 全量 + 本轮增量 |
| `data/pretrain_mix_10g.jsonl` | 10.15 GiB | 3,362,521 | **未改动** |
| `data/pretrain_mix_5g.jsonl` | 4.94 GiB | 1,834,462 | **未改动** |
| `data/pretrain_mix_12g_stats.json` | — | — | 本轮统计 |

构建：`cp pretrain_mix_10g.jsonl → pretrain_mix_12g.jsonl`，再追加 extras（seed={SEED}）。

## 本轮增量

| 源 | 纳入字节 | 行数 | 说明 |
|----|----------|------|------|
| Fineweb-Edu-Chinese-V2.1 `4_5` 续 shard（{len(zh_shards_used)} 个） | {zh_bytes:,} | {zh_lines:,} | 接续 10g 已用的 000000–000035 |
| SmolLM cosmopedia-v2 `train-00000` 截取 | {cosmo_bytes:,} | {cosmo_lines:,} | 合成教材风英文 |
| **增量合计** | **{appended_bytes:,}** | **{appended_lines:,}** | |
| **= 10g + 增量** | **{final_bytes:,}** | **{final_lines:,}** | |

## 原始语料

1. [opencsg/Fineweb-Edu-Chinese-V2.1](https://huggingface.co/datasets/opencsg/Fineweb-Edu-Chinese-V2.1) `4_5/` 续下（Clash SOCKS `socks5h://127.0.0.1:7897`）
2. [HuggingFaceTB/smollm-corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) `cosmopedia-v2/train-00000-of-00104.parquet`（按字节预算截取）

## 处理规则

- 行格式：`{{"text": "<正文>\\n<|im_end|>"}}`
- 短文本 `<64` 字符丢弃
- 未覆盖 5g / 10g；未杀训练进程
- 转换后已删本轮 parquet 以腾磁盘

## 磁盘

组装后 `/` 空闲约 **{free_bytes()/1024**3:.0f}G**。
"""
    SOURCES.write_text(sources_md, encoding="utf-8")
    log(f"DONE final={stats['final_giB']}GiB extra={appended_bytes/1024**3:.2f}GiB free={free_bytes()/1024**3:.1f}G")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        log("FATAL")
        raise
