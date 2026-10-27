#!/usr/bin/env python3
"""Build ~5-6GB mixed pretrain jsonl: ~80% wiki+web, ~20% books/code/other."""
from __future__ import annotations
import json, os, random, struct, sys, time
from pathlib import Path

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None
try:
    import zstandard as zstd
except ImportError:
    zstd = None

ROOT = Path('/home/wyf/miniseek/data')
RAW = ROOT / 'raw'
BUCKETS = RAW / 'buckets'
EOS = '<|im_end|>'
TARGET_TOTAL = int(5.5 * 1024**3)  # 5.5 GiB
# 80% wiki+web, 20% other
BUDGET = {
    'wiki': int(0.50 * TARGET_TOTAL),   # 50% of mix
    'web':  int(0.30 * TARGET_TOTAL),   # 30% of mix  -> wiki+web=80%
    'books': int(0.10 * TARGET_TOTAL),  # 10%
    'code':  int(0.06 * TARGET_TOTAL),  # 6%
    'other': int(0.04 * TARGET_TOTAL),  # 4%
}

INSTR_MARKERS = (
    '请你', '请帮', '请根据', '根据以下', '你是一个', 'You are an',
    'Human:', 'Assistant:', '### Instruction', '### Response',
    '<|im_start|>user', '给即将到来', '帮我想一些',
)

def ensure_eos(text: str) -> str:
    text = (text or '').replace('\x00', '').strip()
    if not text:
        return ''
    if not text.endswith(EOS):
        text = text.rstrip() + '\n' + EOS
    return text

def looks_instruction(text: str) -> bool:
    head = text[:200]
    hits = sum(1 for m in INSTR_MARKERS if m in head)
    return hits >= 1 and ('？' in head or '?' in head or '请' in head[:40])

def write_jsonl_line(f, text: str) -> int:
    text = ensure_eos(text)
    if len(text) < 40:
        return 0
    line = json.dumps({'text': text}, ensure_ascii=False) + '\n'
    b = line.encode('utf-8')
    f.write(b)
    return len(b)

def convert_wiki(out_path: Path, budget: int, seed: int = 42) -> dict:
    assert pq is not None
    files = sorted((RAW / 'wikipedia-zh').glob('train-*.parquet'))
    rng = random.Random(seed)
    # reservoir-ish: randomly keep rows with probability tuned by est size
    # Full est ~28GB; keep fraction budget/28GB
    keep_frac = min(1.0, budget / (28 * 1024**3) * 1.15)
    written = 0
    n_in = n_out = 0
    samples = []
    with out_path.open('wb') as out:
        for fp in files:
            if written >= budget:
                break
            pf = pq.ParquetFile(fp)
            for rg in range(pf.num_row_groups):
                if written >= budget:
                    break
                tbl = pf.read_row_group(rg, columns=['text'])
                for text in tbl.column('text').to_pylist():
                    n_in += 1
                    if rng.random() > keep_frac:
                        continue
                    text = str(text or '')
                    if looks_instruction(text):
                        continue
                    nbytes = write_jsonl_line(out, text)
                    if not nbytes:
                        continue
                    written += nbytes
                    n_out += 1
                    if len(samples) < 2:
                        samples.append(text[:180].replace('\n', ' '))
                    if written >= budget:
                        break
    return {'bytes': written, 'rows_in': n_in, 'rows_out': n_out, 'samples': samples, 'keep_frac': keep_frac}

def convert_fineweb_zst(path: Path, out_path: Path, budget: int, seed: int = 43) -> dict:
    assert zstd is not None
    rng = random.Random(seed)
    written = n_in = n_out = 0
    samples = []
    # fineweb mainland ~1.24GB zst; assume ~3-4x expansion → sample
    keep_frac = 0.55
    dctx = zstd.ZstdDecompressor()
    with path.open('rb') as fin, dctx.stream_reader(fin) as reader, out_path.open('wb') as out:
        # text mode wrapper
        import io
        text_stream = io.TextIOWrapper(reader, encoding='utf-8', errors='replace')
        for line in text_stream:
            if written >= budget:
                break
            n_in += 1
            if rng.random() > keep_frac:
                continue
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            text = obj.get('text') or obj.get('content') or ''
            if looks_instruction(text):
                continue
            # prefer mainland-ish length
            if len(text) < 80:
                continue
            nbytes = write_jsonl_line(out, text)
            if not nbytes:
                continue
            written += nbytes
            n_out += 1
            if len(samples) < 2:
                samples.append(text[:180].replace('\n', ' '))
            # adaptive: if running hot past budget early, tighten
            if n_in > 50000 and written > 0:
                # noop; keep_frac fixed
                pass
    return {'bytes': written, 'rows_in': n_in, 'rows_out': n_out, 'samples': samples}

def convert_firefly_jsonl(path: Path, out_path: Path, budget: int, field_candidates=('text','content','summary','article','input'), seed: int = 44, filter_instr: bool = True) -> dict:
    rng = random.Random(seed)
    written = n_in = n_out = 0
    samples = []
    with path.open('r', encoding='utf-8', errors='replace') as fin, out_path.open('wb') as out:
        for line in fin:
            if written >= budget:
                break
            n_in += 1
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            text = ''
            if isinstance(obj, dict):
                for k in field_candidates:
                    if k in obj and obj[k]:
                        text = str(obj[k])
                        break
                if not text:
                    # common firefly: maybe only one string field
                    for v in obj.values():
                        if isinstance(v, str) and len(v) > 40:
                            text = v
                            break
            elif isinstance(obj, str):
                text = obj
            if filter_instr and looks_instruction(text):
                continue
            if len(text) < 40:
                continue
            nbytes = write_jsonl_line(out, text)
            if not nbytes:
                continue
            written += nbytes
            n_out += 1
            if len(samples) < 2:
                samples.append(text[:180].replace('\n', ' '))
    return {'bytes': written, 'rows_in': n_in, 'rows_out': n_out, 'samples': samples}

def convert_code_parquet(paths, out_path: Path, budget: int, seed: int = 45) -> dict:
    assert pq is not None
    rng = random.Random(seed)
    written = n_in = n_out = 0
    samples = []
    with out_path.open('wb') as out:
        for path in paths:
            if not Path(path).exists():
                continue
            if written >= budget:
                break
            pf = pq.ParquetFile(path)
            cols = [c for c in ('content', 'text', 'code') if c in pf.schema_arrow.names]
            if not cols:
                cols = [pf.schema_arrow.names[0]]
            for rg in range(pf.num_row_groups):
                if written >= budget:
                    break
                tbl = pf.read_row_group(rg, columns=cols)
                col = cols[0]
                for text in tbl.column(col).to_pylist():
                    n_in += 1
                    text = str(text or '')
                    if len(text) < 40 or len(text) > 200000:
                        continue
                    nbytes = write_jsonl_line(out, text)
                    if not nbytes:
                        continue
                    written += nbytes
                    n_out += 1
                    if len(samples) < 2:
                        samples.append(text[:180].replace('\n', ' '))
                    if written >= budget:
                        break
    return {'bytes': written, 'rows_in': n_in, 'rows_out': n_out, 'samples': samples}

def shuffle_merge(bucket_paths, final_path: Path, valid_path: Path, valid_bytes: int = 8*1024*1024, seed: int = 7):
    """Two-pass-ish: load offsets, shuffle order of lines via chunked shuffle."""
    rng = random.Random(seed)
    # Read all lines into temp chunk files of ~200MB then shuffle chunk order + lines
    chunk_dir = BUCKETS / '_shuffle_chunks'
    chunk_dir.mkdir(parents=True, exist_ok=True)
    for old in chunk_dir.glob('*.jsonl'):
        old.unlink()
    chunk_size = 200 * 1024 * 1024
    chunk_idx = 0
    cur_bytes = 0
    cur = None
    total_lines = 0
    def open_new():
        nonlocal chunk_idx, cur, cur_bytes
        if cur:
            cur.close()
        p = chunk_dir / f'chunk_{chunk_idx:04d}.jsonl'
        chunk_idx += 1
        cur = p.open('wb')
        cur_bytes = 0
        return p
    open_new()
    for bp in bucket_paths:
        with open(bp, 'rb') as fin:
            for line in fin:
                if cur_bytes >= chunk_size:
                    open_new()
                cur.write(line)
                cur_bytes += len(line)
                total_lines += 1
    if cur:
        cur.close()
    chunks = sorted(chunk_dir.glob('chunk_*.jsonl'))
    rng.shuffle(chunks)
    # write final + holdout
    valid_written = 0
    final_written = 0
    n_final = n_valid = 0
    with final_path.open('wb') as fout, valid_path.open('wb') as vout:
        for ch in chunks:
            lines = ch.read_bytes().splitlines(keepends=True)
            rng.shuffle(lines)
            for line in lines:
                # ~0.15% to valid until budget
                if valid_written < valid_bytes and rng.random() < 0.002:
                    vout.write(line)
                    valid_written += len(line)
                    n_valid += 1
                else:
                    fout.write(line)
                    final_written += len(line)
                    n_final += 1
            ch.unlink()
    try:
        chunk_dir.rmdir()
    except OSError:
        pass
    return {'final_bytes': final_written, 'valid_bytes': valid_written, 'n_final': n_final, 'n_valid': n_valid, 'total_lines_in': total_lines}

def main():
    BUCKETS.mkdir(parents=True, exist_ok=True)
    stats = {}
    t0 = time.time()

    wiki_out = BUCKETS / 'wiki.jsonl'
    print('=== wiki ===', flush=True)
    stats['wiki'] = convert_wiki(wiki_out, BUDGET['wiki'])
    print(stats['wiki'], flush=True)

    web_parts = []
    fw = RAW / 'fineweb2-zh' / 'MAINLAND_CHINA.jsonl.zst'
    web_out = BUCKETS / 'web.jsonl'
    # remove old
    if web_out.exists():
        web_out.unlink()
    remain = BUDGET['web']
    if fw.exists():
        print('=== fineweb ===', flush=True)
        part = BUCKETS / 'web_fineweb.jsonl'
        st = convert_fineweb_zst(fw, part, remain)
        stats['web_fineweb'] = st
        print(st, flush=True)
        remain = max(0, remain - st['bytes'])
        web_parts.append(part)
    cnew = RAW / 'firefly' / 'CNewSum_v2.jsonl'
    if cnew.exists() and remain > 0:
        print('=== cnewsum ===', flush=True)
        part = BUCKETS / 'web_cnewsum.jsonl'
        st = convert_firefly_jsonl(cnew, part, remain, field_candidates=('text','summary','article','content'), seed=46)
        stats['web_cnewsum'] = st
        print(st, flush=True)
        remain = max(0, remain - st['bytes'])
        web_parts.append(part)
    # concat web parts
    with web_out.open('wb') as out:
        for p in web_parts:
            with p.open('rb') as fin:
                while True:
                    buf = fin.read(8*1024*1024)
                    if not buf:
                        break
                    out.write(buf)
            p.unlink(missing_ok=True)
    stats['web'] = {'bytes': web_out.stat().st_size if web_out.exists() else 0}

    print('=== books ===', flush=True)
    books_out = BUCKETS / 'books.jsonl'
    if books_out.exists():
        books_out.unlink()
    books_remain = BUDGET['books']
    book_stats = []
    for name, seed in [('chinese-poem.jsonl', 50), ('classical_chinese.jsonl', 51), ('jinyong.jsonl', 52)]:
        src = RAW / 'firefly' / name
        if not src.exists():
            continue
        part = BUCKETS / f'books_{name}'
        # allocate remaining proportionally by file size
        alloc = min(books_remain, max(int(src.stat().st_size * 1.05), 1))
        st = convert_firefly_jsonl(src, part, alloc, seed=seed, filter_instr=False)
        book_stats.append((name, st))
        books_remain = max(0, books_remain - st['bytes'])
        with books_out.open('ab') as out, part.open('rb') as fin:
            out.write(fin.read())
        part.unlink(missing_ok=True)
        if books_remain <= 0:
            break
    stats['books'] = {'parts': book_stats, 'bytes': books_out.stat().st_size if books_out.exists() else 0}
    print(stats['books'], flush=True)

    print('=== code ===', flush=True)
    code_out = BUCKETS / 'code.jsonl'
    code_paths = [
        RAW / 'code' / 'stack_smol_python.parquet',
        RAW / 'code' / 'the_stack_smol_python.parquet',
    ]
    stats['code'] = convert_code_parquet(code_paths, code_out, BUDGET['code'])
    print(stats['code'], flush=True)

    print('=== other (csl) ===', flush=True)
    other_out = BUCKETS / 'other.jsonl'
    csl = RAW / 'firefly' / 'csl_camera_readly.jsonl'
    if csl.exists():
        stats['other'] = convert_firefly_jsonl(csl, other_out, BUDGET['other'], field_candidates=('text','abstract','title','content'), seed=60, filter_instr=False)
    else:
        stats['other'] = {'bytes': 0, 'rows_out': 0, 'samples': []}
        other_out.write_bytes(b'')
    print(stats['other'], flush=True)

    # If wiki+web under 80% or total under 5GB, top up wiki
    def sz(p):
        return p.stat().st_size if p.exists() else 0
    total_now = sum(sz(BUCKETS / n) for n in ('wiki.jsonl','web.jsonl','books.jsonl','code.jsonl','other.jsonl'))
    print('buckets_total_gb', round(total_now/1024**3, 3), flush=True)
    if total_now < int(5.0 * 1024**3):
        need = int(5.2 * 1024**3) - total_now
        print(f'top-up wiki by {need} bytes', flush=True)
        # append more wiki with different seed
        extra = BUCKETS / 'wiki_extra.jsonl'
        st = convert_wiki(extra, need, seed=99)
        with wiki_out.open('ab') as out, extra.open('rb') as fin:
            out.write(fin.read())
        extra.unlink(missing_ok=True)
        stats['wiki_topup'] = st
        print(st, flush=True)

    print('=== shuffle merge ===', flush=True)
    final = ROOT / 'pretrain_mix_5g.jsonl'
    valid = ROOT / 'valid_mix.jsonl'
    bucket_list = [BUCKETS / n for n in ('wiki.jsonl','web.jsonl','books.jsonl','code.jsonl','other.jsonl') if (BUCKETS / n).exists() and (BUCKETS / n).stat().st_size > 0]
    merge_stats = shuffle_merge(bucket_list, final, valid)
    stats['merge'] = merge_stats
    stats['elapsed_sec'] = round(time.time() - t0, 1)
    stats['final_gb'] = round(final.stat().st_size / 1024**3, 3)
    stats['valid_mb'] = round(valid.stat().st_size / 1024**2, 2)
    # per-bucket share
    shares = {n: round(sz(BUCKETS / f'{n}.jsonl') / max(total_now,1), 4) for n in ('wiki','web','books','code','other')}
    # recompute total after topup
    total_now = sum(sz(BUCKETS / n) for n in ('wiki.jsonl','web.jsonl','books.jsonl','code.jsonl','other.jsonl'))
    shares = {}
    for n in ('wiki','web','books','code','other'):
        shares[n] = round(sz(BUCKETS / f'{n}.jsonl') / max(total_now,1), 4)
    shares['wiki_plus_web'] = round(shares.get('wiki',0)+shares.get('web',0), 4)
    stats['bucket_shares'] = shares
    stats['bucket_bytes'] = {n: sz(BUCKETS / f'{n}.jsonl') for n in ('wiki','web','books','code','other')}
    (ROOT / 'pretrain_mix_5g_stats.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'final_gb': stats['final_gb'], 'valid_mb': stats['valid_mb'], 'shares': shares, 'elapsed': stats['elapsed_sec']}, ensure_ascii=False), flush=True)

if __name__ == '__main__':
    main()
