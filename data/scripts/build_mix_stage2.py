#!/usr/bin/env python3
"""Convert remaining HF buckets and shuffle-merge to 5-6GB mix. Wiki bucket expected ready."""
from __future__ import annotations
import io, json, random, time
from pathlib import Path
import pyarrow.parquet as pq
import zstandard as zstd

ROOT = Path('/home/wyf/miniseek/data')
RAW = ROOT / 'raw'
BUCKETS = RAW / 'buckets'
EOS = '<|im_end|>'
TARGET = int(5.5 * 1024**3)
BUDGET = {
    'wiki': int(0.50 * TARGET),
    'web': int(0.30 * TARGET),
    'books': int(0.10 * TARGET),
    'code': int(0.06 * TARGET),
    'other': int(0.04 * TARGET),
}
INSTR = ('请你','请帮','请根据','根据以下','你是一个','You are an','Human:','Assistant:','### Instruction','<|im_start|>user')

def ensure_eos(text: str) -> str:
    text = (text or '').replace('\x00','').strip()
    if not text:
        return ''
    if not text.endswith(EOS):
        text = text + '\n' + EOS
    return text

def looks_instr(text: str) -> bool:
    head = text[:200]
    return sum(1 for m in INSTR if m in head) >= 1 and ('请' in head[:50] or '?' in head or '？' in head)

def wline(f, text: str) -> int:
    text = ensure_eos(text)
    if len(text) < 40:
        return 0
    b = (json.dumps({'text': text}, ensure_ascii=False) + '\n').encode()
    f.write(b)
    return len(b)

def convert_fineweb(path: Path, out: Path, budget: int, seed=43):
    rng = random.Random(seed)
    # decompress ~3x; want budget from ~1.18G zst
    keep = 0.45
    written = n_in = n_out = 0
    samples = []
    dctx = zstd.ZstdDecompressor()
    with path.open('rb') as fin, dctx.stream_reader(fin) as reader, out.open('wb') as fout:
        ts = io.TextIOWrapper(reader, encoding='utf-8', errors='replace')
        for line in ts:
            if written >= budget:
                break
            n_in += 1
            if rng.random() > keep:
                continue
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            text = str(obj.get('text') or obj.get('content') or '')
            if len(text) < 80 or looks_instr(text):
                continue
            n = wline(fout, text)
            if not n:
                continue
            written += n; n_out += 1
            if len(samples) < 2:
                samples.append(text[:160].replace('\n',' '))
    return {'bytes': written, 'rows_in': n_in, 'rows_out': n_out, 'samples': samples}

def convert_jsonl(path: Path, out: Path, budget: int, fields=('text','content','summary','article','abstract'), seed=44, filter_instr=True):
    rng = random.Random(seed)
    written = n_in = n_out = 0
    samples = []
    with path.open('r', encoding='utf-8', errors='replace') as fin, out.open('wb') as fout:
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
                for k in fields:
                    if obj.get(k):
                        text = str(obj[k]); break
                if not text:
                    for v in obj.values():
                        if isinstance(v, str) and len(v) > 40:
                            text = v; break
            elif isinstance(obj, str):
                text = obj
            if filter_instr and looks_instr(text):
                continue
            if len(text) < 40:
                continue
            n = wline(fout, text)
            if not n:
                continue
            written += n; n_out += 1
            if len(samples) < 2:
                samples.append(text[:160].replace('\n',' '))
    return {'bytes': written, 'rows_in': n_in, 'rows_out': n_out, 'samples': samples}

def convert_code(paths, out: Path, budget: int):
    written = n_in = n_out = 0
    samples = []
    with out.open('wb') as fout:
        for path in paths:
            p = Path(path)
            if not p.exists():
                continue
            pf = pq.ParquetFile(p)
            cols = [c for c in ('content','text','code') if c in pf.schema_arrow.names] or [pf.schema_arrow.names[0]]
            for rg in range(pf.num_row_groups):
                if written >= budget:
                    break
                tbl = pf.read_row_group(rg, columns=cols)
                for text in tbl.column(cols[0]).to_pylist():
                    n_in += 1
                    text = str(text or '')
                    if len(text) < 40 or len(text) > 200000:
                        continue
                    n = wline(fout, text)
                    if not n:
                        continue
                    written += n; n_out += 1
                    if len(samples) < 2:
                        samples.append(text[:160].replace('\n',' '))
                    if written >= budget:
                        break
    return {'bytes': written, 'rows_in': n_in, 'rows_out': n_out, 'samples': samples}

def shuffle_merge(bucket_paths, final_path: Path, valid_path: Path, valid_budget=8*1024*1024, seed=7):
    rng = random.Random(seed)
    chunk_dir = BUCKETS / '_shuffle_chunks'
    chunk_dir.mkdir(parents=True, exist_ok=True)
    for old in chunk_dir.glob('*.jsonl'):
        old.unlink()
    chunk_size = 200 * 1024 * 1024
    chunk_idx = 0
    cur = None
    cur_bytes = 0
    total_lines = 0
    def new_chunk():
        nonlocal chunk_idx, cur, cur_bytes
        if cur:
            cur.close()
        p = chunk_dir / f'chunk_{chunk_idx:04d}.jsonl'
        chunk_idx += 1
        cur = p.open('wb'); cur_bytes = 0
        return p
    new_chunk()
    for bp in bucket_paths:
        with open(bp, 'rb') as fin:
            for line in fin:
                if cur_bytes >= chunk_size:
                    new_chunk()
                cur.write(line); cur_bytes += len(line); total_lines += 1
    if cur:
        cur.close()
    chunks = sorted(chunk_dir.glob('chunk_*.jsonl'))
    rng.shuffle(chunks)
    valid_w = final_w = n_final = n_valid = 0
    with final_path.open('wb') as fout, valid_path.open('wb') as vout:
        for ch in chunks:
            lines = ch.read_bytes().splitlines(keepends=True)
            rng.shuffle(lines)
            for line in lines:
                if valid_w < valid_budget and rng.random() < 0.002:
                    vout.write(line); valid_w += len(line); n_valid += 1
                else:
                    fout.write(line); final_w += len(line); n_final += 1
            ch.unlink()
    try:
        chunk_dir.rmdir()
    except OSError:
        pass
    return {'final_bytes': final_w, 'valid_bytes': valid_w, 'n_final': n_final, 'n_valid': n_valid, 'total_lines_in': total_lines}

def sz(p: Path) -> int:
    return p.stat().st_size if p.exists() else 0

def main():
    t0 = time.time()
    BUCKETS.mkdir(parents=True, exist_ok=True)
    stats = {}
    wiki = BUCKETS / 'wiki.jsonl'
    assert wiki.exists() and wiki.stat().st_size > 100_000_000, f'wiki missing/small: {sz(wiki)}'
    stats['wiki'] = json.loads((BUCKETS/'wiki_stats.json').read_text()) if (BUCKETS/'wiki_stats.json').exists() else {'bytes': sz(wiki)}
    print('wiki_gb', round(sz(wiki)/1024**3,3), flush=True)

    # WEB
    web_budget = BUDGET['web']
    web_out = BUCKETS / 'web.jsonl'
    if web_out.exists():
        web_out.unlink()
    remain = web_budget
    fw = RAW/'fineweb2-zh'/'MAINLAND_CHINA.jsonl.zst'
    if fw.exists() and remain > 0:
        part = BUCKETS/'web_fineweb.jsonl'
        print('=== fineweb ===', flush=True)
        st = convert_fineweb(fw, part, remain)
        stats['web_fineweb'] = st; print(st, flush=True)
        remain = max(0, remain - st['bytes'])
        with web_out.open('ab') as o, part.open('rb') as i:
            o.write(i.read())
        part.unlink(missing_ok=True)
    cnew = RAW/'firefly'/'CNewSum_v2.jsonl'
    if cnew.exists() and remain > 0:
        part = BUCKETS/'web_cnewsum.jsonl'
        print('=== cnewsum ===', flush=True)
        st = convert_jsonl(cnew, part, remain, fields=('text','summary','article','content'), seed=46)
        stats['web_cnewsum'] = st; print(st, flush=True)
        remain = max(0, remain - st['bytes'])
        with web_out.open('ab') as o, part.open('rb') as i:
            o.write(i.read())
        part.unlink(missing_ok=True)
    stats['web_bytes'] = sz(web_out)
    print('web_gb', round(sz(web_out)/1024**3,3), flush=True)

    # BOOKS
    books_out = BUCKETS/'books.jsonl'
    if books_out.exists(): books_out.unlink()
    books_remain = BUDGET['books']
    book_parts = []
    for name, seed in [('chinese-poem.jsonl',50),('classical_chinese.jsonl',51),('jinyong.jsonl',52)]:
        src = RAW/'firefly'/name
        if not src.exists() or books_remain <= 0:
            continue
        part = BUCKETS/f'books_{name}'
        alloc = min(books_remain, int(src.stat().st_size * 1.1) + 10_000_000)
        print('=== book', name, '===', flush=True)
        st = convert_jsonl(src, part, alloc, seed=seed, filter_instr=False)
        book_parts.append((name, st)); print(st, flush=True)
        books_remain = max(0, books_remain - st['bytes'])
        with books_out.open('ab') as o, part.open('rb') as i:
            o.write(i.read())
        part.unlink(missing_ok=True)
    stats['books'] = {'parts': book_parts, 'bytes': sz(books_out)}

    # CODE
    print('=== code ===', flush=True)
    code_out = BUCKETS/'code.jsonl'
    stats['code'] = convert_code([RAW/'code'/'stack_smol_python.parquet'], code_out, BUDGET['code'])
    print(stats['code'], flush=True)

    # OTHER
    print('=== other csl ===', flush=True)
    other_out = BUCKETS/'other.jsonl'
    csl = RAW/'firefly'/'csl_camera_readly.jsonl'
    if csl.exists():
        stats['other'] = convert_jsonl(csl, other_out, BUDGET['other'], fields=('text','abstract','title','content'), seed=60, filter_instr=False)
    else:
        other_out.write_bytes(b''); stats['other'] = {'bytes':0}
    print(stats['other'], flush=True)

    # Top-up if under 5.0GB: prefer more wiki then more web
    total = sum(sz(BUCKETS/n) for n in ('wiki.jsonl','web.jsonl','books.jsonl','code.jsonl','other.jsonl'))
    print('pre_merge_gb', round(total/1024**3,3), flush=True)
    if total < int(5.0 * 1024**3):
        need = int(5.2 * 1024**3) - total
        # If wiki under budget, we already used keep_frac; append more from remaining wiki with new seed via higher keep - skip complex; use leftover CNewSum/fineweb
        print('need_topup', need, flush=True)
        if fw.exists() and need > 0:
            part = BUCKETS/'web_topup.jsonl'
            st = convert_fineweb(fw, part, need, seed=77)
            stats['web_topup'] = st
            with web_out.open('ab') as o, part.open('rb') as i:
                o.write(i.read())
            part.unlink(missing_ok=True)
            print(st, flush=True)

    total = sum(sz(BUCKETS/n) for n in ('wiki.jsonl','web.jsonl','books.jsonl','code.jsonl','other.jsonl'))
    shares = {n: round(sz(BUCKETS/f'{n}.jsonl')/max(total,1),4) for n in ('wiki','web','books','code','other')}
    shares['wiki_plus_web'] = round(shares['wiki']+shares['web'],4)
    print('shares', shares, 'total_gb', round(total/1024**3,3), flush=True)

    print('=== shuffle ===', flush=True)
    final = ROOT/'pretrain_mix_5g.jsonl'
    valid = ROOT/'valid_mix.jsonl'
    blist = [BUCKETS/n for n in ('wiki.jsonl','web.jsonl','books.jsonl','code.jsonl','other.jsonl') if sz(BUCKETS/n)>0]
    merge = shuffle_merge(blist, final, valid)
    stats['merge'] = merge
    stats['bucket_shares'] = shares
    stats['bucket_bytes'] = {n: sz(BUCKETS/f'{n}.jsonl') for n in ('wiki','web','books','code','other')}
    stats['final_gb'] = round(sz(final)/1024**3,3)
    stats['valid_mb'] = round(sz(valid)/1024**2,2)
    stats['elapsed_sec'] = round(time.time()-t0,1)
    (ROOT/'pretrain_mix_5g_stats.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'final_gb':stats['final_gb'],'valid_mb':stats['valid_mb'],'shares':shares,'elapsed':stats['elapsed_sec']}, ensure_ascii=False), flush=True)

if __name__ == '__main__':
    main()
