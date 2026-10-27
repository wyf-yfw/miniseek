#!/usr/bin/env python3
import json, random
from pathlib import Path
import pyarrow.parquet as pq

ROOT = Path('/home/wyf/miniseek/data')
RAW = ROOT / 'raw'
BUCKETS = RAW / 'buckets'
BUCKETS.mkdir(parents=True, exist_ok=True)
EOS = '<|im_end|>'
budget = int(0.50 * 5.5 * 1024**3)  # ~2.75 GiB of final mix
# empirical: keep_frac 0.118 -> 497MB => full ~4.22GB
keep_frac = min(1.0, budget / (4.22 * 1024**3) * 1.05)
rng = random.Random(42)
out_path = BUCKETS / 'wiki.jsonl'
INSTR = ('请你','请帮','请根据','根据以下','你是一个','You are an','Human:','Assistant:','### Instruction')

def ok(text):
    head = text[:200]
    if sum(1 for m in INSTR if m in head) >= 1 and ('请' in head[:40]):
        return False
    return len(text) >= 40

written = n_in = n_out = 0
samples = []
files = sorted((RAW/'wikipedia-zh').glob('train-*.parquet'))
print('keep_frac', keep_frac, 'budget', budget, flush=True)
with out_path.open('wb') as out:
    for fp in files:
        if written >= budget:
            break
        pf = pq.ParquetFile(fp)
        print('file', fp.name, flush=True)
        for rg in range(pf.num_row_groups):
            if written >= budget:
                break
            tbl = pf.read_row_group(rg, columns=['text'])
            for text in tbl.column('text').to_pylist():
                n_in += 1
                if rng.random() > keep_frac:
                    continue
                text = str(text or '').replace('\x00','').strip()
                if not ok(text):
                    continue
                if not text.endswith(EOS):
                    text = text + '\n' + EOS
                line = (json.dumps({'text': text}, ensure_ascii=False) + '\n').encode()
                out.write(line)
                written += len(line)
                n_out += 1
                if len(samples) < 2:
                    samples.append(text[:160].replace('\n',' '))
                if written >= budget:
                    break
        print('progress bytes', written, 'rows', n_out, flush=True)
meta = {'bytes': written, 'rows_in': n_in, 'rows_out': n_out, 'samples': samples, 'keep_frac': keep_frac}
(BUCKETS/'wiki_stats.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
print('DONE', meta, flush=True)
