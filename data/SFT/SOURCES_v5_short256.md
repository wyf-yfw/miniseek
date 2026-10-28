# sft_miniseek_v5_short256

- Built: from `sft_miniseek_v5_10g.jsonl`
- Length rule: **prefer** total conversation content chars ≤ 256; **soft max** ≤ 280 (≈10% over preferred). Longer rows **dropped** (no truncate).
- Capability rebalance: **not applied** — 10G rows have no per-sample bucket tags; distribution is the natural short-sample mix
- Format: Conversations JSONL `{"conversations":[{"role","content"},...]}`
- Relation: independent filter of 10G; does **not** modify 5G/10G; not a 5G subset
- Training tip: use `max_seq_len=256` (or ≤512). SFT tokenization truncates overflow from the **end** (keep prompt/question head).

## Pack (`sft_miniseek_v5_short256.jsonl`)

- rows_kept: 2018458
- rows_preferred_le_256: 1882126 (93.25% of kept)
- rows_soft_only_257_280: 136332 (6.75% of kept)
- rows_source_10g: 5963117
- rows_bad_json_or_empty: 0
- keep_rate: 0.3385 (33.85% of 10G)
- size_bytes: 900273266 (~0.838 GiB)
- source_size_bytes: 10737420299 (~10.000 GiB)
- mean_content_chars_kept: 136.6

### Length buckets (kept only)

- {'129-192': 470279, '193-256': 394209, '257-280': 136332, '65-128': 604365, '<=64': 413273}

### Language (heuristic CJK)

- zh: 1891242 (93.70%)
- en: 127216 (6.30%)

### Turns

- single: 1759178 (87.15%)
- multi: 259280 (12.85%)

## Notes

- v5 capability targets are **not** re-enforced here.
- WebUI / train defaults still point at 10G unless you select this file.
- 5G and 10G packs were not modified.
