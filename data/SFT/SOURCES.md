# sft_miniseek_v5

- Seed: `20261028`
- Format: Conversations JSONL `{"conversations":[{"role","content"},...]}`
- Capability targets: dialogue43 / knowledge22 / reasoning15 / creative12 / code8
- Turns: single75 / multi25; Language: zh80 / en20
- Route A: deepctrl **full on-disk** (ModelScope zh+en) + Fineweb Pure QA (`sft/cleaned`) + Magpie-Qwen2-Pro-200K-Chinese + Congliu R1 **final only** (`content`, not `reasoning_content`)
- 10G is a superset of 5G (same seed family); v1 deleted; Hub cut over to v5 (train only)

## 10G pack (`sft_miniseek_v5_10g.jsonl`)

- rows: 5963117
- size_bytes: 10737420299 (~10.00 GiB)
- bucket: {"dialogue": 0.43, "knowledge": 0.22, "reasoning": 0.15, "creative": 0.12, "code": 0.08}
- lang: {"zh": 0.8, "en": 0.2}
- turn: {"single": 0.75, "multi": 0.25}
- sources: {"deepctrl-zh": 4095718, "deepctrl-en": 1192624, "fineweb-pure-qa": 455039, "congliu-r1-final": 77022, "magpie-cn": 142714}

## 5G pack (`sft_miniseek_v5_5g.jsonl`) — subset of 10G

- rows: 3112273
- size_bytes: 5368712857 (~5.00 GiB)
- bucket: {"dialogue": 0.43, "knowledge": 0.22, "reasoning": 0.15, "creative": 0.12, "code": 0.08}
- lang: {"zh": 0.8, "en": 0.2}
- turn: {"single": 0.75, "multi": 0.25}
- sources: {"deepctrl-zh": 2152851, "deepctrl-en": 622455, "fineweb-pure-qa": 219903, "magpie-cn": 76308, "congliu-r1-final": 40756}

## Notes

- free_gb_before: 68.61
- free_gb_after: 39.17
- deepctrl full zh+en retained under `data/_raw_sft_v5/deepctrl/`; pool built via large stratified reservoir from local files
- Fineweb Pure QA path: `opencsg/Fineweb-Edu-Chinese-V2.2` `sft/cleaned/*.jsonl`

## Cutover (2026-10-28)

- **Default train file**: `sft_miniseek_v5_10g.jsonl` (WebUI / `algo.train_sft` / API).
- `sft_miniseek_v5_5g.jsonl` kept for smaller runs.
- `sft_miniseek_v1.jsonl` removed.
- Raw build caches under `data/_raw_sft_v5/` removed after packs landed.
- Hub: train files only (10g + 5g + README/SOURCES); no separate valid split.

## short256 (2026-10-28)

- File: `sft_miniseek_v5_short256.jsonl` — filtered from 10G; prefer content-chars ≤256, soft max ≤280; drop longer.
- Details: `SOURCES_v5_short256.md`. WebUI default remains 10G.
- SFT overflow truncation: keep-head (prompt first).
