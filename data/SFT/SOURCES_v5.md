# sft_miniseek_v5

- Seed: `20261028`
- Format: Conversations JSONL `{"conversations":[{"role","content"},...]}`
- Capability targets: dialogue43 / knowledge22 / reasoning15 / creative12 / code8
- Turns: single75 / multi25; Language: zh80 / en20
- Route A: deepctrl stream-sample (ModelScope) + Fineweb Pure QA (`sft/cleaned`) + Magpie-Qwen2-Pro-200K-Chinese + Congliu R1 **final only** (`content`, not `reasoning_content`)
- 10G is a superset of 5G (same seed family); v1 not deleted; no HF upload in this step

## 10G pack (`sft_miniseek_v5_10g.jsonl`)

- rows: 3594121
- size_bytes: 6954850641 (~6.48 GiB)
- bucket: {"dialogue": 0.4294, "knowledge": 0.27, "reasoning": 0.1448, "creative": 0.0948, "code": 0.0611}
- lang: {"zh": 0.8383, "en": 0.1617}
- turn: {"single": 0.76, "multi": 0.24}
- sources: {"deepctrl-zh": 2139976, "deepctrl-en": 581185, "congliu-r1-final": 109965, "magpie-cn": 199994, "fineweb-pure-qa": 563001}

## 5G pack (`sft_miniseek_v5_5g.jsonl`) — subset of 10G

- rows: 2778909
- size_bytes: 5368709349 (~5.00 GiB)
- bucket: {"dialogue": 0.4307, "knowledge": 0.2207, "reasoning": 0.1507, "creative": 0.1189, "code": 0.079}
- lang: {"zh": 0.8012, "en": 0.1988}
- turn: {"single": 0.7485, "multi": 0.2515}
- sources: {"deepctrl-zh": 1682437, "deepctrl-en": 552366, "congliu-r1-final": 91075, "fineweb-pure-qa": 307587, "magpie-cn": 145444}

## Notes

- free_gb_before: 99.03
- free_gb_after: 79.58
- deepctrl full 27G not retained (stream + reservoir)
- Fineweb Pure QA path: `opencsg/Fineweb-Edu-Chinese-V2.2` `sft/cleaned/*.jsonl`
