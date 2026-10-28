# sft_miniseek_v1

- Magpie-Align/Magpie-Qwen2-Pro-200K-Chinese: kept 198977/200000
- Congliu/Chinese-DeepSeek-R1-Distill-data-110k: selected 30000/110000 (final answer only, think stripped)
- Filters: drop empty / identity-ish keywords / overlong assistant
- MiniSeek identity/persona (appended 2026-10-27, trainer=wyf-yfw): 149 lines — 名字 MiniSeek；训练者 wyf-yfw；超小中文对话；Dense≈28M / MoE≈150M；拒绝冒充 ChatGPT/Qwen 等
- Format: {"conversations":[{"role","content"},...]}
- Total lines: 229126
