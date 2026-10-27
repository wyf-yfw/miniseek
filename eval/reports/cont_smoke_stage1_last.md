# Continuation smoke — stage1 MoE last.pt

- time: 2026-10-25 19:12 Asia/Shanghai
- settings: seed=42(+i), temp=0.8, top_k=50, max_new=48
- prompts: 12 Chinese continuation stems (eval/cont_smoke_stage1.py)
- MoE: checkpoints/stage1/last.pt step=79390 loss≈2.073
- Dense spot-check: checkpoints/stage0/last.pt step=158780 loss≈2.495

## Summary metrics

| ckpt | avg_cjk | avg_rep3 | eos | avg_new | high_rep |
|---|---:|---:|---:|---:|---|
| MoE last | 0.873 | 0.118 | 1/12 | 44.9 | 3,4,6 |
| Dense last | 0.883 | 0.091 | 2/12 | 46.0 | 1,7,8 |

## Qualitative (MoE)

- Fluency: 中等偏上；完整中文句多，偶发截断/乱码尾。
- Chinese ratio: 高（短 chat 才容易漂英文；续写 stem 基本稳住）。
- Repetition: 明显（「学习要靠」左右走循环；山/行星列表重复）。
- Coherence: 好坏参半——天气/春天/首都尚可；AI/机器学习易漂成问答或幻想。
- EOS: 几乎不收束（1/12），多数顶满 48 token。
- vs Dense: MoE 事实续写略稳（八大行星/首都）；Dense 也会自相矛盾。差距不足以证明同数据再刷一轮能质变。

## Sample pairs (MoE)

- **今天天气很好，我想** → 去公园散步。天气不错，因为你是要去公园散步。但是，如果你在考虑去公园散步，请避免在其他地方，例如公园、公园或公园里的任何地方。考虑到
- **北京是中国的** → 首都，也是一所拥有许多历史和文化底蕴的城市。北京是中国的首都，也是中国的首都。它有着悠久的历史，如外滩、外滩、城�
- **人工智能是** → 如何影响全球经济发展的？人工智能（AI）已经在全球范围内产生了广泛影响，在全球经济中扮演着至关重要的角色。以下是一些可能的影响：1. 自动化和智能化：人工智能
- **学习要靠** → 近你自己，你就要先从左走开始，到右走结束，就要先从左走开始，到右走结束，就要先从左走开始，到右走结束，就要先先
- **太阳系一共有** → 八大行星，分别为：水星、金星、火星、木星、土星、天王星和海王星。这八大行星中的最大行星是：水星、金星、火
- **中国的首都是** → 哪个城市？中国的首都是北京。

## Recommendation

**选 (c)：可以开第二轮 resume，但必须加/换更好中文数据；不要同数据降 LR 硬刷。**

理由：续写中文已基本可用、loss 平台期后同数据边际收益低；重复与话题漂移靠同语料再训难消。
若目标是「网页短问答别漂英文」→ 优先 SFT/对话数据（更接近 a），而不是再刷一轮 pretrain。

