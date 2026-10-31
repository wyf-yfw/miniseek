# MiniSeek-M0.1：完整复现 DeepSeek-V4.1-Flash 架构（3090 可训）

> **命名更新（2026-10-31）**：产品线已改为 **MiniSeek-V / MiniSeek-V1**；包路径 `algo/model/miniseek_v/`，CLI `--arch miniseek-v`。下文仍保留当时的 MiniSeek-M / miniseek-m 史料写法；MiniSeek-M0.1 现指简单 MoE 对照线（代码不在 main）。

- 日期：2026-10-29
- 状态：第 1–8 部分已逐段经用户确认；第 9–10 部分及所有未决细节按用户指示「全部按推荐来」定稿
- 仓库：`wyf-yfw/miniseek`

## 0. 目标与约束

| 项 | 决定 |
|----|------|
| 目标 | 层内算子与整体框架完整对齐 DeepSeek-V4.1-Flash（arXiv:2609.19969）；宽度、层数、专家数、表大小可缩小 |
| 范围（做） | 文本主干（CED + SWA/CSA2 三模式 + 分层稀疏索引 + Single-Pass mHC + MoE + Engram）、训练配方、推理侧（KV cache、CED 预填充、Bounded Replay、前缀缓存）、DSpark |
| 范围（不做，后续阶段） | 多模态（DeepSeek-ViT + Aligner + 图文分开的路由 bias） |
| 硬件 | 单卡 RTX 3090（Ampere，24GB）。全部 BF16；KV / indexer 的 FP4、FP8 与 QAT **不做**（用户确认保留此差异） |
| 参数预算 | 主干 + Engram 表合计 ≤ 200M；DSpark 草稿头另算 |
| 预训练长度 | 1024 |
| 兼容 | 不保留旧版兼容：M0.1 `MoELM` 删除，旧 MoE checkpoint 不再可加载 |
| 产品名 | arch id `v41`，`MiniSeek-M0.1`（F = Flash）；Dense / Tiny 两档保持不动作为对照 |

## 1. 依据与优先级

1. 官方参考实现（MIT）：HF `deepseek-ai/DeepSeek-V4.1-Flash` 的 `inference/model.py`、`engram.py`、`kernel.py`、`config.json`。**逐算子以此为准。**
2. M 论文 arXiv:2609.19969（§2 架构、§2.5 优化、§3.2.2 Bounded Replay、§4.2 配置）。
3. M 声明沿用或引用的原文：V4 arXiv:2606.19348（CSA、mHC 参数化、Muon 算法）、V3.2 arXiv:2512.02556（DSA indexer 的 KL 训练）、mHC arXiv:2512.24880（α 初值 0.01）、Engram arXiv:2601.07372、DSpark arXiv:2607.05147、DeepSeek-V3（序列均衡损失公式）。
4. 以上都没写的，由本设计决定，统一标「推断」，汇总在 §13。

## 2. 与论文的已知差异（有意保留）

| 论文 | 本项目 | 原因 |
|------|--------|------|
| 主 KV FP4（E2M1 + E4M3 scale/16）、滑窗 KV FP8、indexer Q/K FP4，后训练 QAT | 全部 BF16，无 QAT | 3090 无 FP8/FP4；用户决定 |
| FP8 权重 / FP4 专家权重 | BF16 / fp32 主权重 | 同上 |
| 多模态、图文分开的路由 bias | 不做 | 范围外 |
| 64K 起步、1M 外推（YaRN） | 1024；YaRN 实现但默认关 | 3090 与数据规模 |
| TileLang 融合 kernel（sparse_attn、Mega-mHC 等） | PyTorch 实现同一数学 | 只对齐算子语义，不对齐 kernel |
| 尺寸（见 §3.2） | 缩小 | 预算 ≤200M |

## 3. 总体框架

### 3.1 代码组织

新包 `algo/model/miniseek_v/`，按官方 `model.py` 的模块边界拆分：

| 文件 | 内容 |
|------|------|
| `config.py` | `FlashConfig`，字段名沿用官方 `ModelArgs` |
| `rope.py` | `precompute_freqs_cis`（含 YaRN）、`apply_rotary_emb`（相邻两维成对，支持反向） |
| `attention.py` | `Attention`（SWA + CSA2）、`Compressor`、`Indexer`、`select_candidate_blocks`、核心稀疏注意力、跨层共享状态 |
| `mhc.py` | `hc_mixes`、`hc_split_sinkhorn`、`hc_pre`、`hc_post` |
| `moe.py` | `Gate`、`Expert`、`MoE`、序列均衡损失、bias 更新 |
| `engram.py` | 压缩 token 表、`EngramLayout`、`NgramHashState`、`Engram` |
| `model.py` | `Block`、`Transformer`（CED 接线、损失） |
| `cache.py` | 推理缓存、CED 预填充、Bounded Replay、前缀缓存 |
| `dspark.py` | DSpark 草稿头、投机解码 |

优化器放 `algo/optim/`（`muon.py`、`sinkhorn.py`、`param_groups.py`）。`build_model` 按 `arch` 派发。

### 3.2 配置取值（`FlashConfig` 默认值 = MiniSeek-M0.1）

| 字段 | M0.1 | 官方 DeepSeek-V4.1-Flash |
|------|------|-----------------|
| `vocab_size` | 6400（MiniMind tokenizer） | 129280 |
| `dim` / `n_layers` | 512 / 8（编码 0–3，解码 4–7） | 5120 / 40 |
| `n_heads` × `head_dim` | 16 × 128（K=V 单头共享） | 64 × 512 |
| `rope_head_dim` | 32 | 64 |
| `q_lora_rank` | 128 | 1280 |
| `o_groups` × `o_lora_rank` | 2 × 256 | 8 × 1024 |
| `window_size` | 128 | 128 |
| `compress_ratios` | `[0,0,2,2,1,1,1,1]` | `[0,0,2×18,1×20]` |
| `kv_source_layers` | `[2,4]` | `[2,8,14,20]` |
| `index_source_layers` | `[2,4,6]` | `[2,8,14,20,24,28,32,36]` |
| `index_n_heads` × `index_head_dim` | 8 × 64 | 32 × 128 |
| `index_topk` | 64 | 512 |
| `candidate_source_layer` / `candidate_topk_blocks` × `candidate_block_size` | 4 / 32 × 8（仅后训练启用） | 20 / 2048 × 8 |
| MoE | 1 共享 + 64 路由，top-6，`moe_inter_dim=176` | 1 + 384，top-6，2304 |
| `score_func` / `route_scale` / `swiglu_limit` | `sqrtsoftplus` / 1.5 / 10 | 同 |
| `hc_mult` / `hc_sinkhorn_iters` / `hc_eps` | 4 / 20 / 1e-6 | 同 |
| `norm_eps` | 1e-20 | 同 |
| `rope_theta` / `compress_rope_theta` | 1e4 / 1.6e5 | 同 |
| YaRN（`original_seq_len`, `rope_factor`, `beta_fast`, `beta_slow`） | 0（关）, 16, 32, 1 | 65536, 16, 32, 1 |
| Engram | 层 `[1,3]`，`max_ngram_size=4`，`n_heads=8`，`head_dim=32`，每桶质数 ≈16K | 层 `[1,14]`，4，8，256，≈16M |
| `tie_word_embeddings` | False | False |
| `max_seq_len` | 1024 | — |

参数量（设计公式）：embedding + head 6.6M，8 层 151.5M，Engram 约 29M（表约 25M），合计约 187M。实现后以 `numel()` 为准，测试断言 ≤200M。

### 3.3 层表

| 层 | 半区 | 注意力 | 共享 | Engram |
|----|------|--------|------|--------|
| 0 | 编码 | 只有滑窗（compress_ratio 0） | — | |
| 1 | 编码 | 只有滑窗 | — | ✓ |
| 2 | 编码 | CSA2(m=2, Full) | 产出主 KV、indexer K、top-k | |
| 3 | 编码 | CSA2(m=2, Reuse) | 复用第 2 层主 KV + top-k | ✓ |
| 4 | 解码 | CSA2(m=1, Full) | 输入 = 编码器输出（CED）；产出解码器主 KV、indexer K、top-k；后训练时建候选池 | |
| 5 | 解码 | CSA2(m=1, Reuse) | 复用第 4 层 | |
| 6 | 解码 | CSA2(m=1, Reindex) | 复用第 4 层主 KV + indexer K，自算 top-k | |
| 7 | 解码 | CSA2(m=1, Reuse) | 复用第 4 层主 KV + 第 6 层 top-k | |

### 3.4 前向数据流（照官方 `Transformer.forward`）

1. `h = embed(ids)`，复制成 `[B,T,4,d]`；首个子层的 `pre_mix` 为 one-hot `[1,0,0,0]`。
2. 逐层：若该层有 Engram，先 `h = engram(h, hash_ids)`；再 `h, pre_mix = block(h, pre_mix, ...)`。
3. 末层后 `h = hc_pre(h, pre_mix)`，`final_norm`，`head`（fp32 logits）。
4. 跨层共享（主 KV、indexer K、top-k、候选池）放在每次 forward 新建的共享状态对象里，来源层写、后续层读（对应官方 `SharedAttentionRuntime`）。
5. 训练返回 `(logits, ce, extras)`，`extras = {"balance": ..., "indexer_kl": ...}`。

### 3.5 精度

BF16 autocast。以下保持 fp32（与官方一致）：路由打分、mHC 系数与 Sinkhorn、m=2 压缩器的 `wkv`/`wgate` 与 softmax 池化、RMSNorm 内部、Engram 门控、MoE 输出累加、head/logits。

## 4. 注意力（SWA + CSA2）

### 4.1 每层都有（官方 `Attention`）

- Q：`qr = q_norm(wq_a(x))`（512→128，RMSNorm）；`q = wq_b(qr).view(B,T,16,128)`；对最后 32 维做 RoPE（相邻两维成复数）。
- 滑窗 KV：`kv = kv_norm(wkv(x))`（512→128），最后 32 维做 RoPE；该向量同时作 K 与 V，16 头共享（MQA）。位置 t 看 `[max(0,t−127), t]`。
- RoPE 频率：compress_ratio 为 0 的层用 `rope_theta=1e4` 且不开 YaRN；其余层（含其滑窗分支）用 `compress_rope_theta=1.6e5`（+YaRN 若开启）。
- 核心注意力（`sparse_attn` 语义）：每个 query 的条目集合 = 滑窗条目 ∪ 选中的压缩条目；`logit = q_h·kv × head_dim^-0.5`；softmax 分母额外加 `exp(attn_sink_h)`（sink 的 value 为 0）；条目索引 −1 表示空；没有任何有效条目的行输出 0。
- 输出：对 `o` 的最后 32 维按 query 位置做反向 RoPE；按 8 头一组分 2 组，`wo_a` 为块对角（每组 1024→256，`einsum("bsgd,grd->bsgr")`），再 `wo_b`（512→512）。

### 4.2 压缩器（仅 `kv_source_layers`）

- m=1：`latent = norm(wkv_c(x))`。
- m=2：`kv = wkv_c(x)`、`score = wgate(x)`（fp32），按位置不重叠两两成组，组内按通道 `softmax(score)` 加权求和，再 `norm`。无重叠、无位置偏置（CSA2 相对 CSA 的简化）。
- 只输出完整的组；第 j 个条目代表位置 `j·m`，RoPE 用该位置；query t 可见条目数为 `(t+1)//m`。
- indexer K 由旋转前的 latent 投影：`k = k_norm(wk(latent))`（128→64），再对最后 32 维按 `j·m` 做 RoPE。

### 4.3 三种模式

| 模式 | 层 | 行为 |
|------|----|------|
| Full | 2、4 | 本层压缩器产出主 KV 与 indexer K；本层 indexer 产出 top-k；三者发布到共享状态 |
| Reindex | 6 | 读共享的主 KV 与 indexer K；本层 indexer（只有 `wq_b`、`weights_proj`，无 `wk`）重新打分，发布新 top-k |
| Reuse | 3、5、7 | 读共享主 KV 与最新 top-k，不跑 indexer |

每层都保留自己的 `wq_a/q_norm/wq_b`、`wkv/kv_norm`、`attn_sink`、`wo_a/wo_b`。Reuse/Reindex 层没有压缩器参数。

### 4.4 Indexer（官方 `Indexer`）

- `q_I = wq_b_I(qr).view(B,T,8,64)`，最后 32 维按 t 做 RoPE。
- `w = weights_proj(x) × 64^-0.5 × 8^-0.5`。
- `score[t,s] = Σ_h w[t,h] · ReLU(q_I[t,h]·k[s])`；不可见条目 −inf。
- 候选池（仅后训练）：第 4 层把分数补齐到 8 的倍数、按块取最大值，钉住含最新可见条目的块，取前 32 块，展开成位置掩码；第 6 层把候选外的分数置 −inf。
- `topk = min(64, 可见上限)`；`topk(...).indices.sort()`；不可见的置 −1。严格照官方语义（包括 −inf 条目在可见范围内仍会被保留这一行为）。

### 4.5 Indexer 训练（沿用 V3.2 DSA，V4/M 继承）

- 目标：本层主注意力在其选中压缩条目上的概率，16 头相加后在选中集合内 L1 归一化，`detach`。
- 预测：indexer 分数在选中集合上的 softmax。
- 损失：`KL(target ‖ pred)`，对有选中条目的 token 求平均；三个来源层（2、4、6）再取平均（推断：Reuse 层使用的 top-k 也算在来源层名下，目标只用来源层自己的注意力）。
- indexer 的输入 `x`、`qr`、压缩 latent 全部 `detach`：indexer 只吃 KL 梯度，主模型只吃 LM 与均衡损失。
- 从头就稀疏训练，无稠密 warmup（M §1）。

### 4.6 训练实现

按官方 `sparse_attn` 的「按索引 gather + 带 sink 的 softmax」实现：每个 query 只计算 128 + 64 个条目；索引张量把滑窗位置与压缩条目位置（加偏移）拼在一起，−1 表示空。同时得到 KL 所需的注意力概率。显存约每层 600MB（batch 8 × 1024）。FlashAttention 不适用于本架构（`--flash-attn` 对 v41 无效，日志打印 n/a）。后续若换 FlexAttention 提速，必须与此实现逐元素对拍。

## 5. CED（因果编码器-解码器）

- 编码器 0–3 层，解码器 4–7 层。解码器只有第 4 层有压缩器，其输入是编码器输出（经第 3 层 MoE 子层给出的系数合成、第 4 层 `attn_norm`），即论文式 (1) 的 `H_{L/2}`；5–7 层复用。每个解码器层的滑窗 KV 仍由本层隐状态生成。
- 预训练：8 层完整 forward，所有位置都算损失。
- 推理预填充语义（实现见 §10）：编码器跑整段 prompt；用第 4 层 `attn_norm` + 压缩器算出所有位置的解码器全局 KV 与 indexer K；解码器 4–7 层只跑 prompt 最后 128 个 token（Decoder SWA Bounded Replay），重放起点 `s = P−128`，重放段内位置 i 的滑窗只看 `[max(s, i−127), i]`。
- SFT 中模拟（默认开，预训练关）：对每个回答起点 P（labels 由 −100 变为有效值的位置），解码器层中位置落在 `[P−128, P)` 的 query 看不到 `< P−128` 的滑窗 key。多轮样本每个回答起点各截断一次（推断：截断区若含上一轮回答尾部，其损失在截断上下文下计算，属近似）。
- Encoder SWA Bounded Replay：只用于前缀缓存命中、编码器滑窗缓存缺失时，见 §10。

## 6. Single-Pass mHC

- 残差流 `[B,T,4,512]`，BF16 存储，系数 fp32 计算。
- 每个子层（注意力、MoE 各一套参数 `fn[24, 2048]`、`base[24]`、`scale[3]`）：
  - `mixes = (flatten(X) @ fnᵀ) × rsqrt(mean(flatten(X)²) + 1e-20)`（RMSNorm 权重折进 `fn`）
  - `pre = σ(mixes[0:4]·scale₀ + base[0:4]) + 1e-6`
  - `post = 2σ(mixes[4:8]·scale₁ + base[4:8])`
  - `comb = mixes[8:24].view(4,4)·scale₂ + base[8:24]`；行 softmax + 1e-6；列归一化（分母 +1e-6）；再 19 轮「行归一化、列归一化」（分母 +1e-6）
- 单遍化：子层输入 `u = Σ_k pre_prev[k]·X[k]`（`pre_prev` 来自上一个子层，首个子层 one-hot）；输出 `X_new[k] = post[k]·y + Σ_j comb[j,k]·X[j]`；本子层的 `pre` 交给下一个子层；末层后用最后的 `pre` 合成。
- 初始化：`scale` = 0.01（mHC 论文）；`base` = 0（推断：开局等价 Pre-Norm 残差）；`fn` ~ N(0, 0.02)（推断）。
- 不写融合 kernel；`torch.compile` 负责融合 Sinkhorn 小算子。

## 7. MoE

- `Gate`：`s = sqrt(softplus(x·Wᵀ))`（fp32）；选择 `topk(s + bias, 6)`；权重 = 选中的 `s` / (Σ + 1e-20) × 1.5。
- `bias`：fp32，每专家一个，不参与梯度；每个优化器 step 后按整个 step（累计所有 micro-batch）的负载：高于均值 −0.001，低于均值 +0.001；存入 checkpoint。
- `Expert`：`w1`（gate）、`w3`（up）、`w2`（down），无 bias；`gate.clamp(max=10)`、`up.clamp(-10,10)`；`silu(gate)·up` 用 fp32；路由权重在 `w2` 之前相乘。
- 输出：路由专家加权和 + 共享专家（不加权），fp32 累加。
- 序列均衡损失（V3 公式）：`f_i = N/(K·T)·count_i`，`P_i = mean_t s′_{i,t}`，`s′ = s / Σ_j s_j`（用不带 bias 的 `s`）；每层 `Σ_i f_i P_i` 对 batch 取平均；跨层求和（推断）；权重 1e-4。
- 不做：V4 前几层 Hash 路由（M 已去掉）、图像专用 bias、V4 的 Anticipatory Routing（M 未提）。
- 分发沿用排序 + 每专家一次 GEMM + `index_add_`；保留 `begin_expert_usage` / `end_expert_usage` / `probe_expert_load` 探针接口。

## 8. Engram

- 压缩 token 表（官方 `build_compressed_token_map`）：逐 token 解码，`NFKC → NFD → StripAccents → Lowercase → 合并空白 → 单空格哨兵 → Strip`；规范化结果相同者共用 id；含 `\ufffd` 的半字节 token 用原始 token 字符串作 key。压缩词表大小写入配置并在加载时断言。
- 哈希（官方 `NgramHashState`）：每位置取当前及前 3 个压缩 id；回看遇序列开头或文档边界停止，空位填 tokenizer pad token 的压缩 id；每层乘子由 `np.random.default_rng(10007·layer_id)` 生成、奇数、受 int64 溢出上界约束；逐次异或得到 2/3/4-gram 哈希；每个（阶，头）用互不重复的质数取模并加偏移；每层 24 个桶区间拼成一张表。质数从 `engram_vocab_size − 1` 往上依次取（M0.1 取 16000），自写判质函数，不依赖 sympy。
- 写入（官方 `Engram.forward`，在 block 之前，作用于 4 条流）：查表 `[B,T,24,32]` → 展平 768 → `wkv` 投影出 4 个 key（每流一个）+ 1 个共享 value；`weight = q_weight ⊙ k_weight`；`dot_m = Σ(h_m ⊙ weight_m ⊙ key_m) · rsqrt(mean h_m²+eps) · rsqrt(mean key_m²+eps) / √d`；`gate_m = σ(sign(dot)·√max(|dot|, 1e-6))`；`X_m += gate_m·value`（fp32）。无短卷积（M 去掉）。
- 初始化（推断）：表与 `wkv` 均 N(0, 0.02)，`q_weight`、`k_weight` 为 1，不做任何置零。原版 Engram 只把短卷积置 0（卷积分支恒等，门控值从第一步就写入残差）；M 去掉了卷积，所以没有对应的置零项。
- 尺寸：两层合计约 29M（表约 25M）。

## 9. 训练配方

### 9.1 损失

`total = ce + 1e-4 × Σ_layers balance + indexer_kl`。indexer 与主模型梯度不相交，KL 直接相加。不做 MTP。日志三项分开打印。

### 9.2 优化器（M §2.5、§4.2.2；Muon 算法见 V4 Algorithm 1）

| 组 | 参数 | 规则 |
|----|------|------|
| Muon | 所有线性层权重矩阵：注意力（`wq_a`、`wq_b`、`wkv`、`wo_a`、`wo_b`）、压缩器、indexer（`wq_b`、`weights_proj`、`wk`）、路由器、专家、mHC `fn`、Engram `wkv` | `M = μM + G`；`O′ = HybridNS(μM + G)`；`O = O′·√max(n,m)·0.18`；`W ← W(1−ηλ) − ηO`；μ=0.95，λ=0.1；HybridNS：`M₀ = M/‖M‖_F`，10 轮 `aM + b(MMᵀ)M + c(MMᵀ)²M`，前 8 轮 (3.4445, −4.7750, 2.0315)，后 2 轮 (2, −1.5, 0.5) |
| AdamW | RMSNorm 权重、`attn_sink`、mHC `base`/`scale`、Engram `q_weight`/`k_weight` | β=(0.9, 0.95)，eps=1e-20；仅 norm 权重 wd=0.1，其余 wd=0 |
| Sinkhorn 动量 | embedding、head、Engram 表（DSpark 的 Markov 表同理） | M Algorithm 1：`M = βM + (1−β)G`，`Ĝ = βM + (1−β)G`；`ρ_i ≤ τρ̄` 的行置零（τ=1e-3）；奇数步行 ℓ2 归一化、偶数步列 ℓ2 归一化，共 K=11 次（分母 +1e-20）；`Δ = √n·U`；`W ← W − γηΔ`，γ=0.18，β=0.95，无 wd；Engram 表 lr ×5 |

Muon「逻辑独立矩阵」拆分：`wq_b` 按 16 头拆、indexer `wq_b` 按 8 头拆（head-wise Muon；K 为单头无需拆）；每个专家各自独立；`wo_a` 按组拆；Engram `wkv` 拆成 4 个 key + 1 个 value（推断）。同形状矩阵堆叠做批量 Newton–Schulz。学习率调度沿用仓库 warmup + cosine。

### 9.3 样本级 attention mask（打包训练）

- 构建 token 流时每篇文档（含其 EOS）补 0–1 个 pad 使长度为偶数；pad 的 label 为 −100，归属该文档。切块长度 1024 为偶数，因此所有文档段都从偶数位置开始：m=2 的组不跨文档，分组方式与推理（从 0 开始两两成组）一致。
- batch 携带逐 token 的 `doc_ids`：滑窗、压缩条目可见性、indexer、KL 目标都限制在同一文档；Engram 回看到文档边界停止。
- RoPE 使用切块内全局位置（Q·K 与反向旋转后的输出只依赖相对位置，结果等价于按文档重置位置）。

### 9.4 后训练（SFT）开关

候选池默认开；Decoder Bounded Replay 模拟默认开（回答起点由 labels 推出）；FP4/FP8 QAT 不做。

### 9.5 初始化

线性层 N(0, 0.02)，残差出口投影（`wo_b`、专家 `w2`）再乘 1/√(2L)（沿用仓库习惯）；mHC `scale`=0.01、`base`=0；`attn_sink`=0（推断）；Engram 见 §8；专家 bias 0；norm 权重 1。

### 9.6 3090 默认

micro-batch 8 × 1024、BF16 autocast、`torch.compile` 默认开；新增按层 activation checkpointing 开关（显存不够时开）。日志：启动打印每层模式与共享关系；训练中打印 ce / balance / indexer_kl、各层专家负载、tokens/s、峰值显存。

## 10. 推理

### 10.1 缓存（对应官方各 buffer）

| 缓存 | 层 | 内容 |
|------|----|------|
| 滑窗环形缓存 | 每层 | 128 槽 × 128 维 |
| 压缩主 KV | 2、4 | 每条目 128 维；第 2 层另存未满组的 `kv_state`/`score_state`（fp32） |
| indexer K | 2、4 | 每条目 64 维 |
| 共享状态 | 每步 | 本步主 KV、indexer K、top-k、候选池 |
| Engram 历史 | — | 压缩 id 序列 |

### 10.2 生成

1. 预填充（CED）：编码器 0–3 层跑整段 prompt；第 4 层 `attn_norm` + 压缩器算出所有位置的解码器全局 KV 与 indexer K；解码器 4–7 层只跑最后 128 个 token（Decoder Bounded Replay），取末位 logits。
2. 解码：每个新 token 完整走 8 层，同步更新环形缓存、压缩器中间状态（m=2 每两步产出一个条目）、indexer K、Engram 历史。
3. 超出 `max_seq_len`：用最后 `max_seq_len` 个 token 重新预填充。
4. 对外接口不变：`generate`、`generate_stream`、`generate_stream_debug`；停止机制沿用 `/api/infer/stop`。

### 10.3 多轮前缀缓存（M 部署策略）

推理引擎保留一个槽位：上一轮的 token、全局 KV、indexer K、压缩器状态、Engram 历史；**不保留滑窗缓存**（对应论文「持久缓存不存 SWA KV」）。新请求 token 前缀完全一致时复用全局 KV，用 Encoder SWA Bounded Replay（只重放前缀最后 128 个 token、只重建滑窗 KV、不覆盖全局 KV）恢复编码器滑窗缓存，再对新增后缀做 CED 预填充；不一致则清空重来。

### 10.4 保持全量 forward 的地方

验证集 PPL、lm-eval、专家负载探针。

## 11. DSpark（推荐方案，用户授权按推荐定）

### 11.1 结构（照官方 `DSparkBlock` / `DSparkAttention` / Markov 头 / 置信度头）

- 3 个 block，每个是完整的 mHC Block：注意力为纯滑窗（窗口 128）的 DSpark 变体，MoE 为 1 共享 + 16 路由、top-3、中间维 176（官方 128 路由 top-3）。
- 输入：草稿块 `[anchor, noise×4]`（`dspark_block_size = 5`），用主干的 embedding（冻结、共享）；`noise` 取 tokenizer 中不出现在正常文本里的特殊 token（MiniMind 取 pad token，推断）。
- 上下文特征：主干第 5、6、7 层注意力输入（4 条流取平均）拼接 → 第一个 block 的 `main_proj`（1536→512）+ `main_norm` → 作为上下文 KV 写入草稿头的滑窗环形缓存；草稿块内部双向可见（官方 `get_dspark_topk_idxs`）。
- 输出：最后一个 block 后 `hc_pre` → `norm` → 主干 head（冻结、共享）得到 5 个位置的基础 logits；Markov 头（rank 64：`embed[6400,64]`、`head[6400,64]`）按上一个 token 加转移偏置，逐位采样；置信度头 `Linear(512+64→1)`（fp32），`c_k = σ(w·[h_k; W1[x_{k−1}]])`。
- 参数约 19M，不计入 200M。

### 11.2 训练（DSpark 论文 §3.3；M §2.4.3）

- 独立阶段（`algo/train_dspark.py`）：预训练后，主干冻结；数据用预训练数据（打包 1024）；主干 no-grad forward 得到目标层隐状态与目标分布；每条序列随机采样多个 anchor（默认 32）组成草稿块；Markov 偏置用真实前一个 token（teacher forcing）。
- 损失：`0.1·L_ce + 0.9·L_tv + 1.0·L_conf`，位置权重 `w_k = exp(−(k−1)/γ)`；`L_tv = Σ w_k‖p_d − p_t‖₁`；`L_conf` 为 BCE，目标 `c*_k = 1 − ½‖p_d − p_t‖₁`（detach）。
- 优化器：草稿头线性层 Muon、norm/标量 AdamW、Markov 表 Sinkhorn 动量。
- SFT 阶段可选继续联合训练草稿头，梯度不回传主干。
- 校准：STS（Sequential Temperature Scaling）——在验证切分上从左到右逐位置一维网格搜索温度，最小化累计乘积的 ECE；温度存进草稿头 checkpoint。

### 11.3 投机解码（官方仓库未实现，按论文写）

1. 主干出一个 token 作 anchor → 草稿头一次并行 + Markov 顺序采样出 5 个草稿 token 与置信度。
2. 调度：论文 Algorithm 1，R=1；`SPS(B)` 在加载时于当前 GPU 上实测主干 B=1..6 的单步吞吐；按存活概率 `a_j = Π_{i≤j} c_i` 逐个加入，吞吐不再上升即停（保持无损所需的提前停止）。
3. 验证：主干一次 forward 验证调度出的前缀；温度 > 0 用标准无损接受/拒绝采样（接受概率 `min(1, p_t/p_d)`，拒绝时从 `norm(max(0, p_t − p_d))` 采样），温度 0 要求与 argmax 一致；全部接受时补一个 bonus token。
4. 回滚：拒绝后把滑窗环形缓存（恢复被覆盖的槽）、压缩器中间状态、压缩 KV / indexer K 长度、Engram 历史、草稿头上下文缓存回退到已接受长度。

## 12. 接入、文档与测试

### 12.1 接入

- 删除 `algo/model/moe.py`（M0.1）及 `arch="moe"` 预设；新增 `arch="miniseek-m"`；Dense / Tiny 不动。
- checkpoint：`{"model", "config"(含 arch="miniseek-m"), ...}`，加载按 `config.arch` 派发；DSpark 单独存 `dspark.pt`。
- `train_pretrain.py` / `train_sft.py`：`--arch miniseek-m`；损失组合；优化器工厂；bias 按 step 累计更新；打包对齐 + `doc_ids`；SFT 回答起点；新增开关 `--candidate-pool`、`--bounded-replay`、`--act-ckpt`；`--flash-attn` 对 v41 无效。
- `generate.py`、`api/app.py`：v41 加载、带缓存生成、前缀缓存、有草稿头时可开投机解码。
- `algo/eval/`（`valid_ppl`、`miniseek_lm` 等）：适配 `(logits, ce, extras)` 返回值。
- Web UI：训练表单架构选项 Tiny / Dense / M（M0.1）；架构页新增 `v41` 可训练预设，沿用自下而上括号图，画出 CED 分区、逐层模式（SWA / Full / Reuse / Reindex）、mHC 4 流、Engram（L1、L3）与 KV 流转（L2→L3；编码器输出→L4→L5–7；L4 候选池→L6；L6 top-k→L7）；专家负载页适配 64 专家 top-6。
- 文档：`docs/STAGE_PLAN.md`、`README.md`、`README_en.md`、`AGENTS.md`（改正「sigmoid 选专家」为 sqrt(softplus)；更新架构状态与绘图规则；训练默认 1024、Muon 等）。

### 12.2 测试（CPU 可跑；用小尺寸配置）

在 `tests/miniseek_v_ref/` 把官方函数的语义移植成纯 torch 参考实现（注明来源与 MIT 许可），逐元素对拍：

| 对象 | 断言 |
|------|------|
| RoPE / YaRN | 与官方 `precompute_freqs_cis`、`apply_rotary_emb`（含反向）一致 |
| 注意力 | 滑窗索引、压缩器（m=1、m=2）、indexer 打分与 top-k、候选块选择、`sparse_attn`（含 sink、−1、空行）与参考一致；改未来 token 不影响当前输出；Reuse 层无压缩器 / indexer 参数 |
| indexer KL | KL 梯度只到 indexer 参数；LM 损失梯度不到 indexer 参数 |
| CED | 解码器仅第 4 层有压缩器且输入为编码器输出 |
| mHC | 与官方 `hc_split_sinkhorn` 一致；行列和≈1；单遍错位生效；`scale=0, base=0` 时等价 Pre-Norm |
| MoE | 与官方 `Gate`/`Expert`/`MoE` 一致；bias 只影响选择；均衡损失手算；bias 更新方向 |
| Engram | 哈希 id 与官方一致；压缩表合并大小写与重音；门控一致；开局即写入残差且表能拿到梯度 |
| 优化器 | NS 系数与轮数；head-wise 拆分等价逐头计算；Sinkhorn 更新与 Algorithm 1 直译一致、屏蔽行不更新；每个参数恰属一组 |
| 打包 | 文档段都从偶数位置开始；`doc_ids` 掩码正确 |
| 推理 | P ≤ 128 时缓存生成与全量 forward 逐 token 一致；P > 128 时与带重放掩码的训练 forward 一致；环形缓存绕回正确；m=2 跨组正确；前缀缓存命中不重算全局 KV |
| DSpark | 草稿块注意力索引与官方一致；损失数学；温度 0 时投机解码输出与普通贪心完全一致；拒绝后缓存与普通解码一致 |
| 规模 | M0.1 主干 + Engram 的 `numel()` ≤ 200M（meta device 构建） |

### 12.3 实施阶段（每阶段测试通过后提交）

1. 基础算子：`FlashConfig`、RoPE/YaRN、注意力（SWA + CSA2 + indexer + 候选池 + KL）及对拍测试。
2. 主干组装：mHC、MoE、Engram、CED 接线、损失、参数量。
3. 训练：优化器、打包对齐与 `doc_ids`、SFT 回答起点、训练脚本接入。
4. 推理：缓存、CED 预填充 + Bounded Replay、解码、前缀缓存、API 接入。
5. DSpark：模型、训练脚本、STS 校准、投机解码与回滚。
6. 收尾：删除旧 MoE、UI、文档与 `AGENTS.md`。

## 13. 推断项汇总（论文与官方代码均未明确）

1. indexer KL 的目标取来源层自己的主注意力；token 平均；三层平均；权重 1。
2. mHC `base` 初值 0、`fn` ~ N(0, 0.02)。
3. Engram 表与 `wkv` 均 N(0, 0.02)、不置零；Muon 中 `wkv` 拆成 4 key + 1 value。
4. `attn_sink` 初值 0。
5. 序列均衡损失跨层求和。
6. 多轮 SFT 每个回答起点各做一次 Decoder Bounded Replay 截断。
7. 打包时补 pad 使文档段从偶数位置开始；RoPE 用切块内全局位置。
8. DSpark 尺寸（1+16 top-3、rank 64）、noise token 取 pad、每序列 32 个 anchor、STS 网格、单请求 `SPS` 实测。
9. 残差出口 1/√(2L) 缩放（仓库习惯）。

## 14. 验收

1. §12.2 全部单测在 CPU 上通过。
2. M0.1 `numel()`（主干 + Engram）≤ 200M，启动时打印。
3. 3090 冒烟预训练（1024 × 8，几百步）不 OOM，三项损失与 tokens/s 正常输出，loss 下降。
4. 贪心解码：缓存生成 = 全量生成（P ≤ 128）；投机解码 = 普通解码。
