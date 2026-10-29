# MiniSeek-M0.1（DeepSeek-V4.1-Flash 全架构）实施计划

> 执行方式：同一会话内联执行（用户授权全程不再审核）。逐任务 TDD：先写对拍/行为测试，再实现，测试通过后提交。代码以提交为准，本文锁定文件、接口、测试断言与提交节点。

**目标：** 按 `docs/superpowers/specs/2026-10-29-v41-flash-full-arch-design.md` 实现可在单卡 3090 训练的 MiniSeek-M0.1。

**架构：** 新包 `algo/model/miniseek_m/`（模块边界同官方 `inference/model.py`）+ `algo/optim/` 三类优化器；训练 / 推理 / API / UI 按 `arch="miniseek-m"` 派发；删除旧 M0.1 MoE。

**技术栈：** PyTorch（训练 BF16 autocast；测试 CPU fp32）、numpy、tokenizers、pytest。

## 全局约束

- 本机测试：`.\.venv\Scripts\python.exe -m pytest -q --ignore=_sync_stash`（CPU；基线 127 passed / 3 skipped）。
- 全部 BF16；不做 FP4/FP8/QAT；不做多模态。
- M0.1：主干 + Engram `numel()` ≤ 200M；DSpark 另算。
- 官方语义参考移植放 `tests/miniseek_m_ref/`，注明来源 HF `deepseek-ai/DeepSeek-V4.1-Flash/inference`（MIT）。
- 提交信息：`<type>(<scope>): <summary>`，除 docs 外带正文；issue 编号留空；不加 Co-Authored-By；不提交 `_sync_stash/`、数据、权重。

## 文件结构

| 文件 | 职责 |
|------|------|
| `algo/model/miniseek_m/__init__.py` | 导出 `FlashConfig`、`Transformer`、`build_miniseek_m` |
| `algo/model/miniseek_m/config.py` | `FlashConfig`（官方字段名）、`tiny()` 测试配置、序列化 |
| `algo/model/miniseek_m/rope.py` | `precompute_freqs_cis`、`apply_rotary_emb`、`rotate_tail`、`layer_freqs` |
| `algo/model/miniseek_m/attention.py` | `RMSNorm`、`AttnState`、`window_indices`、`sparse_attention`、`select_candidate_blocks`、`Compressor`、`Indexer`、`Attention` |
| `algo/model/miniseek_m/mhc.py` | `hc_split_sinkhorn`、`hc_mixes`、`hc_pre`、`hc_post` |
| `algo/model/miniseek_m/moe.py` | `Gate`、`Expert`、`MoE`、`sequence_balance_loss` |
| `algo/model/miniseek_m/engram.py` | 压缩 token 表、`EngramLayout`、`compute_hash_multipliers`、`NgramHasher`、`Engram` |
| `algo/model/miniseek_m/model.py` | `Block`、`Transformer`、`bounded_replay_floor` |
| `algo/model/miniseek_m/cache.py` | `InferenceCache`、分块 forward、CED 预填充、解码、前缀缓存 |
| `algo/model/miniseek_m/dspark.py` | DSpark 草稿头、训练损失、调度器、投机解码 |
| `algo/optim/muon.py` / `sinkhorn.py` / `param_groups.py` | 三类优化器与参数分组、组合优化器 |
| `algo/train_dspark.py`、`algo/dspark_calibrate.py` | DSpark 训练与 STS 校准 |
| 修改：`algo/model/__init__.py`、`algo/batching.py`、`algo/train_pretrain.py`、`algo/train_sft.py`、`algo/generate.py`、`algo/eval/*`、`api/*`、`frontend/index.html`、文档 | 接入 |
| 删除：`algo/model/moe.py`、`tests/test_moe.py` | 旧 M0.1 |

## 接口（后续任务以此为准）

```python
# config.py
@dataclass
class FlashConfig:  # 字段与默认值见 spec §3.2；另含 arch="miniseek-m", model_family="MiniSeek-M", model_version="0.1"
    def to_dict(self) -> dict: ...
    @classmethod
    def from_dict(cls, d: dict) -> "FlashConfig": ...
    @classmethod
    def tiny(cls) -> "FlashConfig": ...          # 8 层同模式、dim 32，CPU 测试用
    @property
    def release_name(self) -> str: ...           # "MiniSeek-M0.1"
    def is_decoder(self, layer_id: int) -> bool: ...  # layer_id >= n_layers // 2

# rope.py
def precompute_freqs_cis(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow) -> Tensor  # complex64 [seqlen, dim//2]
def apply_rotary_emb(x: Tensor, freqs_cis: Tensor, inverse: bool = False) -> Tensor  # 不原地；x [B,S,D] / [B,S,H,D]
def rotate_tail(x: Tensor, freqs_cis: Tensor, rope_dim: int, inverse: bool = False) -> Tensor
def layer_freqs(cfg: FlashConfig, layer_id: int, seqlen: int) -> Tensor

# attention.py
@dataclass
class AttnState:
    compress_kv: Optional[Tensor] = None   # [B,Tc,hd]（已 RoPE）
    compress_doc: Optional[Tensor] = None  # [B,Tc] 条目所属文档
    index_k: Optional[Tensor] = None       # [B,Tc,Ihd]
    topk: Optional[Tensor] = None          # [B,T,K] long，压缩条目下标，−1 无效
    candidates: Optional[Tensor] = None    # [B,T,Tc] bool
    use_candidates: bool = False
    kl: list = field(default_factory=list)
    probs_hook: Optional[Callable] = None
def window_indices(T: int, window: int, device) -> Tensor                       # [T,W]，−1 无效
def sparse_attention(q, kv, idx, sink, scale) -> tuple[Tensor, Tensor]          # (o [B,T,H,D], probs [B,T,H,K] fp32)
def select_candidate_blocks(scores, compress_lens, topk_blocks, block_size) -> Tensor  # bool，同 scores 形状
class Attention(nn.Module):
    def __init__(self, cfg: FlashConfig, layer_id: int): ...
    def forward(self, x, state: AttnState, doc_ids=None, swa_floor=None) -> Tensor  # 训练/全量 forward

# mhc.py
def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, iters=20, eps=1e-6) -> tuple[Tensor, Tensor, Tensor]
def hc_mixes(x, fn, scale, base, norm_eps, hc_mult, iters, eps) -> tuple[Tensor, Tensor, Tensor]
def hc_pre(x, pre) -> Tensor
def hc_post(y, residual, post, comb) -> Tensor

# moe.py
class Gate(nn.Module):   def forward(self, x) -> tuple[weights, indices, scores]
class Expert(nn.Module): def forward(self, x, weights=None) -> Tensor
class MoE(nn.Module):    def forward(self, x) -> tuple[Tensor, Tensor]   # (y, balance)
    def update_expert_bias(self, speed: float) -> None
def sequence_balance_loss(scores, indices, n_experts) -> Tensor

# engram.py
def build_compressed_token_map(tokenizer) -> tuple[list[int], int]
class EngramLayout:  @classmethod from_config(cls, cfg) -> Optional["EngramLayout"]
def compute_hash_multipliers(layer_ids, max_ngram_size, vocab_size) -> Tensor
class NgramHasher(nn.Module): def forward(self, input_ids, doc_ids=None) -> Tensor  # [B,T,n_eng_layers,n_cols]
class Engram(nn.Module):      def forward(self, x, hash_ids) -> Tensor

# model.py
class Transformer(nn.Module):
    def __init__(self, cfg: FlashConfig, token_map: Optional[list[int]] = None): ...
    def forward(self, input_ids, labels=None, *, doc_ids=None, swa_floor=None,
                use_candidate_pool=False, return_target_hidden=False) -> tuple[logits, ce, extras]
    def update_expert_bias(self) -> None
    def begin_expert_usage(self) -> None; def end_expert_usage(self) -> Tensor
    def probe_expert_load(self, input_ids) -> tuple[Tensor, int]
def bounded_replay_floor(labels, doc_ids, window) -> Tensor  # [B,T] long

# optim
class Muon(torch.optim.Optimizer)             # 组字段：lr, lr_mult, weight_decay, momentum, ns_steps；参数属性 _muon_splits
class SinkhornMomentum(torch.optim.Optimizer) # 组字段：lr, lr_mult, beta, gamma, k_iters, tau, eps
class CompositeOptimizer                      # param_groups / step / zero_grad / state_dict / load_state_dict
def build_miniseek_m_optimizer(model, lr, *, weight_decay=0.1, engram_lr_mult=5.0) -> CompositeOptimizer
```

---

## 阶段 1：基础算子

### 任务 1：配置、RoPE、官方参考移植

**文件：** 新建 `algo/model/miniseek_m/{__init__,config,rope}.py`、`tests/miniseek_m_ref/{__init__,official}.py`、`tests/test_v41_rope.py`

- [ ] `tests/miniseek_m_ref/official.py`：把官方 `precompute_freqs_cis`、`apply_rotary_emb`（原地版）、`get_window_topk_idxs`、`Compressor`（prefill 分支）、`Indexer` 打分 + top-k、`select_candidate_blocks`、`sparse_attn`（kernel 数学）、`hc_split_sinkhorn`（kernel 数学）、`Gate`、`Expert`、`MoE`、`Engram.forward`、`EngramLayout`、`compute_hash_multipliers`、`NgramHashState.forward`、`get_dspark_topk_idxs` 移植为纯 torch / numpy 函数。
- [ ] 测试：`test_freqs_match_official`（含 YaRN 开/关）；`test_rotate_matches_official`；`test_inverse_rotation_roundtrip`；`test_flash_config_roundtrip`；`test_f01_defaults`（层模式、来源层、尺寸与 spec 一致）。
- [ ] 运行 `pytest tests/test_v41_rope.py -q` 通过；提交 `feat(v41): add FlashConfig, RoPE and official reference ports`。

### 任务 2：注意力（SWA + CSA2 + indexer + 候选池 + KL）

**文件：** 新建 `algo/model/miniseek_m/attention.py`、`tests/test_v41_attention.py`

- [ ] 测试（fp32、`FlashConfig.tiny()`）：
  - `test_window_indices_match_official`
  - `test_compressor_matches_official`（m=1、m=2）
  - `test_sparse_attention_matches_official`（含 sink、−1、全无效行输出 0）
  - `test_indexer_topk_matches_official`；`test_candidate_blocks_match_official`
  - `test_attention_modes_match_reference`：Full / Reuse / Reindex 三层串起来，与官方逻辑拼出的参考一致
  - `test_attention_causal`：改未来 token 不影响当前输出
  - `test_doc_mask_isolation`：改文档 A 不影响文档 B
  - `test_reuse_layer_has_no_compressor_or_indexer`
  - `test_indexer_grad_isolation`：KL 梯度只到 indexer；LM 损失梯度不到 indexer
  - `test_swa_floor_truncates_window`
- [ ] 实现后运行通过；提交 `feat(v41): add SWA + CSA2 attention with indexer and KL`。

## 阶段 2：主干组装

### 任务 3：mHC

**文件：** `algo/model/miniseek_m/mhc.py`、`tests/test_v41_mhc.py`

- [ ] 测试：`test_sinkhorn_matches_official`；`test_comb_doubly_stochastic`（行列和误差 < 1e-3）；`test_hc_post_formula`；`test_zero_scale_is_prenorm`（`scale=0, base=0` 时多流 Block 等价单流 Pre-Norm）。
- [ ] 提交 `feat(v41): add single-pass mHC`。

### 任务 4：MoE

**文件：** `algo/model/miniseek_m/moe.py`、`tests/test_v41_moe.py`

- [ ] 测试：`test_gate_matches_official`；`test_moe_matches_official`；`test_bias_only_changes_selection`；`test_balance_loss_hand_example`（手算 2 专家 / top-1 小例）；`test_bias_update_direction`；`test_counts_accumulate_over_microbatches`。
- [ ] 提交 `feat(v41): add sqrt-softplus MoE with seq balance loss`。

### 任务 5：Engram

**文件：** `algo/model/miniseek_m/engram.py`、`tests/test_v41_engram.py`

- [ ] 测试：`test_primes_and_layout_match_official`；`test_hash_ids_match_official`；`test_doc_boundary_blocks_lookback`；`test_compressed_map_merges_case`（用 `tokenizers` 在内存里构造的小词表）；`test_engram_forward_matches_official`；`test_engram_identity_at_init`。
- [ ] 提交 `feat(v41): add Engram n-gram hash memory`。

### 任务 6：Transformer、CED 接线、损失、参数量

**文件：** `algo/model/miniseek_m/model.py`、修改 `algo/model/__init__.py`、`tests/test_v41_model.py`

- [ ] 测试：`test_forward_backward_tiny`（三项损失有限且可反传）；`test_ced_decoder_kv_from_encoder_output`（解码器仅第 4 层有压缩器，且其 latent 等于由编码器输出单独计算的结果）；`test_doc_isolation_full_model`；`test_replay_floor_from_labels`；`test_f01_param_budget`（meta device，`numel` ≤ 200M 并打印）；`test_build_model_dispatch`。
- [ ] 提交 `feat(v41): assemble CED transformer with losses`。

## 阶段 3：训练

### 任务 7：优化器

**文件：** `algo/optim/{__init__,muon,sinkhorn,param_groups}.py`、`tests/test_v41_optim.py`

- [ ] 测试：`test_hybrid_ns_coeffs`（10 轮、系数正确、输出奇异值≈1）；`test_muon_update_rule`（与 V4 Algorithm 1 直译一致）；`test_headwise_split_equivalence`；`test_sinkhorn_matches_algorithm`；`test_sinkhorn_masks_zero_rows`；`test_every_param_in_one_group`；`test_group_assignment_examples`；`test_lr_mult_applied`。
- [ ] 提交 `feat(optim): add Muon, Sinkhorn momentum and v41 param groups`。

### 任务 8：打包对齐与 doc_ids

**文件：** 修改 `algo/batching.py`、新测试 `tests/test_v41_batching.py`

- [ ] `PackedTextBatchIter` / `PackedSFTBatchIter` 新增 `doc_align: int = 1`、`with_doc_ids: bool = False`；开启时产出 `(x, y, doc_ids)`；补齐 pad 与跨文档预测位置的 label 为 −100。
- [ ] 测试：`test_docs_start_even`；`test_doc_ids_and_cross_doc_labels`；`test_sft_pack_doc_ids`；`test_default_behavior_unchanged`。
- [ ] 提交 `feat(batching): align packed docs and emit doc ids for v41`。

### 任务 9：训练脚本接入

**文件：** 修改 `algo/train_pretrain.py`、`algo/train_sft.py`（及其共用的 checkpoint 加载工具），测试 `tests/test_v41_train_smoke.py`

- [ ] `--arch miniseek-m`：构建 `FlashConfig`（含 tokenizer 压缩表）、组合优化器、损失组合、按 step 更新 bias、doc_ids / swa_floor、`--candidate-pool`、`--bounded-replay`、`--act-ckpt`；checkpoint 存取 `FlashConfig`；日志打印层模式与三项损失。
- [ ] 测试：CPU 上 tiny 配置跑 3 步预训练和 3 步 SFT，loss 有限、ckpt 可续训。
- [ ] 提交 `feat(train): wire v41 into pretrain and sft`。

## 阶段 4：推理

### 任务 10：缓存、CED 预填充、解码

**文件：** `algo/model/miniseek_m/cache.py`、修改 `model.py`、`tests/test_v41_infer.py`

- [ ] 测试：`test_cached_equals_full_short_prompt`（P ≤ 窗口，贪心逐 token 一致）；`test_ced_prefill_matches_replay_forward`（P > 窗口时，与带 `swa_floor` 的 forward 一致）；`test_ring_wraparound`；`test_m2_partial_group_across_steps`；`test_generate_stream_interface`。
- [ ] 提交 `feat(v41): add KV caches, CED prefill and cached decoding`。

### 任务 11：前缀缓存与 API 接入

**文件：** 修改 `cache.py`、`algo/generate.py`、`api/app.py` 等；测试 `tests/test_v41_prefix_cache.py`

- [ ] 测试：`test_prefix_hit_skips_global_kv`；`test_prefix_miss_resets`；`test_engine_loads_v41_ckpt`。
- [ ] 提交 `feat(infer): add v41 prefix cache with encoder bounded replay`。

## 阶段 5：DSpark

### 任务 12：草稿头与训练

**文件：** `algo/model/miniseek_m/dspark.py`、`algo/train_dspark.py`、`tests/test_v41_dspark.py`

- [ ] 测试：`test_dspark_topk_idxs_match_official`；`test_losses_math`；`test_backbone_frozen`；`test_train_dspark_smoke`。
- [ ] 提交 `feat(dspark): add DSpark drafter and training stage`。

### 任务 13：投机解码与 STS 校准

**文件：** 修改 `dspark.py`、`cache.py`；新建 `algo/dspark_calibrate.py`；测试追加

- [ ] 测试：`test_scheduler_early_stop`；`test_greedy_spec_equals_greedy`；`test_rollback_restores_cache`；`test_sampling_acceptance_math`；`test_sts_calibration_reduces_ece`。
- [ ] 提交 `feat(dspark): add confidence-scheduled speculative decoding`。

## 阶段 6：收尾

### 任务 14：删除旧 MoE、UI 接入

**文件：** 删除 `algo/model/moe.py`、`tests/test_moe.py`；修改 `api/arch_inspect.py`、`api/app.py`、`api/task_runner.py`、`frontend/index.html`、相关测试

- [ ] 架构页新增 `v41` 可训练预设（CED 分区、逐层模式、mHC 4 流、Engram、KV 流转）；训练表单选项 Tiny / Dense / M；专家负载页适配。
- [ ] 全量测试通过；提交 `feat(ui): replace M0.1 MoE with v41 preset`。

### 任务 15：文档与收尾

**文件：** `docs/STAGE_PLAN.md`、`README.md`、`README_en.md`、`AGENTS.md`

- [ ] 更新架构状态、MoE 规则（sqrt-softplus）、绘图规则、训练默认值与命令；全量测试通过；`git pull --rebase` 后 push。
- [ ] 提交 `docs: document MiniSeek-M0.1 architecture and workflow`。

## 自查记录

- spec §3–§12 每节均有对应任务（§3→1/6，§4→2，§5→6/10，§6→3，§7→4，§8→5，§9→7/8/9，§10→10/11，§11→12/13，§12→14/15）。
- 接口名在各任务间一致（`AttnState`、`bounded_replay_floor`、`build_miniseek_m_optimizer`、`CompositeOptimizer`）。
