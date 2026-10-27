# 训练速度优化：逐项真实收益基准 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 plan 里写下的每一项训练优化，都能用同一套基准脚本测出**相对 base 的真实吞吐收益**（有效 token/s），而不是绑在一起或测错路径的估算。

**Architecture:** 固定一份 tokenize 样本 + 固定一份 MoE 权重；`base` 复现「未优化」行为；每个变体只打开**一项**开关（`all` 另跑作参考）；指标统一为有效 token/s（label≠-100），并记录 pad%、step 中位数、峰值显存。语义会变的项（分桶、packing）速度合格后另做短程 loss 对照，不并进「等价收益」表。

**Tech Stack:** PyTorch CUDA / MoE（`algo/model/moe.py`）/ `algo/train_pretrain.py` / `scripts/bench_train_opt.py` / 数据 `data/pretrain_mix_5g.jsonl`

## Global Constraints

- 只在 wyf GPU 空闲时跑（当前训练结束后，或 `--wait-pid`）；**禁止**为了 bench 停正在跑的 MoE，除非用户明确说停。
- 吞吐主指标：**有效 token/s** = 测量步内 `(labels != -100)` 总数 / 测量步墙钟合计；padding 不计入分子。
- **单变量**：除 `base` / `all` 外，每个 variant 相对 base **只改一个开关**；禁止把 #1+#2 合成一个「moe_fast」当作两项收益。
- 日志：计时区间内**禁止** `print`/`flush`（避免把 I/O 算进优化收益）；需要模拟 log 同步时只用 `.item()` / `.tolist()`，不打印。
- `torch.compile` 变体：warmup ≥ 30 步或直到连续 3 步 step_ms 波动 <10%；测量步默认 50；其余变体 warmup 20 + 测 40，每个变体 **重复 2 轮**取 tok/s 中位数。
- 判定（相对 base 的 tok/s）：&lt;5% 视为噪声不上；5%–15% 看复杂度；&gt;15% 等价项直接合入。#6/#7 即使 &gt;15% 也必须过 loss 短对照才决策。
- 本 plan **不**把「加大 batch」列为收益项（util 已满时优先测计算/同步/组批）；若显存仍空，可在附录另跑 `batch64` 探路，不进主表。

---

## 现状快照（已测量，写入对照用）

| 指标 | 值 |
|---|---|
| GPU | RTX 3090 24G，util ~100%，显存 ~9.9 / 23.5 GB |
| 线上速度 | ~3.4 step/s |
| 有效 token | p50≈426，mean≈362，p80 触顶 512 |
| padding（随机组批） | ~29.3%；长度分桶后可到 ~0.7% |
| 截断 | ~58.5% 样本被 max_seq=512 截断 |
| 同步热点（旧 MoE） | 每步大量 `.item()`（逐专家）+ 每层 usage `.cpu()` + 每步 `loss.item()` + log 时专家窗 `.tolist()` |

---

## 优化清单与「真实收益」定义

每一项必须有：**改动点**、**base 行为**、**variant 唯一开关**、**bench 必须模拟的路径**、**合格才算测到了这项**。

| # | 名称 | variant id | 语义 | 文件 |
|---|---|---|---|---|
| 1 | MoE 边界一次 `offsets.tolist()`，去掉逐专家 `.item()` | `moe_bounds` | 等价 | `algo/model/moe.py` |
| 2 | 负载统计留 GPU 累加，去掉每层每步 `.cpu()` | `moe_usage_gpu` | 等价 | `algo/model/moe.py` |
| 3 | AdamW `fused=True` | `fused` | 数值等价 | `algo/train_pretrain.py` |
| 4 | `torch.compile` | `compile` | 等价（可能重编译） | 训练入口 |
| 5 | `log_every` 1→10（专家窗降频） | `log10` | 只影响日志粒度 | 任务/`--log-every` |
| 6 | 长度分桶组批 | `bucket` | 改变批内长度分布 | `algo/train_pretrain.py` 采样/组批 |
| 7 | 样本 packing | `pack` | 改变跨文档注意力边界 | 数据/组批 |

另：`base`（全关）、`all_equiv`（只开 #1+#2+#3+#4+#5，**不含** 6/7）、`all_bucket`（等价项 + bucket，可选）。

### 逐项测量契约（必须写进 bench）

**共同 harness**

- 数据：对 `data/pretrain_mix_5g.jsonl` 蓄水池抽 2048 条，**只 tokenize 一次**；`batch_size=32`，`max_seq=512`（pack 见下）。
- 权重：优先 `tasks/<当前或指定>/checkpoints/best.pt`；没有则同配置随机 init 但**所有变体共用同一 state_dict**。
- 步循环：`cuda.synchronize` → 计时 → forward(bf16 autocast) → backward → clip → opt.step → zero_grad →（按契约做同步模拟）→ `synchronize` → 记 dt。
- 输出行：`variant, step_ms_p50, tok_per_s, pad_pct, vram_gb, vs_base, runs`。

**#1 `moe_bounds`**

- base：`MoELayer.forward` 使用「每个专家 `int(offsets[e].item())` / `offsets[e+1].item()`」。
- 本变体：仅改为 `bounds = offsets.tolist()` 后切片；**usage 路径与 base 相同**。
- 合格：相对 base 的加速只能归因于边界同步减少。

**#2 `moe_usage_gpu`**

- **强制**全程 `_record_usage=True`（bench 开头 `begin_expert_usage()`，结束再 `end_expert_usage`），否则这项收益为 0、测了也作废。
- base：usage 分支里 `bincount(...).detach().cpu()` 再累加到 CPU tensor（旧行为）。
- 本变体：bincount 留 GPU 累加；**专家循环仍用 base 的逐专家 `.item()` 边界**（只开 #2）。
- 合格：对比时两边都在记 usage；禁止关 `_record_usage` 测 #2。

**#3 `fused`**

- base：`AdamW(..., fused=False)`（或不传）。
- 本变体：仅 `fused=True`；MoE forward = base（旧边界+旧 usage 或与线上一致的「慢 forward」快照，与其它单项同一 base）。
- 模型已 `.to(cuda)` 后再建优化器。

**#4 `compile`**

- 本变体：仅 `model = torch.compile(model)`；加长 warmup（见 Global Constraints）。
- 收益以测量段 tok/s 相对 base；不把首次编译耗时算进测量段。

**#5 `log10`（必须对齐真训练）**

真训练（`algo/train_pretrain.py`）行为：

- **每步**都有 `float(loss.item())`。
- **仅当** `global_step % log_every == 0`（或 step==1）时额外：`ce.item()`、`aux.item()`、以及 MoE 的 `end_expert_usage()` → `.tolist()` → 再 `begin_expert_usage()`。

bench 必须模拟：

- **所有变体、每步**：一次 `float(total.item())`（对应每步 loss 同步）。
- **base / 非 log10**：每步再额外模拟 log 窗：`float(ce.item())`、若有 aux 则 `float(aux.item())`，且 MoE 下每步 `end_expert_usage(); begin_expert_usage()`（tolist 同步）。
- **`log10`**：同上「每步 loss.item」保留；仅当 `step % 10 == 0` 才做 ce/aux/expert 窗同步。
- 禁止：把 log10 做成「只有打 log 才 item」——那会高估收益。
- 计时时不 `print`。

**#6 `bucket`**

- base：`random` shuffle 后 pad 组批（现有 `collate_pad`，pad 标 -100）。
- 本变体：按 `len` 排序再切 batch，再打乱 **batch 顺序**；其余同 base。
- 主看 tok/s 与 pad%；pad% 应明显下降才算实现正确。

**#7 `pack`**

- 定义（本 plan 锁定）：将样本 token 用 `eos(id=2)` 拼接成流，切成长度 `max_seq` 的块；**每个训练 step 仍使用 `batch_size` 条 packed 序列**（`[B, max_seq]`），与 base 同 B、同设备占用量级。禁止 `B=1` 的「假 packing」对比。
- 标签：块内移位预测；padding 无（或仅最后一块短垫 -100，统计时如实记）。
- 本版 **不**做跨文档 attention mask（与「改变跨文档边界」一致）；loss 短对照后再决定是否上线。
- `all_*` **默认不含 pack**；pack 单独决策。

---

## 文件职责

| 路径 | 职责 |
|---|---|
| `scripts/bench_train_opt.py` | 唯一 A/B 入口；实现上表全部 variant 与契约 |
| `algo/model/moe.py` | 提供可切换的 forward/usage 行为，或由 bench monkeypatch 慢路径 |
| `algo/train_pretrain.py` | 线上默认；bench 对齐其 log/expert 同步语义；落地 fused/bucket/pack 时改这里 |
| `docs/superpowers/plans/2026-10-27-training-speed-bench.md` | 本 plan |
| `/tmp/bench_train_opt.json`（或 `logs/bench_*.json`） | 结果表 |

---

### Task 1: 重写 bench 变体表与慢路径快照

**Files:**
- Modify: `scripts/bench_train_opt.py`
- Test: 本地 `python3 -c` 导入 + 干跑 CPU 跳过；真 GPU 在 Task 4

**Interfaces:**
- Produces: `VARIANTS` 含 `base`, `moe_bounds`, `moe_usage_gpu`, `fused`, `compile`, `log10`, `bucket`, `pack`, `all_equiv`
- Produces: `set_moe_mode(mode: str)` — `"slow"` \| `"bounds_fast"` \| `"usage_gpu"` \| `"both_fast"`（both 仅给 all_equiv）
- Produces: `simulate_log_sync(model, total, ce, aux, step, log_every, force_expert: bool)` 按 #5 契约

- [ ] **Step 1: 删掉把 #1+#2 绑在一起的 `moe_fast` 单一开关**

`VARIANTS` 改为上表 id；`ORDER` 与之一致。

- [ ] **Step 2: 实现两个可叠加的慢/快补丁**

- `slow_bounds`：循环内 `offsets[e].item()` 两次。
- `slow_usage`：`_record_usage` 时 `.detach().cpu()` 累加。
- 当前仓库已落地的 fast：`tolist` 边界 + GPU usage —— 作为对应快路径。
- `moe_bounds`：快边界 + **慢 usage**（且 `_record_usage=True`）。
- `moe_usage_gpu`：慢边界 + **快 usage**（且 `_record_usage=True`）。
- `base`：慢边界 + 慢 usage，`_record_usage=True`。
- 非 MoE 统计项的变体：与 base 相同 MoE 模式，避免混杂。

- [ ] **Step 3: 实现 #5 同步模拟（无 print）**

每步 `total.item()`；若 `step % log_every == 0`（base 的 log_every=1，log10 的=10）则 ce/aux item + expert end/begin。

- [ ] **Step 4: 修正 pack 组批为 `[B, S]`**

```python
# 伪代码：stream 切块 -> 列表 chunks；再按 batch_size 堆叠
# 短块右侧 pad，y 的 pad 位为 -100
```

- [ ] **Step 5: 去掉计时区内 print；compile/warmup/重复轮次按 Global Constraints**

- [ ] **Step 6: 提交**

```bash
git add scripts/bench_train_opt.py
git commit -m "fix(bench): per-item train speed variants with real sync contracts"
```

---

### Task 2: 单元级校验「测的是不是那一项」

**Files:**
- Create: `tests/test_bench_train_opt_contract.py`
- Modify: `scripts/bench_train_opt.py`（若需导出纯函数）

**Interfaces:**
- Consumes: `build_batches`, `simulate_log_sync` 逻辑（可抽成可测函数）
- Produces: 不依赖 GPU 的契约测试

- [ ] **Step 1: 写失败测试 — pack 的 batch 维必须是 batch_size**

```python
def test_pack_batch_shape():
    enc = [[1, 2, 3, 4]] * 100
    batches = build_batches(enc, "pack", batch_size=32, max_seq=32)
    x, y = batches[0]
    assert x.shape[0] == 32
    assert y.shape[0] == 32
```

- [ ] **Step 2: 写失败测试 — log10 不会取消每步 loss 同步次数**

用 mock counter：`log_every=10` 跑 20 步 → `total.item` 调用 20 次，expert_end 调用 2 次；`log_every=1` → expert_end 20 次。

- [ ] **Step 3: 跑测至通过并提交**

```bash
pytest tests/test_bench_train_opt_contract.py -v
git add tests/test_bench_train_opt_contract.py scripts/bench_train_opt.py
git commit -m "test(bench): lock per-item speed bench contracts"
```

---

### Task 3:（可选落地准备）确认线上 #1/#2 与 bench 快路径一致

**Files:**
- Modify: `algo/model/moe.py`（仅当与 bench 快路径漂移时）

- [ ] **Step 1: diff 当前 `MoELayer.forward` 与 bench 快路径注释点**（`tolist`、GPU usage）
- [ ] **Step 2: 若不一致，以「线上 = both_fast」为准改 bench 快照，或改回线上**
- [ ] **Step 3: 提交（若有改动）**

---

### Task 4: 在 wyf 上空闲 GPU 跑完全表并出结果

**Files:**
- Create: `logs/bench_train_opt_YYYYMMDD.json`（或 `/tmp/...` 再拷回 `logs/`）

- [ ] **Step 1: 等训练结束或 `--wait-pid <train_pid>`**

```bash
cd /home/wyf/miniseek
python3 scripts/bench_train_opt.py \
  --ckpt tasks/<id>/checkpoints/best.pt \
  --data data/pretrain_mix_5g.jsonl \
  --variants base,moe_bounds,moe_usage_gpu,fused,compile,log10,bucket,pack,all_equiv \
  --out logs/bench_train_opt.json
```

- [ ] **Step 2: 填结果表（写入本 plan 末尾「结果」节或单独 json）**

| variant | step_ms | tok/s | pad% | VRAM | vs base |
|---|---|---|---|---|---|

- [ ] **Step 3: 按判定标准标「上 / 不上 / 待 loss 对照」**

---

### Task 5: 对 #6/#7 做短程 loss 对照（仅速度过线时）

**Files:**
- 用现有 Web/CLI 训练：同数据子集、同 step 预算（如 2k step）、同 seed 权重 init 或同 resume 点

- [ ] **Step 1: 若 bucket vs_base tok/s ≥ 5%，跑 bucket vs random 短训，比 valid loss**
- [ ] **Step 2: 若 pack ≥ 5%，跑 pack vs random；确认可接受再写入默认**
- [ ] **Step 3: 记录结论；不要把未过 loss 门的项写进默认训练**

---

### Task 6: 把判定为「上」的等价项合入训练默认

**Files:**
- Modify: `algo/train_pretrain.py`（`AdamW fused=True` 若 #3 过线；compile/log_every 默认若过线）
- Modify: Web/API 默认配置（若 log_every / compile 走配置）
- #1/#2 若已在 `moe.py` 落地则无需再改；以 Task 4 数字确认保留

- [ ] **Step 1: 只合入等价项且 vs_base≥阈值者**
- [ ] **Step 2: 部署 wyf：`./scripts/deploy.sh`（训练结束后再 `--restart` worker）**
- [ ] **Step 3: 提交与简短结果备注**

---

## 结果（跑完 Task 4 后填）

| variant | step_ms | tok/s | pad% | VRAM | vs base | 决策 |
|---|---|---|---|---|---|---|
| base | | | | | 1.00x | — |
| moe_bounds | | | | | | |
| moe_usage_gpu | | | | | | |
| fused | | | | | | |
| compile | | | | | | |
| log10 | | | | | | |
| bucket | | | | | | 待 loss |
| pack | | | | | | 待 loss |
| all_equiv | | | | | | 参考 |

---

## Self-review

- 覆盖：原 7 项均有独立 variant 与测量契约；旧脚本 #2 未开 usage、#5 测错、pack B=1、#1+#2 捆绑等问题均有对应 Task。
- 无 TBD：判定阈值、命令、形状断言已写死。
- `all_equiv` 不含 6/7，避免把语义变化混进「等价合计」。

---

## 执行方式（选定后开工）

Plan 已保存到 `docs/superpowers/plans/2026-10-27-training-speed-bench.md`。

1. **Subagent-Driven（推荐）** — 每 Task 一个子代理，Task 间 review  
2. **本会话 Inline** — 按 executing-plans 连续做完  

要哪一种？
