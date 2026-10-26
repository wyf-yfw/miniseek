# MiniSeek 架构页 — 残差旁路连线设计

日期：2026-10-26  
状态：待用户审阅 spec  
关联：`2026-10-26-model-architecture-viz-design.md`（架构页已改为设计规格 Tiny/Dense/MoE，不再依赖 `.pt`）

## 目标

在「架构」页 **详细** 视图中，用教科书式 **旁路绕线（方案 A）** 画出 Pre-Norm 残差，让人一眼看出残差的 `x` 来自子块入口，而不是一张「Residual + Attn」公式卡。

## 已拍板

| 项 | 决定 |
| --- | --- |
| 画法 | A · 旁路绕线：主路径竖排，橙色线从入口右侧绕到 ⊕ |
| 视图范围 | **仅详细**；简洁 / 标准不变 |
| 公式卡 | **去掉** `Residual + Attn` / `Residual + FFN` 零件卡；只留旁路线 + ⊕ |
| 实现路线 | DOM 主路径块 + 层内绝对定位 SVG 画旁路与 ⊕（不整层改 SVG） |

## 非目标

- 简洁 / 标准视图画残差旁路
- 整图或整层纯 SVG 重写
- 前向激活动画、注意力热力图
- 改变参数量公式或设计规格（Tiny/Dense/MoE）本身

## 详细视图布局（一层）

主路径自上而下：

1. RMSNorm (pre-Attn) → Attention → **⊕_attn**
2. RMSNorm (pre-FFN) → FFN 或 MoE（Router / Shared / Routed）→ **⊕_ffn**

两条橙色旁路：

- Attn 残差：从「该 Attn 子块入口 `x`」（Norm 之前的层内输入）右侧水平伸出，向下绕到 ⊕_attn，线旁小号标注 `x`
- FFN 残差：从 ⊕_attn 之后、FFN Norm 之前的 `x` 同样旁路到 ⊕_ffn

视觉约定：

- 旁路描边橙色（与现有结构强调色一致即可），箭头指向 ⊕
- ⊕ 为绿色描边圆 + 「+」字，主路径与旁路在此汇合
- 层头仍可点开抽屉；抽屉 **不再** 列出 residual 零件

## 数据（设计 JSON）

`inspect_design` / `inspect_from_config`：

- **删除** parts 中的 `residual_attn`、`residual_ffn`（`params: 0` 公式卡）
- 每层增加可选字段（不算参数）：

```json
"residuals": [
  { "id": "attn", "from": "layer_in", "to": "add_attn", "label": "x" },
  { "id": "ffn", "from": "post_attn", "to": "add_ffn", "label": "x" }
]
```

- 前端也可在无 `residuals` 时按固定顺序推断（Norm→Attn→add→Norm→FFN→add）；有字段则优先用字段，便于测试与 MoE 对齐
- checkpoint 路径的 `inspect`（若仍保留）详细视图同样不渲染 residual 公式卡；旁路线仅设计页详细视图保证支持（首版以 `/api/arch/design` 为准）

## UI 实现要点

- `.arch-layer-group { position: relative; }`，右侧留出旁路 gutter（约 48–64px）
- 主路径节点带稳定 `data-node`（如 `attn_norm`、`attn`、`add_attn`、`ffn_norm`、`ffn`/`router`…、`add_ffn`）
- 旁路起点映射到**可见块右中**，不用零高锚点：`layer_in` → `attn_norm`，`post_attn` → `add_attn`；终点仍为对应 ⊕ 中心
- 渲染完一层后测量节点相对层组的坐标，插入 `<svg class="arch-residual-svg">` 画折线与 ⊕；窗口 resize / 视图切换时重算
- ⊕ 可作为 SVG 圆，或小号 DOM 节点插入主路径流；推荐 **主路径流里放 DOM ⊕**，SVG 只画旁路折线接到 ⊕ 中心，避免两套布局打架
- 简洁 / 标准：逻辑与现网一致，不画 SVG 旁路

## 测试

- 单元：设计 JSON 无 `residual_*` parts；Dense/Tiny/MoE 层含约定的 `residuals`（或文档化的推断顺序）
- 手工：详细视图 Dense / MoE 各一层旁路可见且 `x` 指向入口；切到标准/简洁无旁路、无公式卡；刷新 `:7860` 后 Tiny/Dense/MoE 按钮仍可用

## 验收

1. 详细视图看不到 Residual 公式卡。
2. 每层两条橙色旁路 + 两个 ⊕，能直观看出残差 `x` 是谁。
3. 简洁 / 标准与改前一致（无旁路线）。
4. 参数量与设计规格不变。

## 图上信息密度（补充，2026-10-26）

详细视图主图里，每个模块块 **只显示名称**（如 `Attention`、`RMSNorm`、`FFN`）；**不**在块下方展示参数量、`n_heads`、`head_dim`、`qkv`/`o_proj` 等 detail 字段。

这些明细仅在用户 **点击** 该层（或该模块所属层）后，于 **右侧抽屉** 展示（沿用现有 `arch-drawer`：层标题、本层参数、各 part 的 params + detail）。

层头可保留一层总参量（可选）；若与「图面干净」冲突，层头也只留 `Layer i`，总参放进抽屉首行（推荐：层头仅 `Layer i`，总参进抽屉）。

