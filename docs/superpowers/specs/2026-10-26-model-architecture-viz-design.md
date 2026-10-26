# MiniSeek Web UI — 架构可视化页设计

日期：2026-10-26  
状态：待用户审阅后进入 implementation plan

## 目标

在 MiniSeek Task 控制台增加顶栏页签 **「架构」**：用户选择训练产出的 `.pt`（或 HF 导出目录 / `pytorch_model.bin`），查看该 checkpoint 的**可交互结构图**；点击某一层可展开 Attention / FFN 细节；若为 MoE，再展示 shared / routed 专家与 top-k。

## 非目标（首版不做）

- 前向激活、注意力热力图、logit lens
- 3D / Token-Print 式生成过程动画
- 导出 ONNX 或内嵌 Netron
- MoE 实时路由动画（训练页已有专家负载即可）

## 用户流程

1. 打开「架构」页签。
2. 左侧：浏览 / 粘贴 checkpoint 路径（复用现有 `/api/pick-file`），可选刷新全局 ckpt 列表。
3. 点击「解析」→ `POST /api/arch/inspect`。
4. 右侧主区渲染纵列结构图；顶栏摘要显示 arch、关键超参、总参数、step/loss（若有）。
5. 点击某一 `Layer i` → 右侧抽屉展示该层模块与参数量；再点可切换层；点空白或关闭收回抽屉。

## API

### `POST /api/arch/inspect`

请求：

```json
{ "ckpt_path": "D:\\miniseek\\checkpoints\\hf\\MiniSeek-D0.1-base.pt" }
```

行为：

- 路径不存在 → 404。
- 支持与推理加载相同的格式归一化：
  - 训练 blob：`{ "config", "model", ... }`
  - HF：目录或 `pytorch_model.bin` + 旁路 `config.json`
- **不**将整模搬到 GPU；只读 config + 各权重 `numel`（`torch.load` map_location=cpu）。
- 按 key 前缀聚合参数量：`tok_emb`、`layers.{i}.*`（再拆 `attn` / `ffn` 或 MoE experts）、`norm`、`lm_head`（tie 时注明共享、不重复计入总参或按惯例只计一份并标注）。

响应（示意）：

```json
{
  "path": "...",
  "config": { "arch": "dense", "d_model": 512, "n_layers": 8, "...": "..." },
  "meta": { "step": 158035, "loss": 0.77, "source": "optional" },
  "total_params": 28451328,
  "tie_word_embeddings": true,
  "blocks": [
    { "id": "embed", "kind": "embedding", "label": "Token Embedding", "params": 3276800 },
    {
      "id": "layer-0",
      "kind": "layer",
      "index": 0,
      "label": "Layer 0",
      "params": ...,
      "parts": [
        { "id": "attn", "label": "Attention", "params": ..., "detail": { "n_heads": 8, "head_dim": 64 } },
        { "id": "ffn", "label": "FFN", "params": ..., "detail": { "intermediate_size": 2048 } }
      ]
    }
  ],
  "tail": [
    { "id": "norm", "kind": "norm", "label": "Final RMSNorm", "params": ... },
    { "id": "lm_head", "kind": "lm_head", "label": "LM Head", "params": ..., "tied": true }
  ]
}
```

MoE 时 `parts` 增加 router / shared experts / routed experts（含 `n_routed_experts`、`num_experts_per_tok`、每专家 params）。

## UI

- 顶栏页签顺序：`训练 | 对话 | 续写 | 架构`（四列单行，适当加宽 `page-tabs`）。
- 布局：左栏选文件 + 摘要；中部纵列 SVG/HTML 框图；右侧抽屉（默认收起）。
- 视觉风格对齐现有控制台（卡片、圆角、muted 提示）；层块可 hover/选中高亮。
- 错误：坏文件、缺 config、无法解析 keys → 页内红色提示，不弹系统 alert（可选）。

## 实现要点

- 解析逻辑放在 `web/arch_inspect.py`（纯函数，便于单测）；`web_ui.py` 只挂路由。
- 单测：用 tiny 或合成 state_dict / 现有 `MiniSeek-D0.1-base` 转换后的 `.pt` 断言层数、总参、tie 标注。
- 不依赖新前端框架；原生 JS + CSS，与现有 `index.html` 一致。

## 成功标准

- 选 `MiniSeek-D0.1-base.pt` 能看到 8 层 Dense 结构，总参约 28.45M，点 Layer 可见 Attn/FFN。
- Tiny ckpt：2 层、d=64 正确。
- MoE ckpt（若有）：层内可见专家数量与 top-k；无 MoE 文件时 Dense 路径不受影响。
- 解析大文件时 UI 有「解析中…」；失败信息可读。

## 参考（借鉴交互，不整仓引入）

- Netron：点选节点看细节
- MoE-Visualizer：专家列表展开
- 本仓已有：文件浏览、`load_model` 的 HF/训练 blob 归一化
