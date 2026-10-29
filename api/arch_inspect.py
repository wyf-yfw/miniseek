"""Architecture diagram data from MiniSeek design configs (and optional ckpt inspect)."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from algo.model.config import ModelConfig
from algo.model.v41.config import FlashConfig

_LAYER_RE = re.compile(r"^layers\.(\d+)\.(.+)$")

PRESET_ARCHS = ("tiny", "dense", "v41", "v41-flash")

# Reference-only aliases (DeepSeek-V4.1-Flash paper diagram; not ModelConfig.preset).
_V41_FLASH_ALIASES = frozenset({
    "v41-flash",
    "v4.1-flash",
    "deepseek-v4.1-flash",
    "v41flash",
})


def preset_config(arch: str):
    a = (arch or "").strip().lower()
    if a == "tiny":
        return ModelConfig.tiny()
    if a == "dense":
        return ModelConfig()  # D0.1 defaults
    if a == "v41":
        return FlashConfig()  # MiniSeek-F0.1
    raise ValueError(
        f"未知架构: {arch}（可选: tiny / dense / v41 / v41-flash）"
    )


def _attn_params(d: int) -> int:
    # qkv + o_proj (no bias)
    return 3 * d * d + d * d


def _dense_ffn_params(d: int, intermediate: int) -> int:
    return d * intermediate + intermediate * d


def _expert_ffn_params(d: int, moe_intermediate: int) -> int:
    # gate + up + down，clamp SwiGLU
    return 3 * d * moe_intermediate


def _shape(
    cfg: ModelConfig,
    *,
    vocab: bool = False,
    tokens: bool = False,
) -> str:
    """Activation shapes: batch and seq symbolic (b, t); d/v from ModelConfig."""
    d = int(cfg.d_model)
    v = int(cfg.vocab_size)
    if tokens:
        return "[b,t]"
    if vocab:
        return f"[b,t,{v}]"
    return f"[b,t,{d}]"


def _w(name: str, dims: List[int] | str, *, note: str = "") -> Dict[str, str]:
    """One weight tensor entry for the drawer."""
    if isinstance(dims, str):
        shape = dims
    else:
        shape = "[" + ",".join(str(int(x)) for x in dims) + "]"
    if note:
        shape = f"{shape} {note}"
    return {"name": name, "shape": shape}


def inspect_from_config(cfg: ModelConfig, *, source: str = "design") -> Dict[str, Any]:
    """Build viz JSON from ModelConfig (no checkpoint). Param counts are design formulas."""
    c = cfg.to_dict()
    d = int(cfg.d_model)
    vocab = int(cfg.vocab_size)
    n_layers = int(cfg.n_layers)
    tie = bool(cfg.tie_word_embeddings)
    is_moe = bool(cfg.arch == "moe" or cfg.use_moe)

    embed_params = vocab * d
    bt = _shape(cfg, tokens=True)
    btd = _shape(cfg)
    btv = _shape(cfg, vocab=True)

    blocks: List[Dict[str, Any]] = [
        {
            "id": "embed",
            "kind": "embedding",
            "label": "Token Embedding",
            "params": embed_params,
            "shape_in": bt,
            "shape_out": btd,
            "weight_shapes": [_w("tok_emb.weight", [vocab, d])],
        }
    ]

    attn_p = _attn_params(d)
    attn_norm_p = d
    ffn_norm_p = d

    for i in range(n_layers):
        parts: List[Dict[str, Any]] = [
            {
                "id": "attn_norm",
                "label": "RMSNorm (pre-Attn)",
                "params": attn_norm_p,
                "shape_in": btd,
                "shape_out": btd,
                "weight_shapes": [_w("weight", [d])],
                "detail": {"eps": cfg.rms_norm_eps},
            },
            {
                "id": "attn",
                "label": "Attention",
                "params": attn_p,
                "shape_in": btd,
                "shape_out": btd,
                "weight_shapes": [
                    _w("qkv.weight", [3 * d, d]),
                    _w("o_proj.weight", [d, d]),
                ],
                "detail": {
                    "n_heads": cfg.n_heads,
                    "head_dim": cfg.head_dim,
                    "qkv": f"Linear({d}→{3*d})",
                    "o_proj": f"Linear({d}→{d})",
                },
            },
            {
                "id": "add_attn",
                "label": "⊕",
                "params": 0,
                "kind": "add",
                "shape_in": btd,
                "shape_out": btd,
                "weight_shapes": [],
            },
            {
                "id": "ffn_norm",
                "label": "RMSNorm (pre-FFN)",
                "params": ffn_norm_p,
                "shape_in": btd,
                "shape_out": btd,
                "weight_shapes": [_w("weight", [d])],
                "detail": {"eps": cfg.rms_norm_eps},
            },
        ]
        if is_moe:
            n_shared = int(cfg.n_shared_experts)
            n_routed = int(cfg.n_routed_experts)
            topk = int(cfg.num_experts_per_tok)
            mid = int(cfg.moe_intermediate_size)
            gate_p = n_routed * d
            one_exp = _expert_ffn_params(d, mid)
            shared_p = n_shared * one_exp
            routed_p = n_routed * one_exp
            parts.extend(
                [
                    {
                        "id": "router",
                        "label": "Router / Gate",
                        "params": gate_p,
                        "shape_in": btd,
                        "shape_out": btd,
                        "weight_shapes": [_w("gate.weight", [n_routed, d])],
                        "detail": {
                            "n_routed_experts": n_routed,
                            "num_experts_per_tok": topk,
                        },
                    },
                    {
                        "id": "shared_experts",
                        "label": f"Shared Experts ×{n_shared}",
                        "params": shared_p,
                        "shape_in": btd,
                        "shape_out": btd,
                        "weight_shapes": [
                            _w("gate_proj.weight", [mid, d], note=f"×{n_shared}"),
                            _w("up_proj.weight", [mid, d], note=f"×{n_shared}"),
                            _w("down_proj.weight", [d, mid], note=f"×{n_shared}"),
                        ],
                        "detail": {
                            "n_shared_experts": n_shared,
                            "moe_intermediate_size": mid,
                            "params_per_expert": one_exp,
                        },
                    },
                    {
                        "id": "routed_experts",
                        "label": f"Routed Experts ×{n_routed}",
                        "params": routed_p,
                        "shape_in": btd,
                        "shape_out": btd,
                        "weight_shapes": [
                            _w("gate_proj.weight", [mid, d], note=f"×{n_routed}"),
                            _w("up_proj.weight", [mid, d], note=f"×{n_routed}"),
                            _w("down_proj.weight", [d, mid], note=f"×{n_routed}"),
                        ],
                        "detail": {
                            "n_routed_experts": n_routed,
                            "num_experts_per_tok": topk,
                            "moe_intermediate_size": mid,
                            "params_per_expert": one_exp,
                        },
                    },
                ]
            )
            ffn_block_p = gate_p + shared_p + routed_p
        else:
            mid = int(cfg.intermediate_size)
            ffn_p = _dense_ffn_params(d, mid)
            parts.append(
                {
                    "id": "ffn",
                    "label": "FFN (GELU)",
                    "params": ffn_p,
                    "shape_in": btd,
                    "shape_out": btd,
                    "weight_shapes": [
                        _w("fc1.weight", [mid, d]),
                        _w("fc2.weight", [d, mid]),
                    ],
                    "detail": {
                        "intermediate_size": mid,
                        "fc1": f"Linear({d}→{mid})",
                        "fc2": f"Linear({mid}→{d})",
                    },
                }
            )
            ffn_block_p = ffn_p

        parts.append(
            {
                "id": "add_ffn",
                "label": "⊕",
                "params": 0,
                "kind": "add",
                "shape_in": btd,
                "shape_out": btd,
                "weight_shapes": [],
            }
        )
        layer_params = attn_norm_p + attn_p + ffn_norm_p + ffn_block_p
        blocks.append(
            {
                "id": f"layer-{i}",
                "kind": "layer",
                "index": i,
                "label": f"Layer {i}",
                "params": layer_params,
                "parts": parts,
                "residuals": [
                    {"id": "attn", "from": "layer_in", "to": "add_attn", "label": "x"},
                    {"id": "ffn", "from": "post_attn", "to": "add_ffn", "label": "x"},
                ],
            }
        )

    norm_p = d
    lm_p = vocab * d
    tail = [
        {
            "id": "norm",
            "kind": "norm",
            "label": "Final RMSNorm",
            "params": norm_p,
            "shape_in": btd,
            "shape_out": btd,
            "weight_shapes": [_w("weight", [d])],
        },
        {
            "id": "lm_head",
            "kind": "lm_head",
            "label": "LM Head",
            "params": lm_p,
            "tied": tie,
            "shape_in": btd,
            "shape_out": btv,
            "weight_shapes": [_w("lm_head.weight", [vocab, d], note="(tied)" if tie else "")],
        },
    ]

    total = embed_params + sum(b["params"] for b in blocks if b["kind"] == "layer") + norm_p
    if not tie:
        total += lm_p

    return {
        "path": f"design:{cfg.arch}",
        "config": c,
        "meta": {
            "step": None,
            "loss": None,
            "source": source,
            "release_name": cfg.release_name,
        },
        "total_params": total,
        "tie_word_embeddings": tie,
        "blocks": blocks,
        "tail": tail,
    }


def inspect_design(arch: str) -> Dict[str, Any]:
    a = (arch or "").strip().lower()
    if a in _V41_FLASH_ALIASES:
        return inspect_v41_flash()
    cfg = preset_config(arch)
    if isinstance(cfg, FlashConfig):
        return inspect_v41_model(cfg, source="design")
    return inspect_from_config(cfg, source="design")


# ---------------------------------------------------------------- MiniSeek-F（v41，可训练）----

_MODE_COLORS = {"Full": "#86efac", "Reuse": "#fde047", "Reindex": "#93c5fd", "SWA": "#fda4af"}


def _v41_param_counts(cfg: FlashConfig) -> Tuple[Dict[str, Tuple[int, Tuple[int, ...]]], int]:
    """在 meta device 上构建主干与 DSpark，返回 {参数名: (numel, shape)} 与草稿头总参。"""
    from algo.model.v41.dspark import DSparkDrafter
    from algo.model.v41.model import Transformer

    with torch.device("meta"):
        model = Transformer(cfg)
        drafter = DSparkDrafter(cfg)
    named = {n: (int(p.numel()), tuple(p.shape)) for n, p in model.named_parameters()}
    return named, int(sum(p.numel() for p in drafter.parameters()))


def _sum(named: Dict[str, Tuple[int, Tuple[int, ...]]], prefix: str) -> int:
    return sum(v[0] for k, v in named.items() if k.startswith(prefix))


def _shapes(named, prefix: str, keys: List[str]) -> List[Dict[str, str]]:
    out = []
    for k in keys:
        hit = named.get(prefix + k)
        if hit:
            out.append(_w(k, list(hit[1])))
    return out


def _v41_groups(cfg: FlashConfig, lo: int, hi: int, prefix: str) -> List[Dict[str, Any]]:
    """把 [lo, hi) 层按模式归组（自下而上）：连续 SWA 一组；Full/Reindex 带上其后的 Reuse。

    解码器的 Full 单独成组（渲染器把它画在解码器最底部、接 CED 箭头），其后的 Reuse 另成一组。
    """
    modes = cfg.layer_modes()
    groups: List[Dict[str, Any]] = []
    i = lo

    def moe_row(rid: str) -> Dict[str, Any]:
        return {"id": rid, "type": "moe", "label": "MoE"}

    while i < hi:
        m = modes[i]
        r = cfg.compress_ratios[i]
        if m == "SWA":
            j = i
            while j < hi and modes[j] == "SWA":
                j += 1
            groups.append({
                "id": f"{prefix}_swa",
                "repeat": j - i,
                "kind": "swa",
                "rows": [{"id": f"{prefix}_swa_attn", "type": "swa", "label": "SWA", "mode": "SWA"}, moe_row(f"{prefix}_swa_moe")],
                "note": f"L{i}–L{j - 1} 只有滑窗（窗口 {cfg.window_size}）",
            })
            i = j
            continue
        j = i + 1
        while j < hi and modes[j] == "Reuse":
            j += 1
        n_reuse = j - i - 1
        if m == "Reuse":
            groups.append({
                "id": f"{prefix}_reuse_g{i}",
                "repeat": max(1, n_reuse + 1),
                "kind": "csa2-reuse",
                "rows": [{"id": f"{prefix}_reuse", "type": "csa2-reuse", "label": f"CSA2({r}, Reuse)", "mode": "Reuse", "m": r}, moe_row(f"{prefix}_reuse_moe")],
                "note": f"L{i}–L{j - 1} 复用主 KV 与 top-k",
            })
            i = j
            continue
        rid = f"{prefix}_{'full' if m == 'Full' else 'reindex'}"
        head = {
            "id": f"{rid}_g{i}",
            "repeat": 1,
            "kind": f"csa2-{m.lower()}",
            "rows": [{"id": rid, "type": f"csa2-{m.lower()}", "label": f"CSA2({r}, {m})", "mode": m, "m": r}, moe_row(f"{rid}_moe")],
            "note": f"L{i}",
        }
        decoder_full = prefix == "dec" and m == "Full"
        if n_reuse and not decoder_full:
            head["nested"] = {
                "repeat": n_reuse,
                "rows": [{"id": f"{prefix}_reuse", "type": "csa2-reuse", "label": f"CSA2({r}, Reuse)", "mode": "Reuse", "m": r}, moe_row(f"{prefix}_reuse_moe")],
            }
            head["note"] = f"L{i} {m} + L{i + 1}–L{j - 1} Reuse"
            i = j
        else:
            i += 1
        groups.append(head)
    return groups


def inspect_v41_model(
    cfg: FlashConfig,
    *,
    source: str = "design",
    path: str = "",
    step: Any = None,
    loss: Any = None,
) -> Dict[str, Any]:
    """MiniSeek-F（DeepSeek-V4.1-Flash 全架构的小尺寸版）：Fig.3 风格示意 + 逐层参数。"""
    named, dspark_params = _v41_param_counts(cfg)
    d, V, E = cfg.dim, cfg.vocab_size, cfg.n_encoder_layers
    btd, bt = f"[b,t,{d}]", "[b,t]"
    modes = cfg.layer_modes()

    blocks: List[Dict[str, Any]] = [{
        "id": "embed",
        "kind": "embedding",
        "label": "Token Embedding",
        "params": _sum(named, "embed."),
        "shape_in": bt,
        "shape_out": btd,
        "weight_shapes": _shapes(named, "", ["embed.weight"]),
    }]
    for i in range(cfg.n_layers):
        pre = f"layers.{i}."
        mode = modes[i]
        attn_total = _sum(named, pre + "attn.")
        comp = _sum(named, pre + "attn.compressor.")
        idx = _sum(named, pre + "attn.indexer.")
        parts: List[Dict[str, Any]] = [
            {"id": "hc_attn", "label": "Single-Pass mHC（注意力子层系数）", "params": _sum(named, pre + "hc_attn_"),
             "detail": {"hc_mult": cfg.hc_mult, "sinkhorn_iters": cfg.hc_sinkhorn_iters}},
            {"id": "attn_norm", "label": "RMSNorm (pre-Attn)", "params": _sum(named, pre + "attn_norm.")},
            {"id": "attn", "label": f"{'SWA' if mode == 'SWA' else 'CSA2(' + str(cfg.compress_ratios[i]) + ', ' + mode + ')'} 注意力",
             "params": attn_total - comp - idx,
             "weight_shapes": _shapes(named, pre + "attn.", ["wq_a.weight", "wq_b.weight", "wkv.weight", "wo_a.weight", "wo_b.weight", "attn_sink"]),
             "detail": {"mode": mode, "n_heads": cfg.n_heads, "head_dim(shared KV)": cfg.head_dim,
                        "rope_head_dim": cfg.rope_head_dim, "window": cfg.window_size,
                        "kv_source": cfg.kv_source_for(i), "o_groups": cfg.o_groups}},
        ]
        if comp:
            parts.append({"id": "compressor", "label": f"Compressor (m={cfg.compress_ratios[i]})", "params": comp,
                          "weight_shapes": _shapes(named, pre + "attn.compressor.", ["wkv.weight", "wgate.weight"])})
        if idx:
            parts.append({"id": "indexer", "label": "Indexer", "params": idx,
                          "detail": {"heads": cfg.index_n_heads, "head_dim": cfg.index_head_dim, "topk": cfg.index_topk,
                                     "candidate_source": i == cfg.candidate_source_layer}})
        parts += [
            {"id": "hc_ffn", "label": "Single-Pass mHC（MoE 子层系数）", "params": _sum(named, pre + "hc_ffn_")},
            {"id": "ffn_norm", "label": "RMSNorm (pre-MoE)", "params": _sum(named, pre + "ffn_norm.")},
            {"id": "router", "label": "Router (sqrt-softplus + bias)", "params": _sum(named, pre + "ffn.gate."),
             "detail": {"n_routed_experts": cfg.n_routed_experts, "top_k": cfg.n_activated_experts, "route_scale": cfg.route_scale}},
            {"id": "shared_experts", "label": "Shared Expert ×1", "params": _sum(named, pre + "ffn.shared_experts."),
             "detail": {"inter_dim": cfg.moe_inter_dim, "swiglu_limit": cfg.swiglu_limit}},
            {"id": "routed_experts", "label": f"Routed Experts ×{cfg.n_routed_experts}", "params": _sum(named, pre + "ffn.experts.")},
        ]
        eng = _sum(named, pre + "engram.")
        if eng:
            parts.append({"id": "engram", "label": "Engram（block 之前写入 4 条流）", "params": eng,
                          "detail": {"ngram": f"2..{cfg.engram_max_ngram_size}", "heads": cfg.engram_n_heads, "head_dim": cfg.engram_head_dim}})
        blocks.append({
            "id": f"layer-{i}",
            "kind": "layer",
            "index": i,
            "label": f"L{i} · {mode}" + (" · Engram" if eng else ""),
            "params": _sum(named, pre),
            "parts": parts,
            "detail": {"section": "decoder" if cfg.is_decoder(i) else "encoder", "mode": mode,
                       "compress_ratio": cfg.compress_ratios[i]},
        })

    tail = [
        {"id": "norm", "kind": "norm", "label": "Final RMSNorm", "params": _sum(named, "norm."), "shape_in": btd, "shape_out": btd},
        {"id": "lm_head", "kind": "lm_head", "label": "LM Head（不与 embedding 共享）", "params": _sum(named, "head."), "tied": False,
         "shape_in": btd, "shape_out": f"[b,t,{V}]", "weight_shapes": _shapes(named, "", ["head.weight"])},
        {"id": "dspark", "kind": "dspark", "label": "DSpark 草稿头（另训，不计入主干）", "params": dspark_params,
         "detail": {"blocks": cfg.n_mtp_layers, "gamma": cfg.dspark_block_size, "markov_rank": cfg.dspark_markov_rank,
                    "experts": f"1+{cfg.dspark_n_routed_experts} top-{cfg.dspark_n_activated_experts}"}},
    ]
    total = sum(v[0] for v in named.values())
    eng_layers = " · ".join(f"@L{i}" for i in cfg.engram_layer_ids)
    reuse_target = next((i for i in cfg.engram_layer_ids if modes[i] == "Reuse"), None)
    fig3 = {
        "title": f"{cfg.release_name} | DeepSeek-V4.1-Flash 架构（{cfg.n_layers} 层：编码 {E} + 解码 {cfg.n_layers - E}）",
        "encoder": {"id": "causal_encoder", "label": f"Causal Encoder ({E}L)", "n_layers": E,
                    "groups": _v41_groups(cfg, 0, E, "enc"), "output": {"id": "enc_hidden", "label": "Enc. Hidden States"}},
        "decoder": {"id": "decoder", "label": f"Decoder ({cfg.n_layers - E}L)", "n_layers": cfg.n_layers - E,
                    "groups": _v41_groups(cfg, E, cfg.n_layers, "dec"), "output": {"id": "dspark", "label": "DSpark"}},
        "csa2_modes": [
            {"mode": "Full", "color": _MODE_COLORS["Full"], "kv": "本层压缩器产出主 KV、indexer K 与 top-k（编码器 m=2，解码器 m=1）"},
            {"mode": "Reuse", "color": _MODE_COLORS["Reuse"], "kv": "复用最近来源层的主 KV 与 top-k，不跑 indexer"},
            {"mode": "Reindex", "color": _MODE_COLORS["Reindex"], "kv": "复用主 KV 与 indexer K，在候选池里重选 top-k"},
            {"mode": "SWA", "color": _MODE_COLORS["SWA"], "kv": f"只有滑窗（窗口 {cfg.window_size}）；其余层也都带本层滑窗"},
        ],
        "kv_flows": [
            {"id": "ced", "label": "CED", "from": "enc_hidden", "to": "dec_full", "style": "ced",
             "note": f"编码器输出只在 L{E}（解码器 Full）投影成全局 KV；之后的解码器层复用。每层仍有自己的 Q 与滑窗 KV。"},
            {"id": "indexer", "label": "Hierarchical Sparse Indexer", "from": "dec_full", "to": "candidate_pool", "style": "indexer",
             "note": f"L{cfg.candidate_source_layer} 按 {cfg.candidate_block_size} 个条目一块取最大分，保留 {cfg.candidate_topk_blocks} 块（后训练启用）。"},
            {"id": "reindex", "label": "Reindex Top-K", "from": "candidate_pool", "to": "dec_reindex", "style": "reindex",
             "note": f"Reindex 层只在候选池里重选 top-{cfg.index_topk}。"},
            {"id": "engram_feed", "label": "Engram", "from": "engram", "to": "enc_swa_attn", "style": "engram",
             "note": f"Engram 写入 {eng_layers} 之前的 {cfg.hc_mult} 条残差流。"},
        ],
        "engram_sub": eng_layers,
        "engram_targets": ["enc_swa_attn", "enc_reuse" if reuse_target is not None else "enc_full"],
        "dspark_sub": f"投机解码草稿头：{cfg.n_mtp_layers} 个 block，另训，不计入 {cfg.n_layers} 层",
        "show_vision": False,
        "notes_list": [
            "自下而上计算。注意力在下，MoE 在上；括号右侧 ×N 是这一括号重复的次数。",
            f"CED：编码器输出 H_{{L/2}} 只在解码器最底下的 CSA2(1, Full)（L{E}）投影成全局 KV，之后的 Reuse / Reindex 复用；每层仍有自己的 Q 与本层滑窗 KV。",
            f"候选池：L{cfg.candidate_source_layer} 建池（{cfg.candidate_topk_blocks} 块 × {cfg.candidate_block_size}），Reindex 只在池内重选 top-{cfg.index_topk}；论文在后训练才开，SFT 默认启用。",
            f"Engram 在 {eng_layers} 的 block 之前按流门控写入；Single-Pass mHC 为 {cfg.hc_mult} 条残差流、Sinkhorn {cfg.hc_sinkhorn_iters} 次。",
            "与论文的有意差异：KV / indexer 全部 BF16（论文主 KV 为 FP4、滑窗 KV 为 FP8，3090 不支持）；没有视觉分支；尺寸缩小。",
        ],
    }
    config = cfg.to_dict()
    meta = {"step": step, "loss": loss, "source": source, "diagram": "fig3", "release_name": cfg.release_name,
            "trainable": True, "dspark_params": dspark_params}
    return {
        "path": path or f"design:{cfg.arch}",
        "config": config,
        "meta": meta,
        "fig3": fig3,
        "kv_flows": fig3["kv_flows"],
        "sections": {"encoder": fig3["encoder"], "decoder": fig3["decoder"]},
        "total_params": total,
        "tie_word_embeddings": False,
        "blocks": blocks,
        "tail": tail,
    }


def inspect_v41_flash() -> Dict[str, Any]:
    """DeepSeek-V4.1-Flash reference architecture (arXiv:2609.19969 §2.1 / Fig.3 / §4.2.1).

    Reference-only design diagram — not a trainable MiniSeek ModelConfig.
    Layer blocks are *group-level* (matching Fig.3 readability), not 40 expanded stacks.
    """
    d = 5120
    n_layers = 40  # CED: 20 causal encoder + 20 decoder
    n_heads = 64
    head_dim = 512
    btd = f"[b,t,{d}]"
    bt = "[b,t]"

    # Approx expert FFN (SwiGLU gate+up+down): 3 * d * intermediate
    moe_inter = 2304
    n_shared, n_routed, topk = 1, 384, 6
    one_exp = 3 * d * moe_inter
    moe_per_layer = n_shared * one_exp + n_routed * one_exp  # ~13.6B illustrative
    backbone = 552_000_000_000
    engram_total = 196_000_000_000

    def moe_parts(layer_note: str = "") -> List[Dict[str, Any]]:
        detail = {
            "n_shared_experts": n_shared,
            "n_routed_experts": n_routed,
            "num_experts_per_tok": topk,
            "moe_intermediate_size": moe_inter,
            "activation": "SwiGLU (clamp 10)",
            "note": "DeepSeekMoE every Transformer layer",
        }
        if layer_note:
            detail["layer_note"] = layer_note
        return [
            {
                "id": "moe",
                "label": f"DeepSeekMoE 1+{n_routed} top-{topk}",
                "params": moe_per_layer,
                "shape_in": btd,
                "shape_out": btd,
                "weight_shapes": [
                    _w("gate.weight", [n_routed, d]),
                    _w("expert.w_gate/up/down", [moe_inter, d], note=f"×{n_shared + n_routed} SwiGLU"),
                ],
                "detail": detail,
            }
        ]

    def mhc_part() -> Dict[str, Any]:
        return {
            "id": "mhc",
            "label": "Single-Pass mHC (×4)",
            "params": 0,
            "shape_in": btd,
            "shape_out": btd,
            "weight_shapes": [],
            "detail": {
                "variant": "Single-Pass mHC",
                "expansion_factor": 4,
                "note": "manifold-constrained hyper-connections",
            },
        }

    def attn_swa(*, layers: str) -> Dict[str, Any]:
        return {
            "id": "attn",
            "label": f"SWA only (n_win=128) · {layers}",
            "params": 0,
            "shape_in": btd,
            "shape_out": btd,
            "weight_shapes": [],
            "detail": {
                "attn": "SWA",
                "n_win": 128,
                "layers": layers,
                "n_heads": n_heads,
                "head_dim": head_dim,
                "kv_dtype": "FP8 (SWA KV)",
            },
        }

    def attn_csa2(
        *,
        mode: str,
        m: int,
        layers: str,
        pattern: str,
        ced_note: str = "",
    ) -> Dict[str, Any]:
        detail: Dict[str, Any] = {
            "attn": f"CSA2(m={m})",
            "mode": mode,
            "pattern": pattern,
            "layers": layers,
            "n_heads": n_heads,
            "head_dim": head_dim,
            "query_compression_dim": 1280,
            "out_proj_groups": 8,
            "attn_out_intermediate": 1024,
            "indexer_q_heads": 32,
            "indexer_head_dim": 128,
            "attention_topk": 512,
            "own_global_Q": True,
            "own_swa_KV": True,
            "main_kv_dtype": "FP4 (post-train QAT)",
            "swa_kv_dtype": "FP8",
            "note": "All CSA2 layers keep own global Q + SWA KV; modes differ for main KV / indexer K / Top-K",
        }
        if mode == "Reindex":
            detail["hierarchical_sparse_indexer"] = (
                "First Full builds candidate pool (≤2048 blocks × 8 = 16384); "
                "later Reindex select Top-512 from pool"
            )
        if ced_note:
            detail["CED"] = ced_note
        return {
            "id": "attn",
            "label": f"CSA2(m={m}) {mode} · {pattern}",
            "params": 0,
            "shape_in": btd,
            "shape_out": btd,
            "weight_shapes": [],
            "detail": detail,
        }

    def engram_part(layer_idx: int) -> Dict[str, Any]:
        return {
            "id": "engram",
            "label": f"Engram @{layer_idx}",
            "params": engram_total // 2,  # 196B total at L1 and L14
            "shape_in": btd,
            "shape_out": btd,
            "weight_shapes": [
                _w("engram.emb", "N-gram{2,3,4} × 8 hash heads × dim 2048/order"),
            ],
            "detail": {
                "layers_0indexed": [1, 14],
                "this_layer": layer_idx,
                "total_params": engram_total,
                "n_gram_orders": [2, 3, 4],
                "hash_heads": 8,
                "emb_dim_per_order": 2048,
            },
        }

    def layer_group(
        *,
        gid: str,
        label: str,
        index: int,
        n_in_group: int,
        parts: List[Dict[str, Any]],
        detail: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        # Illustrative share of backbone (not exact FLOPs / param accounting).
        share = int(backbone * n_in_group / n_layers)
        # Add Engram share into displayed params when present
        eng = sum(int(p.get("params") or 0) for p in parts if p.get("id") == "engram")
        blk: Dict[str, Any] = {
            "id": gid,
            "kind": "layer",
            "index": index,
            "label": label,
            "params": share + eng,
            "parts": parts,
            "residuals": [
                {"id": "attn", "from": "layer_in", "to": "add_attn", "label": "x"},
                {"id": "ffn", "from": "post_attn", "to": "add_ffn", "label": "x"},
            ],
        }
        if detail:
            blk["detail"] = detail
        return blk

    # --- early blocks: text embed + vision ---
    blocks: List[Dict[str, Any]] = [
        {
            "id": "embed",
            "kind": "embedding",
            "label": "Token Embedding",
            "params": 0,
            "shape_in": bt,
            "shape_out": btd,
            "weight_shapes": [_w("tok_emb.weight", f"[vocab,{d}]")],
            "detail": {"d_model": d, "note": "Multimodal MoE Transformer backbone"},
        },
        {
            "id": "vision",
            "kind": "vision",
            "label": "DeepSeek-ViT → Projector",
            "params": 0,
            "shape_in": "[b,H,W,3]",
            "shape_out": btd,
            "weight_shapes": [
                _w("vit", "32L d=1024 heads=16 patch=14 2D-RoPE RMSNorm SwiGLU"),
                _w("pixel_unshuffle", "3×3"),
                _w("mlp_projector", f"2-layer MLP hidden={d}"),
            ],
            "detail": {
                "vit_layers": 32,
                "vit_d_model": 1024,
                "vit_heads": 16,
                "patch_size": 14,
                "rope": "2D-RoPE",
                "norm": "RMSNorm",
                "ffn": "SwiGLU",
                "pixel_unshuffle": "3×3",
                "projector": f"2-layer MLP hidden {d} → visual embeddings into sequence",
            },
        },
    ]

    # L0–1: SWA only; Engram @1
    parts_01: List[Dict[str, Any]] = [
        {
            "id": "attn_norm",
            "label": "RMSNorm (pre-Attn)",
            "params": d,
            "shape_in": btd,
            "shape_out": btd,
            "weight_shapes": [_w("weight", [d])],
        },
        attn_swa(layers="L0–1"),
        {
            "id": "add_attn",
            "label": "⊕",
            "params": 0,
            "kind": "add",
            "shape_in": btd,
            "shape_out": btd,
            "weight_shapes": [],
        },
        {
            "id": "ffn_norm",
            "label": "RMSNorm (pre-MoE)",
            "params": d,
            "shape_in": btd,
            "shape_out": btd,
            "weight_shapes": [_w("weight", [d])],
        },
        *moe_parts("L0–1"),
        mhc_part(),
        engram_part(1),
        {
            "id": "add_ffn",
            "label": "⊕",
            "params": 0,
            "kind": "add",
            "shape_in": btd,
            "shape_out": btd,
            "weight_shapes": [],
        },
    ]
    blocks.append(
        layer_group(
            gid="layer-swa-0-1",
            label="L0–1 · SWA (Engram@1)",
            index=0,
            n_in_group=2,
            parts=parts_01,
            detail={"section": "encoder-early", "attn": "SWA", "engram": 1},
        )
    )

    ced_kv = "CED: decoder global KV projected from encoder final hidden H_{L/2}=H_20 (not each decoder H_l); SWA still layer-local"

    def csa2_group(
        *,
        gid: str,
        label: str,
        index: int,
        layers: str,
        mode: str,
        m: int,
        pattern: str,
        n_in_group: int,
        engram_at: Optional[int] = None,
        is_decoder: bool = False,
    ) -> Dict[str, Any]:
        parts: List[Dict[str, Any]] = [
            {
                "id": "attn_norm",
                "label": "RMSNorm (pre-Attn)",
                "params": d,
                "shape_in": btd,
                "shape_out": btd,
                "weight_shapes": [_w("weight", [d])],
            },
            attn_csa2(
                mode=mode,
                m=m,
                layers=layers,
                pattern=pattern,
                ced_note=ced_kv if is_decoder else "",
            ),
            {
                "id": "add_attn",
                "label": "⊕",
                "params": 0,
                "kind": "add",
                "shape_in": btd,
                "shape_out": btd,
                "weight_shapes": [],
            },
            {
                "id": "ffn_norm",
                "label": "RMSNorm (pre-MoE)",
                "params": d,
                "shape_in": btd,
                "shape_out": btd,
                "weight_shapes": [_w("weight", [d])],
            },
            *moe_parts(layers),
            mhc_part(),
        ]
        if engram_at is not None:
            parts.append(engram_part(engram_at))
        parts.append(
            {
                "id": "add_ffn",
                "label": "⊕",
                "params": 0,
                "kind": "add",
                "shape_in": btd,
                "shape_out": btd,
                "weight_shapes": [],
            }
        )
        detail: Dict[str, Any] = {
            "section": "decoder" if is_decoder else "encoder",
            "csa2_m": m,
            "mode": mode,
            "pattern": pattern,
            "layers": layers,
        }
        if is_decoder:
            detail["CED"] = ced_kv
        if engram_at is not None:
            detail["engram"] = engram_at
        return layer_group(
            gid=gid,
            label=label,
            index=index,
            n_in_group=n_in_group,
            parts=parts,
            detail=detail,
        )

    # Encoder L2–19: CSA2 m=2, three groups of 6: Full + Reuse×5
    blocks.append(
        csa2_group(
            gid="layer-enc-2-7",
            label="L2–7 · Enc CSA2(m=2) Full+Reuse×5",
            index=2,
            layers="L2–7",
            mode="Full",
            m=2,
            pattern="Full + Reuse×5",
            n_in_group=6,
        )
    )
    blocks.append(
        csa2_group(
            gid="layer-enc-8-13",
            label="L8–13 · Enc CSA2(m=2) Full+Reuse×5",
            index=8,
            layers="L8–13",
            mode="Full",
            m=2,
            pattern="Full + Reuse×5",
            n_in_group=6,
        )
    )
    blocks.append(
        csa2_group(
            gid="layer-enc-14-19",
            label="L14–19 · Enc CSA2(m=2) Full+Reuse×5 (Engram@14)",
            index=14,
            layers="L14–19",
            mode="Full",
            m=2,
            pattern="Full + Reuse×5",
            n_in_group=6,
            engram_at=14,
        )
    )

    # Decoder L20–39: CSA2 m=1, five groups of 4
    blocks.append(
        csa2_group(
            gid="layer-dec-20-23",
            label="L20–23 · Dec CSA2(m=1) Full+Reuse×3",
            index=20,
            layers="L20–23",
            mode="Full",
            m=1,
            pattern="Full + Reuse×3",
            n_in_group=4,
            is_decoder=True,
        )
    )
    for start, gid_suffix in ((24, "24-27"), (28, "28-31"), (32, "32-35"), (36, "36-39")):
        end = start + 3
        blocks.append(
            csa2_group(
                gid=f"layer-dec-{gid_suffix}",
                label=f"L{start}–{end} · Dec CSA2(m=1) Reindex+Reuse×3",
                index=start,
                layers=f"L{start}–{end}",
                mode="Reindex",
                m=1,
                pattern="Reindex + Reuse×3",
                n_in_group=4,
                is_decoder=True,
            )
        )

    tail: List[Dict[str, Any]] = [
        {
            "id": "norm",
            "kind": "norm",
            "label": "Final RMSNorm",
            "params": d,
            "shape_in": btd,
            "shape_out": btd,
            "weight_shapes": [_w("weight", [d])],
        },
        {
            "id": "lm_head",
            "kind": "lm_head",
            "label": "LM Head",
            "params": 0,
            "tied": False,
            "shape_in": btd,
            "shape_out": "[b,t,vocab]",
            "weight_shapes": [_w("lm_head.weight", f"[vocab,{d}]")],
        },
        {
            "id": "dspark",
            "kind": "dspark",
            "label": "DSpark (post-hoc)",
            "params": 0,
            "shape_in": btd,
            "shape_out": btd,
            "weight_shapes": [],
            "detail": {
                "blocks": 3,
                "attn": "SWA 128",
                "draft": "semi-AR draft + confidence-scheduled verify",
                "note": "Post-hoc speculative decoding; not in backbone pretrain",
            },
        },
    ]

    config: Dict[str, Any] = {
        "arch": "deepseek-v4.1-flash",
        "release_name": "DeepSeek-V4.1-Flash",
        "d_model": d,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "head_dim": head_dim,
        "use_moe": True,
        "n_shared_experts": n_shared,
        "n_routed_experts": n_routed,
        "num_experts_per_tok": topk,
        "moe_intermediate_size": moe_inter,
        "ced": "20 causal encoder + 20 decoder",
        "csa2": "encoder m=2 (Full+Reuse×5 ×3); decoder m=1 (Full+Reuse×3, then Reindex+Reuse×3 ×4)",
        "swa_early": "L0–1 SWA n_win=128",
        "engram_layers": [1, 14],
        "mhc": "Single-Pass mHC expansion=4",
        "context_max": 1_000_000,
        "act_prefill_params": 8_000_000_000,
        "act_decode_params": 16_000_000_000,
        "backbone_params": backbone,
        "engram_params": engram_total,
        "note": "Reference diagram only (arXiv:2609.19969); not a MiniSeek trainable preset",
    }

    meta: Dict[str, Any] = {
        "step": None,
        "loss": None,
        "source": "reference",
        "diagram": "fig3",
        "release_name": "DeepSeek-V4.1-Flash",
        "paper": "arXiv:2609.19969",
        "backbone_params": backbone,
        "act_prefill": 8_000_000_000,
        "act_decode": 16_000_000_000,
        "engram_params": engram_total,
    }

    # Dedicated Fig.3 schematic payload (frontend renders this instead of flat cards).
    fig3: Dict[str, Any] = {
        "title": "Figure 3 | Overall architecture of DeepSeek-V4.1-Flash",
        "caption": (
            "40-layer Multimodal MoE Transformer = Causal Encoder 20L + Decoder 20L (CED). "
            "All FFN = DeepSeekMoE. L0–1 SWA; remaining layers CSA2. "
            "CSA2(ratio, mode): compression ratio + Full / Reuse / Reindex. "
            "Includes Single-Pass mHC, Engram, DSpark, Hierarchical Sparse Indexer."
        ),
        "act_note": "Activated params ≈ 8B prefill / 16B decode because CED (decoder mostly skipped in prefill).",
        "inputs": [
            {"id": "vision_enc", "label": "Vision Encoder"},
            {"id": "vision_emb", "label": "Vision Embedding"},
            {"id": "text_emb", "label": "Text Embedding"},
            {"id": "mhc", "label": "Single-Pass mHC"},
            {"id": "embedding", "label": "Embedding"},
            {"id": "engram", "label": "Engram @L1 & @L14"},
        ],
        "encoder": {
            "id": "causal_encoder",
            "label": "Causal Encoder (20L)",
            "n_layers": 20,
            "groups": [
                {
                    "id": "enc_swa",
                    "repeat": 2,
                    "kind": "swa",
                    "rows": [
                        {"id": "enc_swa_attn", "type": "swa", "label": "SWA", "mode": "SWA"},
                        {"id": "enc_swa_moe", "type": "moe", "label": "MoE"},
                    ],
                    "note": "L0–L1 · 窗口 128 · Engram@L1",
                },
                {
                    "id": "enc_csa2",
                    "repeat": 3,
                    "kind": "csa2-group",
                    "rows": [
                        {"id": "enc_full", "type": "csa2-full", "label": "CSA2(2, Full)", "mode": "Full", "m": 2},
                        {"id": "enc_full_moe", "type": "moe", "label": "MoE"},
                    ],
                    "nested": {
                        "repeat": 5,
                        "rows": [
                            {"id": "enc_reuse", "type": "csa2-reuse", "label": "CSA2(2, Reuse)", "mode": "Reuse", "m": 2},
                            {"id": "enc_reuse_moe", "type": "moe", "label": "MoE"},
                        ],
                    },
                    "note": "L2–L19 · 三组 Full + Reuse×5 · Engram@L14",
                },
            ],
            "output": {"id": "enc_hidden", "label": "Enc. Hidden States"},
        },
        "decoder": {
            "id": "decoder",
            "label": "Decoder (20L)",
            "n_layers": 20,
            "groups": [
                {
                    "id": "dec_full_g",
                    "repeat": 1,
                    "kind": "csa2-full",
                    "rows": [
                        {"id": "dec_full", "type": "csa2-full", "label": "CSA2(1, Full)", "mode": "Full", "m": 1},
                        {"id": "dec_full_moe", "type": "moe", "label": "MoE"},
                    ],
                    "note": "L20 · 产出 main KV、indexer K、Top-K 和候选池",
                },
                {
                    "id": "dec_reuse_early",
                    "repeat": 3,
                    "kind": "csa2-reuse",
                    "rows": [
                        {"id": "dec_reuse_e", "type": "csa2-reuse", "label": "CSA2(1, Reuse)", "mode": "Reuse", "m": 1},
                        {"id": "dec_reuse_e_moe", "type": "moe", "label": "MoE"},
                    ],
                    "note": "L21–L23 · 复用 KV 和已有 Top-K",
                },
                {
                    "id": "dec_reindex_g",
                    "repeat": 4,
                    "kind": "csa2-reindex-group",
                    "rows": [
                        {"id": "dec_reindex", "type": "csa2-reindex", "label": "CSA2(1, Reindex)", "mode": "Reindex", "m": 1},
                        {"id": "dec_reindex_moe", "type": "moe", "label": "MoE"},
                    ],
                    "nested": {
                        "repeat": 3,
                        "rows": [
                            {"id": "dec_reuse_r", "type": "csa2-reuse", "label": "CSA2(1, Reuse)", "mode": "Reuse", "m": 1},
                            {"id": "dec_reuse_r_moe", "type": "moe", "label": "MoE"},
                        ],
                    },
                    "note": "L24–L39 · 四组 Reindex + Reuse×3",
                },
            ],
            "output": {"id": "dspark", "label": "DSpark"},
        },
        "side": [
            {"id": "candidate_pool", "label": "Candidate Pool"},
        ],
        "kv_flows": [
            {
                "id": "ced",
                "label": "CED",
                "from": "enc_hidden",
                "to": "dec_full",
                "style": "ced",
                "note": "解码器全局 KV 只由编码器最终隐状态 H_L/2 投影到最底 CSA2(1, Full)。Reuse 和 Reindex 复用这份 KV。每层仍有自己的 global Q 和 SWA KV。",
            },
            {
                "id": "indexer",
                "label": "Hierarchical Sparse Indexer",
                "from": "dec_full",
                "to": "candidate_pool",
                "style": "indexer",
                "note": "Full 建立候选池（最多 2048 块 × 8 = 16384 token）。",
            },
            {
                "id": "reindex",
                "label": "Reindex Top-K",
                "from": "candidate_pool",
                "to": "dec_reindex",
                "style": "reindex",
                "note": "Reindex 复用 main KV 与 indexer K，只在候选池里重选 Top-512。",
            },
            {
                "id": "engram_feed",
                "label": "Engram",
                "from": "engram",
                "to": "enc_swa_attn",
                "style": "engram",
                "note": "Engram 打进第 1 层 SWA（层号从 0 起）。",
            },
            {
                "id": "engram_feed_2",
                "label": "Engram",
                "from": "engram",
                "to": "enc_full",
                "style": "engram",
                "note": "Engram @L14（0-indexed）打进编码器第三组的 CSA2 Full，不是每一组 Full。",
            },
        ],
        "csa2_modes": [
            {
                "mode": "Full",
                "color": "#86efac",
                "kv": "本层产出 main KV、indexer K 和 Top-K",
            },
            {
                "mode": "Reuse",
                "color": "#fde047",
                "kv": "复用最近一次 Full/Reindex 的 main KV 与 Top-K，不再索引",
            },
            {
                "mode": "Reindex",
                "color": "#93c5fd",
                "kv": "复用 main KV 与 indexer K，只在候选池里重选 Top-K",
            },
            {
                "mode": "SWA",
                "color": "#fda4af",
                "kv": "仅 L0–L1 是纯滑窗。其余层仍有本层 SWA KV（n_win=128）",
            },
        ],
        "notes": [
            "CED：解码器全局 KV 来自编码器最终隐状态，不是每一层解码器隐状态。",
            "每个 CSA2 层仍有自己的 global Q 和本层 SWA KV。",
            "只有 L0–L1 是纯滑窗；其余层是 CSA2，同时还有窗口 128 的本层滑窗。",
            "预填充激活约 8B（编码器为主），解码约 16B。",
            "论文参考图，不是 MiniSeek 可训练预设。",
        ],
    }

    return {
        "path": "reference:deepseek-v4.1-flash",
        "config": config,
        "meta": meta,
        "fig3": fig3,
        "kv_flows": fig3["kv_flows"],
        "sections": {
            "encoder": fig3["encoder"],
            "decoder": fig3["decoder"],
        },
        "total_params": backbone,
        "tie_word_embeddings": False,
        "blocks": blocks,
        "tail": tail,
    }


def load_ckpt_blob(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))

    weight_path: Optional[Path] = None
    cfg_path: Optional[Path] = None
    if path.is_dir() and (path / "pytorch_model.bin").exists():
        weight_path = path / "pytorch_model.bin"
        cfg_path = path / "config.json"
    elif path.name == "pytorch_model.bin":
        weight_path = path
        cfg_path = path.parent / "config.json"

    if weight_path is not None:
        sd = torch.load(weight_path, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "model" in sd:
            blob = sd
            blob.setdefault("source", "hf-blob")
        else:
            cfg_raw: Dict[str, Any] = {}
            if cfg_path and cfg_path.exists():
                cfg_raw = json.loads(cfg_path.read_text(encoding="utf-8"))
            allowed = set(ModelConfig.__dataclass_fields__)  # type: ignore[attr-defined]
            cfg_dict = {k: v for k, v in cfg_raw.items() if k in allowed}
            blob = {
                "model": sd,
                "config": cfg_dict,
                "step": cfg_raw.get("train_step", cfg_raw.get("step")),
                "loss": cfg_raw.get("train_loss", cfg_raw.get("loss")),
                "source": "hf",
            }
        return blob

    blob = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(blob, dict) or "model" not in blob:
        raise ValueError(f"无法识别的 checkpoint 格式: {path}")
    blob.setdefault("source", "training")
    return blob


def _numel(t: torch.Tensor) -> int:
    return int(t.numel())


def _sum_prefix(sd: Dict[str, torch.Tensor], prefix: str) -> int:
    return sum(_numel(v) for k, v in sd.items() if k.startswith(prefix))


def inspect_blob(blob: Dict[str, Any], *, path: str = "") -> Dict[str, Any]:
    if "model" not in blob:
        raise ValueError("checkpoint 缺少 model 权重")
    sd: Dict[str, torch.Tensor] = blob["model"]
    if not isinstance(sd, dict):
        raise ValueError("model 不是 state_dict")

    cfg_raw = blob.get("config") or {}
    if not isinstance(cfg_raw, dict):
        cfg_raw = {}
    if cfg_raw.get("arch") == "v41":
        return inspect_v41_model(
            FlashConfig.from_dict(cfg_raw),
            source=str(blob.get("source") or "training"),
            path=path,
            step=blob.get("step"),
            loss=blob.get("loss"),
        )
    try:
        cfg = ModelConfig.from_dict(cfg_raw)
        cfg_out = cfg.to_dict()
    except Exception:
        cfg_out = dict(cfg_raw)

    tie = bool(cfg_out.get("tie_word_embeddings", True))
    if "tok_emb.weight" in sd and "lm_head.weight" in sd:
        try:
            if sd["tok_emb.weight"].data_ptr() == sd["lm_head.weight"].data_ptr():
                tie = True
        except Exception:
            pass

    n_layers = int(cfg_out.get("n_layers") or 0)
    if not n_layers:
        idxs = []
        for k in sd:
            m = _LAYER_RE.match(k)
            if m:
                idxs.append(int(m.group(1)))
        n_layers = (max(idxs) + 1) if idxs else 0

    blocks: List[Dict[str, Any]] = []
    embed_params = _sum_prefix(sd, "tok_emb.")
    blocks.append(
        {
            "id": "embed",
            "kind": "embedding",
            "label": "Token Embedding",
            "params": embed_params,
        }
    )

    is_moe = bool(
        (cfg_out.get("arch") == "moe")
        or cfg_out.get("use_moe")
        or any(".moe.experts." in k or ".moe.shared." in k for k in sd)
    )

    for i in range(n_layers):
        pref = f"layers.{i}."
        layer_keys = {k: v for k, v in sd.items() if k.startswith(pref)}
        attn_params = sum(
            _numel(v)
            for k, v in layer_keys.items()
            if k.startswith(pref + "attn")
        )
        parts: List[Dict[str, Any]] = [
            {
                "id": "attn",
                "label": "Attention",
                "params": attn_params,
                "detail": {
                    "n_heads": cfg_out.get("n_heads"),
                    "head_dim": cfg_out.get("head_dim"),
                },
            }
        ]
        if is_moe:
            gate_params = _sum_prefix(sd, pref + "moe.router.")
            shared_params = _sum_prefix(sd, pref + "moe.shared.")
            routed_params = _sum_prefix(sd, pref + "moe.experts.")
            ffn_norm = _sum_prefix(sd, pref + "ffn_norm.")
            parts.append(
                {
                    "id": "router",
                    "label": "Router / Gate",
                    "params": gate_params + ffn_norm,
                    "detail": {
                        "num_experts_per_tok": cfg_out.get("num_experts_per_tok"),
                    },
                }
            )
            parts.append(
                {
                    "id": "shared_experts",
                    "label": "Shared Experts",
                    "params": shared_params,
                    "detail": {
                        "n_shared_experts": cfg_out.get("n_shared_experts"),
                        "moe_intermediate_size": cfg_out.get("moe_intermediate_size"),
                    },
                }
            )
            n_routed = int(cfg_out.get("n_routed_experts") or 0)
            per = (routed_params // n_routed) if n_routed else routed_params
            parts.append(
                {
                    "id": "routed_experts",
                    "label": "Routed Experts",
                    "params": routed_params,
                    "detail": {
                        "n_routed_experts": n_routed,
                        "num_experts_per_tok": cfg_out.get("num_experts_per_tok"),
                        "params_per_expert": per,
                        "moe_intermediate_size": cfg_out.get("moe_intermediate_size"),
                    },
                }
            )
        else:
            ffn_params = sum(
                _numel(v)
                for k, v in layer_keys.items()
                if k.startswith(pref + "ffn")
            )
            parts.append(
                {
                    "id": "ffn",
                    "label": "FFN",
                    "params": ffn_params,
                    "detail": {
                        "intermediate_size": cfg_out.get("intermediate_size"),
                    },
                }
            )
        layer_params = sum(_numel(v) for v in layer_keys.values())
        blocks.append(
            {
                "id": f"layer-{i}",
                "kind": "layer",
                "index": i,
                "label": f"Layer {i}",
                "params": layer_params,
                "parts": parts,
            }
        )

    norm_params = _sum_prefix(sd, "norm.")
    lm_params = _sum_prefix(sd, "lm_head.")
    tail = [
        {
            "id": "norm",
            "kind": "norm",
            "label": "Final RMSNorm",
            "params": norm_params,
        },
        {
            "id": "lm_head",
            "kind": "lm_head",
            "label": "LM Head",
            "params": lm_params,
            "tied": tie,
        },
    ]

    total = 0
    seen_ptrs: set[int] = set()
    for k, v in sd.items():
        if tie and k.startswith("lm_head."):
            continue
        try:
            ptr = v.data_ptr()
            if ptr in seen_ptrs:
                continue
            seen_ptrs.add(ptr)
        except Exception:
            pass
        total += _numel(v)

    meta = {
        "step": blob.get("step"),
        "loss": blob.get("loss"),
        "source": blob.get("source"),
    }
    return {
        "path": path,
        "config": cfg_out,
        "meta": meta,
        "total_params": total,
        "tie_word_embeddings": tie,
        "blocks": blocks,
        "tail": tail,
    }


def inspect_path(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    blob = load_ckpt_blob(path)
    return inspect_blob(blob, path=str(path))
