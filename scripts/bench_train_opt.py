#!/usr/bin/env python3
"""训练速度优化：逐项真实收益 A/B 基准。

每一项优化单独相对 base 测量（有效 token/s；padding 不算）。
测量契约见 docs/superpowers/plans/2026-10-27-training-speed-bench.md。

用法：
    python3 scripts/bench_train_opt.py --steps 40 --warmup 20 --runs 2
    python3 scripts/bench_train_opt.py --wait-pid 56159 --out logs/bench_train_opt.json
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from algo.model import build_model
from algo.model.config import ModelConfig
from algo.model.moe import MoELayer, route_tokens, sequence_balance_loss
from algo.model.tokenizer import load_tokenizer
from algo.train_pretrain import collate_pad


# --------------------------------------------------------------------------- 数据
def sample_texts(path: str, n: int, seed: int = 0) -> list[str]:
    """蓄水池抽样，避免把大 jsonl 整个读进内存。"""
    rng = random.Random(seed)
    out: list[str] = []
    seen = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            t = obj.get("text") or obj.get("content") or ""
            if not t:
                continue
            seen += 1
            if len(out) < n:
                out.append(t)
            else:
                j = rng.randrange(seen)
                if j < n:
                    out[j] = t
    return out


def build_batches(
    enc: list[list[int]],
    mode: str,
    batch_size: int,
    max_seq: int,
    seed: int = 0,
) -> list[Tuple[torch.Tensor, torch.Tensor]]:
    """返回 [(x, y)]。

    - random / bucket: pad 组批，y 的 pad=-100
    - pack: eos=2 拼流 → 切 max_seq 块 → 再按 batch_size 堆成 [B, S]
      （短块右侧 pad；y pad=-100）。禁止 B=1 假 packing。
    """
    rng = random.Random(seed)
    pad_id = 0
    eos = 2

    if mode == "pack":
        stream: list[int] = []
        for ids in enc:
            if not ids:
                continue
            stream.extend(ids)
            stream.append(eos)
        # 切成长度 max_seq 的块（至少 2 token 才能做 next-token）
        chunks: list[list[int]] = []
        for i in range(0, max(0, len(stream) - 1), max_seq):
            chunk = stream[i : i + max_seq]
            if len(chunk) < 2:
                continue
            chunks.append(chunk)
        batches: list[Tuple[torch.Tensor, torch.Tensor]] = []
        for i in range(0, len(chunks) - batch_size + 1, batch_size):
            group = chunks[i : i + batch_size]
            xs: list[torch.Tensor] = []
            ys: list[torch.Tensor] = []
            for chunk in group:
                # x = chunk[:-1] padded to max_seq-1? Plan: cut max_seq chunks,
                # stack [B, max_seq]. Use full chunk as input ids length S=len(chunk),
                # predict next within chunk: x=chunk[:-1] or treat as causal LM on
                # packed seq of length max_seq with labels shifted.
                # Lock: each packed sequence length == max_seq (pad short last).
                ids = list(chunk)
                if len(ids) < max_seq:
                    ids = ids + [pad_id] * (max_seq - len(ids))
                else:
                    ids = ids[:max_seq]
                x_ids = ids[:-1]
                y_ids = ids[1:]
                # pad positions in y → -100 (ids that were pad_id on input side
                # corresponding to padded region of original chunk)
                y_t = []
                for j, tok in enumerate(y_ids):
                    # position j+1 in ids; if that was pad filler beyond real len
                    real_len = min(len(chunk), max_seq)
                    if j + 1 >= real_len:
                        y_t.append(-100)
                    else:
                        y_t.append(tok)
                xs.append(torch.tensor(x_ids, dtype=torch.long))
                ys.append(torch.tensor(y_t, dtype=torch.long))
            # All same length max_seq-1; stack directly (or collate_pad)
            x = torch.stack(xs, dim=0)
            y = torch.stack(ys, dim=0)
            batches.append((x, y))
        return batches

    seqs = [ids[:max_seq] for ids in enc]
    if mode == "bucket":
        seqs.sort(key=len)
    else:
        rng.shuffle(seqs)
    batches = []
    for i in range(0, len(seqs) - batch_size + 1, batch_size):
        chunk = seqs[i : i + batch_size]
        xs, ys = [], []
        for ids in chunk:
            if len(ids) < 2:
                ids = [eos, eos]
            xs.append(torch.tensor(ids[:-1], dtype=torch.long))
            ys.append(torch.tensor(ids[1:], dtype=torch.long))
        batches.append(collate_pad(list(zip(xs, ys)), pad_id=pad_id))
    if mode == "bucket":
        rng.shuffle(batches)
    return batches


# ------------------------------------------------------- MoE forward 快慢组合
def _make_moe_forward(*, bounds_fast: bool, usage_gpu: bool) -> Callable:
    """组合边界同步与 usage 累加路径，供单变量 A/B。"""

    def _forward(self, x):
        b, t, d = x.shape
        flat = x.reshape(b * t, d)
        n_tok = flat.size(0)
        shared_out = self.shared(flat)

        _scores, topk_prob, topk_idx = route_tokens(
            self.router(flat), self.expert_bias, self.top_k
        )

        if self._record_usage:
            if usage_gpu:
                batch_counts = torch.bincount(
                    topk_idx.reshape(-1), minlength=self.n_routed
                )
            else:
                batch_counts = torch.bincount(
                    topk_idx.reshape(-1), minlength=self.n_routed
                ).detach().cpu()
            if self._usage_counts is None:
                self._usage_counts = batch_counts
            else:
                self._usage_counts += batch_counts

        k = self.top_k
        expert_ids = topk_idx.reshape(-1)
        weights = topk_prob.reshape(-1, 1)
        token_ids = (
            torch.arange(n_tok, device=flat.device)
            .unsqueeze(1)
            .expand(n_tok, k)
            .reshape(-1)
        )
        order = torch.argsort(expert_ids, stable=True)
        token_sorted = token_ids[order]
        weight_sorted = weights[order]
        expert_sorted = expert_ids[order]
        counts = torch.bincount(expert_sorted, minlength=self.n_routed)
        offsets = torch.zeros(self.n_routed + 1, dtype=torch.long, device=flat.device)
        offsets[1:] = counts.cumsum(0)
        gathered = flat.index_select(0, token_sorted)
        expert_out = torch.empty_like(gathered)

        if bounds_fast:
            bounds = offsets.tolist()
            for e in range(self.n_routed):
                start, end = bounds[e], bounds[e + 1]
                if start == end:
                    continue
                expert_out[start:end] = self.experts[e](gathered[start:end])
        else:
            for e in range(self.n_routed):
                start = int(offsets[e].item())
                end = int(offsets[e + 1].item())
                if start == end:
                    continue
                expert_out[start:end] = self.experts[e](gathered[start:end])

        weighted = expert_out * weight_sorted
        routed = torch.zeros_like(flat)
        routed.index_add_(0, token_sorted, weighted)
        y = (shared_out + routed).view(b, t, d)
        aux = sequence_balance_loss(
            _scores.view(b, t, -1), topk_idx.view(b, t, -1), self.n_routed
        )
        return y, aux

    return _forward


# 当前仓库 moe.py = both_fast（tolist 边界 + GPU usage）
_FAST_BOTH = MoELayer.forward
_FORWARD_CACHE = {
    "slow": _make_moe_forward(bounds_fast=False, usage_gpu=False),
    "bounds_fast": _make_moe_forward(bounds_fast=True, usage_gpu=False),
    "usage_gpu": _make_moe_forward(bounds_fast=False, usage_gpu=True),
    "both_fast": _FAST_BOTH,
}


def set_moe_mode(mode: str) -> None:
    """mode: slow | bounds_fast | usage_gpu | both_fast"""
    if mode not in _FORWARD_CACHE:
        raise ValueError(f"unknown moe mode: {mode}")
    MoELayer.forward = _FORWARD_CACHE[mode]


# --------------------------------------------------------------------- #5 契约
def count_log_syncs(log_every: int, n_steps: int) -> Tuple[int, int]:
    """返回 (n_loss_items, n_expert_ends)。

    与 simulate_log_sync / 真训练对齐：
    - 每步一次 loss.item
    - 仅当 step % log_every == 0 时做 expert end/begin（及 ce/aux item）
    """
    if log_every < 1:
        raise ValueError("log_every must be >= 1")
    n_loss = n_steps
    n_expert = sum(1 for s in range(n_steps) if s % log_every == 0)
    return n_loss, n_expert


# 别名，方便测试导入
simulate_log_schedule = count_log_syncs


def simulate_log_sync(
    model: Any,
    total: torch.Tensor,
    ce: Optional[torch.Tensor],
    aux: Optional[torch.Tensor],
    step: int,
    log_every: int,
    *,
    record_usage: bool,
) -> None:
    """按 #5 契约做同步模拟；不计时区内 print。

    - 每步：float(total.item())
    - step % log_every == 0：ce/aux .item()；若 record_usage 则 end→tolist 语义→begin
    """
    _ = float(total.item())
    if step % log_every != 0:
        return
    if ce is not None:
        _ = float(ce.item())
    if aux is not None:
        _ = float(aux.item())
    if record_usage and hasattr(model, "end_expert_usage") and hasattr(
        model, "begin_expert_usage"
    ):
        counts = model.end_expert_usage()
        if counts is not None:
            _ = counts.tolist() if hasattr(counts, "tolist") else counts
        model.begin_expert_usage()


# --------------------------------------------------------------------- 变体定义
# moe_mode: slow=慢边界+慢usage；bounds_fast / usage_gpu / both_fast
# 非 MoE 单项（fused/compile/log10/bucket/pack）一律 slow，保证单变量。
VARIANTS: Dict[str, dict] = {
    "base": dict(
        moe_mode="slow", fused=False, compile=False, batch="random", log_every=1
    ),
    "moe_bounds": dict(
        moe_mode="bounds_fast", fused=False, compile=False, batch="random", log_every=1
    ),
    "moe_usage_gpu": dict(
        moe_mode="usage_gpu", fused=False, compile=False, batch="random", log_every=1
    ),
    "fused": dict(
        moe_mode="slow", fused=True, compile=False, batch="random", log_every=1
    ),
    "compile": dict(
        moe_mode="slow", fused=False, compile=True, batch="random", log_every=1
    ),
    "log10": dict(
        moe_mode="slow", fused=False, compile=False, batch="random", log_every=10
    ),
    "bucket": dict(
        moe_mode="slow", fused=False, compile=False, batch="bucket", log_every=1
    ),
    "pack": dict(
        moe_mode="slow", fused=False, compile=False, batch="pack", log_every=1
    ),
    "all_equiv": dict(
        moe_mode="both_fast",
        fused=True,
        compile=True,
        batch="random",
        log_every=10,
    ),
}
ORDER = [
    "base",
    "moe_bounds",
    "moe_usage_gpu",
    "fused",
    "compile",
    "log10",
    "bucket",
    "pack",
    "all_equiv",
]


def _unwrap(model: Any) -> Any:
    return getattr(model, "_orig_mod", model)


def bench_one(
    name: str,
    cfg: dict,
    base_state: dict,
    model_cfg: ModelConfig,
    batches: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    steps: int,
    warmup: int,
    aux_coef: float,
    lr: float,
    device: torch.device,
) -> dict:
    torch.manual_seed(0)
    set_moe_mode(cfg["moe_mode"])
    model = build_model(model_cfg)
    model.load_state_dict(base_state, strict=True)
    model.to(device).train()
    if cfg["compile"]:
        model = torch.compile(model)
    opt = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=0.01, fused=bool(cfg["fused"])
    )

    # MoE：全程开 usage，使 #2 与 base 可比；end 在 log 窗里做
    raw = _unwrap(model)
    record_usage = hasattr(raw, "begin_expert_usage")
    if record_usage:
        raw.begin_expert_usage()

    times: list[float] = []
    real_tok = 0
    pad_tok = 0
    n_steps = warmup + steps
    torch.cuda.reset_peak_memory_stats()
    for i in range(n_steps):
        x, y = batches[i % len(batches)]
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(x, labels=y)
            ce = out[1]
            aux = out[2] if len(out) == 3 else None
            total = ce + aux_coef * aux if aux is not None else ce
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        # #5：无 print；每步 loss.item；log 窗额外 ce/aux + expert
        simulate_log_sync(
            raw,
            total,
            ce,
            aux,
            i,
            cfg["log_every"],
            record_usage=record_usage,
        )
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        if i >= warmup:
            times.append(dt)
            real_tok += int((y != -100).sum())
            pad_tok += int((y == -100).sum())

    med = statistics.median(times)
    total_t = sum(times)
    return dict(
        variant=name,
        step_ms=round(med * 1000, 1),
        tok_per_s=round(real_tok / total_t) if total_t > 0 else 0,
        pad_pct=round(100 * pad_tok / max(1, real_tok + pad_tok), 1),
        vram_gb=round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        steps=steps,
        moe_mode=cfg["moe_mode"],
        log_every=cfg["log_every"],
    )


def _median_result(runs: List[dict]) -> dict:
    """多轮取 tok_per_s 中位数对应的那一轮；并列取 step_ms 中位。"""
    if len(runs) == 1:
        out = dict(runs[0])
        out["runs"] = 1
        return out
    toks = sorted(r["tok_per_s"] for r in runs)
    med_tok = toks[len(toks) // 2]
    # 选 tok_per_s 最接近中位的 run
    best = min(runs, key=lambda r: abs(r["tok_per_s"] - med_tok))
    out = dict(best)
    out["tok_per_s"] = med_tok
    out["step_ms"] = round(statistics.median(r["step_ms"] for r in runs), 1)
    out["runs"] = len(runs)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/pretrain_mix_5g.jsonl")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-seq", type=int, default=512)
    ap.add_argument("--samples", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument(
        "--compile-warmup",
        type=int,
        default=30,
        help="compile 变体单独加长 warmup（默认 30）",
    )
    ap.add_argument("--runs", type=int, default=2, help="每变体重复轮次，取 tok/s 中位")
    ap.add_argument("--aux-coef", type=float, default=0.01)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--variants", default=",".join(ORDER))
    ap.add_argument(
        "--wait-pid",
        type=int,
        default=0,
        help="先等这个 pid 退出（训练占着 GPU 时用）",
    )
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.wait_pid:
        import os

        print(f"[bench] 等待 pid {args.wait_pid} 结束…", flush=True)
        while True:
            try:
                os.kill(args.wait_pid, 0)
            except OSError:
                break
            time.sleep(15)
        time.sleep(10)
        print("[bench] GPU 空闲，开始", flush=True)

    if not torch.cuda.is_available():
        raise SystemExit("需要 CUDA")
    device = torch.device("cuda")

    tok = load_tokenizer(None, allow_fallback=True)
    texts = sample_texts(args.data, args.samples)
    enc = [tok.encode(t, add_special_tokens=True) for t in texts]
    print(
        f"[bench] samples={len(enc)} avg_tokens={sum(len(x) for x in enc)/max(1,len(enc)):.0f}",
        flush=True,
    )

    if args.ckpt and Path(args.ckpt).exists():
        blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        model_cfg = ModelConfig.from_dict(blob["config"])
        base_state = blob["model"]
    else:
        model_cfg = ModelConfig(arch="moe")
        base_state = build_model(model_cfg).state_dict()
    print(
        f"[bench] arch={model_cfg.arch} batch={args.batch_size} seq={args.max_seq} runs={args.runs}",
        flush=True,
    )

    cache: dict = {}
    results = []
    for name in args.variants.split(","):
        name = name.strip()
        if not name:
            continue
        if name not in VARIANTS:
            raise SystemExit(f"unknown variant: {name}; choose from {ORDER}")
        cfg = VARIANTS[name]
        key = (cfg["batch"],)
        if key not in cache:
            cache[key] = build_batches(
                enc, cfg["batch"], args.batch_size, args.max_seq
            )
            if not cache[key]:
                raise SystemExit(f"no batches for mode={cfg['batch']}")
            bx, by = cache[key][0]
            print(
                f"[bench] batch_mode={cfg['batch']} first_shape=x{tuple(bx.shape)} y{tuple(by.shape)} n={len(cache[key])}",
                flush=True,
            )
        warmup = args.compile_warmup if cfg["compile"] else args.warmup
        print(f"[bench] === {name} === {cfg} warmup={warmup}", flush=True)
        run_rows = []
        for run_i in range(max(1, args.runs)):
            r = bench_one(
                name,
                cfg,
                base_state,
                model_cfg,
                cache[key],
                args.steps,
                warmup,
                args.aux_coef,
                args.lr,
                device,
            )
            run_rows.append(r)
            print(
                f"[bench] run={run_i+1}/{args.runs} "
                + json.dumps(r, ensure_ascii=False),
                flush=True,
            )
        agg = _median_result(run_rows)
        results.append(agg)
        print("[bench] median " + json.dumps(agg, ensure_ascii=False), flush=True)

    base = next((r for r in results if r["variant"] == "base"), None)
    print("\n=== 汇总 ===", flush=True)
    print(
        f"{'variant':14s} {'step_ms':>8s} {'tok/s':>9s} {'pad%':>6s} {'VRAM':>6s} {'vs base':>9s} {'runs':>4s}",
        flush=True,
    )
    for r in results:
        rel = (
            ""
            if not base or r["variant"] == "base"
            else f"{r['tok_per_s']/max(1, base['tok_per_s']):.2f}x"
        )
        print(
            f"{r['variant']:14s} {r['step_ms']:8.1f} {r['tok_per_s']:9d} {r['pad_pct']:6.1f} "
            f"{r['vram_gb']:6.2f} {rel:>9s} {r.get('runs', 1):4d}",
            flush=True,
        )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[bench] 结果写入 {args.out}", flush=True)


if __name__ == "__main__":
    main()
