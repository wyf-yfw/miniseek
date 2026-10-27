#!/usr/bin/env python3
"""训练速度优化 A/B 基准。

在 GPU 机上跑，训练空闲时执行。测的是**有效 token 吞吐**——padding 不算 token。

每一项优化都单独叠加在基线之上，用来量化它自己的贡献，最后再给一个全部打开的合计。

用法：
    python3 scripts/bench_train_opt.py --steps 30 --warmup 10
    python3 scripts/bench_train_opt.py --wait-pid 56159 --out /tmp/bench.json
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from algo.model import build_model
from algo.model.config import ModelConfig
from algo.model.moe import MoELayer, load_balance_loss
from algo.model.tokenizer import load_tokenizer
from algo.train_pretrain import collate_pad


# --------------------------------------------------------------------------- 数据
def sample_texts(path: str, n: int, seed: int = 0) -> list[str]:
    """蓄水池抽样，避免把 4GB jsonl 整个读进内存。"""
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


def build_batches(enc: list[list[int]], mode: str, batch_size: int, max_seq: int, seed: int = 0):
    """返回 [(x, y)]；random/bucket 会 padding，pack 不会。"""
    rng = random.Random(seed)
    pad_id = 0
    eos = 2

    if mode == "pack":
        stream: list[int] = []
        for ids in enc:
            stream.extend(ids)
            stream.append(eos)
        batches = []
        for i in range(0, len(stream) - max_seq, max_seq):
            chunk = stream[i : i + max_seq]
            x = torch.tensor(chunk[:-1], dtype=torch.long).unsqueeze(0)
            y = torch.tensor(chunk[1:], dtype=torch.long).unsqueeze(0)
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
        rng.shuffle(batches)  # 桶内顺序已经打散，批顺序再打散一次
    return batches


# ------------------------------------------------------- 老版 MoE forward（A/B 用）
def slow_moe_forward(self, x):
    """改造前的实现对快照：每专家两次 .item()，负载统计每层 .cpu()。"""
    b, t, d = x.shape
    flat = x.reshape(b * t, d)
    n_tok = flat.size(0)
    shared_out = self.shared(flat)

    logits = self.router(flat)
    probs = F.softmax(logits, dim=-1)
    topk_prob, topk_idx = torch.topk(probs, self.top_k, dim=-1)
    topk_prob = topk_prob / topk_prob.sum(dim=-1, keepdim=True).clamp_min(1e-9)

    if self._record_usage:
        batch_counts = torch.bincount(topk_idx.reshape(-1), minlength=self.n_routed).detach().cpu()
        if self._usage_counts is None:
            self._usage_counts = batch_counts
        else:
            self._usage_counts += batch_counts

    k = self.top_k
    expert_ids = topk_idx.reshape(-1)
    weights = topk_prob.reshape(-1, 1)
    token_ids = torch.arange(n_tok, device=flat.device).unsqueeze(1).expand(n_tok, k).reshape(-1)
    order = torch.argsort(expert_ids, stable=True)
    expert_sorted = expert_ids[order]
    token_sorted = token_ids[order]
    weight_sorted = weights[order]
    counts = torch.bincount(expert_sorted, minlength=self.n_routed)
    offsets = torch.zeros(self.n_routed + 1, dtype=torch.long, device=flat.device)
    offsets[1:] = counts.cumsum(0)
    gathered = flat.index_select(0, token_sorted)
    expert_out = torch.empty_like(gathered)
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
    aux = load_balance_loss(probs, topk_idx, self.n_routed)
    return y, aux


FAST_FORWARD = MoELayer.forward


def set_moe_fast(fast: bool) -> None:
    MoELayer.forward = FAST_FORWARD if fast else slow_moe_forward


# --------------------------------------------------------------------- 变体定义
VARIANTS = {
    # name          moe_fast fused compile batch     log_every
    "base": dict(moe_fast=False, fused=False, compile=False, batch="random", log_every=1),
    "moe_fast": dict(moe_fast=True, fused=False, compile=False, batch="random", log_every=1),
    "fused": dict(moe_fast=False, fused=True, compile=False, batch="random", log_every=1),
    "compile": dict(moe_fast=False, fused=False, compile=True, batch="random", log_every=1),
    "log10": dict(moe_fast=False, fused=False, compile=False, batch="random", log_every=10),
    "bucket": dict(moe_fast=False, fused=False, compile=False, batch="bucket", log_every=1),
    "pack": dict(moe_fast=False, fused=False, compile=False, batch="pack", log_every=1),
    "all": dict(moe_fast=True, fused=True, compile=True, batch="bucket", log_every=10),
}
ORDER = ["base", "moe_fast", "fused", "compile", "log10", "bucket", "pack", "all"]


def bench_one(name, cfg, base_state, model_cfg, batches, steps, warmup, aux_coef, lr, device):
    torch.manual_seed(0)
    set_moe_fast(cfg["moe_fast"])
    model = build_model(model_cfg)
    model.load_state_dict(base_state, strict=True)
    model.to(device).train()
    if cfg["compile"]:
        model = torch.compile(model)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, fused=cfg["fused"])

    times, real_tok, pad_tok = [], 0, 0
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
            total = ce + aux_coef * out[2] if len(out) == 3 and out[2] is not None else ce
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if i % cfg["log_every"] == 0:
            _ = float(total.item())  # 模拟 log_every 的同步
            print(f"  step={i} loss={_:.4f}", flush=True)
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
        tok_per_s=round(real_tok / total_t),
        pad_pct=round(100 * pad_tok / max(1, real_tok + pad_tok), 1),
        vram_gb=round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        steps=steps,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/pretrain_mix_5g.jsonl")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-seq", type=int, default=512)
    ap.add_argument("--samples", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--aux-coef", type=float, default=0.01)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--variants", default=",".join(ORDER))
    ap.add_argument("--wait-pid", type=int, default=0, help="先等这个 pid 退出（训练占着 GPU 时用）")
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
    print(f"[bench] samples={len(enc)} avg_tokens={sum(len(x) for x in enc)/len(enc):.0f}", flush=True)

    if args.ckpt and Path(args.ckpt).exists():
        blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        model_cfg = ModelConfig.from_dict(blob["config"])
        base_state = blob["model"]
    else:
        model_cfg = ModelConfig(arch="moe")
        base_state = build_model(model_cfg).state_dict()
    print(f"[bench] arch={model_cfg.arch} batch={args.batch_size} seq={args.max_seq}", flush=True)

    cache = {}
    results = []
    for name in args.variants.split(","):
        name = name.strip()
        if not name:
            continue
        cfg = VARIANTS[name]
        key = (cfg["batch"],)
        if key not in cache:
            cache[key] = build_batches(enc, cfg["batch"], args.batch_size, args.max_seq)
        print(f"[bench] === {name} === {cfg}", flush=True)
        r = bench_one(
            name, cfg, base_state, model_cfg, cache[key],
            args.steps, args.warmup, args.aux_coef, args.lr, device,
        )
        results.append(r)
        print("[bench] " + json.dumps(r, ensure_ascii=False), flush=True)

    base = next((r for r in results if r["variant"] == "base"), None)
    print("\n=== 汇总 ===", flush=True)
    print(f"{'variant':10s} {'step_ms':>8s} {'tok/s':>9s} {'pad%':>6s} {'VRAM':>6s} {'vs base':>9s}", flush=True)
    for r in results:
        rel = "" if not base or r["variant"] == "base" else f"{r['tok_per_s']/base['tok_per_s']:.2f}x"
        print(
            f"{r['variant']:10s} {r['step_ms']:8.1f} {r['tok_per_s']:9d} {r['pad_pct']:6.1f} "
            f"{r['vram_gb']:6.2f} {rel:>9s}",
            flush=True,
        )

    if args.out:
        Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[bench] 结果写入 {args.out}", flush=True)


if __name__ == "__main__":
    main()
