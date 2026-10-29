#!/usr/bin/env python3
"""DSpark 草稿头训练（预训练之后的独立阶段，主干冻结）。

示例::

    python -m algo.train_dspark --backbone tasks/<pt>/last.pt \\
        --data data/PT/pretrain_mix_12g_clean.jsonl --out-dir tasks/_dspark --max-steps 2000

产出 ``<out-dir>/dspark.pt``：{"dspark", "config", "backbone", "step", "loss", "optimizer", "stage"}。
校准置信度（STS）见 ``python -m algo.dspark_calibrate``。
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

import argparse
import time

import torch

from algo import v41_train as v41t
from algo.batching import PackedTextBatchIter
from algo.model import load_model_from_blob
from algo.model.tokenizer import load_tokenizer
from algo.model.v41 import FlashConfig, step_expert_bias
from algo.model.v41.dspark import DSparkDrafter, dspark_training_losses
from algo.optim import build_v41_optimizer
from algo.train_pretrain import JsonlTextDataset, lr_at_step, save_ckpt, set_lr, set_seed


def save_dspark(path: Path, drafter: DSparkDrafter, opt, backbone_path: str, step: int, loss: float) -> None:
    save_ckpt(
        path,
        {
            "dspark": v41t.state_for_ckpt(drafter),
            "config": drafter.cfg.to_dict(),
            "backbone": str(backbone_path),
            "step": int(step),
            "loss": float(loss),
            "optimizer": opt.state_dict(),
            "stage": "dspark",
        },
    )


def load_drafter(blob: dict) -> DSparkDrafter:
    drafter = DSparkDrafter(FlashConfig.from_dict(blob["config"]))
    drafter.load_state_dict(v41t.clean_state_dict(blob["dspark"]), strict=True)
    return drafter


def main() -> None:
    p = argparse.ArgumentParser(description="MiniSeek-F DSpark drafter training (backbone frozen)")
    p.add_argument("--backbone", type=str, required=True, help="v41 主干 checkpoint（PT 或 SFT）")
    p.add_argument("--data", type=str, default="data/PT/pretrain_mix_12g_clean.jsonl")
    p.add_argument("--tokenizer", type=str, default=None)
    p.add_argument("--out-dir", type=str, default="tasks/_default_dspark")
    p.add_argument("--max-seq-len", type=int, default=None, help="默认取主干配置")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--anchors", type=int, default=32, help="每条序列随机取多少个草稿块起点")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-lr", type=float, default=3e-5)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--aux-coef", type=float, default=0.0001)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-last-every", type=int, default=1000)
    p.add_argument("--resume", type=str, default=None, help="从 dspark.pt 续训（含 optimizer）")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    blob = torch.load(args.backbone, map_location="cpu", weights_only=False)
    backbone, cfg = load_model_from_blob(blob)
    if not isinstance(cfg, FlashConfig):
        raise SystemExit("DSpark 只支持 v41 主干")
    backbone.to(device).eval()
    for prm in backbone.parameters():
        prm.requires_grad_(False)
    max_len = int(args.max_seq_len or cfg.max_seq_len)

    if args.resume:
        rblob = torch.load(args.resume, map_location="cpu", weights_only=False)
        drafter = load_drafter(rblob).to(device)
    else:
        rblob = None
        drafter = DSparkDrafter(cfg).to(device)
    opt = build_v41_optimizer(drafter, lr=args.lr, cfg=cfg)
    if rblob is not None and "optimizer" in rblob:
        opt.load_state_dict(rblob["optimizer"])
    print(
        f"[dspark] backbone={args.backbone} drafter_params={drafter.num_params():,} "
        f"blocks={cfg.n_mtp_layers} gamma={cfg.dspark_block_size} targets={list(cfg.dspark_target_layer_ids)} "
        f"experts=1+{cfg.dspark_n_routed_experts} top{cfg.dspark_n_activated_experts} markov_rank={cfg.dspark_markov_rank}",
        flush=True,
    )

    tok = load_tokenizer(args.tokenizer, allow_fallback=True)
    pad_id = int(getattr(tok, "pad_token_id", 0) or 0)
    eos_id = int(getattr(tok, "eos_token_id", 2) or 2)
    ds = JsonlTextDataset(args.data, tok, max_seq_len=max_len, max_samples=args.max_samples, seed=args.seed)

    def epoch_iter(epoch: int):
        return PackedTextBatchIter(
            ds.samples,
            tok,
            max_seq_len=max_len,
            batch_size=args.batch_size,
            pad_id=pad_id,
            eos_id=eos_id,
            seed=args.seed + epoch,
            doc_align=v41t.doc_align(cfg),
            with_doc_ids=True,
        )

    total_steps = int(args.max_steps or max(1, epoch_iter(0).estimate_len() * max(1, args.epochs)))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator().manual_seed(args.seed)
    step, last = 0, 0.0
    t0 = time.time()
    drafter.train()
    for epoch in range(max(1, args.epochs)):
        for batch in epoch_iter(epoch):
            if step >= total_steps:
                break
            ids, labels, docs = v41t.unpack_batch(batch, device)
            set_lr(opt, lr_at_step(step + 1, base_lr=args.lr, min_lr=args.min_lr, warmup_steps=args.warmup_steps, total_steps=total_steps))
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                terms = dspark_training_losses(drafter, backbone, ids, labels, docs, n_anchors=args.anchors, generator=gen)
                loss = terms["total"] + args.aux_coef * terms["balance"]
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(drafter.parameters(), args.grad_clip)
            opt.step()
            step_expert_bias(drafter)
            step += 1
            last = float(loss.item())
            if step % args.log_every == 0 or step == 1:
                print(
                    f"epoch={epoch} step={step} loss={last:.4f} ce={float(terms['ce']):.4f} "
                    f"tv={float(terms['tv']):.4f} conf={float(terms['conf']):.4f} "
                    f"accept={float(terms['accept']):.4f} lr={opt.param_groups[0]['lr']:.2e}",
                    flush=True,
                )
            if args.save_last_every > 0 and step % args.save_last_every == 0:
                save_dspark(out_dir / "dspark.pt", drafter, opt, args.backbone, step, last)
        if step >= total_steps:
            break
    save_dspark(out_dir / "dspark.pt", drafter, opt, args.backbone, step, last)
    print(f"[dspark] done steps={step} loss={last:.4f} elapsed={time.time() - t0:.1f}s -> {out_dir / 'dspark.pt'}", flush=True)


if __name__ == "__main__":
    main()
