#!/usr/bin/env python3
"""Generate smoke: load a MiniSeek-V checkpoint and sample a short continuation."""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
from algo.model import load_model_from_blob
from algo.model.tokenizer import load_tokenizer

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--prompt", default="你好")
    p.add_argument("--max-new-tokens", type=int, default=32)
    args = p.parse_args()
    tok = load_tokenizer(allow_fallback=True)
    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model, cfg = load_model_from_blob(blob)
    model.eval()
    ids = torch.tensor([tok.encode(args.prompt, add_special_tokens=True)], dtype=torch.long)
    out = model.generate(ids, max_new_tokens=args.max_new_tokens)
    print(tok.decode(out[0].tolist(), skip_special_tokens=True))
    print(f"[gen_smoke] arch={cfg.arch} ok")

if __name__ == "__main__":
    main()
