#!/usr/bin/env python3
"""Upload MiniSeek-V1 weights. Proxy for HF API; direct for S3 LFS (proxy breaks large PUTs)."""
from __future__ import annotations
import os, time, traceback, sys

os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7897"
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7897"
os.environ["ALL_PROXY"] = "http://127.0.0.1:7897"
# Large LFS parts to s3-accelerate break SSL via clash/mitm proxy — go direct.
os.environ["NO_PROXY"] = "amazonaws.com,.amazonaws.com,s3.amazonaws.com,s3-accelerate.amazonaws.com,hf-hub-lfs-us-east-1.s3-accelerate.amazonaws.com,cloudfront.net,.cloudfront.net"
os.environ["no_proxy"] = os.environ["NO_PROXY"]
os.environ["HF_ENDPOINT"] = "https://huggingface.co"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ.pop("HF_XET_HIGH_PERFORMANCE", None)
os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)

from huggingface_hub import HfApi

api = HfApi(endpoint="https://huggingface.co")
print("who", api.whoami()["name"], "XET=off NO_PROXY=s3", flush=True)

jobs = [
    ("wyf-yfw/MiniSeek-V1-base",
     "/home/wyf/miniseek/release/MiniSeek-V1-base/model.safetensors",
     "Add MiniSeek-V1-base model.safetensors (PT last)"),
    ("wyf-yfw/MiniSeek-V1",
     "/home/wyf/miniseek/release/MiniSeek-V1/model.safetensors",
     "Add MiniSeek-V1 model.safetensors (SFT best)"),
]

for repo, path, msg in jobs:
    size = os.path.getsize(path)
    for attempt in range(1, 16):
        print(f"[{time.strftime('%F %T')}] {repo} {size/1e9:.2f}GB try {attempt}", flush=True)
        t0 = time.time()
        try:
            url = api.upload_file(
                path_or_fileobj=path,
                path_in_repo="model.safetensors",
                repo_id=repo,
                repo_type="model",
                commit_message=msg,
            )
            print(f"DONE {repo} in {time.time()-t0:.1f}s {url}", flush=True)
            break
        except Exception as e:
            wait = min(180, 8 * attempt)
            print(f"FAIL {attempt}: {e!r}", flush=True)
            traceback.print_exc()
            if attempt == 15:
                sys.exit(1)
            time.sleep(wait)
    else:
        sys.exit(1)

print("ALL_DONE", time.strftime("%F %T"), flush=True)
