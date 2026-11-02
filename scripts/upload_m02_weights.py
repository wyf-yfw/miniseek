#!/usr/bin/env python3
"""Upload MiniSeek-V0.2 weights with Xet disabled (proxy breaks xethub CAS)."""
from __future__ import annotations
import os, time, traceback, sys

os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7897"
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7897"
os.environ["ALL_PROXY"] = "http://127.0.0.1:7897"
os.environ["HF_ENDPOINT"] = "https://huggingface.co"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ.pop("HF_XET_HIGH_PERFORMANCE", None)
os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)

from huggingface_hub import HfApi

api = HfApi()
print("who", api.whoami()["name"], "HF_HUB_DISABLE_XET=1", flush=True)

jobs = [
    ("wyf-yfw/MiniSeek-V0.2-base",
     "/home/wyf/miniseek/release/MiniSeek-V0.2-base/model.safetensors",
     "Add MiniSeek-V0.2-base weights (PT last@10000)"),
    ("wyf-yfw/MiniSeek-V0.2",
     "/home/wyf/miniseek/release/MiniSeek-V0.2/model.safetensors",
     "Add MiniSeek-V0.2 weights (SFT best@36428)"),
]

for repo, path, msg in jobs:
    size = os.path.getsize(path)
    for attempt in range(1, 12):
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
            wait = min(120, 5 * attempt)
            print(f"FAIL {attempt}: {e!r}", flush=True)
            traceback.print_exc()
            if attempt == 11:
                sys.exit(1)
            time.sleep(wait)

print("ALL_DONE", time.strftime("%F %T"), flush=True)
