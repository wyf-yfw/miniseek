#!/usr/bin/env python3
"""Upload MiniSeek SFT v5 packs to HF. Safe to re-run; retries on SSL blips."""
from __future__ import annotations

import time
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

REPO = "wyf-yfw/MiniSeek-SFT"
ROOT = Path(__file__).resolve().parents[1] / "data" / "SFT"
FILES = [
    "sft_miniseek_v5_10g.jsonl",
    "sft_miniseek_v5_5g.jsonl",
    "README.md",
    "SOURCES.md",
    "SOURCES_v5.md",
]


def upload_with_retry(api: HfApi, name: str, attempts: int = 8) -> None:
    path = ROOT / name
    if not path.exists():
        raise SystemExit(f"missing {path}")
    size = path.stat().st_size
    for i in range(1, attempts + 1):
        print(f"uploading {name} ({size / 1024**3:.2f} GiB) try {i}/{attempts} ...", flush=True)
        t0 = time.time()
        try:
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=name,
                repo_id=REPO,
                repo_type="dataset",
                commit_message=f"Add/update {name} (SFT v5 cutover)",
            )
            print(f"  done {name} in {time.time() - t0:.1f}s", flush=True)
            return
        except Exception as e:
            wait = min(60, 2 ** i)
            print(f"  fail {name}: {e!r}; sleep {wait}s", flush=True)
            if i == attempts:
                raise
            time.sleep(wait)


def main() -> None:
    api = HfApi()
    print("start", time.strftime("%F %T"), flush=True)
    # Prefer remaining small/failed first if env SKIP_10G=1
    import os

    files = list(FILES)
    if os.environ.get("SKIP_10G") == "1":
        files = [f for f in files if f != "sft_miniseek_v5_10g.jsonl"]
        print("SKIP_10G=1 — skipping already-uploaded 10g", flush=True)
    for name in files:
        upload_with_retry(api, name)
    for _ in range(5):
        try:
            api.delete_file(
                path_in_repo="sft_miniseek_v1.jsonl",
                repo_id=REPO,
                repo_type="dataset",
                commit_message="Remove retired sft_miniseek_v1.jsonl",
            )
            print("deleted sft_miniseek_v1.jsonl from Hub", flush=True)
            break
        except Exception as e:
            msg = str(e).lower()
            if "404" in msg or "not found" in msg or "does not exist" in msg:
                print("v1 already absent on Hub", flush=True)
                break
            print(f"delete v1 retry: {e!r}", flush=True)
            time.sleep(5)
    print("ALL_DONE", time.strftime("%F %T"), flush=True)


if __name__ == "__main__":
    main()
