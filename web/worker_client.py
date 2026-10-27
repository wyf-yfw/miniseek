"""HTTP client from API → worker."""
from __future__ import annotations

import os
import sys
from typing import Any, Optional

import httpx
from fastapi import HTTPException

DEFAULT_WORKER_URL = "http://127.0.0.1:8001"


def worker_base_url() -> str:
    return (os.environ.get("MINISEEK_WORKER_URL") or DEFAULT_WORKER_URL).rstrip("/")


def use_monolith() -> bool:
    # Windows default monolith; set MINISEEK_MONOLITH=0 + run worker to split on Windows.
    flag = (os.environ.get("MINISEEK_MONOLITH") or "").strip().lower()
    if flag in {"1", "true", "yes", "on"}:
        return True
    if flag in {"0", "false", "no", "off"}:
        return False
    # Default: monolith only on Windows so local uvicorn still works without worker.
    return sys.platform == "win32"


def forward(method: str, path: str, *, json: Optional[dict] = None, timeout: float = 60.0) -> Any:
    url = worker_base_url() + path
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.request(method, url, json=json)
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"worker unreachable: {exc}") from exc
    if resp.status_code >= 400:
        detail: Any
        try:
            payload = resp.json()
            detail = payload.get("detail", payload)
        except Exception:  # noqa: BLE001
            detail = resp.text
        raise HTTPException(status_code=resp.status_code, detail=detail)
    if resp.status_code == 204 or not resp.content:
        return {"ok": True}
    return resp.json()
