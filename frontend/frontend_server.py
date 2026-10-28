"""Static frontend + /api reverse proxy (SSE-safe)."""
from __future__ import annotations

import os
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = Path(__file__).resolve().parent
API_URL = (os.environ.get("MINISEEK_API_URL") or "http://127.0.0.1:8000").rstrip("/")
INDEX_HTML = WEB_DIR / "index.html"

HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "content-encoding",
    "content-length",
}

NO_STORE = {
    "Cache-Control": "no-store, no-cache, must-revalidate",
    "Pragma": "no-cache",
    "Expires": "0",
}

app = FastAPI(title="MiniSeek Web")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(INDEX_HTML, headers=dict(NO_STORE))


@app.get("/__dev/mtime")
def dev_mtime() -> JSONResponse:
    """Poll target for frontend live-reload (mtime of index.html)."""
    try:
        mtime = INDEX_HTML.stat().st_mtime
    except OSError:
        mtime = 0.0
    return JSONResponse(
        {"mtime": float(mtime), "path": "index.html"},
        headers=dict(NO_STORE),
    )


@app.api_route(
    "/api/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
)
async def proxy_api(path: str, request: Request):
    url = f"{API_URL}/api/{path}"
    if request.url.query:
        url = f"{url}?{request.url.query}"
    headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in HOP_BY_HOP and k.lower() != "host"
    }
    body = await request.body()

    client = httpx.AsyncClient(timeout=None)
    req = client.build_request(request.method, url, headers=headers, content=body)
    upstream = await client.send(req, stream=True)

    out_headers = {
        k: v for k, v in upstream.headers.items() if k.lower() not in HOP_BY_HOP
    }
    # Disable intermediary buffering hints for SSE
    if "text/event-stream" in (upstream.headers.get("content-type") or ""):
        out_headers["Cache-Control"] = "no-cache"
        out_headers["X-Accel-Buffering"] = "no"

    async def stream():
        try:
            async for chunk in upstream.aiter_bytes():
                yield chunk
        finally:
            await upstream.aclose()
            await client.aclose()

    return StreamingResponse(
        stream(),
        status_code=upstream.status_code,
        headers=out_headers,
        media_type=upstream.headers.get("content-type"),
    )


# Static assets after routes so /, /__dev/mtime and /api win
app.mount("/", StaticFiles(directory=str(WEB_DIR), html=False), name="static")
