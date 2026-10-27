# tests/test_frontend_proxy.py
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(monkeypatch, tmp_path):
    # Point WEB_DIR at a tiny fixture copy of index.html
    web_dir = tmp_path / "web"
    web_dir.mkdir()
    (web_dir / "index.html").write_text("<html>ok</html>", encoding="utf-8")
    monkeypatch.setenv("MINISEEK_API_URL", "http://127.0.0.1:8000")
    import importlib
    import web.frontend_server as fs

    monkeypatch.setattr(fs, "WEB_DIR", web_dir)
    importlib.reload(fs)
    monkeypatch.setattr(fs, "WEB_DIR", web_dir)
    return TestClient(fs.app)


def test_serves_index(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "ok" in r.text


def test_api_proxy_forwards(client, monkeypatch):
    import web.frontend_server as fs

    class FakeResp:
        status_code = 200
        headers = {"content-type": "application/json"}

        async def aiter_bytes(self):
            yield b'{"ok":true}'

        async def aclose(self):
            return None

    class FakeClient:
        def build_request(self, *a, **k):
            return object()

        async def send(self, *a, **k):
            return FakeResp()

        async def aclose(self):
            return None

    monkeypatch.setattr(fs.httpx, "AsyncClient", lambda **kw: FakeClient())
    r = client.get("/api/defaults")
    assert r.status_code == 200
    assert r.json() == {"ok": True}
