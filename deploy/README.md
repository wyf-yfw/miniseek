# MiniSeek three-service deploy (wyf)

User systemd units for **frontend / API / worker**. Restarting web or API must not kill training/eval processes owned by the worker.

| Unit | Port | App |
|------|------|-----|
| `miniseek-web.service` | `0.0.0.0:7860` | `frontend.frontend_server:app` (static + `/api` proxy) |
| `miniseek-api.service` | `127.0.0.1:8000` | `api.app:app` (`MINISEEK_MONOLITH=0`) |
| `miniseek-worker.service` | `127.0.0.1:8001` | `api.worker_app:app` |

Funnel / public reverse proxy still points at **`:7860`**. Do **not** enable both `miniseek-webui` and `miniseek-web` (port clash).

## Cutover from `miniseek-webui`

```bash
# From repo root on wyf
mkdir -p ~/.config/systemd/user
cp deploy/miniseek-*.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user disable --now miniseek-webui.service || true
systemctl --user enable --now miniseek-worker.service miniseek-api.service miniseek-web.service
systemctl --user status miniseek-worker miniseek-api miniseek-web --no-pager
curl -s http://127.0.0.1:8001/health
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8000/api/defaults
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:7860/
ss -ltnp | grep -E ':7860|:8000|:8001'
```

## Windows (dev monolith)

```bash
MINISEEK_MONOLITH=1 uvicorn api.app:app --host 0.0.0.0 --port 7860
```
