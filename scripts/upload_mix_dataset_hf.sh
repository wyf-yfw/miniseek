#!/usr/bin/env bash
# Upload MiniSeek mix dataset to existing HF repo wyf-yfw/MiniSeek-dataset
# Requires: hf auth login (as wyf-yfw) OR HF_TOKEN in env
set -euo pipefail
cd /home/wyf/miniseek
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
# Prefer official hub unless HF_ENDPOINT already set (prior attempt used hf-mirror.com)
STAGING=/home/wyf/miniseek/data/hf_upload_MiniSeek-dataset_real
LOG=/home/wyf/miniseek/logs/hf_upload_MiniSeek-dataset.log
mkdir -p /home/wyf/miniseek/logs /home/wyf/miniseek/scripts
if ! hf auth whoami >/dev/null 2>&1; then
  echo "ERROR: not logged in. Run: hf auth login   (or export HF_TOKEN=...)" >&2
  exit 1
fi
hf auth whoami
echo "Uploading from $STAGING -> wyf-yfw/MiniSeek-dataset (dataset)"
# hf upload is the supported path (upload-large-folder deprecated)
nohup hf upload wyf-yfw/MiniSeek-dataset "$STAGING" --repo-type dataset \
  >"$LOG" 2>&1 &
echo $! > /home/wyf/miniseek/logs/hf_upload_MiniSeek-dataset.pid
echo "started pid=$(cat /home/wyf/miniseek/logs/hf_upload_MiniSeek-dataset.pid) log=$LOG"
