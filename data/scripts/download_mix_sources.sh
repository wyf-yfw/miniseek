#!/usr/bin/env bash
set -u
download() {
  local url="$1" out="$2"
  mkdir -p "$(dirname "$out")"
  if [ -f "$out" ] && [ -s "$out" ]; then
    echo "SKIP $out ($(du -h "$out" | cut -f1))"
    return 0
  fi
  echo "GET $out"
  # resume partial if present
  local tmp="$out.partial"
  curl -L --fail --retry 8 --retry-delay 4 -C - \
    --connect-timeout 30 --speed-time 60 --speed-limit 1000 \
    -o "$tmp" "$url"
  local rc=$?
  if [ $rc -eq 0 ]; then
    mv "$tmp" "$out"
    echo "OK $out ($(du -h "$out" | cut -f1))"
  else
    echo "FAIL rc=$rc $out"
    return $rc
  fi
}

BASE=https://hf-mirror.com

download "$BASE/datasets/agentlans/fineweb2-chinese/resolve/main/MAINLAND_CHINA.jsonl.zst" \
  /home/wyf/miniseek/data/raw/fineweb2-zh/MAINLAND_CHINA.jsonl.zst

download "$BASE/datasets/YeungNLP/firefly-pretrain-dataset/resolve/main/CNewSum_v2.jsonl" \
  /home/wyf/miniseek/data/raw/firefly/CNewSum_v2.jsonl
download "$BASE/datasets/YeungNLP/firefly-pretrain-dataset/resolve/main/chinese-poem.jsonl" \
  /home/wyf/miniseek/data/raw/firefly/chinese-poem.jsonl
download "$BASE/datasets/YeungNLP/firefly-pretrain-dataset/resolve/main/classical_chinese.jsonl" \
  /home/wyf/miniseek/data/raw/firefly/classical_chinese.jsonl
download "$BASE/datasets/YeungNLP/firefly-pretrain-dataset/resolve/main/jinyong.jsonl" \
  /home/wyf/miniseek/data/raw/firefly/jinyong.jsonl
download "$BASE/datasets/YeungNLP/firefly-pretrain-dataset/resolve/main/csl_camera_readly.jsonl" \
  /home/wyf/miniseek/data/raw/firefly/csl_camera_readly.jsonl

# optional extra web if needed later: sample of thucnews via range? skip for now
echo ALL_DOWNLOADS_FINISHED
