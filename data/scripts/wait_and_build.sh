#!/usr/bin/env bash
set -u
need=(
  /home/wyf/miniseek/data/raw/fineweb2-zh/MAINLAND_CHINA.jsonl.zst
  /home/wyf/miniseek/data/raw/firefly/CNewSum_v2.jsonl
  /home/wyf/miniseek/data/raw/firefly/chinese-poem.jsonl
  /home/wyf/miniseek/data/raw/firefly/classical_chinese.jsonl
  /home/wyf/miniseek/data/raw/firefly/jinyong.jsonl
  /home/wyf/miniseek/data/raw/firefly/csl_camera_readly.jsonl
  /home/wyf/miniseek/data/raw/code/stack_smol_python.parquet
)
echo "waiting for downloads + wiki..."
for i in $(seq 1 180); do
  wiki_done=0
  if grep -q '^DONE' /home/wyf/miniseek/logs/convert_wiki.log 2>/dev/null; then wiki_done=1; fi
  # also accept wiki file >= 2GB
  wbytes=$(stat -c%s /home/wyf/miniseek/data/raw/buckets/wiki.jsonl 2>/dev/null || echo 0)
  if [ "$wbytes" -ge 2000000000 ]; then wiki_done=1; fi
  all=1
  for f in "${need[@]}"; do
    [ -f "$f" ] || all=0
  done
  echo "[$i $(date +%H:%M:%S)] wiki_done=$wiki_done wiki_bytes=$wbytes all_dl=$all curls=$(pgrep -c curl || true)"
  if [ "$wiki_done" = 1 ] && [ "$all" = 1 ]; then
    echo READY
    # wait a moment for wiki process to flush stats
    sleep 2
    python3 /home/wyf/miniseek/data/scripts/build_mix_stage2.py
    ec=$?
    echo BUILD_EXIT=$ec
    exit $ec
  fi
  # if no curls and missing files for long, fail
  if [ "$i" -gt 5 ] && ! pgrep -x curl >/dev/null && [ "$all" = 0 ]; then
    echo 'NO_CURLS_AND_MISSING'
    ls -lah /home/wyf/miniseek/data/raw/fineweb2-zh/ /home/wyf/miniseek/data/raw/firefly/ || true
    # try resume once
    /home/wyf/miniseek/data/scripts/download_one.sh \
      "https://hf-mirror.com/datasets/agentlans/fineweb2-chinese/resolve/main/MAINLAND_CHINA.jsonl.zst" \
      /home/wyf/miniseek/data/raw/fineweb2-zh/MAINLAND_CHINA.jsonl.zst >>/home/wyf/miniseek/logs/dl_fineweb.log 2>&1 &
    /home/wyf/miniseek/data/scripts/download_one.sh \
      "https://hf-mirror.com/datasets/YeungNLP/firefly-pretrain-dataset/resolve/main/CNewSum_v2.jsonl" \
      /home/wyf/miniseek/data/raw/firefly/CNewSum_v2.jsonl >>/home/wyf/miniseek/logs/dl_cnewsum.log 2>&1 &
    /home/wyf/miniseek/data/scripts/download_one.sh \
      "https://hf-mirror.com/datasets/YeungNLP/firefly-pretrain-dataset/resolve/main/chinese-poem.jsonl" \
      /home/wyf/miniseek/data/raw/firefly/chinese-poem.jsonl >>/home/wyf/miniseek/logs/dl_poem.log 2>&1 &
    /home/wyf/miniseek/data/scripts/download_one.sh \
      "https://hf-mirror.com/datasets/YeungNLP/firefly-pretrain-dataset/resolve/main/classical_chinese.jsonl" \
      /home/wyf/miniseek/data/raw/firefly/classical_chinese.jsonl >>/home/wyf/miniseek/logs/dl_classical.log 2>&1 &
    /home/wyf/miniseek/data/scripts/download_one.sh \
      "https://hf-mirror.com/datasets/YeungNLP/firefly-pretrain-dataset/resolve/main/csl_camera_readly.jsonl" \
      /home/wyf/miniseek/data/raw/firefly/csl_camera_readly.jsonl >>/home/wyf/miniseek/logs/dl_csl.log 2>&1 &
  fi
  sleep 15
done
echo TIMEOUT
exit 1
