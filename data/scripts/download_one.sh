#!/usr/bin/env bash
url="$1"; out="$2"
mkdir -p "$(dirname "$out")"
if [ -f "$out" ] && [ -s "$out" ]; then
  echo "SKIP $out"; exit 0
fi
tmp="$out.partial"
echo "GET $out"
curl -L --fail --retry 10 --retry-delay 5 -C - \
  --connect-timeout 30 --speed-time 120 --speed-limit 500 \
  -o "$tmp" "$url"
rc=$?
if [ $rc -eq 0 ]; then mv "$tmp" "$out"; echo "OK $out"; else echo "FAIL $rc $out"; exit $rc; fi
