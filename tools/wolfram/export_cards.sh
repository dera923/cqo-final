#!/usr/bin/env bash
set -euo pipefail
out="reports/figures"
mkdir -p "$out"
echo "[wolfram] exporting cards -> $out"
status=0
shopt -s nullglob
for w in tools/wolfram/*.wls; do
  echo "==> $w"
  if command -v wolframscript >/dev/null 2>&1; then
    if ! wolframscript -file "$w"; then
      echo "[warn] failed: $w" >&2
      status=1
    fi
  else
    echo "[skip] wolframscript not found" >&2
    status=0
  fi
done
exit $status
