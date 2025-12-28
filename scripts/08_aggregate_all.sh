#!/usr/bin/env bash
set -e

RUNS_DIR=${1:-runs}
for d in "$RUNS_DIR"/*; do
  if [ -d "$d" ]; then
    if [ -f "$d/summary.json" ]; then
      continue
    fi
    python -m pr2.eval.aggregate --runs_dir "$d" --out "$d/summary.json" || true
  fi
done
