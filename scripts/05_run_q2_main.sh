#!/usr/bin/env bash
set -e

CONFIG=${1:-configs/default.yaml}
TASKS=${2:-assets/demo_tasks/tier1_demo_tasks.jsonl}
OUT=${3:-runs/q2_main}

python -m pr2.eval.run_suite --config "$CONFIG" --tasks "$TASKS" --out "$OUT"
python -m pr2.eval.aggregate --runs_dir "$OUT" --out "$OUT/summary.json"
