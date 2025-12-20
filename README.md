# PR2-Drag Lite (CPU)

A lightweight, pretrained-first reliability controller prototype:
- Stage1: precompute per-seq trajectories + evidence features from DAVIS (480p)
- Stage2: build train/val datasets, handle stale/mismatch cache
- Stage3: emission (logistic) + optional isotonic calibration + HMM smoothing + AoB (abstain/bridge) + analysis outputs

## Quick start

1) Install deps:
```bash
pip install -r requirements.txt
Run all:

python -m pr2drag.cli --config configs/davis2016_cpu.yaml --cmd run_all
Outputs go to:
<base_out>/<stage_dirs...> and final analysis in:
<base_out>/<stage3_out>/_analysis/*

Notes
This repo is designed for robustness: atomic writes, signatures for cache compatibility, and clear logs.

feat_set: a|b toggles feature dimensionality (5 vs 7).

aob.eps_gate = 1.0 guarantees "pure abstain" (bridge disabled) as a strict control.

---