# pr2drag-lite (DAVIS2016 CPU)

A minimal, structured pipeline:

- Stage1: precompute DAVIS sequence npz from GT masks
- Stage2: optical flow mask propagation -> z_obs + evidence E + chi (weak label by IoU)
- Stage3: logistic emission + HMM smoothing + AoB (bridge/abstain) + metrics

## Colab quickstart
1. Mount Drive and place DAVIS at:
   `/content/drive/MyDrive/DAVIS_unzipped/DAVIS`
2. Run:
```bash
pip -q install -r requirements.txt
pip -q install -e .
python -m pr2drag.cli --config configs/davis2016_cpu.yaml run_all
