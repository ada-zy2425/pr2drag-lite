#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from pr2.posterior.model import SklearnLogRegPosterior, TorchMLPPosterior
from pr2.posterior.calibrate import IdentityCalibrator, TemperatureScalingCalibrator, IsotonicCalibrator
from pr2.posterior.bundle import PosteriorBundle
from pr2.posterior.metrics import ece, brier, risk_coverage_curve
from pr2.utils.io import iter_jsonl, atomic_write_json


def load_xy(jsonl_path: str) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for row in iter_jsonl(jsonl_path):
        # 支持多种字段名：x / E / evidence
        x = row.get("x", None)
        if x is None:
            x = row.get("E", None)
        if x is None:
            x = row.get("evidence", None)
        if x is None:
            raise KeyError("Tier-0 jsonl 每行需要包含 x/E/evidence 字段（list of float）")
        X.append(x)
        y.append(int(row["y"]))
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    return X, y


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="train jsonl (x,y)")
    ap.add_argument("--calib", required=True, help="calib jsonl (x,y)")
    ap.add_argument("--out", required=True, help="output dir for bundle + reports")
    ap.add_argument("--model", default="logreg", choices=["logreg", "mlp"])
    ap.add_argument("--calibrator", default="temperature", choices=["identity", "temperature", "isotonic"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    Xtr, ytr = load_xy(args.train)
    Xc, yc = load_xy(args.calib)

    if args.model == "logreg":
        model = SklearnLogRegPosterior()
    else:
        model = TorchMLPPosterior(seed=args.seed)

    model.fit(Xtr, ytr)
    p_raw = model.predict_proba(Xc)

    if args.calibrator == "identity":
        cal = IdentityCalibrator()
    elif args.calibrator == "temperature":
        cal = TemperatureScalingCalibrator()
    else:
        cal = IsotonicCalibrator()

    cal.fit(p_raw, yc)
    p = cal.predict(p_raw)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = PosteriorBundle(model=model, calibrator=cal)
    bundle.save(out_dir)

    report = {
        "ece": ece(p, yc),
        "brier": brier(p, yc),
        "risk_coverage": {k: v.tolist() for k, v in risk_coverage_curve(p, yc).items()},
        "n_train": int(len(Xtr)),
        "n_calib": int(len(Xc)),
        "model": args.model,
        "calibrator": args.calibrator,
    }
    atomic_write_json(out_dir / "calib_report.json", report, indent=2)
    print(f"[train_posterior] saved bundle to {out_dir} and report to calib_report.json")


if __name__ == "__main__":
    main()
