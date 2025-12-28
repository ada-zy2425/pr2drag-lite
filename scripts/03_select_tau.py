#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np

from pr2.posterior.bundle import PosteriorBundle
from pr2.posterior.crc_threshold import select_tau_by_risk_cp
from pr2.utils.io import iter_jsonl, atomic_write_json


def load_xy(jsonl_path: str) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for row in iter_jsonl(jsonl_path):
        x = row.get("x", row.get("E", row.get("evidence", None)))
        if x is None:
            raise KeyError("Tier-0 jsonl 每行需要包含 x/E/evidence 字段（list of float）")
        X.append(x)
        y.append(int(row["y"]))
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True, help="posterior bundle dir (contains bundle.pkl)")
    ap.add_argument("--calib", required=True, help="calib jsonl (x,y)")
    ap.add_argument("--out", required=True, help="output policy_instantiation.json")
    ap.add_argument("--target_risk", type=float, default=0.10)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--min_selected", type=int, default=50)
    args = ap.parse_args()

    bundle = PosteriorBundle.load(args.bundle)
    Xc, yc = load_xy(args.calib)
    w = bundle.predict_wtilde(Xc)

    sel = select_tau_by_risk_cp(w, yc, target_risk=args.target_risk, alpha=args.alpha, min_selected=args.min_selected)
    payload = {
        "tau": sel.tau,
        "target_risk": sel.target_risk,
        "alpha": sel.alpha,
        "coverage": sel.coverage,
        "n_selected": sel.n_selected,
        "n_total": sel.n_total,
        "n_errors": sel.n_errors,
        "method": sel.method,
        "details": sel.details,
    }
    atomic_write_json(args.out, payload, indent=2)
    print(f"[select_tau] wrote: {args.out}")


if __name__ == "__main__":
    main()
