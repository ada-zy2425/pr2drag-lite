from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from pr2.utils.io import read_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib_report", required=True, help="tier0_calibration_report.json")
    ap.add_argument("--out", required=True, help="output png path")
    args = ap.parse_args()

    rep = read_json(args.calib_report)
    rc = rep.get("risk_coverage", {})
    tau = np.array(rc.get("tau", []), dtype=np.float32)
    coverage = np.array(rc.get("coverage", []), dtype=np.float32)
    risk = np.array(rc.get("risk", []), dtype=np.float32)

    plt.figure()
    plt.plot(coverage, risk)
    plt.xlabel("Coverage (p >= tau)")
    plt.ylabel("Risk (error rate)")
    plt.title("Risk-Coverage Curve")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"[visualize] wrote: {args.out}")


if __name__ == "__main__":
    main()
