from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from pr2.metrics.intent import compute_intent_success
from pr2.metrics.safety import compute_catastrophic
from pr2.metrics.flicker import compute_flicker_p95
from pr2.metrics.coverage import compute_coverage
from pr2.utils.io import read_json, atomic_write_json


def bootstrap_ci(values: np.ndarray, fn=np.mean, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, values.size, size=values.size)
        stats.append(fn(values[idx]))
    stats = np.sort(np.asarray(stats))
    lo = np.quantile(stats, alpha / 2)
    hi = np.quantile(stats, 1 - alpha / 2)
    return float(fn(values)), float(lo), float(hi)


def aggregate_run(runs_dir: Path) -> Dict[str, Any]:
    rows = []
    for task_dir in sorted(runs_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        res_path = task_dir / "result.json"
        if not res_path.exists():
            continue
        rows.append(read_json(res_path))

    df = pd.DataFrame(rows)
    if df.empty:
        return {"n_tasks": 0, "warning": "no results found"}

    intent = df.apply(compute_intent_success, axis=1).to_numpy(dtype=np.float32)
    catastroph = df.apply(compute_catastrophic, axis=1).to_numpy(dtype=np.float32)
    flicker = df.apply(compute_flicker_p95, axis=1).to_numpy(dtype=np.float32)
    coverage = df.apply(compute_coverage, axis=1).to_numpy(dtype=np.float32)

    summary = {
        "n_tasks": int(len(df)),
        "intent_success_rate": float(np.mean(intent)),
        "catastrophic_rate": float(np.mean(catastroph)),
        "flicker_p95_mean": float(np.mean(flicker)),
        "coverage_mean": float(np.mean(coverage)),
        "intent_success_rate_ci": bootstrap_ci(intent, fn=np.mean),
        "catastrophic_rate_ci": bootstrap_ci(catastroph, fn=np.mean),
        "flicker_p95_mean_ci": bootstrap_ci(flicker, fn=np.mean),
        "coverage_mean_ci": bootstrap_ci(coverage, fn=np.mean),
        "by_difficulty": {},
    }

    if "difficulty" in df.columns:
        for diff, g in df.groupby("difficulty"):
            gi = g.apply(compute_intent_success, axis=1).to_numpy(dtype=np.float32)
            gc = g.apply(compute_catastrophic, axis=1).to_numpy(dtype=np.float32)
            gf = g.apply(compute_flicker_p95, axis=1).to_numpy(dtype=np.float32)
            gv = g.apply(compute_coverage, axis=1).to_numpy(dtype=np.float32)
            summary["by_difficulty"][str(diff)] = {
                "n": int(len(g)),
                "intent_success_rate": float(np.mean(gi)),
                "catastrophic_rate": float(np.mean(gc)),
                "flicker_p95_mean": float(np.mean(gf)),
                "coverage_mean": float(np.mean(gv)),
            }

    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", required=True, help="run output directory (contains task subdirs)")
    ap.add_argument("--out", required=True, help="output summary json path")
    args = ap.parse_args()

    summary = aggregate_run(Path(args.runs_dir))
    atomic_write_json(args.out, summary, indent=2)
    print(f"[aggregate] wrote: {args.out}")


if __name__ == "__main__":
    main()
