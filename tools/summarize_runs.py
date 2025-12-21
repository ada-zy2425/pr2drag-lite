# tools/summarize_runs.py
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


REQUIRED_COLS = [
    "seq", "tau", "low_frac",
    "all_obs_p95", "all_fin_p95",
    "all_obs_fail", "all_fin_fail",
    "bridge_n", "abst_n",
]

OPTIONAL_COLS = [
    "T",
]

def _safe_read_csv(p: Path) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(p)
        if df.empty:
            return None
        return df
    except Exception as e:
        print(f"[WARN] failed to read {p}: {e}")
        return None

def _infer_run_name(run_dir: Path) -> str:
    return run_dir.name

def _find_table_csvs(base_out: Path) -> List[Path]:
    # match: <run_dir>/_analysis/table_v1.csv
    return sorted(base_out.glob("davis2016_*/_analysis/table_v1.csv"))

def _extract_run_row(run_dir: Path, table_csv: Path) -> Optional[Dict]:
    df = _safe_read_csv(table_csv)
    if df is None:
        return None

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        print(f"[WARN] {table_csv} missing columns: {missing}")
        return None

    # Aggregate across sequences (mean) + keep worst-case
    row: Dict = {}
    row["run"] = _infer_run_name(run_dir)
    row["table_csv"] = str(table_csv)

    # means
    row["mean_fin_p95"] = float(df["all_fin_p95"].mean())
    row["mean_fin_fail"] = float(df["all_fin_fail"].mean())
    row["mean_obs_p95"] = float(df["all_obs_p95"].mean())
    row["mean_obs_fail"] = float(df["all_obs_fail"].mean())
    row["mean_low_frac"] = float(df["low_frac"].mean())
    row["mean_tau"] = float(df["tau"].mean())

    # worst sequences by fin_fail then fin_p95
    worst = df.sort_values(["all_fin_fail", "all_fin_p95"], ascending=[False, False]).head(1).iloc[0]
    row["worst_seq"] = str(worst["seq"])
    row["worst_fin_fail"] = float(worst["all_fin_fail"])
    row["worst_fin_p95"] = float(worst["all_fin_p95"])

    # coverage proxy (1 - low_frac) is NOT exact; we only store mean(low_frac) here.
    # If you logged coverage in stdout only, keep it separately; or extend pipeline to write it into table.
    return row

def _collect_worst_sequences(base_out: Path, topk: int = 10) -> pd.DataFrame:
    rows = []
    for table_csv in _find_table_csvs(base_out):
        run_dir = table_csv.parent.parent
        df = _safe_read_csv(table_csv)
        if df is None:
            continue
        if not set(REQUIRED_COLS).issubset(set(df.columns)):
            continue
        df = df.copy()
        df["run"] = _infer_run_name(run_dir)
        rows.append(df[["run"] + REQUIRED_COLS])
    if not rows:
        return pd.DataFrame()
    all_df = pd.concat(rows, ignore_index=True)
    all_df = all_df.sort_values(["all_fin_fail", "all_fin_p95"], ascending=[False, False])
    return all_df.head(topk)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_out", type=str, required=True,
                    help="e.g. /content/drive/MyDrive/pr2drag_data")
    ap.add_argument("--out", type=str, default=None,
                    help="output directory; default: <base_out>/_summary")
    ap.add_argument("--topk", type=int, default=15)
    args = ap.parse_args()

    base_out = Path(args.base_out).expanduser().resolve()
    if not base_out.exists():
        raise FileNotFoundError(f"base_out not found: {base_out}")

    out_dir = Path(args.out).expanduser().resolve() if args.out else (base_out / "_summary")
    out_dir.mkdir(parents=True, exist_ok=True)

    table_csvs = _find_table_csvs(base_out)
    if not table_csvs:
        print(f"[ERROR] no table_v1.csv found under {base_out}/davis2016_*/_analysis/")
        return

    run_rows = []
    for table_csv in table_csvs:
        run_dir = table_csv.parent.parent
        r = _extract_run_row(run_dir, table_csv)
        if r is not None:
            run_rows.append(r)

    if not run_rows:
        print("[ERROR] no valid runs parsed.")
        return

    summary = pd.DataFrame(run_rows)
    summary = summary.sort_values(["mean_fin_fail", "mean_fin_p95"], ascending=[True, True])

    summary_path = out_dir / "runs_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[OK] wrote {summary_path}")

    worst_df = _collect_worst_sequences(base_out, topk=args.topk)
    worst_path = out_dir / f"worst_sequences_top{args.topk}.csv"
    worst_df.to_csv(worst_path, index=False)
    print(f"[OK] wrote {worst_path}")

    # also print best run
    best = summary.iloc[0]
    print("\n=== BEST RUN ===")
    print(best.to_string())

if __name__ == "__main__":
    main()