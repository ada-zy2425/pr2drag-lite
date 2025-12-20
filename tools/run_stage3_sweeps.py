# tools/run_stage3_sweeps.py
from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

# --- YAML loader with fallback ---
def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "PyYAML not found. Install one of:\n"
            "  pip install pyyaml\n"
            "or in Colab:\n"
            "  !pip -q install pyyaml\n"
        ) from e
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _dump_yaml(obj: Dict[str, Any], path: Path) -> None:
    import yaml  # type: ignore
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)

def _deep_get(d: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def _deep_set(d: Dict[str, Any], keys: List[str], value: Any) -> None:
    cur: Any = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value

def _slug(x: str) -> str:
    x = str(x)
    x = x.replace(".", "p")
    x = re.sub(r"[^a-zA-Z0-9_\-]+", "_", x)
    x = re.sub(r"_+", "_", x).strip("_")
    return x

def _maybe_git_commit(repo_root: Path) -> Optional[str]:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo_root)).decode().strip()
        return out
    except Exception:
        return None

def _find_reliability_bins_csv(analysis_dir: Path) -> Optional[Path]:
    """
    Find a CSV that looks like:
    lo,hi,center,count,pos_rate,ci_low,ci_high
    """
    for p in sorted(analysis_dir.glob("*.csv")):
        try:
            df = pd.read_csv(p, nrows=5)
        except Exception:
            continue
        cols = set(df.columns.astype(str).tolist())
        need = {"lo", "hi", "center", "count", "pos_rate"}
        if need.issubset(cols):
            return p
    return None

def _compute_gate_from_bins(bins_csv: Path, tau: float) -> Dict[str, float]:
    df = pd.read_csv(bins_csv)
    for c in ["center", "count", "pos_rate"]:
        if c not in df.columns:
            return {}
    center = df["center"].astype(float).values
    count = df["count"].astype(float).values
    pos = df["pos_rate"].astype(float).values

    n = float(count.sum())
    if n <= 0:
        return {"tau": float(tau), "coverage": float("nan"), "risk": float("nan"), "kept_n": 0.0, "n": 0.0}

    kept = center >= float(tau)
    kept_n = float(count[kept].sum())
    if kept_n <= 0:
        return {"tau": float(tau), "coverage": 0.0, "risk": float("nan"), "kept_n": 0.0, "n": n}

    # risk = P(y=0 | kept) = 1 - P(y=1 | kept)
    # Using bin-aggregated pos_rate
    good = float((count[kept] * pos[kept]).sum())
    risk = float(1.0 - good / kept_n)
    coverage = float(kept_n / n)
    return {"tau": float(tau), "coverage": coverage, "risk": risk, "kept_n": kept_n, "n": n}

def _read_table_metrics(table_csv: Path) -> Dict[str, float]:
    df = pd.read_csv(table_csv)
    out: Dict[str, float] = {}
    if len(df) == 0:
        return out
    # Main aggregate metrics (you can extend later)
    for k in ["all_obs_p95", "all_fin_p95", "all_obs_fail", "all_fin_fail", "low_frac", "bridge_n", "abst_n"]:
        if k in df.columns:
            out[f"mean_{k}"] = float(df[k].mean())
            out[f"max_{k}"] = float(df[k].max())
    return out

@dataclass(frozen=True)
class RunSpec:
    tau_risk: float
    eps_gate: float
    max_bridge_len: int
    abstain_mode: str
    wcalib: str  # "off" | "isotonic" | "binning"
    out_tag: str

def _build_runs(
    base_cfg: Dict[str, Any],
    tau_risk_list: List[float],
    eps_gate_list: List[float],
    max_bridge_len_list: List[int],
    abstain_mode_list: List[str],
    wcalib_list: List[str],
    out_prefix: str,
) -> List[RunSpec]:
    # If a list is empty => use base config value (single)
    s3 = base_cfg.get("stage3", {})
    aob = s3.get("aob", {})
    wcal = s3.get("w_calibration", {})

    if not tau_risk_list:
        tau_risk_list = [float(s3.get("tau_risk", 0.02))]
    if not eps_gate_list:
        eps_gate_list = [float(aob.get("eps_gate", 0.1))]
    if not max_bridge_len_list:
        max_bridge_len_list = [int(aob.get("max_bridge_len", 40))]
    if not abstain_mode_list:
        abstain_mode_list = [str(aob.get("abstain_mode", "linear"))]
    if not wcalib_list:
        # infer from base cfg
        if bool(wcal.get("enable", True)):
            wcalib_list = [str(wcal.get("method", "isotonic"))]
        else:
            wcalib_list = ["off"]

    runs: List[RunSpec] = []
    for r0, eg, mbl, am, wc in itertools.product(
        tau_risk_list, eps_gate_list, max_bridge_len_list, abstain_mode_list, wcalib_list
    ):
        tag = f"{out_prefix}_r0{_slug(f'{r0:.4g}')}_eg{_slug(f'{eg:.4g}')}_mbl{mbl}_am{_slug(am)}_wc{_slug(wc)}"
        runs.append(
            RunSpec(
                tau_risk=float(r0),
                eps_gate=float(eg),
                max_bridge_len=int(mbl),
                abstain_mode=str(am),
                wcalib=str(wc).lower(),
                out_tag=tag,
            )
        )
    return runs

def _apply_runspec_to_cfg(cfg: Dict[str, Any], spec: RunSpec, wcalib_bins: int = 50) -> Dict[str, Any]:
    cfg = json.loads(json.dumps(cfg))  # deep copy via json
    # stage3 out dir
    _deep_set(cfg, ["stage3", "out_dir"], spec.out_tag)

    # tau risk
    _deep_set(cfg, ["stage3", "tau_mode"], "risk")
    _deep_set(cfg, ["stage3", "tau_risk"], float(spec.tau_risk))

    # AoB
    _deep_set(cfg, ["stage3", "aob", "eps_gate"], float(spec.eps_gate))
    _deep_set(cfg, ["stage3", "aob", "max_bridge_len"], int(spec.max_bridge_len))
    _deep_set(cfg, ["stage3", "aob", "abstain_mode"], str(spec.abstain_mode))

    # w calibration
    wc = spec.wcalib
    if wc in ("off", "false", "0", "none"):
        _deep_set(cfg, ["stage3", "w_calibration", "enable"], False)
    elif wc in ("isotonic", "binning"):
        _deep_set(cfg, ["stage3", "w_calibration", "enable"], True)
        _deep_set(cfg, ["stage3", "w_calibration", "method"], wc)
        _deep_set(cfg, ["stage3", "w_calibration", "bins"], int(wcalib_bins))
    else:
        raise ValueError(f"Unknown wcalib spec: {spec.wcalib} (use off|isotonic|binning)")
    return cfg

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_config", type=str, required=True, help="Path to base yaml config (e.g., configs/davis2016_cpu.yaml)")
    ap.add_argument("--out_prefix", type=str, default="s3", help="Prefix for stage3 out_dir name")
    ap.add_argument("--overwrite", action="store_true", help="If set, delete existing stage3 out_dir before running")
    ap.add_argument("--dry_run", action="store_true", help="If set, only print planned runs, do not execute")
    ap.add_argument("--wcalib_bins", type=int, default=50)

    # sweep params (cartesian product)
    ap.add_argument("--tau_risk", type=float, nargs="*", default=[], help="List of r0 to sweep, e.g. 0.01 0.02 0.05 0.08")
    ap.add_argument("--eps_gate", type=float, nargs="*", default=[], help="List of eps_gate to sweep, e.g. 0.05 0.1 0.2")
    ap.add_argument("--max_bridge_len", type=int, nargs="*", default=[], help="List of max_bridge_len to sweep, e.g. 20 40 60")
    ap.add_argument("--abstain_mode", type=str, nargs="*", default=[], help="List of abstain_mode, e.g. linear hold")
    ap.add_argument("--wcalib", type=str, nargs="*", default=[], help="List: off isotonic binning")

    args = ap.parse_args()

    base_path = Path(args.base_config)
    base_cfg = _load_yaml(base_path)

    base_out = Path(base_cfg["base_out"])
    stage2_train = base_out / "davis2016_stage2_fixed"
    stage2_val = base_out / "davis2016_val_stage2"

    if not stage2_train.is_dir() or not stage2_val.is_dir():
        raise FileNotFoundError(
            "Stage2 dirs not found. You should run Stage1+Stage2 once first.\n"
            f"  expected train: {stage2_train}\n"
            f"  expected val  : {stage2_val}\n"
        )

    runs = _build_runs(
        base_cfg=base_cfg,
        tau_risk_list=list(args.tau_risk),
        eps_gate_list=list(args.eps_gate),
        max_bridge_len_list=list(args.max_bridge_len),
        abstain_mode_list=list(args.abstain_mode),
        wcalib_list=list(args.wcalib),
        out_prefix=args.out_prefix,
    )

    print(f"[Plan] num_runs={len(runs)}")
    for i, r in enumerate(runs[:20]):
        print(f"  - {i:02d} {r.out_tag}")
    if len(runs) > 20:
        print("  ...")

    if args.dry_run:
        return

    # import here so script can be used even if pr2drag isn't installed globally
    from pr2drag.pipeline import stage3_train_eval  # type: ignore

    repo_root = Path(__file__).resolve().parents[1]
    git_commit = _maybe_git_commit(repo_root)

    sweep_dir = base_out / "_sweeps" / f"{args.out_prefix}"
    cfg_dir = sweep_dir / "configs"
    log_dir = sweep_dir / "logs"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    cfg_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, Any]] = []

    for idx, spec in enumerate(runs):
        cfg_run = _apply_runspec_to_cfg(base_cfg, spec, wcalib_bins=int(args.wcalib_bins))
        out_dir = base_out / cfg_run["stage3"]["out_dir"]
        analysis_dir = out_dir / "_analysis"

        cfg_path = cfg_dir / f"{spec.out_tag}.yaml"
        _dump_yaml(cfg_run, cfg_path)

        if out_dir.exists() and args.overwrite:
            shutil.rmtree(out_dir)

        log_path = log_dir / f"{spec.out_tag}.log"
        print(f"\n[Run {idx+1}/{len(runs)}] out_dir={out_dir}")

        # run stage3 and tee stdout/stderr
        with log_path.open("w", encoding="utf-8") as f:
            try:
                # stage3_train_eval writes a lot to stdout; we also mirror it to log
                # simplest: temporarily redirect prints by duplicating OS-level fds is overkill
                # here we just run and rely on internal prints + log for exceptions
                stage3_train_eval(cfg_run, stage2_train=stage2_train, stage2_val=stage2_val, out_dir=out_dir)
                f.write("[OK] stage3_train_eval finished\n")
            except Exception as e:
                f.write(f"[ERR] {repr(e)}\n")
                print(f"[ERR] failed: {e}")
                # still record failure
                summary_rows.append(
                    {
                        "out_tag": spec.out_tag,
                        **asdict(spec),
                        "status": "failed",
                        "error": repr(e),
                        "out_dir": str(out_dir),
                        "git_commit": git_commit,
                    }
                )
                continue

        # collect metrics
        row: Dict[str, Any] = {
            "out_tag": spec.out_tag,
            **asdict(spec),
            "status": "ok",
            "out_dir": str(out_dir),
            "git_commit": git_commit,
        }

        table_csv = analysis_dir / "table_v1.csv"
        if table_csv.exists():
            row.update(_read_table_metrics(table_csv))

        # try read val gate summary if pipeline wrote it; otherwise compute from bins csv
        gate_summary_json = analysis_dir / "val_gate_summary.json"
        if gate_summary_json.exists():
            try:
                row.update(json.loads(gate_summary_json.read_text(encoding="utf-8")))
            except Exception:
                pass
        else:
            bins_csv = _find_reliability_bins_csv(analysis_dir)
            if bins_csv is not None:
                row.update(_compute_gate_from_bins(bins_csv, tau=float(_deep_get(cfg_run, ["stage3", "tau_risk"], 0.0))))

        # also store the exact cfg used (for reproducibility)
        row["cfg_path"] = str(cfg_path)

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = sweep_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n[Done] wrote summary: {summary_path}")
    if len(summary_df) > 0:
        # quick print top configs by mean_all_fin_fail
        if "mean_all_fin_fail" in summary_df.columns:
            top = summary_df.sort_values("mean_all_fin_fail", ascending=True).head(10)
            print("\n[Top-10 by mean_all_fin_fail]")
            print(top[["out_tag", "tau_risk", "eps_gate", "max_bridge_len", "abstain_mode", "wcalib", "mean_all_fin_fail", "mean_all_fin_p95"]].to_string(index=False))


if __name__ == "__main__":
    main()
