from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

from .utils import ensure_dir, save_json, get_logger


def _ece_binary(probs: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    probs = np.asarray(probs, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    n = len(y)
    if n == 0:
        return float("nan")
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        if i == bins - 1:
            idx = (probs >= lo) & (probs <= hi)
        else:
            idx = (probs >= lo) & (probs < hi)
        m = int(idx.sum())
        if m == 0:
            continue
        acc = float(y[idx].mean())
        conf = float(probs[idx].mean())
        ece += (m / n) * abs(acc - conf)
    return float(ece)


def risk_at_coverage(probs: np.ndarray, y: np.ndarray, coverage: float = 0.5) -> float:
    """
    coverage=0.5: keep top 50% confident frames (largest probs),
    compute risk=1-accuracy on that subset.
    """
    probs = np.asarray(probs, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    n = len(y)
    if n == 0:
        return float("nan")
    k = int(np.ceil(n * coverage))
    if k <= 0:
        return float("nan")
    idx = np.argsort(-probs)[:k]
    acc = float(y[idx].mean())
    return float(1.0 - acc)


def p95(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return float("nan")
    return float(np.quantile(x, 0.95))


def fail_rate(err: np.ndarray, thr: float) -> float:
    err = np.asarray(err, dtype=np.float64)
    if err.size == 0:
        return float("nan")
    return float(np.mean(err > thr))


def per_seq_summary(
    seq: str,
    tau: float,
    w: np.ndarray,
    err_obs: np.ndarray,
    err_fin: np.ndarray,
    low_mask: np.ndarray,
    bridged_mask: np.ndarray,
    abstained_mask: np.ndarray,
    fail_thr: float,
) -> Dict[str, Any]:
    T = int(len(err_obs))
    all_obs_p95 = p95(err_obs)
    all_fin_p95 = p95(err_fin)
    all_obs_fail = fail_rate(err_obs, fail_thr)
    all_fin_fail = fail_rate(err_fin, fail_thr)

    low_n = int(low_mask.sum())
    bridge_n = int(bridged_mask.sum())
    abst_n = int(abstained_mask.sum())

    def _cond_stats(mask: np.ndarray) -> Tuple[float, float, float, float]:
        idx = np.where(mask)[0]
        if idx.size == 0:
            return float("nan"), float("nan"), float("nan"), float("nan")
        return (
            p95(err_obs[idx]),
            p95(err_fin[idx]),
            fail_rate(err_obs[idx], fail_thr),
            fail_rate(err_fin[idx], fail_thr),
        )

    low_obs_p95, low_fin_p95, low_obs_fail, low_fin_fail = _cond_stats(low_mask)
    bridge_obs_p95, bridge_fin_p95, bridge_obs_fail, bridge_fin_fail = _cond_stats(bridged_mask)
    abst_obs_p95, abst_fin_p95, abst_obs_fail, abst_fin_fail = _cond_stats(abstained_mask)

    return {
        "seq": seq,
        "T": T,
        "tau": float(tau),
        "low_frac": float(np.mean(low_mask)) if T > 0 else 0.0,
        "all_obs_p95": all_obs_p95,
        "all_fin_p95": all_fin_p95,
        "all_obs_fail": all_obs_fail,
        "all_fin_fail": all_fin_fail,
        "low_n": low_n,
        "low_obs_p95": low_obs_p95,
        "low_fin_p95": low_fin_p95,
        "low_obs_fail": low_obs_fail,
        "low_fin_fail": low_fin_fail,
        "bridge_n": bridge_n,
        "bridge_obs_p95": bridge_obs_p95,
        "bridge_fin_p95": bridge_fin_p95,
        "bridge_obs_fail": bridge_obs_fail,
        "bridge_fin_fail": bridge_fin_fail,
        "abst_n": abst_n,
        "abst_obs_p95": abst_obs_p95,
        "abst_fin_p95": abst_fin_p95,
        "abst_obs_fail": abst_obs_fail,
        "abst_fin_fail": abst_fin_fail,
    }


def write_analysis(
    out_dir: str,
    table: pd.DataFrame,
    w_all: np.ndarray,
    err_all: np.ndarray,
    y_all: np.ndarray,
    probs_all: np.ndarray,
    tau: float,
    feat_stats: Dict[str, Any],
    cfg_analysis: Dict[str, Any],
) -> None:
    logger = get_logger()
    analysis_dir = os.path.join(out_dir, "_analysis")
    ensure_dir(analysis_dir)

    # Save table
    csv_path = os.path.join(analysis_dir, "table_v1.csv")
    table.to_csv(csv_path, index=False)
    logger.info(f"[OK] wrote: {csv_path}")

    # Corr
    corr_path = os.path.join(analysis_dir, "corr_wtil_err.txt")
    rho = float("nan")
    try:
        rho = float(spearmanr(w_all, -err_all, nan_policy="omit").correlation)
    except Exception:
        rho = float("nan")
    with open(corr_path, "w", encoding="utf-8") as f:
        f.write(f"spearman(wtil, -err_obs) = {rho}\n")
    logger.info(f"[OK] wrote: {corr_path}")

    # Save feat stats
    feat_path = os.path.join(analysis_dir, "feat_stats.json")
    save_json(feat_path, feat_stats)
    logger.info(f"[OK] wrote: {feat_path}")

    if not bool(cfg_analysis.get("make_plots", True)):
        return

    bins = int(cfg_analysis.get("bins", 10))

    # Gate bins plot: prob vs err
    fig1 = os.path.join(analysis_dir, "fig_gate_bins.png")
    try:
        edges = np.linspace(0, 1, bins + 1)
        xs, ys = [], []
        for i in range(bins):
            lo, hi = edges[i], edges[i + 1]
            if i == bins - 1:
                idx = (probs_all >= lo) & (probs_all <= hi)
            else:
                idx = (probs_all >= lo) & (probs_all < hi)
            if idx.sum() == 0:
                continue
            xs.append((lo + hi) / 2.0)
            ys.append(np.nanmedian(err_all[idx]))
        plt.figure()
        plt.plot(xs, ys, marker="o")
        plt.xlabel("Emission prob (calibrated)")
        plt.ylabel("Median obs error (px)")
        plt.title("Gate bins: prob vs obs error")
        plt.tight_layout()
        plt.savefig(fig1, dpi=150)
        plt.close()
        logger.info(f"[OK] wrote: {fig1}")
    except Exception as e:
        logger.warning(f"[WARN] failed to write {fig1}: {e}")

    # Tau hist
    fig2 = os.path.join(analysis_dir, "fig_tau_hist.png")
    try:
        plt.figure()
        plt.hist(w_all, bins=50)
        plt.axvline(float(tau), linestyle="--")
        plt.xlabel("wtil")
        plt.ylabel("count")
        plt.title("Histogram of wtil with tau")
        plt.tight_layout()
        plt.savefig(fig2, dpi=150)
        plt.close()
        logger.info(f"[OK] wrote: {fig2}")
    except Exception as e:
        logger.warning(f"[WARN] failed to write {fig2}: {e}")

    # low_frac hist
    fig3 = os.path.join(analysis_dir, "fig_lowfrac_hist.png")
    try:
        plt.figure()
        plt.hist(table["low_frac"].values, bins=20)
        plt.xlabel("low_frac per seq")
        plt.ylabel("count")
        plt.title("Histogram of low_frac (val)")
        plt.tight_layout()
        plt.savefig(fig3, dpi=150)
        plt.close()
        logger.info(f"[OK] wrote: {fig3}")
    except Exception as e:
        logger.warning(f"[WARN] failed to write {fig3}: {e}")

    # feat hist placeholder (stored in stats; plot dims distribution)
    fig4 = os.path.join(analysis_dir, "fig_feat_hist.png")
    try:
        # Plot first 7 dims distribution summary from stats (if any)
        dims = sorted([k for k in feat_stats.keys() if k.startswith("dim")])
        plt.figure(figsize=(8, 4))
        vals = [feat_stats[d]["mean"] for d in dims]
        plt.bar(range(len(vals)), vals)
        plt.xticks(range(len(vals)), dims, rotation=45)
        plt.ylabel("mean")
        plt.title("Feature means (train)")
        plt.tight_layout()
        plt.savefig(fig4, dpi=150)
        plt.close()
        logger.info(f"[OK] wrote: {fig4}")
    except Exception as e:
        logger.warning(f"[WARN] failed to write {fig4}: {e}")


def compute_emission_metrics(probs: np.ndarray, y: np.ndarray, bins: int = 10) -> Dict[str, float]:
    probs = np.asarray(probs, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    out = {}
    try:
        out["AUROC"] = float(roc_auc_score(y, probs))
    except Exception:
        out["AUROC"] = float("nan")
    out["ECE"] = float(_ece_binary(probs, y, bins=bins))
    out["risk@50%"] = float(risk_at_coverage(probs, y, coverage=0.5))
    return out
