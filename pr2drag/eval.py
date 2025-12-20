from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


def l2_err(z: np.ndarray, z_ref: np.ndarray) -> np.ndarray:
    d = z.astype(np.float64) - z_ref.astype(np.float64)
    return np.sqrt((d * d).sum(axis=1)).astype(np.float32)


def pctl(x: np.ndarray, q: float) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return float("nan")
    return float(np.quantile(x, q))


@dataclass
class SeqMetrics:
    seq: str
    T: int
    tau: float
    low_frac: float

    all_obs_p95: float
    all_fin_p95: float
    all_obs_fail: float
    all_fin_fail: float

    low_n: int
    low_obs_p95: float
    low_fin_p95: float
    low_obs_fail: float
    low_fin_fail: float

    bridge_n: int
    bridge_obs_p95: float
    bridge_fin_p95: float
    bridge_obs_fail: float
    bridge_fin_fail: float

    abst_n: int
    abst_obs_p95: float
    abst_fin_p95: float
    abst_obs_fail: float
    abst_fin_fail: float


def compute_seq_metrics(
    *,
    seq: str,
    z_gt: np.ndarray,
    z_obs: np.ndarray,
    z_fin: np.ndarray,
    w: np.ndarray,
    tau: float,
    is_bridge: np.ndarray,
    is_abst: np.ndarray,
    fail_px: float,
) -> SeqMetrics:
    T = int(z_gt.shape[0])
    err_obs = l2_err(z_obs, z_gt)
    err_fin = l2_err(z_fin, z_gt)

    low = w < tau
    low_n = int(low.sum())
    bridge_n = int(is_bridge.sum())
    abst_n = int(is_abst.sum())

    def _fail_rate(err: np.ndarray, mask: np.ndarray) -> float:
        if mask.sum() == 0:
            return float("nan")
        return float((err[mask] > fail_px).mean())

    m_all = np.ones((T,), dtype=bool)

    return SeqMetrics(
        seq=seq,
        T=T,
        tau=float(tau),
        low_frac=float(low.mean()),

        all_obs_p95=pctl(err_obs[m_all], 0.95),
        all_fin_p95=pctl(err_fin[m_all], 0.95),
        all_obs_fail=_fail_rate(err_obs, m_all),
        all_fin_fail=_fail_rate(err_fin, m_all),

        low_n=low_n,
        low_obs_p95=pctl(err_obs[low], 0.95) if low_n else float("nan"),
        low_fin_p95=pctl(err_fin[low], 0.95) if low_n else float("nan"),
        low_obs_fail=_fail_rate(err_obs, low),
        low_fin_fail=_fail_rate(err_fin, low),

        bridge_n=bridge_n,
        bridge_obs_p95=pctl(err_obs[is_bridge], 0.95) if bridge_n else float("nan"),
        bridge_fin_p95=pctl(err_fin[is_bridge], 0.95) if bridge_n else float("nan"),
        bridge_obs_fail=_fail_rate(err_obs, is_bridge),
        bridge_fin_fail=_fail_rate(err_fin, is_bridge),

        abst_n=abst_n,
        abst_obs_p95=pctl(err_obs[is_abst], 0.95) if abst_n else float("nan"),
        abst_fin_p95=pctl(err_fin[is_abst], 0.95) if abst_n else float("nan"),
        abst_obs_fail=_fail_rate(err_obs, is_abst),
        abst_fin_fail=_fail_rate(err_fin, is_abst),
    )


def metrics_to_row(m: SeqMetrics) -> Dict[str, float]:
    return {
        "seq": m.seq,
        "T": m.T,
        "tau": m.tau,
        "low_frac": m.low_frac,

        "all_obs_p95": m.all_obs_p95,
        "all_fin_p95": m.all_fin_p95,
        "all_obs_fail": m.all_obs_fail,
        "all_fin_fail": m.all_fin_fail,

        "low_n": m.low_n,
        "low_obs_p95": m.low_obs_p95,
        "low_fin_p95": m.low_fin_p95,
        "low_obs_fail": m.low_obs_fail,
        "low_fin_fail": m.low_fin_fail,

        "bridge_n": m.bridge_n,
        "bridge_obs_p95": m.bridge_obs_p95,
        "bridge_fin_p95": m.bridge_fin_p95,
        "bridge_obs_fail": m.bridge_obs_fail,
        "bridge_fin_fail": m.bridge_fin_fail,

        "abst_n": m.abst_n,
        "abst_obs_p95": m.abst_obs_p95,
        "abst_fin_p95": m.abst_fin_p95,
        "abst_obs_fail": m.abst_obs_fail,
        "abst_fin_fail": m.abst_fin_fail,
    }
