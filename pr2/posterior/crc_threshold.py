from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from scipy.stats import beta


@dataclass
class ThresholdSelectionResult:
    tau: float
    target_risk: float
    alpha: float
    coverage: float
    n_selected: int
    n_total: int
    n_errors: int
    method: str
    details: Dict[str, float]


def clopper_pearson_ucb(k: int, n: int, alpha: float) -> float:
    """
    Clopper-Pearson 上置信界：P(err_rate <= u) >= 1-alpha.
    """
    if n <= 0:
        return 0.0
    if k < 0 or k > n:
        raise ValueError("k must satisfy 0 <= k <= n.")
    if k == n:
        return 1.0
    return float(beta.ppf(1 - alpha, k + 1, n - k))


def select_tau_by_risk_cp(
    p: np.ndarray,
    y: np.ndarray,
    target_risk: float,
    alpha: float = 0.1,
    min_selected: int = 50,
) -> ThresholdSelectionResult:
    """
    单次阈值选择（审计友好）：
    - 对每个候选 tau，计算选中集合 S={p>=tau}
    - 计算 error rate 的 Clopper-Pearson 上界 UCB
    - 选 coverage 最大（tau 最小）且 UCB <= target_risk 的 tau
    """
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    if p.shape[0] != y.shape[0]:
        raise ValueError("p and y must have same length.")

    cand = np.unique(np.clip(p, 0.0, 1.0))
    cand = np.concatenate([cand, np.array([1.0])])
    cand = np.unique(cand)
    cand.sort()

    best_tau = 1.0
    best_cov = -1.0
    best_stats = None

    N = len(p)
    for tau in cand:
        sel = p >= tau
        n = int(sel.sum())
        if n < min_selected:
            continue
        k = int(np.sum(1 - y[sel]))  # errors
        ucb = clopper_pearson_ucb(k=k, n=n, alpha=alpha)
        cov = n / N
        if ucb <= target_risk:
            if cov > best_cov or (abs(cov - best_cov) < 1e-12 and tau < best_tau):
                best_tau = float(tau)
                best_cov = float(cov)
                best_stats = {"ucb": float(ucb), "k": float(k), "n": float(n)}

    if best_stats is None:
        sel = p >= 1.0
        n = int(sel.sum())
        k = int(np.sum(1 - y[sel])) if n > 0 else 0
        best_stats = {"ucb": 0.0, "k": float(k), "n": float(n)}
        best_cov = float(n / N)
        best_tau = 1.0

    return ThresholdSelectionResult(
        tau=float(best_tau),
        target_risk=float(target_risk),
        alpha=float(alpha),
        coverage=float(best_cov),
        n_selected=int(best_stats["n"]),
        n_total=int(N),
        n_errors=int(best_stats["k"]),
        method="cp_ucb",
        details=best_stats,
    )
