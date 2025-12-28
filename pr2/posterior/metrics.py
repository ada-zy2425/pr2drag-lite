from __future__ import annotations

from typing import Dict

import numpy as np


def ece(p: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """
    Expected Calibration Error (ECE) for binary classification.
    """
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece_val = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if mask.sum() == 0:
            continue
        acc = y[mask].mean()
        conf = p[mask].mean()
        ece_val += (mask.mean()) * abs(acc - conf)
    return float(ece_val)


def brier(p: np.ndarray, y: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return float(np.mean((p - y) ** 2))


def risk_coverage_curve(p: np.ndarray, y: np.ndarray, num_points: int = 101) -> Dict[str, np.ndarray]:
    """
    输出 (coverage, risk) 曲线：
    - coverage: 被选中的样本比例（p >= tau）
    - risk: 选中样本的错误率（1-y 的均值）
    """
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    taus = np.linspace(0.0, 1.0, num_points)
    coverage = np.zeros_like(taus)
    risk = np.zeros_like(taus)
    for i, tau in enumerate(taus):
        sel = p >= tau
        coverage[i] = sel.mean()
        risk[i] = 0.0 if sel.sum() == 0 else (1.0 - y[sel]).mean()
    return {"tau": taus.astype(np.float32), "coverage": coverage.astype(np.float32), "risk": risk.astype(np.float32)}
