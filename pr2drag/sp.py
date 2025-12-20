# pr2drag/sp.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    # numerically stable sigmoid
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    expx = np.exp(x[~pos])
    out[~pos] = expx / (1.0 + expx)
    return out


def _clip_prob(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    return np.clip(p, eps, 1.0 - eps)


def ece_score(p: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if p.size == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = float(len(p))
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if m.sum() == 0:
            continue
        acc = float(y[m].mean())
        conf = float(p[m].mean())
        ece += (float(m.sum()) / n) * abs(acc - conf)
    return float(ece)


def risk_at_coverage(p: np.ndarray, y: np.ndarray, coverage: float = 0.5) -> float:
    """
    Keep top coverage by confidence p, compute risk = 1-accuracy on kept subset.
    """
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    n = int(len(p))
    if n == 0:
        return float("nan")
    k = int(round(n * float(coverage)))
    k = max(1, min(n, k))
    idx = np.argsort(-p)[:k]
    acc = float((y[idx] == 1).mean())
    return float(1.0 - acc)


@dataclass
class HMMParams:
    prior_good: float
    p_stay_good: float
    p_stay_bad: float
    eps: float = 1e-12


def forward_backward_binary(emission_p_good: np.ndarray, hmm: HMMParams) -> np.ndarray:
    """
    Two-state HMM with states {bad=0, good=1}
    emission_p_good[t] is a proxy for p(state=good | x_t).
    We do pragmatic smoothing:
      likelihoods L_good = p, L_bad = 1-p.
    """
    p = np.clip(np.asarray(emission_p_good, dtype=np.float64), hmm.eps, 1.0 - hmm.eps)
    T = int(p.shape[0])
    if T == 0:
        return np.zeros((0,), dtype=np.float32)

    # Transition matrix A[s_prev, s_next] in order [bad, good]
    a11 = float(hmm.p_stay_bad)
    a10 = 1.0 - a11
    a00 = float(hmm.p_stay_good)
    a01 = 1.0 - a00
    A = np.array([[a11, a10], [a01, a00]], dtype=np.float64)

    pi_good = float(np.clip(hmm.prior_good, hmm.eps, 1.0 - hmm.eps))
    pi = np.array([1.0 - pi_good, pi_good], dtype=np.float64)

    # log-likelihoods
    ll = np.stack([np.log(1.0 - p), np.log(p)], axis=1)  # (T,2)

    # forward
    alpha = np.zeros((T, 2), dtype=np.float64)
    alpha[0] = np.log(pi) + ll[0]
    logA = np.log(A + hmm.eps)
    for t in range(1, T):
        prev = alpha[t - 1][:, None] + logA  # (2,2)
        m = prev.max(axis=0)
        alpha[t] = m + np.log(np.exp(prev - m).sum(axis=0)) + ll[t]

    # backward
    beta = np.zeros((T, 2), dtype=np.float64)
    beta[T - 1] = 0.0
    for t in range(T - 2, -1, -1):
        nxt = beta[t + 1] + ll[t + 1]  # (2,)
        tmp = logA + nxt[None, :]      # (2,2)
        m = tmp.max(axis=1)
        beta[t] = m + np.log(np.exp(tmp - m[:, None]).sum(axis=1))

    # posterior
    post = alpha + beta
    m = post.max(axis=1, keepdims=True)
    post = np.exp(post - m)
    post = post / (post.sum(axis=1, keepdims=True) + hmm.eps)
    w = post[:, 1]  # good
    return w.astype(np.float32)


def fit_emission_model(X: np.ndarray, y: np.ndarray, *, c: float = 1.0, max_iter: int = 2000) -> LogisticRegression:
    clf = LogisticRegression(
        C=float(c),
        max_iter=int(max_iter),
        solver="lbfgs",
        n_jobs=1,
    )
    clf.fit(X, y)
    return clf


def emission_metrics(p: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    if p.size == 0:
        return {"auroc": float("nan"), "ece": float("nan"), "risk@50%": float("nan")}
    auroc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")
    ece = ece_score(p, y)
    r50 = risk_at_coverage(p, y, coverage=0.5)
    return {"auroc": float(auroc), "ece": float(ece), "risk@50%": float(r50)}


# -------------------------------
# Calibration: temperature scaling
# -------------------------------

def apply_temperature_scaling(p_raw: np.ndarray, T: float, *, eps: float = 1e-6) -> np.ndarray:
    """
    p = sigmoid(logit(p_raw) / T)
    T>1 => less confident (flatter).
    """
    p = _clip_prob(p_raw, eps=eps)
    logit = np.log(p) - np.log(1.0 - p)
    T = float(max(T, 1e-6))
    p_cal = _sigmoid(logit / T)
    return p_cal.astype(np.float32)


def _nll_bernoulli(p: np.ndarray, y: np.ndarray, eps: float = 1e-6) -> float:
    p = _clip_prob(p, eps=eps)
    y = np.asarray(y, dtype=np.float64)
    # NLL = -mean( y log p + (1-y) log(1-p) )
    return float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean())


def fit_temperature_grid(
    p_raw: np.ndarray,
    y: np.ndarray,
    *,
    grid: Optional[np.ndarray] = None,
    eps: float = 1e-6,
) -> Tuple[float, Dict[str, float]]:
    """
    Fit temperature T by minimizing NLL on (p_raw, y).
    Pure numpy grid search => stable, dependency-free.

    Returns:
      best_T, diagnostics dict
    """
    p_raw = np.asarray(p_raw, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    if p_raw.size == 0 or y.size == 0 or p_raw.size != y.size:
        return 1.0, {"nll_raw": float("nan"), "nll_cal": float("nan")}

    if grid is None:
        # log-spaced grid covers typical overconfidence ranges
        grid = np.exp(np.linspace(np.log(0.25), np.log(10.0), 60)).astype(np.float64)

    nll_raw = _nll_bernoulli(p_raw, y, eps=eps)

    best_T = 1.0
    best_nll = nll_raw
    for T in grid:
        p_cal = apply_temperature_scaling(p_raw, float(T), eps=eps)
        nll = _nll_bernoulli(p_cal, y, eps=eps)
        if np.isfinite(nll) and nll < best_nll:
            best_nll = nll
            best_T = float(T)

    return best_T, {"nll_raw": float(nll_raw), "nll_cal": float(best_nll)}
