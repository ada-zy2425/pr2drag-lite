# pr2drag/sp.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Union

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.dummy import DummyClassifier
from sklearn.base import ClassifierMixin


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    expx = np.exp(x[~pos])
    out[~pos] = expx / (1.0 + expx)
    return out


def _clip_prob(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    if np.any(~np.isfinite(p)):
        p = np.nan_to_num(p, nan=0.5, posinf=1.0 - eps, neginf=eps)
    return np.clip(p, eps, 1.0 - eps)


def _sanitize_Xy(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (N,F); got shape={X.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X,y length mismatch: X has N={X.shape[0]}, y has N={y.shape[0]}")
    if np.any(~np.isfinite(X)):
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, y


def ece_score(p: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if p.size == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
    ece = 0.0
    n = float(len(p))
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if int(m.sum()) == 0:
            continue
        acc = float(y[m].mean())
        conf = float(p[m].mean())
        ece += (float(m.sum()) / n) * abs(acc - conf)
    return float(ece)


def risk_at_coverage(p: np.ndarray, y: np.ndarray, coverage: float = 0.5) -> float:
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.int64).reshape(-1)
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
    p = np.clip(np.asarray(emission_p_good, dtype=np.float64).reshape(-1), hmm.eps, 1.0 - hmm.eps)
    T = int(p.shape[0])
    if T == 0:
        return np.zeros((0,), dtype=np.float32)

    a_bb = float(np.clip(hmm.p_stay_bad, hmm.eps, 1.0 - hmm.eps))
    a_bg = 1.0 - a_bb
    a_gg = float(np.clip(hmm.p_stay_good, hmm.eps, 1.0 - hmm.eps))
    a_gb = 1.0 - a_gg
    A = np.array([[a_bb, a_bg], [a_gb, a_gg]], dtype=np.float64)

    pi_good = float(np.clip(hmm.prior_good, hmm.eps, 1.0 - hmm.eps))
    pi = np.array([1.0 - pi_good, pi_good], dtype=np.float64)

    ll = np.stack([np.log(1.0 - p), np.log(p)], axis=1)  # (T,2)
    logA = np.log(A + hmm.eps)

    alpha = np.zeros((T, 2), dtype=np.float64)
    alpha[0] = np.log(pi) + ll[0]
    for t in range(1, T):
        prev = alpha[t - 1][:, None] + logA
        m = prev.max(axis=0)
        alpha[t] = m + np.log(np.exp(prev - m).sum(axis=0)) + ll[t]

    beta = np.zeros((T, 2), dtype=np.float64)
    beta[T - 1] = 0.0
    for t in range(T - 2, -1, -1):
        nxt = beta[t + 1] + ll[t + 1]
        tmp = logA + nxt[None, :]
        m = tmp.max(axis=1)
        beta[t] = m + np.log(np.exp(tmp - m[:, None]).sum(axis=1))

    post = alpha + beta
    m = post.max(axis=1, keepdims=True)
    post = np.exp(post - m)
    post = post / (post.sum(axis=1, keepdims=True) + hmm.eps)
    w = post[:, 1]
    return w.astype(np.float32)


def fit_emission_model(
    X: np.ndarray,
    y: np.ndarray,
    *,
    c: float = 1.0,
    max_iter: int = 2000,
    class_weight: Optional[str] = "balanced",
    random_state: int = 0,
) -> ClassifierMixin:
    X, y = _sanitize_Xy(X, y)

    uniq = np.unique(y)
    if uniq.size < 2:
        # Degenerate split: fall back to constant predictor (still has predict_proba)
        const = int(uniq[0]) if uniq.size == 1 else 1
        dummy = DummyClassifier(strategy="constant", constant=const)
        dummy.fit(X, y)
        return dummy

    clf = LogisticRegression(
        C=float(c),
        max_iter=int(max_iter),
        solver="lbfgs",
        n_jobs=1,
        class_weight=class_weight,
        random_state=int(random_state),
    )
    clf.fit(X, y)
    return clf


def emission_metrics(p: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.int64).reshape(-1)
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
    p = _clip_prob(p_raw, eps=eps)
    logit = np.log(p) - np.log(1.0 - p)
    T = float(max(float(T), 1e-6))
    p_cal = _sigmoid(logit / T)
    return p_cal.astype(np.float32)


def _nll_bernoulli(p: np.ndarray, y: np.ndarray, eps: float = 1e-6) -> float:
    p = _clip_prob(p, eps=eps)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    return float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean())


def fit_temperature_grid(
    p_raw: np.ndarray,
    y: np.ndarray,
    *,
    grid: Optional[np.ndarray] = None,
    eps: float = 1e-6,
) -> Tuple[float, Dict[str, float]]:
    p_raw = np.asarray(p_raw, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    if p_raw.size == 0 or y.size == 0 or p_raw.size != y.size:
        return 1.0, {"nll_raw": float("nan"), "nll_cal": float("nan")}

    if grid is None:
        grid = np.exp(np.linspace(np.log(0.25), np.log(10.0), 60)).astype(np.float64)

    nll_raw = _nll_bernoulli(p_raw, y, eps=eps)
    best_T = 1.0
    best_nll = nll_raw

    for T in grid:
        p_cal = apply_temperature_scaling(p_raw, float(T), eps=eps)
        nll = _nll_bernoulli(p_cal, y, eps=eps)
        if np.isfinite(nll) and nll < best_nll:
            best_nll = float(nll)
            best_T = float(T)

    return best_T, {"nll_raw": float(nll_raw), "nll_cal": float(best_nll)}
