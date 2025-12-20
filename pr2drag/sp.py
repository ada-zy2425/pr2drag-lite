from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def ece_score(p: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    p = np.asarray(p).astype(np.float64)
    y = np.asarray(y).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if m.sum() == 0:
            continue
        acc = y[m].mean()
        conf = p[m].mean()
        ece += (m.sum() / len(p)) * abs(acc - conf)
    return float(ece)


def risk_at_coverage(p: np.ndarray, y: np.ndarray, coverage: float = 0.5) -> float:
    """
    Keep top coverage by confidence p, compute risk = 1-accuracy on kept subset.
    """
    p = np.asarray(p).astype(np.float64)
    y = np.asarray(y).astype(np.int64)
    n = len(p)
    k = int(round(n * coverage))
    k = max(1, min(n, k))
    idx = np.argsort(-p)[:k]
    acc = (y[idx] == 1).mean()
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
    emission_p_good[t] = p(x_t | state=good) proportional to p_good (we treat as Bernoulli prob)
    We interpret emission_p_good as p(state=good | x_t) proxy and do a pragmatic smoothing:
    Use likelihoods L_good = p, L_bad = 1-p.
    """
    p = np.clip(emission_p_good.astype(np.float64), hmm.eps, 1.0 - hmm.eps)
    T = p.shape[0]

    # Transition matrix A[s_prev, s_next]
    a11 = hmm.p_stay_bad
    a10 = 1.0 - a11
    a00 = hmm.p_stay_good
    a01 = 1.0 - a00
    # NOTE: we store in order [bad, good]
    A = np.array([[a11, a10], [a01, a00]], dtype=np.float64)

    pi_good = float(np.clip(hmm.prior_good, hmm.eps, 1.0 - hmm.eps))
    pi = np.array([1.0 - pi_good, pi_good], dtype=np.float64)

    # log-likelihoods
    ll = np.stack([np.log(1.0 - p), np.log(p)], axis=1)  # (T,2)

    # forward
    alpha = np.zeros((T, 2), dtype=np.float64)
    alpha[0] = np.log(pi) + ll[0]
    for t in range(1, T):
        # logsumexp over prev state
        prev = alpha[t - 1][:, None] + np.log(A)  # (2,2)
        m = prev.max(axis=0)
        alpha[t] = m + np.log(np.exp(prev - m).sum(axis=0)) + ll[t]

    # backward
    beta = np.zeros((T, 2), dtype=np.float64)
    beta[T - 1] = 0.0
    for t in range(T - 2, -1, -1):
        nxt = beta[t + 1] + ll[t + 1]  # (2,)
        tmp = np.log(A) + nxt[None, :]  # (2,2)
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
    auroc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")
    ece = ece_score(p, y)
    r50 = risk_at_coverage(p, y, coverage=0.5)
    return {"auroc": float(auroc), "ece": float(ece), "risk@50%": float(r50)}
