from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression


@dataclass
class EmissionModel:
    clf: LogisticRegression
    calibrator: Optional[IsotonicRegression] = None

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = self.clf.predict_proba(X)[:, 1].astype(np.float64)
        p = np.clip(p, 1e-6, 1.0 - 1e-6)
        if self.calibrator is not None:
            p = self.calibrator.transform(p)
            p = np.clip(p, 1e-6, 1.0 - 1e-6)
        return p


def fit_emission(
    X: np.ndarray,
    y: np.ndarray,
    cfg_emission: Dict[str, Any],
) -> EmissionModel:
    model = str(cfg_emission.get("model", "logreg")).lower()
    if model != "logreg":
        raise ValueError(f"Unsupported emission model: {model}")

    C = float(cfg_emission.get("C", 1.0))
    max_iter = int(cfg_emission.get("max_iter", 2000))
    class_weight = cfg_emission.get("class_weight", "balanced")

    clf = LogisticRegression(
        C=C,
        max_iter=max_iter,
        class_weight=class_weight,
        solver="lbfgs",
    )
    clf.fit(X, y)

    calib_cfg = (cfg_emission.get("calibrate") or {})
    enabled = bool(calib_cfg.get("enabled", False))
    method = str(calib_cfg.get("method", "isotonic")).lower()

    calibrator = None
    if enabled:
        if method == "isotonic":
            # Calibrate on train predictions (simple, auditable)
            p_train = np.clip(clf.predict_proba(X)[:, 1], 1e-6, 1.0 - 1e-6)
            calibrator = IsotonicRegression(out_of_bounds="clip")
            calibrator.fit(p_train, y.astype(np.float64))
        elif method in ("none", "off"):
            calibrator = None
        else:
            raise ValueError(f"Unknown calibration method: {method}")

    return EmissionModel(clf=clf, calibrator=calibrator)


def hmm_smooth_binary(
    p1: np.ndarray,
    A: np.ndarray,
    pi: np.ndarray,
) -> np.ndarray:
    """
    Forward-backward for 2-state HMM where emission likelihood is:
      e_t = [P(obs | chi=0), P(obs | chi=1)] ~ [1-p1[t], p1[t]]
    Returns posterior gamma_t = P(chi=1 | obs_1:T)
    Uses scaling for numerical stability.
    """
    p1 = np.asarray(p1, dtype=np.float64)
    T = int(p1.shape[0])
    if T == 0:
        return p1

    p1 = np.clip(p1, 1e-9, 1.0 - 1e-9)
    e = np.stack([1.0 - p1, p1], axis=1)  # (T,2)

    alpha = np.zeros((T, 2), dtype=np.float64)
    c = np.zeros((T,), dtype=np.float64)

    # init
    alpha[0] = pi * e[0]
    c[0] = alpha[0].sum()
    if c[0] <= 0:
        c[0] = 1e-12
    alpha[0] /= c[0]

    # forward
    for t in range(1, T):
        alpha[t] = (alpha[t - 1] @ A) * e[t]
        c[t] = alpha[t].sum()
        if c[t] <= 0:
            c[t] = 1e-12
        alpha[t] /= c[t]

    # backward
    beta = np.zeros((T, 2), dtype=np.float64)
    beta[T - 1] = 1.0
    for t in range(T - 2, -1, -1):
        beta[t] = (A @ (e[t + 1] * beta[t + 1]))
        beta[t] /= c[t + 1]

    gamma = alpha * beta
    denom = gamma.sum(axis=1, keepdims=True)
    denom = np.clip(denom, 1e-12, None)
    gamma /= denom
    return gamma[:, 1]


def build_hmm_params(cfg_hmm: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    A00 = float(cfg_hmm.get("A00", 0.95))
    A01 = float(cfg_hmm.get("A01", 0.05))
    A10 = float(cfg_hmm.get("A10", 0.05))
    A11 = float(cfg_hmm.get("A11", 0.95))
    A = np.array([[A00, A01], [A10, A11]], dtype=np.float64)
    # row-normalize
    A = A / np.clip(A.sum(axis=1, keepdims=True), 1e-12, None)

    pi0 = float(cfg_hmm.get("pi0", 0.5))
    pi1 = float(cfg_hmm.get("pi1", 0.5))
    pi = np.array([pi0, pi1], dtype=np.float64)
    pi = pi / np.clip(pi.sum(), 1e-12, None)
    return A, pi
