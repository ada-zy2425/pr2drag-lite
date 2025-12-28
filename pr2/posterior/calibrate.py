from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


class Calibrator(Protocol):
    def fit(self, p: np.ndarray, y: np.ndarray) -> "Calibrator":
        ...

    def predict(self, p: np.ndarray) -> np.ndarray:
        ...


@dataclass
class IdentityCalibrator:
    def fit(self, p: np.ndarray, y: np.ndarray) -> "IdentityCalibrator":
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=np.float32)
        return np.clip(p, 0.0, 1.0)


@dataclass
class TemperatureScalingCalibrator:
    """
    对 logits 做温度缩放更常见，但我们这里假设输入是概率 p.
    用 logit(p) / T -> sigmoid -> p_cal。
    """
    T: float = 1.0

    def fit(self, p: np.ndarray, y: np.ndarray) -> "TemperatureScalingCalibrator":
        p = np.asarray(p, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        eps = 1e-6
        p = np.clip(p, eps, 1 - eps)
        logits = np.log(p / (1 - p))

        Ts = np.logspace(-1, 1, 41)  # 0.1..10
        best_T, best_nll = 1.0, float("inf")
        for T in Ts:
            z = logits / T
            pc = 1.0 / (1.0 + np.exp(-z))
            nll = -(y * np.log(pc + eps) + (1 - y) * np.log(1 - pc + eps)).mean()
            if nll < best_nll:
                best_nll = nll
                best_T = float(T)

        self.T = best_T
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=np.float64)
        eps = 1e-6
        p = np.clip(p, eps, 1 - eps)
        logits = np.log(p / (1 - p))
        z = logits / float(self.T)
        pc = 1.0 / (1.0 + np.exp(-z))
        return np.clip(pc.astype(np.float32), 0.0, 1.0)


@dataclass
class IsotonicCalibrator:
    _iso: Any = None

    def fit(self, p: np.ndarray, y: np.ndarray) -> "IsotonicCalibrator":
        try:
            from sklearn.isotonic import IsotonicRegression
        except Exception as e:
            raise ImportError("需要 scikit-learn 才能使用 IsotonicCalibrator。请 pip install scikit-learn") from e
        p = np.asarray(p, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p, y)
        self._iso = iso
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        if self._iso is None:
            raise RuntimeError("Calibrator not fitted.")
        p = np.asarray(p, dtype=np.float64)
        return np.clip(self._iso.predict(p).astype(np.float32), 0.0, 1.0)
