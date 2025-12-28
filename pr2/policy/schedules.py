from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RadiusSchedule:
    R_min: float
    R_max: float
    alpha: float = 1.0

    def __call__(self, w: float | np.ndarray) -> float | np.ndarray:
        w = np.asarray(w, dtype=np.float32)
        return self.R_min + (self.R_max - self.R_min) * np.power(w, self.alpha)


@dataclass(frozen=True)
class GMaxSchedule:
    g_min: float
    g_max0: float
    beta: float = 1.0

    def __call__(self, w: float | np.ndarray) -> float | np.ndarray:
        w = np.asarray(w, dtype=np.float32)
        return self.g_min + (self.g_max0 - self.g_min) * np.power(w, self.beta)


@dataclass(frozen=True)
class LambdaSafeSchedule:
    a: float
    b: float

    def __call__(self, w: float | np.ndarray) -> float | np.ndarray:
        w = np.asarray(w, dtype=np.float32)
        x = self.a * (w - self.b)
        return 1.0 / (1.0 + np.exp(-x))
