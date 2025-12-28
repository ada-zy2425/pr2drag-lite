from __future__ import annotations

from typing import Sequence

import numpy as np


def iou(mask_a: np.ndarray, mask_b: np.ndarray, eps: float = 1e-6) -> float:
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / (float(union) + eps)


def mask_iou_series(masks: Sequence[np.ndarray]) -> np.ndarray:
    if len(masks) <= 1:
        return np.zeros((len(masks),), dtype=np.float32)
    out = np.zeros((len(masks),), dtype=np.float32)
    out[0] = 1.0
    for t in range(1, len(masks)):
        out[t] = iou(masks[t - 1], masks[t])
    return out
