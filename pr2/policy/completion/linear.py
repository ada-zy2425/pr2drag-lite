from __future__ import annotations

import numpy as np


def linear_completion(z_prev: np.ndarray, z_next: np.ndarray, seg_len: int) -> np.ndarray:
    if seg_len <= 0:
        return np.zeros((0, z_prev.shape[-1]), dtype=np.float32)
    out = []
    for i in range(1, seg_len + 1):
        a = i / float(seg_len + 1)
        out.append((1 - a) * z_prev + a * z_next)
    return np.stack(out, axis=0).astype(np.float32)
