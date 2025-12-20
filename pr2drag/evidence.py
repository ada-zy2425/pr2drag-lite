from __future__ import annotations

from typing import Dict, Tuple, List, Any
import numpy as np


def build_features(
    feat_set: str,
    peak: np.ndarray,
    cycle: np.ndarray,
    out_of_frame: np.ndarray,
    mask_missing: np.ndarray,
    speed: np.ndarray,
    accel: np.ndarray,
) -> Tuple[np.ndarray, List[str]]:
    """
    Return E[t,d] and feature names.
    feat_set:
      - "a": 5 dims (peak, cycle, out_of_frame, mask_missing, speed_norm_small)
      - "b": 7 dims (peak, cycle, out_of_frame, mask_missing, speed, accel, cycle2)
    """
    feat_set = str(feat_set).lower().strip()
    T = int(len(peak))
    if not (len(cycle) == len(out_of_frame) == len(mask_missing) == len(speed) == len(accel) == T):
        raise ValueError("Feature vectors must have the same length T.")

    # Basic sanitation
    peak = np.nan_to_num(peak, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    cycle = np.nan_to_num(cycle, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    out_of_frame = out_of_frame.astype(np.float32)
    mask_missing = mask_missing.astype(np.float32)
    speed = np.nan_to_num(speed, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    accel = np.nan_to_num(accel, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # A simple bounded transform to keep some features in a reasonable range
    cycle2 = np.clip(cycle, 0.0, 10.0) / 10.0  # [0,1] proxy

    if feat_set == "a":
        # 5 dims
        speed_small = np.clip(speed, 0.0, 50.0) / 50.0
        E = np.stack([peak, cycle, out_of_frame, mask_missing, speed_small], axis=1)
        names = ["peak", "cycle", "out_of_frame", "mask_missing", "speed_small"]
        return E, names

    if feat_set == "b":
        # 7 dims (closer to your logs: 5 base + 2 motion-related)
        E = np.stack([peak, cycle, cycle2, out_of_frame, mask_missing, speed, accel], axis=1)
        names = ["peak", "cycle", "cycle2", "out_of_frame", "mask_missing", "speed", "accel"]
        return E, names

    raise ValueError(f"Unknown feat_set={feat_set}. Use 'a' or 'b'.")
