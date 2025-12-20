from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import numpy as np


@dataclass
class SegmentDecision:
    start: int
    end: int          # inclusive
    kind: str         # "bridge" or "abstain"
    score: float
    L: int
    u: float


def find_low_segments(w: np.ndarray, tau: float) -> List[Tuple[int, int]]:
    """Return inclusive segments where w[t] < tau."""
    w = np.asarray(w, dtype=np.float64)
    T = int(w.shape[0])
    segs = []
    in_seg = False
    s = 0
    for t in range(T):
        if w[t] < tau and not in_seg:
            in_seg = True
            s = t
        if in_seg and (t == T - 1 or w[t + 1] >= tau):
            e = t
            segs.append((s, e))
            in_seg = False
    return segs


def _segment_boundaries(seg: Tuple[int, int], T: int) -> Tuple[Optional[int], Optional[int]]:
    s, e = seg
    left = s - 1 if s - 1 >= 0 else None
    right = e + 1 if e + 1 < T else None
    return left, right


def _compute_u(w: np.ndarray, left: int, right: int) -> float:
    # interior frames (left+1 ... right-1)
    if right <= left + 1:
        return 0.0
    interior = w[left + 1:right]
    return float(np.mean(1.0 - interior))


def decide_segments(
    w: np.ndarray,
    tau: float,
    cfg_aob: Dict[str, Any],
) -> List[SegmentDecision]:
    T = int(len(w))
    segs = find_low_segments(w, tau)
    eta_L = float(cfg_aob.get("eta_L", 0.12))
    eta_u = float(cfg_aob.get("eta_u", 1.0))
    eps_gate = float(cfg_aob.get("eps_gate", 1.0))

    # If eps_gate>=1.0, guarantee pure abstain control (bridge OFF).
    bridge_enabled = bool((cfg_aob.get("bridge") or {}).get("enabled", True)) and (eps_gate < 1.0)

    decisions: List[SegmentDecision] = []
    for (s, e) in segs:
        left, right = _segment_boundaries((s, e), T)
        # If missing boundaries, we abstain (conservative)
        if left is None or right is None:
            decisions.append(SegmentDecision(s, e, "abstain", score=0.0, L=e - s + 1, u=float("inf")))
            continue

        L = (right - left)  # boundary gap length
        u = _compute_u(w, left, right)
        score = float(np.exp(-eta_L * L) * np.exp(-eta_u * u))
        kind = "bridge" if (bridge_enabled and score >= eps_gate) else "abstain"
        decisions.append(SegmentDecision(s, e, kind, score=score, L=L, u=u))

    return decisions


def _abstain_fill(
    z: np.ndarray,
    left: int,
    right: int,
    mode: str,
) -> None:
    """Fill z[left+1:right] (interior) given fixed endpoints z[left], z[right]."""
    mode = str(mode).lower().strip()
    if right <= left + 1:
        return
    if mode == "hold":
        z[left + 1:right] = z[left]
        return
    if mode == "linear":
        L = right - left
        for t in range(left + 1, right):
            a = (t - left) / float(L)
            z[t] = (1.0 - a) * z[left] + a * z[right]
        return
    raise ValueError(f"Unknown abstain_mode={mode}. Use 'hold' or 'linear'.")


def _bridge_bvp_accel(
    z: np.ndarray,
    left: int,
    right: int,
) -> None:
    """
    Solve min sum ||Δ^2 z_t||^2 with hard boundaries at left/right.
    Here we use a simple cubic-spline-like construction via linear interpolation + smoothing.
    For a lightweight repo, we implement a robust approximation that is stable and fast.
    """
    if right <= left + 1:
        return
    # Start from linear and then apply a few Laplacian smoothing iterations on interior.
    _abstain_fill(z, left, right, mode="linear")
    # Smooth interior (keep boundaries fixed)
    for _ in range(20):
        for t in range(left + 1, right):
            z[t] = 0.25 * z[t - 1] + 0.5 * z[t] + 0.25 * z[t + 1]


def apply_aob(
    w: np.ndarray,
    z_bar: np.ndarray,
    tau: float,
    cfg_aob: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Construct final trajectory z_fin based on:
      - reliable frames: z = z_bar
      - low segments: abstain or bridge
    Returns (z_fin, stats)
    """
    w = np.asarray(w, dtype=np.float64)
    z_bar = np.asarray(z_bar, dtype=np.float64)
    T = int(len(w))
    if z_bar.shape != (T, 2):
        raise ValueError(f"z_bar must be (T,2), got {z_bar.shape}")

    abstain_mode = str(cfg_aob.get("abstain_mode", "linear")).lower().strip()

    z = z_bar.copy()
    decisions = decide_segments(w, tau, cfg_aob)

    bridged_frames = 0
    abstained_frames = 0

    # Apply per segment
    for d in decisions:
        left = d.start - 1
        right = d.end + 1
        # Handle missing boundaries by holding nearest available state
        if left < 0:
            # segment starts at 0: hold z[right] backward
            z[d.start:d.end + 1] = z[right] if right < T else z[0]
            abstained_frames += (d.end - d.start + 1)
            continue
        if right >= T:
            z[d.start:d.end + 1] = z[left]
            abstained_frames += (d.end - d.start + 1)
            continue

        if d.kind == "bridge":
            _bridge_bvp_accel(z, left, right)
            bridged_frames += (d.end - d.start + 1)
        else:
            _abstain_fill(z, left, right, mode=abstain_mode)
            abstained_frames += (d.end - d.start + 1)

    stats = {
        "tau": float(tau),
        "low_frac": float(np.mean(w < tau)) if T > 0 else 0.0,
        "bridged_frames": int(bridged_frames),
        "abstained_frames": int(abstained_frames),
        "num_segments": int(len(decisions)),
        "segments": [d.__dict__ for d in decisions],
    }
    return z.astype(np.float32), stats
