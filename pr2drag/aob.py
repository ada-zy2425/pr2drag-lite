# pr2drag/aob.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np


@dataclass
class AoBParams:
    eps_gate: float
    abstain_mode: str  # "hold" | "linear"
    eta_L: float
    eta_u: float
    max_bridge_len: int

    # --- no-harm guards (backward-compatible defaults) ---
    min_low_len: int = 2  # ignore very short low spikes (likely SP noise)
    boundary_tau: Optional[float] = None  # if None, use tau; else stricter boundary requirement

    # Optional clamping to valid coordinate range (if you know image bounds)
    clamp_min: Optional[Tuple[float, float]] = None
    clamp_max: Optional[Tuple[float, float]] = None

    # How to treat NaNs in w
    nan_in_w_as_low: bool = True


def _sanitize_w(w: np.ndarray, *, nan_as_low: bool = True) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    if w.size == 0:
        return w.astype(np.float32)

    if np.any(~np.isfinite(w)):
        # Replace NaN/Inf with 0.0 (treat as low) or 1.0 (treat as high)
        fill = 0.0 if nan_as_low else 1.0
        w = np.nan_to_num(w, nan=fill, posinf=fill, neginf=fill)

    # If user accidentally passes logits or out-of-range values, clip hard.
    w = np.clip(w, 0.0, 1.0)
    return w.astype(np.float32)


def _sanitize_z(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float32)
    if z.ndim != 2:
        raise ValueError(f"z_base must be 2D array (T,D); got shape={z.shape}")
    if z.shape[0] == 0:
        return z
    if np.any(~np.isfinite(z)):
        # conservative: replace NaN/Inf with 0 to avoid propagation crash
        z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return z


def _segments_low(w: np.ndarray, tau: float) -> List[Tuple[int, int]]:
    """
    Return inclusive segments [s,e] where w < tau.
    """
    w = np.asarray(w, dtype=np.float32).reshape(-1)
    T = int(w.shape[0])
    if T == 0:
        return []

    low = w < float(tau)
    segs: List[Tuple[int, int]] = []
    i = 0
    while i < T:
        if not low[i]:
            i += 1
            continue
        s = i
        while i < T and low[i]:
            i += 1
        e = i - 1
        segs.append((s, e))
    return segs


def _bvp_min_accel(zL: np.ndarray, zR: np.ndarray, n_interior: int) -> np.ndarray:
    """
    Discrete minimum acceleration interpolation:
      minimize sum ||Δ^2 z||^2 with endpoints fixed.
    Returns interior points shape (n_interior, D).
    """
    if n_interior <= 0:
        return np.zeros((0, zL.shape[0]), dtype=np.float32)

    zL = np.asarray(zL, dtype=np.float64).reshape(-1)
    zR = np.asarray(zR, dtype=np.float64).reshape(-1)
    if zL.shape != zR.shape:
        raise ValueError(f"zL and zR must have same shape; got {zL.shape} vs {zR.shape}")

    D = int(zL.shape[0])
    N = int(n_interior) + 2  # include endpoints

    # D2 operator: (N-2) x N
    D2 = np.zeros((N - 2, N), dtype=np.float64)
    for r in range(N - 2):
        D2[r, r] = 1.0
        D2[r, r + 1] = -2.0
        D2[r, r + 2] = 1.0
    A = D2.T @ D2  # NxN

    ii = np.arange(1, N - 1)          # interior indices
    bb = np.array([0, N - 1])         # boundary indices
    Aii = A[np.ix_(ii, ii)]
    Aib = A[np.ix_(ii, bb)]

    out = np.zeros((n_interior, D), dtype=np.float32)
    zb = np.stack([zL, zR], axis=1)   # (D,2)

    for d in range(D):
        rhs = -Aib @ zb[d]
        try:
            sol = np.linalg.solve(Aii, rhs)
        except np.linalg.LinAlgError:
            # fallback: least squares then linear if still bad
            try:
                sol = np.linalg.lstsq(Aii, rhs, rcond=None)[0]
            except np.linalg.LinAlgError:
                sol = np.linspace(zL[d], zR[d], N)[1:-1]
        out[:, d] = sol.astype(np.float32)

    return out


def aob_fill(
    z_base: np.ndarray,
    w: np.ndarray,
    tau: float,
    params: AoBParams,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given base trajectory z_base (T,D) and reliability w (T,),
    fill low-reliability segments by bridge (BVP) or abstain.

    Returns:
      z_final (T,D),
      is_bridge (T,) bool,
      is_abstain (T,) bool
    """
    z_base = _sanitize_z(z_base)
    T = int(z_base.shape[0])

    w = _sanitize_w(w, nan_as_low=params.nan_in_w_as_low)
    if int(w.shape[0]) != T:
        raise ValueError(f"Length mismatch: z_base has T={T}, but w has len={len(w)}")

    tau = float(np.clip(float(tau), 0.0, 1.0))
    boundary_thr = float(tau if params.boundary_tau is None else np.clip(params.boundary_tau, 0.0, 1.0))

    z_final = z_base.copy().astype(np.float32)
    is_bridge = np.zeros((T,), dtype=np.uint8)
    is_abst = np.zeros((T,), dtype=np.uint8)

    segs = _segments_low(w, tau)
    for (s, e) in segs:
        seg_len = int(e - s + 1)
        if seg_len < int(params.min_low_len):
            # no-harm guard: ignore short low spikes
            continue

        left = s - 1
        right = e + 1
        has_left = (left >= 0) and (w[left] >= tau)
        has_right = (right < T) and (w[right] >= tau)

        # Lk: use interior length by default; you can choose to include endpoints if you want stronger penalty
        Lk = float(seg_len)

        # u_k: average (1-w) on interior
        interior = slice(s, e + 1)
        uk = float((1.0 - w[interior]).mean()) if seg_len > 0 else 0.0

        a_k = float(np.exp(-params.eta_L * Lk) * np.exp(-params.eta_u * uk))

        # Bridge only if both boundaries exist, boundaries are strong enough, and segment short enough
        do_bridge = bool(
            has_left
            and has_right
            and (w[left] >= boundary_thr)
            and (w[right] >= boundary_thr)
            and (a_k >= params.eps_gate)
            and (seg_len <= int(params.max_bridge_len))
        )

        if do_bridge:
            n_interior = seg_len
            zL = z_final[left]
            zR = z_final[right]
            interior_pts = _bvp_min_accel(zL=zL, zR=zR, n_interior=n_interior)
            z_final[s:right] = interior_pts
            is_bridge[s:right] = 1
        else:
            # abstain
            if params.abstain_mode not in ("hold", "linear"):
                raise ValueError(f"Unknown abstain_mode={params.abstain_mode}. Use hold|linear.")

            if params.abstain_mode == "hold":
                if has_left:
                    z_final[s:e + 1] = z_final[left][None, :]
                elif has_right:
                    z_final[s:e + 1] = z_final[right][None, :]
                else:
                    # nowhere to hold: keep base
                    pass
                is_abst[s:e + 1] = 1
            else:
                # linear
                if has_left and has_right:
                    zL = z_final[left]
                    zR = z_final[right]
                    n = float(right - left)
                    if n <= 0:
                        # degenerate; keep base
                        pass
                    else:
                        for t in range(s, e + 1):
                            alpha = float(t - left) / n
                            z_final[t] = (1.0 - alpha) * zL + alpha * zR
                    is_abst[s:e + 1] = 1
                elif has_left:
                    z_final[s:e + 1] = z_final[left][None, :]
                    is_abst[s:e + 1] = 1
                elif has_right:
                    z_final[s:e + 1] = z_final[right][None, :]
                    is_abst[s:e + 1] = 1
                else:
                    pass

    # Optional clamp
    if params.clamp_min is not None and params.clamp_max is not None and T > 0:
        lo = np.asarray(params.clamp_min, dtype=np.float32).reshape(1, -1)
        hi = np.asarray(params.clamp_max, dtype=np.float32).reshape(1, -1)
        if lo.shape[1] == z_final.shape[1] and hi.shape[1] == z_final.shape[1]:
            z_final = np.minimum(np.maximum(z_final, lo), hi)

    return z_final, is_bridge.astype(bool), is_abst.astype(bool)
