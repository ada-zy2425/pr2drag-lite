from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class AoBParams:
    eps_gate: float
    abstain_mode: str  # "hold" | "linear"
    eta_L: float
    eta_u: float
    max_bridge_len: int


def _segments_low(w: np.ndarray, tau: float) -> List[Tuple[int, int]]:
    """
    Return inclusive segments [s,e] where w < tau
    """
    low = w < tau
    segs: List[Tuple[int, int]] = []
    i = 0
    T = len(w)
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

    D = zL.shape[0]
    N = n_interior + 2  # include endpoints
    # D2 operator: (N-2) x N
    # Each row r has [1, -2, 1] at columns r,r+1,r+2
    D2 = np.zeros((N - 2, N), dtype=np.float64)
    for r in range(N - 2):
        D2[r, r] = 1.0
        D2[r, r + 1] = -2.0
        D2[r, r + 2] = 1.0
    A = D2.T @ D2  # NxN, PSD

    # partition: endpoints fixed => solve for interior indices 1..N-2
    ii = np.arange(1, N - 1)
    bb = np.array([0, N - 1])
    Aii = A[np.ix_(ii, ii)]
    Aib = A[np.ix_(ii, bb)]

    out = np.zeros((n_interior, D), dtype=np.float32)
    zb = np.stack([zL, zR], axis=1).astype(np.float64)  # (D,2)

    # Solve per-dim
    for d in range(D):
        rhs = -Aib @ zb[d]
        # robust solve (Aii should be SPD-ish for N>=4; for very short, fall back)
        try:
            sol = np.linalg.solve(Aii, rhs)
        except np.linalg.LinAlgError:
            # fallback: linear
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
    Given base trajectory z_base (T,2) and reliability w (T,),
    fill low-reliability segments by bridge (BVP) or abstain.
    Returns:
      z_final (T,2),
      is_bridge (T,) bool,
      is_abstain (T,) bool
    """
    T = z_base.shape[0]
    z_final = z_base.copy().astype(np.float32)

    is_bridge = np.zeros((T,), dtype=np.uint8)
    is_abst = np.zeros((T,), dtype=np.uint8)

    segs = _segments_low(w, tau)
    for (s, e) in segs:
        left = s - 1
        right = e + 1
        has_left = left >= 0 and (w[left] >= tau)
        has_right = right < T and (w[right] >= tau)

        # segment score a_k (only meaningful if both boundaries exist)
        Lk = (right - left) if (has_left and has_right) else (e - s + 1)
        interior = slice(s, e + 1)
        # u_k: average (1-w) on interior, normalize like proposal (endpoint-exclusive if possible)
        if has_left and has_right and (right - left) >= 2:
            denom = max((right - left - 1), 1)
            uk = float((1.0 - w[s:right]).sum() / denom)
        else:
            uk = float((1.0 - w[interior]).mean()) if (e >= s) else 0.0

        a_k = float(np.exp(-params.eta_L * float(Lk)) * np.exp(-params.eta_u * float(uk)))
        do_bridge = bool(
            has_left
            and has_right
            and (a_k >= params.eps_gate)
            and ((e - s + 1) <= params.max_bridge_len)
        )

        if do_bridge:
            n_interior = right - left - 1
            zL = z_final[left]
            zR = z_final[right]
            interior_pts = _bvp_min_accel(zL=zL, zR=zR, n_interior=n_interior)
            z_final[s:right] = interior_pts
            is_bridge[s:right] = 1
            continue

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
                n = (right - left)
                # points for indices left..right, we fill s..e
                for t in range(s, e + 1):
                    alpha = float(t - left) / float(n)
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

    return z_final, is_bridge.astype(bool), is_abst.astype(bool)
