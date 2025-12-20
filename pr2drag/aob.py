# pr2drag/aob.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import numpy as np


@dataclass
class AoBParams:
    eps_gate: float
    abstain_mode: str  # "hold" | "linear"
    eta_L: float
    eta_u: float
    max_bridge_len: int
    bridge_mode: str = "min_accel"  # "min_accel" | "linear" | "hermite"
    hermite_scan: int = 6           # how far to scan for a reliable neighbor when estimating velocity
    clamp_hermite: bool = True      # clamp Hermite output to a box around endpoints (safer)
    clamp_margin_px: float = 0.0    # optional extra margin (in px); 0 => auto margin from endpoint distance


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
    With only endpoints constrained, this tends to be close to linear interpolation.
    Returns interior points shape (n_interior, D).
    """
    if n_interior <= 0:
        return np.zeros((0, zL.shape[0]), dtype=np.float32)

    D = zL.shape[0]
    N = n_interior + 2  # include endpoints
    D2 = np.zeros((N - 2, N), dtype=np.float64)
    for r in range(N - 2):
        D2[r, r] = 1.0
        D2[r, r + 1] = -2.0
        D2[r, r + 2] = 1.0
    A = D2.T @ D2  # NxN

    ii = np.arange(1, N - 1)
    bb = np.array([0, N - 1])
    Aii = A[np.ix_(ii, ii)]
    Aib = A[np.ix_(ii, bb)]

    out = np.zeros((n_interior, D), dtype=np.float32)
    zb = np.stack([zL, zR], axis=1).astype(np.float64)  # (D,2)

    for d in range(D):
        rhs = -Aib @ zb[d]
        try:
            sol = np.linalg.solve(Aii, rhs)
        except np.linalg.LinAlgError:
            sol = np.linspace(zL[d], zR[d], N)[1:-1]
        out[:, d] = sol.astype(np.float32)
    return out


def _bridge_linear(zL: np.ndarray, zR: np.ndarray, n_interior: int) -> np.ndarray:
    if n_interior <= 0:
        return np.zeros((0, zL.shape[0]), dtype=np.float32)
    N = n_interior + 2
    pts = np.linspace(zL.astype(np.float64), zR.astype(np.float64), N, axis=0)[1:-1]
    return pts.astype(np.float32)


def _find_reliable_neighbor(
    w: np.ndarray, tau: float, start: int, direction: int, max_scan: int
) -> Optional[int]:
    """
    Find nearest index from start moving in direction (-1 or +1) such that w[idx] >= tau.
    Returns None if not found within max_scan.
    """
    T = int(w.shape[0])
    step = -1 if direction < 0 else 1
    for k in range(1, max_scan + 1):
        idx = start + step * k
        if idx < 0 or idx >= T:
            return None
        if w[idx] >= tau:
            return int(idx)
    return None


def _estimate_velocity(
    z: np.ndarray, w: np.ndarray, tau: float, anchor: int, side: str, max_scan: int
) -> np.ndarray:
    """
    Estimate per-frame velocity at anchor using nearby reliable points.

    side="left": use (anchor - prev) difference
    side="right": use (next - anchor) difference
    If no neighbor found, returns zeros.
    """
    D = int(z.shape[1])
    v = np.zeros((D,), dtype=np.float32)

    if side == "left":
        prev = _find_reliable_neighbor(w, tau, anchor, direction=-1, max_scan=max_scan)
        if prev is None or prev == anchor:
            return v
        dt = float(anchor - prev)
        if dt <= 0:
            return v
        v = ((z[anchor] - z[prev]) / dt).astype(np.float32)
        return v

    if side == "right":
        nxt = _find_reliable_neighbor(w, tau, anchor, direction=+1, max_scan=max_scan)
        if nxt is None or nxt == anchor:
            return v
        dt = float(nxt - anchor)
        if dt <= 0:
            return v
        v = ((z[nxt] - z[anchor]) / dt).astype(np.float32)
        return v

    raise ValueError(f"Unknown side={side}. Use 'left'|'right'.")


def _bridge_hermite(
    zL: np.ndarray,
    zR: np.ndarray,
    vL: np.ndarray,
    vR: np.ndarray,
    dt: int,
    *,
    clamp: bool,
    clamp_margin_px: float = 0.0,
) -> np.ndarray:
    """
    Cubic Hermite interpolation between endpoints with endpoint velocities.
    dt = (right - left) in frames, so interior count = dt-1.
    Returns (dt-1, D).
    """
    n_interior = int(dt - 1)
    if n_interior <= 0:
        return np.zeros((0, zL.shape[0]), dtype=np.float32)

    zL64 = zL.astype(np.float64)
    zR64 = zR.astype(np.float64)
    vL64 = vL.astype(np.float64)
    vR64 = vR.astype(np.float64)

    # parameter u in (0,1) for interior frames
    u = (np.arange(1, dt, dtype=np.float64) / float(dt))[:, None]  # (dt-1,1)

    h00 = (2*u**3 - 3*u**2 + 1)
    h10 = (u**3 - 2*u**2 + u)
    h01 = (-2*u**3 + 3*u**2)
    h11 = (u**3 - u**2)

    # Note: derivative scaling: position(u) uses (dt * v) terms
    mL = float(dt) * vL64
    mR = float(dt) * vR64

    pts = h00 * zL64 + h10 * mL + h01 * zR64 + h11 * mR  # (dt-1,D)

    if clamp:
        lo = np.minimum(zL64, zR64)
        hi = np.maximum(zL64, zR64)

        # auto margin: small fraction of endpoint distance to avoid over-clamping
        dist = float(np.linalg.norm(zR64 - zL64) + 1e-6)
        auto_margin = 0.15 * dist  # safe default
        margin = float(max(clamp_margin_px, auto_margin))

        lo = lo - margin
        hi = hi + margin
        pts = np.clip(pts, lo[None, :], hi[None, :])

    return pts.astype(np.float32)


def aob_fill(
    z_base: np.ndarray,
    w: np.ndarray,
    tau: float,
    params: AoBParams,
    debug: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given base trajectory z_base (T,2) and reliability w (T,),
    fill low-reliability segments by bridge (BVP/Hermite) or abstain.
    Returns:
      z_final (T,2),
      is_bridge (T,) bool,
      is_abstain (T,) bool
    """
    if z_base.ndim != 2:
        raise ValueError(f"z_base must be 2D (T,D). Got shape={z_base.shape}")
    if w.ndim != 1:
        raise ValueError(f"w must be 1D (T,). Got shape={w.shape}")
    T = int(z_base.shape[0])
    if int(w.shape[0]) != T:
        raise ValueError(f"Length mismatch: z_base has T={T}, w has {int(w.shape[0])}")
    if T == 0:
        return z_base.astype(np.float32), np.zeros((0,), dtype=bool), np.zeros((0,), dtype=bool)

    z_final = z_base.copy().astype(np.float32)

    is_bridge = np.zeros((T,), dtype=np.uint8)
    is_abst = np.zeros((T,), dtype=np.uint8)

    segs = _segments_low(w, tau)
    for (s, e) in segs:
        left = s - 1
        right = e + 1
        has_left = left >= 0 and (w[left] >= tau)
        has_right = right < T and (w[right] >= tau)

        # segment score a_k
        Lk = (right - left) if (has_left and has_right) else (e - s + 1)
        interior = slice(s, e + 1)

        # u_k: mean(1-w) on low segment
        if has_left and has_right and (right - left) >= 2:
            denom = max((right - left - 1), 1)
            uk = float((1.0 - w[s:right]).sum() / float(denom))
        else:
            uk = float((1.0 - w[interior]).mean()) if (e >= s) else 0.0

        a_k = float(np.exp(-params.eta_L * float(Lk)) * np.exp(-params.eta_u * float(uk)))
        do_bridge = bool(
            has_left
            and has_right
            and (a_k >= params.eps_gate)
            and ((e - s + 1) <= params.max_bridge_len)
        )

        if debug is not None:
            debug.append({
                "s": int(s), "e": int(e), "len": int(e - s + 1),
                "left": int(left), "right": int(right),
                "has_left": bool(has_left), "has_right": bool(has_right),
                "Lk": float(Lk), "uk": float(uk), "a_k": float(a_k),
                "do_bridge": bool(do_bridge),
                "mode": "bridge" if do_bridge else f"abstain:{params.abstain_mode}",
                "bridge_mode": str(params.bridge_mode),
            })

        if do_bridge:
            dt = int(right - left)
            n_interior = dt - 1
            zL = z_final[left]
            zR = z_final[right]

            bm = str(params.bridge_mode).lower()
            if bm == "linear":
                interior_pts = _bridge_linear(zL=zL, zR=zR, n_interior=n_interior)
            elif bm == "hermite":
                vL = _estimate_velocity(z_final, w, tau, anchor=left, side="left", max_scan=int(params.hermite_scan))
                vR = _estimate_velocity(z_final, w, tau, anchor=right, side="right", max_scan=int(params.hermite_scan))
                interior_pts = _bridge_hermite(
                    zL=zL, zR=zR, vL=vL, vR=vR, dt=dt,
                    clamp=bool(params.clamp_hermite),
                    clamp_margin_px=float(params.clamp_margin_px),
                )
                if debug is not None:
                    debug[-1].update({
                        "vL": vL.astype(float).tolist(),
                        "vR": vR.astype(float).tolist(),
                        "dt": int(dt),
                    })
            else:
                # default: min_accel (near-linear)
                interior_pts = _bvp_min_accel(zL=zL, zR=zR, n_interior=n_interior)

            # fill s..e (right is excluded)
            z_final[s:right] = interior_pts
            is_bridge[s:right] = 1
            continue

        # abstain
        mode = str(params.abstain_mode).lower()
        if mode not in ("hold", "linear"):
            raise ValueError(f"Unknown abstain_mode={params.abstain_mode}. Use hold|linear.")

        if mode == "hold":
            if has_left:
                z_final[s:e + 1] = z_final[left][None, :]
            elif has_right:
                z_final[s:e + 1] = z_final[right][None, :]
            else:
                pass
            is_abst[s:e + 1] = 1
        else:
            # linear abstain
            if has_left and has_right:
                zL = z_final[left]
                zR = z_final[right]
                n = float(right - left)
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

    return z_final, is_bridge.astype(bool), is_abst.astype(bool)
