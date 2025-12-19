from __future__ import annotations
import math
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, Tuple, Optional

# optional scipy banded solver
try:
    from scipy.linalg import solve_banded
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

# -------------------------
# masks / geometry
# -------------------------
def load_mask_uint8(path: Path) -> np.ndarray:
    m = np.array(Image.open(path))
    if m.ndim == 3:
        m = m[..., 0]
    return m.astype(np.uint8)

def choose_target_id(mask0: np.ndarray) -> Optional[int]:
    ids = [i for i in np.unique(mask0).tolist() if i != 0]
    if len(ids) == 0:
        return None
    if len(ids) == 1:
        return int(ids[0])
    areas = [(int(i), int((mask0 == i).sum())) for i in ids]
    areas.sort(key=lambda x: x[1], reverse=True)
    return int(areas[0][0])

def mask_to_binary(mask: np.ndarray, target_id: Optional[int]) -> np.ndarray:
    if target_id is None:
        return np.zeros_like(mask, dtype=np.uint8)
    uniq = np.unique(mask)
    if set(uniq.tolist()).issubset({0, 255}) or (len(uniq) <= 2 and (mask.max() in [1, 255])):
        fg = (mask > 0)
    else:
        fg = (mask == target_id)
    return fg.astype(np.uint8)

def centroid_from_binary(fg: np.ndarray) -> np.ndarray:
    ys, xs = np.where(fg > 0)
    if len(xs) == 0:
        return np.array([np.nan, np.nan], dtype=np.float32)
    return np.array([xs.mean(), ys.mean()], dtype=np.float32)

def bbox_from_binary(fg: np.ndarray) -> np.ndarray:
    ys, xs = np.where(fg > 0)
    if len(xs) == 0:
        return np.array([np.nan, np.nan, np.nan, np.nan], dtype=np.float32)
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    return np.array([x1, y1, x2, y2], dtype=np.float32)

def border_margin_from_bbox(bbox: np.ndarray, H: int, W: int) -> float:
    if np.any(np.isnan(bbox)):
        return 0.0
    x1, y1, x2, y2 = bbox
    m = min(x1, y1, (W - 1) - x2, (H - 1) - y2)
    m = max(float(m), 0.0)
    return float(m / max(H, W))

def iou_binary(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a > 0, b > 0).sum()
    union = np.logical_or(a > 0, b > 0).sum()
    return float(inter / union) if union > 0 else 1.0

# -------------------------
# calibration metrics
# -------------------------
def expected_calibration_error(y_true, p, n_bins=15) -> float:
    y_true = np.asarray(y_true).astype(np.int64)
    p = np.asarray(p).astype(np.float32)
    p = np.clip(p, 0.0, 1.0)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if not np.any(mask):
            continue
        acc = float(np.mean(y_true[mask]))
        conf = float(np.mean(p[mask]))
        w = float(np.mean(mask))
        ece += w * abs(acc - conf)
    return float(ece)

def coverage_risk(y_true, conf):
    y_true = np.asarray(y_true).astype(np.float32)
    conf = np.asarray(conf).astype(np.float32)
    idx = np.argsort(-conf)
    y_sorted = y_true[idx]
    T = len(y_sorted)
    cumsum = np.cumsum(y_sorted)
    cov = np.arange(1, T + 1) / T
    acc = cumsum / np.arange(1, T + 1)
    risk = 1.0 - acc
    return cov.astype(np.float32), risk.astype(np.float32)

# -------------------------
# HMM smoothing (2-state)
# -------------------------
def hmm_smooth(e, p01=0.05, p10=0.05, pi1=0.5) -> np.ndarray:
    e = np.asarray(e, dtype=np.float64)
    e = np.clip(e, 1e-6, 1 - 1e-6)
    T = len(e)

    A = np.array([[1 - p10, p10],
                  [p01, 1 - p01]], dtype=np.float64)
    pi = np.array([1.0 - pi1, pi1], dtype=np.float64)

    B0 = 1.0 - e
    B1 = e

    alpha = np.zeros((T, 2), dtype=np.float64)
    alpha[0, 0] = pi[0] * B0[0]
    alpha[0, 1] = pi[1] * B1[0]
    alpha[0] /= max(alpha[0].sum(), 1e-12)

    for t in range(1, T):
        prev = alpha[t - 1] @ A
        alpha[t, 0] = prev[0] * B0[t]
        alpha[t, 1] = prev[1] * B1[t]
        alpha[t] /= max(alpha[t].sum(), 1e-12)

    beta = np.ones((T, 2), dtype=np.float64)
    for t in range(T - 2, -1, -1):
        bnext = np.array([B0[t + 1], B1[t + 1]], dtype=np.float64)
        beta[t, 0] = np.sum(A[0, :] * bnext * beta[t + 1, :])
        beta[t, 1] = np.sum(A[1, :] * bnext * beta[t + 1, :])
        beta[t] /= max(beta[t].sum(), 1e-12)

    post = alpha * beta
    post /= np.maximum(post.sum(axis=1, keepdims=True), 1e-12)
    return post[:, 1].astype(np.float32)

# -------------------------
# AoB (bridge/abstain)
# -------------------------
def find_segments(low_mask: np.ndarray):
    low_mask = np.asarray(low_mask, dtype=bool)
    T = len(low_mask)
    segs = []
    t = 0
    while t < T:
        if low_mask[t]:
            s = t
            while t + 1 < T and low_mask[t + 1]:
                t += 1
            e = t
            segs.append((s, e))
        t += 1
    return segs

def solve_bvp_min_acc(z_left: np.ndarray, z_right: np.ndarray, n_interior: int) -> np.ndarray:
    if n_interior <= 0:
        return np.zeros((0, 2), dtype=np.float32)

    n = int(n_interior)
    N = n + 2
    D2 = np.zeros((N - 2, N), dtype=np.float64)
    for i in range(N - 2):
        D2[i, i] = 1.0
        D2[i, i + 1] = -2.0
        D2[i, i + 2] = 1.0
    Q = D2.T @ D2
    A = Q[1:-1, 1:-1]
    QbL = Q[1:-1, 0]
    QbR = Q[1:-1, -1]

    b = np.zeros((n, 2), dtype=np.float64)
    b[:, 0] = -(QbL * float(z_left[0]) + QbR * float(z_right[0]))
    b[:, 1] = -(QbL * float(z_left[1]) + QbR * float(z_right[1]))

    if SCIPY_OK:
        l = u = 2
        ab = np.zeros((l + u + 1, n), dtype=np.float64)
        for k in range(-l, u + 1):
            diag = np.diag(A, k=k)
            row = u - k
            if k >= 0:
                ab[row, k:] = diag
            else:
                ab[row, :k] = diag
        x = np.zeros((n, 2), dtype=np.float64)
        x[:, 0] = solve_banded((l, u), ab, b[:, 0])
        x[:, 1] = solve_banded((l, u), ab, b[:, 1])
    else:
        x = np.linalg.solve(A, b)
    return x.astype(np.float32)

def apply_aob(
    z_bar: np.ndarray,
    wtil: np.ndarray,
    tau: float,
    eta_l: float,
    eta_u: float,
    eps_gate: float,
    abstain_mode: str = "hold",
) -> Tuple[np.ndarray, Dict]:
    z_bar = np.asarray(z_bar, dtype=np.float32)
    wtil = np.asarray(wtil, dtype=np.float32)
    T = len(wtil)
    z_final = z_bar.copy()

    low = (wtil < tau)
    segs = find_segments(low)

    bridged_frames = 0
    abstained_frames = 0
    invalid_segments = 0

    def _finite(t: int) -> bool:
        return 0 <= t < T and np.all(np.isfinite(z_final[t]))

    for (s, e) in segs:
        Lk = e - s + 1
        left = s - 1
        right = e + 1

        have_left = (left >= 0) and (not low[left]) and _finite(left)
        have_right = (right < T) and (not low[right]) and _finite(right)

        if not (have_left and have_right):
            invalid_segments += 1
            if have_left:
                z_final[s:e+1] = z_final[left]
            elif have_right:
                z_final[s:e+1] = z_final[right]
            else:
                z_final[s:e+1] = np.array([0.0, 0.0], dtype=np.float32)
            abstained_frames += Lk
            continue

        uk = float(np.mean(1.0 - wtil[s:e+1]))
        ak = math.exp(-eta_l * Lk) * math.exp(-eta_u * uk)

        if ak >= eps_gate:
            interior = solve_bvp_min_acc(z_final[left], z_final[right], Lk)
            z_final[s:e+1] = interior
            bridged_frames += Lk
        else:
            if abstain_mode == "linear":
                zL, zR = z_final[left], z_final[right]
                for i, t in enumerate(range(s, e + 1), start=1):
                    a = i / (Lk + 1)
                    z_final[t] = (1 - a) * zL + a * zR
            else:
                z_final[s:e+1] = z_final[left]
            abstained_frames += Lk

    audit = dict(
        T=int(T),
        tau=float(tau),
        low_frac=float(np.mean(low)),
        bridged_frames=int(bridged_frames),
        abstained_frames=int(abstained_frames),
        invalid_segments=int(invalid_segments),
        num_segments=int(len(segs)),
        wtil_min=float(wtil.min()),
        wtil_mean=float(wtil.mean()),
        wtil_max=float(wtil.max()),
    )
    return z_final, audit

# -------------------------
# trajectory metrics
# -------------------------
def traj_metrics(z_pred: np.ndarray, z_gt: np.ndarray, fail_thresh: float = 20.0) -> Dict[str, float]:
    z_pred = np.asarray(z_pred, dtype=np.float32)
    z_gt = np.asarray(z_gt, dtype=np.float32)
    valid = np.isfinite(z_pred).all(axis=1) & np.isfinite(z_gt).all(axis=1)
    if valid.sum() == 0:
        return dict(mean=float("nan"), p50=float("nan"), p90=float("nan"), p95=float("nan"), fail_rate=float("nan"))
    err = np.linalg.norm(z_pred[valid] - z_gt[valid], axis=1)
    return dict(
        mean=float(np.mean(err)),
        p50=float(np.percentile(err, 50)),
        p90=float(np.percentile(err, 90)),
        p95=float(np.percentile(err, 95)),
        fail_rate=float(np.mean(err > fail_thresh)),
    )
