# pr2drag/tier0/metrics_tapvid.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple, Optional
import numpy as np


DEFAULT_THRESHOLDS_PX = (1, 2, 4, 8, 16)


@dataclass(frozen=True)
class TapVidMetrics:
    aj: float
    oa: float
    delta_x: float
    q: int
    t: int

    # optional breakdown
    jaccard_by_thr: Dict[int, float]
    pck_by_thr: Dict[int, float]


def _valid_mask_from_queries(T: int, queries_txy: np.ndarray) -> np.ndarray:
    # queries_txy: [Q,3], first column is t0
    if queries_txy.ndim != 2 or queries_txy.shape[1] != 3:
        raise ValueError(f"[Metrics] queries_txy must be [Q,3], got {queries_txy.shape}")
    Q = queries_txy.shape[0]
    t0 = np.floor(queries_txy[:, 0]).astype(int)
    t0 = np.clip(t0, 0, T - 1)
    # valid[t,q] = t >= t0[q]
    ts = np.arange(T, dtype=int)[:, None]
    return ts >= t0[None, :]


def _resize_to_256_xy(xy: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    H, W = hw
    if H <= 0 or W <= 0:
        raise ValueError("[Metrics] resize_to_256 requested but video_hw is unknown/invalid")
    sx = 256.0 / float(W)
    sy = 256.0 / float(H)
    out = xy.copy()
    out[..., 0] = out[..., 0] * sx
    out[..., 1] = out[..., 1] * sy
    return out


def compute_tapvid_metrics(
    gt_tracks_xy: np.ndarray,   # [T,Q,2]
    gt_vis: np.ndarray,         # [T,Q]
    pred_tracks_xy: np.ndarray, # [T,Q,2]
    pred_vis: np.ndarray,       # [T,Q]
    queries_txy: np.ndarray,    # [Q,3]
    thresholds_px: Iterable[int] = DEFAULT_THRESHOLDS_PX,
    resize_to_256: bool = False,
    video_hw: Optional[Tuple[int, int]] = None,
) -> TapVidMetrics:
    if gt_tracks_xy.shape != pred_tracks_xy.shape:
        raise ValueError(f"[Metrics] tracks shape mismatch gt={gt_tracks_xy.shape} pred={pred_tracks_xy.shape}")
    if gt_vis.shape != pred_vis.shape:
        raise ValueError(f"[Metrics] vis shape mismatch gt={gt_vis.shape} pred={pred_vis.shape}")

    if gt_tracks_xy.ndim != 3 or gt_tracks_xy.shape[-1] != 2:
        raise ValueError(f"[Metrics] tracks must be [T,Q,2], got {gt_tracks_xy.shape}")
    if gt_vis.ndim != 2:
        raise ValueError(f"[Metrics] vis must be [T,Q], got {gt_vis.shape}")

    T, Q = gt_vis.shape
    if Q == 0 or T == 0:
        raise ValueError("[Metrics] empty sequence (T==0 or Q==0)")

    if resize_to_256:
        if video_hw is None:
            raise ValueError("[Metrics] resize_to_256=True but video_hw is None")
        gt_tracks_xy = _resize_to_256_xy(gt_tracks_xy, video_hw)
        pred_tracks_xy = _resize_to_256_xy(pred_tracks_xy, video_hw)
        # queries xy should also be in resized space for valid mask? valid mask uses t0 only; but we keep consistent anyway
        q2 = queries_txy.copy().astype(np.float32)
        q2[:, 1:] = _resize_to_256_xy(q2[:, None, 1:], video_hw)[:, 0, :]
        queries_txy = q2

    valid = _valid_mask_from_queries(T, queries_txy)  # [T,Q]

    gtV = (gt_vis.astype(bool)) & valid
    prV = (pred_vis.astype(bool)) & valid

    # distances
    dxy = pred_tracks_xy - gt_tracks_xy
    dist = np.sqrt(np.sum(dxy * dxy, axis=-1)).astype(np.float32)  # [T,Q]

    # OA: occlusion accuracy over valid frames
    oa = float(np.mean(((gt_vis.astype(bool) == pred_vis.astype(bool)) & valid) | (~valid)))

    # delta_x: mean distance over frames where both visible (and valid)
    both_vis = gtV & prV
    if np.any(both_vis):
        delta_x = float(np.mean(dist[both_vis]))
    else:
        delta_x = float("nan")

    j_by_thr: Dict[int, float] = {}
    pck_by_thr: Dict[int, float] = {}

    aj_terms = []
    for thr in thresholds_px:
        thr = int(thr)
        correct = dist <= float(thr)

        # Jaccard per query:
        # intersection: pred visible & gt visible & correct
        inter = np.sum((gtV & prV & correct).astype(np.int32), axis=0)  # [Q]
        # union: (gt visible) OR (pred visible) over valid frames
        # wrong predicted visible counts as FP and also leaves FN for gt => naturally handled by union and missing inter
        union = np.sum((gtV | prV).astype(np.int32), axis=0)  # [Q]

        # if union==0 (both always invisible in valid range), define J=1
        j = np.ones((Q,), dtype=np.float32)
        nz = union > 0
        j[nz] = inter[nz] / union[nz]

        j_by_thr[thr] = float(np.mean(j))
        aj_terms.append(np.mean(j))

        # PCK over GT-visible frames (classic)
        gt_count = np.sum(gtV.astype(np.int32), axis=0)  # [Q]
        hit = np.sum((gtV & prV & correct).astype(np.int32), axis=0)
        p = np.ones((Q,), dtype=np.float32)
        nz2 = gt_count > 0
        p[nz2] = hit[nz2] / gt_count[nz2]
        pck_by_thr[thr] = float(np.mean(p))

    aj = float(np.mean(np.asarray(aj_terms, dtype=np.float32)))

    return TapVidMetrics(
        aj=aj,
        oa=oa,
        delta_x=delta_x,
        q=int(Q),
        t=int(T),
        jaccard_by_thr=j_by_thr,
        pck_by_thr=pck_by_thr,
    )