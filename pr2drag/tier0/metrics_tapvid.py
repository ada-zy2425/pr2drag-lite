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
    jaccard_by_thr: Dict[int, float]
    pck_by_thr: Dict[int, float]


def _as_bool(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.dtype == np.bool_:
        return a
    return (a.astype(np.int32) != 0)


def _canonicalize_TQ_tracks(tracks: np.ndarray, T: int, Q: int, name: str) -> np.ndarray:
    tr = np.asarray(tracks)
    if tr.ndim != 3 or tr.shape[-1] != 2:
        raise ValueError(f"[Metrics] {name} must be 3D with last dim=2, got {tr.shape}")
    # accept [T,Q,2] or [Q,T,2]
    if tr.shape[0] == T and tr.shape[1] == Q:
        return tr
    if tr.shape[0] == Q and tr.shape[1] == T:
        return np.transpose(tr, (1, 0, 2))
    raise ValueError(f"[Metrics] {name} shape {tr.shape} not compatible with T={T}, Q={Q}")


def _canonicalize_TQ_vis(vis: np.ndarray, T: int, Q: int, name: str) -> np.ndarray:
    v = _as_bool(vis)
    if v.ndim != 2:
        raise ValueError(f"[Metrics] {name} must be 2D, got {v.shape}")
    # accept [T,Q] or [Q,T]
    if v.shape == (T, Q):
        return v
    if v.shape == (Q, T):
        return v.T
    raise ValueError(f"[Metrics] {name} shape {v.shape} not compatible with T={T}, Q={Q}")


def _resize_to_256_xy_TQ(xy_TQ2: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    H, W = hw
    if H <= 0 or W <= 0:
        raise ValueError("[Metrics] resize_to_256 requested but video_hw is invalid")
    sx = 256.0 / float(W)
    sy = 256.0 / float(H)
    out = xy_TQ2.astype(np.float32, copy=True)
    out[..., 0] *= sx
    out[..., 1] *= sy
    return out


def _valid_mask(T: int, queries_txy: np.ndarray, query_mode: str) -> np.ndarray:
    """
    Return valid[t, q] bool.

    queries_txy: [Q,3] in (t,x,y) or (t,y,x) doesn't matter here; we use only t.
    query_mode:
      - "tapvid" : evaluate t > t0  (exclude query frame and before)
      - "strided": evaluate all t != t0
    """
    if queries_txy.ndim != 2 or queries_txy.shape[1] != 3:
        raise ValueError(f"[Metrics] queries_txy must be [Q,3], got {queries_txy.shape}")
    if query_mode not in ("tapvid", "strided"):
        raise ValueError(f"[Metrics] query_mode must be 'tapvid' or 'strided', got {query_mode!r}")

    Q = queries_txy.shape[0]
    t0 = np.floor(queries_txy[:, 0]).astype(np.int32)
    t0 = np.clip(t0, 0, T - 1)

    ts = np.arange(T, dtype=np.int32)[:, None]  # [T,1]

    if query_mode == "tapvid":
        # strictly after query time
        return ts > t0[None, :]
    else:
        # all except query frame
        return ts != t0[None, :]


def compute_tapvid_metrics(
    gt_tracks_xy: np.ndarray,    # [T,Q,2] or [Q,T,2]
    gt_vis: np.ndarray,          # [T,Q]   or [Q,T]
    pred_tracks_xy: np.ndarray,  # [T,Q,2] or [Q,T,2]
    pred_vis: np.ndarray,        # [T,Q]   or [Q,T]
    queries_txy: np.ndarray,     # [Q,3] (t,*,*)
    query_mode: str = "tapvid",  # "tapvid" | "strided"
    thresholds_px: Iterable[int] = DEFAULT_THRESHOLDS_PX,
    resize_to_256: bool = False,
    video_hw: Optional[Tuple[int, int]] = None,
) -> TapVidMetrics:
    # Infer T,Q from the visibility arrays first (more stable)
    gv = np.asarray(gt_vis)
    pv = np.asarray(pred_vis)
    if gv.ndim != 2 or pv.ndim != 2:
        raise ValueError(f"[Metrics] vis must be 2D, got gt={gv.shape} pred={pv.shape}")

    # accept [T,Q] or [Q,T] for vis — pick one consistent by using queries length
    Q = int(queries_txy.shape[0])
    if Q <= 0:
        raise ValueError("[Metrics] Q==0 (no queries)")

    # determine T using whichever axis matches Q
    if gv.shape[1] == Q:
        T = int(gv.shape[0])
    elif gv.shape[0] == Q:
        T = int(gv.shape[1])
    else:
        raise ValueError(f"[Metrics] gt_vis shape {gv.shape} incompatible with Q={Q}")

    if T <= 0:
        raise ValueError("[Metrics] T==0 (empty sequence)")

    gt_tracks_xy_TQ = _canonicalize_TQ_tracks(gt_tracks_xy, T=T, Q=Q, name="gt_tracks_xy")
    pred_tracks_xy_TQ = _canonicalize_TQ_tracks(pred_tracks_xy, T=T, Q=Q, name="pred_tracks_xy")
    gt_vis_TQ = _canonicalize_TQ_vis(gt_vis, T=T, Q=Q, name="gt_vis")
    pred_vis_TQ = _canonicalize_TQ_vis(pred_vis, T=T, Q=Q, name="pred_vis")

    if resize_to_256:
        if video_hw is None:
            raise ValueError("[Metrics] resize_to_256=True but video_hw is None")
        gt_tracks_xy_TQ = _resize_to_256_xy_TQ(gt_tracks_xy_TQ, video_hw)
        pred_tracks_xy_TQ = _resize_to_256_xy_TQ(pred_tracks_xy_TQ, video_hw)

    valid = _valid_mask(T, queries_txy, query_mode=query_mode)  # [T,Q]

    gtV = gt_vis_TQ & valid
    prV = pred_vis_TQ & valid

    # distances [T,Q]
    dxy = pred_tracks_xy_TQ - gt_tracks_xy_TQ
    dist = np.sqrt(np.sum(dxy * dxy, axis=-1)).astype(np.float32)

    # OA: occlusion accuracy over valid frames only
    denom = np.maximum(valid.sum(), 1)
    oa = float(((gt_vis_TQ == pred_vis_TQ) & valid).sum() / denom)

    # delta_x: mean distance where both visible and valid
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

        # Jaccard per query: intersection over union on valid frames
        inter = np.sum((gtV & prV & correct).astype(np.int32), axis=0)  # [Q]
        union = np.sum((gtV | prV).astype(np.int32), axis=0)            # [Q]

        j = np.ones((Q,), dtype=np.float32)
        nz = union > 0
        j[nz] = inter[nz] / union[nz]

        j_by_thr[thr] = float(np.mean(j))
        aj_terms.append(float(np.mean(j)))

        # PCK: hits over GT-visible valid frames
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