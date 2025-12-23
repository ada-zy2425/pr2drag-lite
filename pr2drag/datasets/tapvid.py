# pr2drag/datasets/tapvid.py
from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np


@dataclass(frozen=True)
class TapVidSeq:
    name: str
    frame_paths: List[str]              # len=T (can be empty if not resolvable)
    video_hw: Tuple[int, int]           # (H, W) original
    gt_tracks_xy: np.ndarray            # float32 [N,T,2] in (x,y)
    gt_occluded: np.ndarray             # bool [N,T] True=occluded
    query_points_tyx: np.ndarray        # float32 [Q,3] (t,y,x)
    query_track_ids: np.ndarray         # int64   [Q]


def load_tapvid_pkl(pkl_path: str) -> Dict[str, Any]:
    p = Path(pkl_path)
    if not p.exists():
        raise FileNotFoundError(f"[tapvid] pkl not found: {p}")
    with p.open("rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"[tapvid] pkl root must be dict[seq->dict], got {type(obj)}")
    return obj


def _resolve_davis_frame_paths(davis_root: str, res: str, seq: str, T: int) -> List[str]:
    """
    DAVIS layout: {davis_root}/JPEGImages/{res}/{seq}/00000.jpg ...
    We try 00000.. or 00001.. both, return empty list if not found.
    """
    root = Path(davis_root) / "JPEGImages" / res / seq
    if not root.exists():
        return []
    # try 00000 indexing
    cand0 = root / "00000.jpg"
    cand1 = root / "00001.jpg"
    if cand0.exists():
        return [str(root / f"{t:05d}.jpg") for t in range(T)]
    if cand1.exists():
        return [str(root / f"{t:05d}.jpg") for t in range(1, T + 1)]
    return []


def _as_bool(a: np.ndarray) -> np.ndarray:
    if a.dtype == np.bool_:
        return a
    return (a.astype(np.int32) != 0)


def _pick_query_for_track(points_xy: np.ndarray, occ: np.ndarray) -> Optional[Tuple[int, float, float]]:
    """
    points_xy: [T,2] (x,y)
    occ: [T] bool True=occluded
    Strategy: first frame where not occluded and coords finite.
    """
    T = points_xy.shape[0]
    for t in range(T):
        if occ[t]:
            continue
        x, y = float(points_xy[t, 0]), float(points_xy[t, 1])
        if np.isfinite(x) and np.isfinite(y):
            return (t, y, x)  # (t,y,x)
    return None


def _resize_coords_to_256(
    pts_xy: np.ndarray,  # [...,2] x,y
    H: int,
    W: int,
    keep_aspect: bool,
) -> np.ndarray:
    pts = pts_xy.astype(np.float32).copy()
    if keep_aspect:
        scale = min(256.0 / float(W), 256.0 / float(H))
        newW = float(W) * scale
        newH = float(H) * scale
        pad_x = (256.0 - newW) * 0.5
        pad_y = (256.0 - newH) * 0.5
        pts[..., 0] = pts[..., 0] * scale + pad_x
        pts[..., 1] = pts[..., 1] * scale + pad_y
    else:
        sx = 256.0 / float(W)
        sy = 256.0 / float(H)
        pts[..., 0] = pts[..., 0] * sx
        pts[..., 1] = pts[..., 1] * sy
    return pts


def build_tapvid_dataset(
    davis_root: str,
    pkl_path: str,
    res: str = "480p",
    split: str = "davis",
    query_mode: str = "tapvid",         # enum: tapvid / strided / first (we treat first specially downstream)
    stride: int = 5,
    max_queries: int = 0,               # 0=unlimited
    resize_to_256: bool = True,
    keep_aspect: bool = True,
    seed: int = 0,
) -> List[TapVidSeq]:
    """
    Returns sequences with:
      - gt_tracks_xy [N,T,2] (x,y)
      - gt_occluded  [N,T]
      - queries: query_points_tyx [Q,3], query_track_ids [Q]
    """
    if split != "davis":
        raise ValueError(f"[tapvid] only split='davis' supported for now (got {split!r})")

    raw = load_tapvid_pkl(pkl_path)

    rng = np.random.default_rng(int(seed))
    seqs: List[TapVidSeq] = []

    for name, item in raw.items():
        if not isinstance(item, dict):
            raise TypeError(f"[tapvid] seq {name!r} must map to dict, got {type(item)}")

        if "points" not in item or "occluded" not in item:
            raise KeyError(f"[tapvid] seq {name!r} missing keys. Need 'points' and 'occluded'")

        pts = np.asarray(item["points"], dtype=np.float32)         # [N,T,2]
        occ = _as_bool(np.asarray(item["occluded"]))               # [N,T]

        if pts.ndim != 3 or pts.shape[-1] != 2:
            raise ValueError(f"[tapvid] {name}: points must be [N,T,2], got {pts.shape}")
        if occ.shape != pts.shape[:2]:
            raise ValueError(f"[tapvid] {name}: occluded shape {occ.shape} != (N,T) {pts.shape[:2]}")

        N, T = int(pts.shape[0]), int(pts.shape[1])

        # video_hw from 'video' if present, else infer from DAVIS res if possible
        H = W = None
        if "video" in item and isinstance(item["video"], np.ndarray):
            vid = item["video"]
            if vid.ndim != 4 or vid.shape[0] != T or vid.shape[-1] != 3:
                raise ValueError(f"[tapvid] {name}: video must be [T,H,W,3], got {vid.shape}")
            H, W = int(vid.shape[1]), int(vid.shape[2])
        else:
            # try infer from DAVIS JPEGImages folder by reading nothing (best-effort)
            # if can't infer, keep coords as-is but require user has 'video' in pkl later.
            # Here we just set placeholders; downstream resize_to_256 needs H,W.
            # We'll attempt to infer from res preset:
            if res == "480p":
                H = 480
                W = 854
            elif res == "1080p":
                H = 1080
                W = 1920
            else:
                # unknown; require video present
                raise ValueError(f"[tapvid] {name}: cannot infer video_hw without 'video' for res={res!r}")

        assert H is not None and W is not None
        frame_paths = _resolve_davis_frame_paths(davis_root, res, str(name), T)

        # build queries
        q_list: List[Tuple[int, float, float]] = []
        q_ids: List[int] = []

        if query_mode == "tapvid":
            # one query per track: first visible point
            for i in range(N):
                q = _pick_query_for_track(pts[i], occ[i])
                if q is None:
                    continue
                q_list.append(q)
                q_ids.append(i)

        elif query_mode == "strided":
            # potentially multiple queries per track (striding on time)
            if stride <= 0:
                raise ValueError("[tapvid] stride must be >0 for query_mode='strided'")
            for i in range(N):
                # choose t = 0, stride, 2*stride...
                for t in range(0, T, int(stride)):
                    if occ[i, t]:
                        continue
                    x, y = float(pts[i, t, 0]), float(pts[i, t, 1])
                    if not (np.isfinite(x) and np.isfinite(y)):
                        continue
                    q_list.append((t, y, x))
                    q_ids.append(i)

        else:
            raise ValueError(f"[tapvid] unknown query_mode={query_mode!r} (expected 'tapvid' or 'strided')")

        if len(q_list) == 0:
            # no valid queries -> skip sequence
            continue

        q_points = np.asarray(q_list, dtype=np.float32)  # [Q,3] (t,y,x)
        q_track_ids = np.asarray(q_ids, dtype=np.int64)  # [Q]

        # max_queries subsample (reproducible)
        if int(max_queries) > 0 and q_points.shape[0] > int(max_queries):
            idx = rng.choice(q_points.shape[0], size=int(max_queries), replace=False)
            idx = np.sort(idx)
            q_points = q_points[idx]
            q_track_ids = q_track_ids[idx]

        # resize coords to 256 if requested
        if resize_to_256:
            pts = _resize_coords_to_256(pts, H=H, W=W, keep_aspect=bool(keep_aspect))
            # q_points: (t,y,x) -> scale x,y consistently
            xy = np.stack([q_points[:, 2], q_points[:, 1]], axis=-1)  # [Q,2] x,y
            xy2 = _resize_coords_to_256(xy, H=H, W=W, keep_aspect=bool(keep_aspect))
            q_points = q_points.copy()
            q_points[:, 1] = xy2[:, 1]  # y
            q_points[:, 2] = xy2[:, 0]  # x

        seqs.append(
            TapVidSeq(
                name=str(name),
                frame_paths=frame_paths,
                video_hw=(H, W),
                gt_tracks_xy=pts.astype(np.float32),         # [N,T,2]
                gt_occluded=_as_bool(occ),
                query_points_tyx=q_points.astype(np.float32),
                query_track_ids=q_track_ids.astype(np.int64),
            )
        )

    # keep deterministic ordering
    seqs.sort(key=lambda s: s.name)
    return seqs