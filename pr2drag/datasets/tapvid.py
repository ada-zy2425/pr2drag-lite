# pr2drag/tapvid.py
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

@dataclass(frozen=True)
class TapVidSeq:
    name: str
    frame_paths: list[str]           # len=T; 允许为空（如果你完全走 pkl.video）
    video_hw: tuple[int, int]        # (H, W)
    gt_tracks_xy: np.ndarray         # float32 [T, N, 2] (x,y)
    gt_vis: np.ndarray               # bool    [T, N]
    queries_txy: np.ndarray          # int/float [Q, 3] (t0, x0, y0)
    query_track_ids: np.ndarray      # int64   [Q]  每个 query 对应哪个 track（0..N-1）

def load_tapvid_pkl(pkl_path: str) -> Dict[str, Any]:
    p = Path(pkl_path)
    if not p.exists():
        raise FileNotFoundError(f"[tapvid] pkl not found: {p}")
    with p.open("rb") as f:
        return pickle.load(f)


def _as_bool(a: np.ndarray) -> np.ndarray:
    if a.dtype == np.bool_:
        return a
    return (a.astype(np.int32) != 0)


def _ensure_points_nt2(points: np.ndarray, vid: str) -> np.ndarray:
    points = np.asarray(points)
    if points.ndim != 3:
        raise ValueError(f"[tapvid:{vid}] points must be 3D, got {points.shape}")
    # accept [T,N,2] -> transpose
    if points.shape[-1] != 2:
        raise ValueError(f"[tapvid:{vid}] points last dim must be 2, got {points.shape}")
    # heuristic: if first dim looks like T and second like N but we want [N,T,2]
    # if points.shape[0] is much larger than points.shape[1], treat as [T,N,2]
    if points.shape[0] > points.shape[1] and points.shape[1] <= 512:
        points = np.transpose(points, (1, 0, 2))
    return points.astype(np.float32)

def _infer_hwT_from_video_field(video: Any, vid: str) -> Tuple[int, int, int]:
    v = np.asarray(video)
    if v.ndim != 4 or v.shape[-1] not in (1, 3, 4):
        raise ValueError(f"[tapvid:{vid}] video field must be [T,H,W,C], got {v.shape}")
    T, H, W, _ = v.shape
    return int(H), int(W), int(T)

def _infer_hw_from_davis(davis_root: str, res: str, vid: str) -> Tuple[int, int, List[Path]]:
    seq_dir = Path(davis_root) / "JPEGImages" / res / vid
    if not seq_dir.exists():
        raise FileNotFoundError(f"[tapvid] DAVIS frames folder not found: {seq_dir}")
    frames = sorted(seq_dir.glob("*.jpg"))
    if not frames:
        raise FileNotFoundError(f"[tapvid] No jpg frames under: {seq_dir}")

    # robust image size probe
    H = W = None
    try:
        from PIL import Image  # type: ignore
        with Image.open(frames[0]) as im:
            W, H = im.size
    except Exception:
        try:
            import imageio.v2 as imageio  # type: ignore
            im = imageio.imread(frames[0])
            H, W = int(im.shape[0]), int(im.shape[1])
        except Exception as e:
            raise RuntimeError(f"[tapvid] Failed to read image size for {frames[0]}: {e}") from e

    return int(H), int(W), frames


def _build_queries_tapvid(points_xy: np.ndarray, occ: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each track n, choose first visible+finite frame as query:
      q = (t0, y, x)
    Returns (query_points_tyx [Q,3], query_track_ids [Q])
    """
    N, T, _ = points_xy.shape
    occ = _as_bool(occ)
    queries: List[List[float]] = []
    q_ids: List[int] = []

    for n in range(N):
        # visible & finite
        finite = np.isfinite(points_xy[n, :, 0]) & np.isfinite(points_xy[n, :, 1])
        good = (~occ[n, :]) & finite
        idx = np.where(good)[0]
        if idx.size == 0:
            continue
        t0 = int(idx[0])
        x0, y0 = float(points_xy[n, t0, 0]), float(points_xy[n, t0, 1])
        queries.append([float(t0), y0, x0])
        q_ids.append(n)

    if not queries:
        return np.zeros((0, 3), np.float32), np.zeros((0,), np.int64)
    return np.asarray(queries, np.float32), np.asarray(q_ids, np.int64)


def _build_queries_strided(points_xy: np.ndarray, occ: np.ndarray, stride: int, max_q: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Strided query set:
      For each track n, for t in {0,stride,2stride,...}, if visible+finite => add query (t,y,x) mapped to that track.
    If max_q>0, subsample globally with RNG(seed).
    """
    N, T, _ = points_xy.shape
    occ = _as_bool(occ)
    cand_q: List[List[float]] = []
    cand_ids: List[int] = []
    for n in range(N):
        for t in range(0, T, stride):
            if occ[n, t]:
                continue
            if not (np.isfinite(points_xy[n, t, 0]) and np.isfinite(points_xy[n, t, 1])):
                continue
            x, y = float(points_xy[n, t, 0]), float(points_xy[n, t, 1])
            cand_q.append([float(t), y, x])
            cand_ids.append(n)

    if not cand_q:
        return np.zeros((0, 3), np.float32), np.zeros((0,), np.int64)

    q = np.asarray(cand_q, np.float32)
    ids = np.asarray(cand_ids, np.int64)

    if max_q > 0 and q.shape[0] > max_q:
        rng = np.random.default_rng(int(seed))
        sel = rng.choice(q.shape[0], size=int(max_q), replace=False)
        sel = np.sort(sel)
        q = q[sel]
        ids = ids[sel]
    return q, ids


def _affine_to_256(H: int, W: int, keep_aspect: bool) -> Tuple[float, float, float, float]:
    """
    Returns (sx, sy, tx, ty) such that:
      x' = sx*x + tx
      y' = sy*y + ty
    """
    if keep_aspect:
        s = 256.0 / float(max(H, W))
        newH = H * s
        newW = W * s
        ty = (256.0 - newH) * 0.5
        tx = (256.0 - newW) * 0.5
        return s, s, tx, ty
    else:
        sx = 256.0 / float(W)
        sy = 256.0 / float(H)
        return sx, sy, 0.0, 0.0


def _apply_affine_tracks(points_xy: np.ndarray, sx: float, sy: float, tx: float, ty: float) -> np.ndarray:
    out = points_xy.copy()
    out[..., 0] = out[..., 0] * sx + tx
    out[..., 1] = out[..., 1] * sy + ty
    return out


def _apply_affine_queries(q_tyx: np.ndarray, sx: float, sy: float, tx: float, ty: float) -> np.ndarray:
    out = q_tyx.copy()
    # q = (t, y, x)
    out[:, 2] = out[:, 2] * sx + tx
    out[:, 1] = out[:, 1] * sy + ty
    return out
    

def build_tapvid_dataset(
    davis_root: str,
    pkl_path: str,
    res: str = "480p",
    split: str = "davis",
    query_mode: str = "tapvid",
    stride: int = 5,
    max_queries: int = 0,
    resize_to_256: bool = True,
    keep_aspect: bool = True,
    seed: int = 0,
) -> List[TapVidSeq]:
    """
    Your pkl layout:
      data[vid] = {"points": [N,T,2], "occluded": [N,T], "video": ...}
    We resolve frames via DAVIS root to get (H,W) and T sanity check.
    """
    data = load_tapvid_pkl(pkl_path)
    if not isinstance(data, dict):
        raise TypeError(f"[tapvid] expected dict at pkl root, got {type(data)}")

    seqs: List[TapVidSeq] = []
    for vid, ex in data.items():
        if not isinstance(ex, dict):
            raise TypeError(f"[tapvid:{vid}] value must be dict, got {type(ex)}")
        if "points" not in ex or "occluded" not in ex:
            raise KeyError(f"[tapvid:{vid}] missing keys: require 'points' and 'occluded'")

        points = _ensure_points_nt2(ex["points"], str(vid))
        occ = _as_bool(np.asarray(ex["occluded"]))
        if occ.ndim == 2 and occ.shape[0] > occ.shape[1] and occ.shape[1] <= 512:
            # accept [T,N] -> [N,T]
            occ = occ.T
        if occ.shape != points.shape[:2]:
            raise ValueError(f"[tapvid:{vid}] occluded shape {occ.shape} != points (N,T) {points.shape[:2]}")

        T = points.shape[1]

        frames = None
        H = W = None
        # Try DAVIS first (best: reproducible frame paths + sanity check)
        try:
            H, W, frames = _infer_hw_from_davis(davis_root, res, str(vid))
            if len(frames) != T:
                raise ValueError(f"frames T={len(frames)} != points T={T}")
        except Exception:
            # Fallback to pkl video tensor if present
            if "video" not in ex:
                raise
            H2, W2, T2 = _infer_hwT_from_video_field(ex["video"], str(vid))
            if T2 != T:
                raise ValueError(f"[tapvid:{vid}] video T={T2} != points T={T}")
            H, W = H2, W2
            
        if q_tyx.shape[0] == 0:
            # allow sequence with no valid queries: skip (better than writing empty npz)
            continue

        if resize_to_256:
            sx, sy, tx, ty = _affine_to_256(H, W, keep_aspect=keep_aspect)
            points = _apply_affine_tracks(points, sx, sy, tx, ty)
            q_tyx = _apply_affine_queries(q_tyx, sx, sy, tx, ty)

        seqs.append(TapVidSeq(
            name=str(vid),
            video_hw=(H, W),
            gt_tracks_xy=points.astype(np.float32),
            gt_occluded=occ.astype(np.bool_),
            query_points_tyx=q_tyx.astype(np.float32),
            query_track_ids=q_ids.astype(np.int64),
        ))

    return seqs