# pr2drag/datasets/tapvid.py
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np


@dataclass(frozen=True)
class TapVidSeq:
    name: str
    frame_paths: List[str]              # optional (can be empty)
    video_hw: Tuple[int, int]           # (H, W) in the SAME coord system as gt_tracks_xy/query_points_tyx
    video: Optional[np.ndarray]         # uint8 [T,H,W,3] in the SAME coord system as above

    gt_tracks_xy: np.ndarray            # float32 [N,T,2] (x,y)
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
    root = Path(davis_root) / "JPEGImages" / res / seq
    if not root.exists():
        return []
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


def _as_uint8_video(video: np.ndarray) -> np.ndarray:
    v = np.asarray(video)
    if v.dtype == np.uint8:
        return v
    if np.issubdtype(v.dtype, np.floating):
        vmax = float(np.nanmax(v)) if v.size else 0.0
        if vmax <= 1.5:
            v = v * 255.0
        v = np.clip(v, 0.0, 255.0).astype(np.uint8)
        return v
    v = np.clip(v, 0, 255).astype(np.uint8)
    return v


def _resize_pad_video_to_256(video_uint8: np.ndarray, keep_aspect: bool, interp: str = "bilinear") -> np.ndarray:
    if interp not in ("bilinear", "nearest"):
        raise ValueError(f"[tapvid] interp must be bilinear/nearest, got {interp!r}")
    try:
        import cv2  # type: ignore
    except Exception as e:
        raise ImportError(
            "[tapvid] resizing video requires opencv-python (cv2). "
            "Install: pip install opencv-python"
        ) from e

    v = np.asarray(video_uint8, dtype=np.uint8)
    if v.ndim != 4 or v.shape[-1] != 3:
        raise ValueError(f"[tapvid] video must be [T,H,W,3], got {v.shape}")
    T, H, W, _ = v.shape

    inter = cv2.INTER_LINEAR if interp == "bilinear" else cv2.INTER_NEAREST

    if not keep_aspect:
        out = np.stack([cv2.resize(v[t], (256, 256), interpolation=inter) for t in range(T)], axis=0)
        return out.astype(np.uint8)

    s = min(256.0 / float(H), 256.0 / float(W))
    Hr = int(round(H * s))
    Wr = int(round(W * s))
    Hr = max(1, min(256, Hr))
    Wr = max(1, min(256, Wr))

    resized = np.stack([cv2.resize(v[t], (Wr, Hr), interpolation=inter) for t in range(T)], axis=0)
    out = np.zeros((T, 256, 256, 3), dtype=np.uint8)

    pad_y = (256 - Hr) * 0.5
    pad_x = (256 - Wr) * 0.5
    y0 = int(np.floor(pad_y))
    x0 = int(np.floor(pad_x))
    out[:, y0:y0 + Hr, x0:x0 + Wr, :] = resized
    return out


def _pick_query_for_track(points_xy: np.ndarray, occ: np.ndarray) -> Optional[Tuple[int, float, float]]:
    T = points_xy.shape[0]
    for t in range(T):
        if occ[t]:
            continue
        x, y = float(points_xy[t, 0]), float(points_xy[t, 1])
        if np.isfinite(x) and np.isfinite(y):
            return (t, y, x)  # (t,y,x)
    return None


def _resize_coords_to_256(pts_xy: np.ndarray, H: int, W: int, keep_aspect: bool) -> np.ndarray:
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
    query_mode: str = "tapvid",   # tapvid / strided
    stride: int = 5,
    max_queries: int = 0,
    resize_to_256: bool = True,
    keep_aspect: bool = True,
    seed: int = 0,
    interp: str = "bilinear",
) -> List[TapVidSeq]:
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

        pts = np.asarray(item["points"], dtype=np.float32)   # [N,T,2]
        occ = _as_bool(np.asarray(item["occluded"]))         # [N,T]
        if pts.ndim != 3 or pts.shape[-1] != 2:
            raise ValueError(f"[tapvid] {name}: points must be [N,T,2], got {pts.shape}")
        if occ.shape != pts.shape[:2]:
            raise ValueError(f"[tapvid] {name}: occluded shape {occ.shape} != (N,T) {pts.shape[:2]}")

        N, T = int(pts.shape[0]), int(pts.shape[1])

        video = None
        H = W = None
        if "video" in item and isinstance(item["video"], np.ndarray):
            v = _as_uint8_video(item["video"])
            if v.ndim != 4 or v.shape[0] != T or v.shape[-1] != 3:
                raise ValueError(f"[tapvid] {name}: video must be [T,H,W,3], got {v.shape}")
            H, W = int(v.shape[1]), int(v.shape[2])
            video = v
        else:
            # best-effort inference
            if res == "480p":
                H, W = 480, 854
            elif res == "1080p":
                H, W = 1080, 1920
            else:
                raise ValueError(f"[tapvid] {name}: cannot infer video_hw without 'video' for res={res!r}")

        assert H is not None and W is not None
        frame_paths = _resolve_davis_frame_paths(davis_root, res, str(name), T)

        # build queries
        q_list: List[Tuple[int, float, float]] = []
        q_ids: List[int] = []

        if query_mode == "tapvid":
            for i in range(N):
                q = _pick_query_for_track(pts[i], occ[i])
                if q is None:
                    continue
                q_list.append(q)
                q_ids.append(i)
        elif query_mode == "strided":
            if stride <= 0:
                raise ValueError("[tapvid] stride must be >0 for query_mode='strided'")
            for i in range(N):
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
            continue

        q_points = np.asarray(q_list, dtype=np.float32)  # [Q,3] (t,y,x)
        q_track_ids = np.asarray(q_ids, dtype=np.int64)  # [Q]

        if int(max_queries) > 0 and q_points.shape[0] > int(max_queries):
            idx = rng.choice(q_points.shape[0], size=int(max_queries), replace=False)
            idx = np.sort(idx)
            q_points = q_points[idx]
            q_track_ids = q_track_ids[idx]

        # If resize_to_256: resize BOTH coords and video (so tracker + eval live in same space)
        if resize_to_256:
            pts = _resize_coords_to_256(pts, H=H, W=W, keep_aspect=bool(keep_aspect))
            xy = np.stack([q_points[:, 2], q_points[:, 1]], axis=-1)  # x,y
            xy2 = _resize_coords_to_256(xy, H=H, W=W, keep_aspect=bool(keep_aspect))
            q_points = q_points.copy()
            q_points[:, 1] = xy2[:, 1]  # y
            q_points[:, 2] = xy2[:, 0]  # x

            if video is not None:
                video = _resize_pad_video_to_256(video, keep_aspect=bool(keep_aspect), interp=interp)
            # in 256-space now
            video_hw = (256, 256)
        else:
            video_hw = (H, W)

        seqs.append(
            TapVidSeq(
                name=str(name),
                frame_paths=frame_paths,
                video_hw=video_hw,
                video=video,
                gt_tracks_xy=pts.astype(np.float32),
                gt_occluded=_as_bool(occ),
                query_points_tyx=q_points.astype(np.float32),
                query_track_ids=q_track_ids.astype(np.int64),
            )
        )

    seqs.sort(key=lambda s: s.name)
    return seqs