# pr2drag/datasets/tapvid.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple
import pickle
import numpy as np

from .davis import resolve_davis_frames


@dataclass(frozen=True)
class TapVidSeq:
    name: str
    frame_paths: list[str]        # len=T
    video_hw: tuple[int, int]     # (H, W)
    gt_tracks_xy: "np.ndarray"    # float32 [T, Q, 2] in (x,y), coords in ORIGINAL
    gt_vis: "np.ndarray"          # bool [T, Q]
    queries_txy: "np.ndarray"     # int/float [Q, 3] (t0,x0,y0) in ORIGINAL


def _as_np(a: Any) -> np.ndarray:
    if isinstance(a, np.ndarray):
        return a
    return np.asarray(a)


def _normalize_tracks_and_vis(tracks: np.ndarray, vis: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    tracks = _as_np(tracks)
    vis = _as_np(vis)

    if tracks.ndim != 3 or tracks.shape[-1] != 2:
        raise ValueError(f"[TapVidPKL] tracks must be [T,N,2] or [N,T,2], got {tracks.shape}")
    if vis.ndim != 2:
        raise ValueError(f"[TapVidPKL] vis must be [T,N] or [N,T], got {vis.shape}")

    # decide time axis by matching vis
    # case A: tracks [T,N,2], vis [T,N]
    if tracks.shape[0] == vis.shape[0] and tracks.shape[1] == vis.shape[1]:
        T, N = vis.shape
        tr = tracks
        vv = vis
    # case B: tracks [N,T,2], vis [N,T]
    elif tracks.shape[1] == vis.shape[1] and tracks.shape[0] == vis.shape[0]:
        # ambiguous; prefer [T,N] if possible
        # if vis looks like [N,T] and tracks [N,T,2], transpose
        tr = np.transpose(tracks, (1, 0, 2))
        vv = np.transpose(vis, (1, 0))
        T, N = vv.shape
    # case C: tracks [N,T,2], vis [T,N]
    elif tracks.shape[0] == vis.shape[1] and tracks.shape[1] == vis.shape[0]:
        tr = np.transpose(tracks, (1, 0, 2))
        vv = vis
        T, N = vv.shape
    # case D: tracks [T,N,2], vis [N,T]
    elif tracks.shape[0] == vis.shape[1] and tracks.shape[1] == vis.shape[0]:
        tr = tracks
        vv = np.transpose(vis, (1, 0))
        T, N = tr.shape[0], tr.shape[1]
    else:
        raise ValueError(f"[TapVidPKL] cannot align tracks {tracks.shape} with vis {vis.shape}")

    tr = tr.astype(np.float32, copy=False)
    vv = vv.astype(bool, copy=False)

    if not np.isfinite(tr).all():
        raise ValueError("[TapVidPKL] tracks contain NaN/Inf")
    return tr, vv


def load_tapvid_pkl(pkl_path: str) -> dict:
    p = Path(pkl_path)
    if not p.exists():
        raise FileNotFoundError(f"[TapVidPKL] not found: {p}")
    with p.open("rb") as f:
        try:
            obj = pickle.load(f)
        except Exception:
            f.seek(0)
            obj = pickle.load(f, encoding="latin1")

    if not isinstance(obj, (dict, list)):
        raise ValueError(f"[TapVidPKL] unsupported top-level type: {type(obj)}")
    return obj  # raw


def _iter_entries(raw: Any) -> List[Dict[str, Any]]:
    # Normalize to list of dict entries
    if isinstance(raw, list):
        entries = raw
    elif isinstance(raw, dict):
        # if dict-of-entries
        if all(isinstance(v, dict) for v in raw.values()):
            entries = list(raw.values())
            # but keep name if key is name
            for k, v in raw.items():
                if isinstance(v, dict) and ("name" not in v and "video_name" not in v):
                    v["_name_from_key"] = k
        # if dict with 'videos' etc
        elif "videos" in raw and isinstance(raw["videos"], list):
            entries = raw["videos"]
        else:
            # maybe split->list
            for kk in ("davis", "tapvid_davis", "val", "train"):
                if kk in raw and isinstance(raw[kk], list):
                    entries = raw[kk]
                    break
            else:
                raise ValueError(f"[TapVidPKL] dict format not recognized. Top keys={list(raw.keys())[:40]}")
    else:
        raise AssertionError("unreachable")
    out: List[Dict[str, Any]] = []
    for e in entries:
        if not isinstance(e, dict):
            raise ValueError(f"[TapVidPKL] entry must be dict, got {type(e)}")
        out.append(e)
    return out


def _extract_one(entry: Dict[str, Any]) -> Tuple[str, np.ndarray, np.ndarray]:
    # name
    name = (
        entry.get("name")
        or entry.get("video_name")
        or entry.get("seq_name")
        or entry.get("_name_from_key")
    )
    if name is None:
        raise ValueError(f"[TapVidPKL] missing name keys in entry. keys={list(entry.keys())[:40]}")
    name = str(name)

    # tracks
    tracks = None
    for k in ("gt_tracks_xy", "tracks_xy", "tracks", "points", "target_points"):
        if k in entry:
            tracks = entry[k]
            break
    if tracks is None:
        raise ValueError(f"[TapVidPKL:{name}] missing tracks. keys={list(entry.keys())[:40]}")

    # visibility
    vis = None
    for k in ("gt_vis", "vis", "visibility", "visible", "occluded"):
        if k in entry:
            vis = entry[k]
            # if occluded mask is provided, invert it
            if k == "occluded":
                vis = ~_as_np(vis).astype(bool)
            break
    if vis is None:
        raise ValueError(f"[TapVidPKL:{name}] missing visibility. keys={list(entry.keys())[:40]}")

    tr, vv = _normalize_tracks_and_vis(_as_np(tracks), _as_np(vis))
    return name, tr, vv


def _build_queries_first(tracks: np.ndarray, vis: np.ndarray) -> np.ndarray:
    # per point: first visible time
    T, N = vis.shape
    qs = np.zeros((N, 3), dtype=np.float32)
    for n in range(N):
        idx = np.where(vis[:, n])[0]
        if idx.size == 0:
            # no visible frames: define t0=0, xy=(0,0); it will be ignored by valid mask anyway
            t0 = 0
            x0, y0 = 0.0, 0.0
        else:
            t0 = int(idx[0])
            x0, y0 = tracks[t0, n, 0], tracks[t0, n, 1]
        qs[n] = (t0, x0, y0)
    return qs


def _expand_strided(tracks: np.ndarray, vis: np.ndarray, stride: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # For every point n, create a query at each time t0 where vis[t0,n]=True and t0 % stride == 0
    T, N = vis.shape
    q_list: List[Tuple[int, float, float, int]] = []  # (t0,x0,y0,n)
    for n in range(N):
        ts = np.where(vis[:, n])[0]
        ts = ts[ts % stride == 0]
        for t0 in ts.tolist():
            x0, y0 = float(tracks[t0, n, 0]), float(tracks[t0, n, 1])
            q_list.append((int(t0), x0, y0, n))

    if not q_list:
        # fallback to "first" if nothing meets stride
        qs = _build_queries_first(tracks, vis)
        return tracks.astype(np.float32), vis.astype(bool), qs

    Q = len(q_list)
    trQ = np.zeros((T, Q, 2), dtype=np.float32)
    vvQ = np.zeros((T, Q), dtype=bool)
    qs = np.zeros((Q, 3), dtype=np.float32)

    for qi, (t0, x0, y0, n) in enumerate(q_list):
        trQ[:, qi, :] = tracks[:, n, :]
        vvQ[:, qi] = vis[:, n]
        qs[qi, :] = (t0, x0, y0)

    return trQ, vvQ, qs


def build_tapvid_dataset(
    davis_root: str,
    pkl_path: str,
    split: str = "davis",
    res: str = "480p",
    query_mode: str = "strided",
    stride: int = 5,
) -> list[TapVidSeq]:
    """Return list of sequences with resolved frame paths and GT in ORIGINAL coords."""
    raw = load_tapvid_pkl(pkl_path)
    entries = _iter_entries(raw)

    out: List[TapVidSeq] = []
    for e in entries:
        name, tracksTN, visTN = _extract_one(e)

        # Resolve frames from DAVIS root (source of truth)
        frame_paths = resolve_davis_frames(davis_root, name, res=res)

        # Infer HW if present, else leave unknown (-1) and handle later
        H = int(e.get("height", e.get("h", -1)))
        W = int(e.get("width", e.get("w", -1)))
        video_hw = (H, W)

        if tracksTN.shape[0] != len(frame_paths):
            # sometimes pkl stores a trimmed range; for DAVIS split we expect full align
            raise ValueError(
                f"[TapVid:{name}] T mismatch: tracksT={tracksTN.shape[0]} frames={len(frame_paths)}"
            )

        if query_mode == "first":
            qs = _build_queries_first(tracksTN, visTN)
            trQ, vvQ = tracksTN.astype(np.float32), visTN.astype(bool)
        elif query_mode == "strided":
            trQ, vvQ, qs = _expand_strided(tracksTN, visTN, stride=stride)
        else:
            raise ValueError(f"[TapVid] unsupported query_mode: {query_mode}")

        out.append(
            TapVidSeq(
                name=name,
                frame_paths=frame_paths,
                video_hw=video_hw,
                gt_tracks_xy=trQ,
                gt_vis=vvQ,
                queries_txy=qs,
            )
        )
    return out