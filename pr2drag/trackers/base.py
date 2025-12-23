# pr2drag/trackers/base.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, Tuple
import numpy as np

from pr2drag.datasets.tapvid import TapVidSeq


@dataclass(frozen=True)
class TapVidPred:
    tracks_xy: np.ndarray  # float32 [T,Q,2]
    vis: np.ndarray        # bool [T,Q]
    queries_tyx: Optional[np.ndarray] = None
    video_hw: Optional[Tuple[int, int]] = None


class BaseTapVidTracker(Protocol):
    name: str
    def predict(self, seq: TapVidSeq) -> TapVidPred: ...


def _as_bool(a: np.ndarray) -> np.ndarray:
    return a if a.dtype == np.bool_ else (a.astype(np.int32) != 0)


def load_pred_npz(npz_path: str) -> TapVidPred:
    p = Path(npz_path)
    if not p.exists():
        raise FileNotFoundError(f"[Pred] npz not found: {p}")
    arr = np.load(str(p), allow_pickle=False)
    keys = set(arr.files)

    # --- New contract
    if "tracks_xy" in keys and "vis" in keys:
        tracks = np.asarray(arr["tracks_xy"], dtype=np.float32)
        vis = _as_bool(np.asarray(arr["vis"]))
    # --- Backward-compat (old contract)
    elif "pred_tracks" in keys and "pred_occluded" in keys:
        pt = np.asarray(arr["pred_tracks"], dtype=np.float32)     # [Q,T,2]
        po = _as_bool(np.asarray(arr["pred_occluded"]))           # [Q,T] occ
        if pt.ndim != 3 or pt.shape[-1] != 2:
            raise ValueError(f"[Pred] pred_tracks must be [Q,T,2], got {pt.shape}")
        if po.shape != pt.shape[:2]:
            raise ValueError(f"[Pred] pred_occluded {po.shape} must match [Q,T]={pt.shape[:2]}")
        tracks = np.transpose(pt, (1, 0, 2))                      # -> [T,Q,2]
        vis = np.transpose(~po, (1, 0))                           # -> [T,Q]
    else:
        raise KeyError(f"[Pred] {p.name} must contain (tracks_xy,vis) or (pred_tracks,pred_occluded). keys={arr.files}")

    if tracks.ndim != 3 or tracks.shape[-1] != 2:
        raise ValueError(f"[Pred] tracks_xy must be [T,Q,2], got {tracks.shape}")
    if vis.ndim != 2 or vis.shape[:2] != tracks.shape[:2]:
        raise ValueError(f"[Pred] vis shape {vis.shape} must match tracks [T,Q]={tracks.shape[:2]}")
    if not np.isfinite(tracks).all():
        raise ValueError(f"[Pred] tracks_xy contains NaN/Inf in {p}")

    queries = None
    if "queries_tyx" in keys:
        queries = np.asarray(arr["queries_tyx"], dtype=np.float32)
        if queries.ndim != 2 or queries.shape[1] != 3:
            raise ValueError(f"[Pred] queries_tyx must be [Q,3], got {queries.shape}")

    video_hw = None
    if "video_hw" in keys:
        hw = np.asarray(arr["video_hw"]).reshape(-1)
        if hw.size == 2:
            video_hw = (int(hw[0]), int(hw[1]))

    return TapVidPred(tracks_xy=tracks, vis=vis, queries_tyx=queries, video_hw=video_hw)