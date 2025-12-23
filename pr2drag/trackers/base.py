# pr2drag/trackers/base.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol
import numpy as np

from pr2drag.datasets.tapvid import TapVidSeq


# ---- Backward/compat output used by trackers (e.g., cotracker_v2.py)
@dataclass(frozen=True)
class TrackerOutput:
    """
    Compatibility container for tracker wrappers.

    tracks_xy:
      - either [Q,T,2] or [T,Q,2] float32, in (x,y) pixel coords of the *input video*.
    occluded:
      - either [Q,T] or [T,Q] bool/{0,1}, True means occluded.
    """
    tracks_xy: np.ndarray
    occluded: np.ndarray


@dataclass(frozen=True)
class TapVidPred:
    """
    Generic prediction container used by evaluation / IO utilities.

    tracks_xy:
      - [T,Q,2] float32 in (x,y)
    vis:
      - [T,Q] bool (True visible)
    """
    tracks_xy: np.ndarray
    vis: np.ndarray
    queries_txy: Optional[np.ndarray] = None


class BaseTapVidTracker(Protocol):
    name: str

    def predict(self, seq: TapVidSeq) -> TrackerOutput:
        ...


def load_pred_npz(npz_path: str) -> TapVidPred:
    """
    Legacy NPZ loader (tracks_xy/vis keys). Keep for compatibility with older dumps.
    New TapVid plumbing in tier0/runner_tapvid.py uses pred_tracks/pred_occluded/query_points.
    """
    p = Path(npz_path)
    if not p.exists():
        raise FileNotFoundError(f"[Pred] npz not found: {p}")

    arr = np.load(str(p), allow_pickle=False)
    keys = list(arr.keys())

    if "tracks_xy" not in arr or "vis" not in arr:
        raise KeyError(f"[Pred] {p.name} must contain keys tracks_xy and vis. keys={keys}")

    tracks = np.asarray(arr["tracks_xy"], dtype=np.float32)
    vis = np.asarray(arr["vis"])
    vis = (vis.astype(np.int32) != 0)

    if tracks.ndim != 3 or tracks.shape[-1] != 2:
        raise ValueError(f"[Pred] tracks_xy must be [T,Q,2], got {tracks.shape}")
    if vis.ndim != 2 or vis.shape[:2] != tracks.shape[:2]:
        raise ValueError(f"[Pred] vis shape {vis.shape} must match tracks [T,Q]={tracks.shape[:2]}")

    if not np.isfinite(tracks).all():
        raise ValueError(f"[Pred] tracks_xy contains NaN/Inf in {p}")

    queries = None
    if "queries_txy" in arr:
        queries = np.asarray(arr["queries_txy"], dtype=np.float32)
        if queries.ndim != 2 or queries.shape[1] != 3:
            raise ValueError(f"[Pred] queries_txy must be [Q,3], got {queries.shape}")

    return TapVidPred(tracks_xy=tracks, vis=vis.astype(bool), queries_txy=queries)