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
    queries_txy: Optional[np.ndarray] = None


class BaseTapVidTracker(Protocol):
    name: str

    def predict(self, seq: TapVidSeq) -> TapVidPred:
        """Return prediction in the SAME coord system as seq GT (original), unless caller enforces resize_to_256."""
        ...


def load_pred_npz(npz_path: str) -> TapVidPred:
    p = Path(npz_path)
    if not p.exists():
        raise FileNotFoundError(f"[Pred] npz not found: {p}")
    arr = np.load(str(p), allow_pickle=False)

    if "tracks_xy" not in arr or "vis" not in arr:
        raise KeyError(f"[Pred] {p.name} must contain keys tracks_xy and vis. keys={list(arr.keys())}")

    tracks = arr["tracks_xy"].astype(np.float32)
    vis = arr["vis"].astype(bool)

    if tracks.ndim != 3 or tracks.shape[-1] != 2:
        raise ValueError(f"[Pred] tracks_xy must be [T,Q,2], got {tracks.shape}")
    if vis.ndim != 2 or vis.shape[0] != tracks.shape[0] or vis.shape[1] != tracks.shape[1]:
        raise ValueError(f"[Pred] vis shape {vis.shape} must match tracks [T,Q]=({tracks.shape[0]},{tracks.shape[1]})")

    if not np.isfinite(tracks).all():
        raise ValueError(f"[Pred] tracks_xy contains NaN/Inf in {p}")

    queries = None
    if "queries_txy" in arr:
        queries = arr["queries_txy"].astype(np.float32)
        if queries.ndim != 2 or queries.shape[1] != 3:
            raise ValueError(f"[Pred] queries_txy must be [Q,3], got {queries.shape}")
    return TapVidPred(tracks_xy=tracks, vis=vis, queries_txy=queries)