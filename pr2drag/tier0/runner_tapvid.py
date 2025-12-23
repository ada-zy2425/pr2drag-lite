# pr2drag/tier0/runner_tapvid.py
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple, Union

import numpy as np


@dataclass(frozen=True)
class TapVidExample:
    video_id: str
    query_points: np.ndarray   # [N,3] (t,y,x)
    gt_tracks: np.ndarray      # [N,T,2] (x,y)
    gt_occluded: np.ndarray    # [N,T] bool


def _as_bool(a: np.ndarray) -> np.ndarray:
    if a.dtype == np.bool_:
        return a
    return (a.astype(np.int32) != 0)


def _load_pickle(path: Union[str, Path]) -> Any:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"[tapvid] pickle not found: {p}")
    with p.open("rb") as f:
        return pickle.load(f)


def _iter_examples_from_pickle(obj: Any) -> Iterable[Tuple[str, Mapping[str, Any]]]:
    if isinstance(obj, dict) and "examples" in obj and isinstance(obj["examples"], (list, tuple)):
        for i, ex in enumerate(obj["examples"]):
            yield str(ex.get("video_id", i)), ex
        return

    if isinstance(obj, dict):
        vals = list(obj.values())
        if vals and isinstance(vals[0], dict):
            for k, v in obj.items():
                yield str(k), v
            return

    if isinstance(obj, (list, tuple)):
        for i, ex in enumerate(obj):
            if not isinstance(ex, dict):
                raise TypeError(f"[tapvid] Example[{i}] must be dict, got {type(ex)}")
            yield str(ex.get("video_id", i)), ex
        return

    raise TypeError(f"[tapvid] unsupported pickle root type: {type(obj)}")


def _auto_scale_to_256_if_normalized(q: np.ndarray, tracks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # q: [N,3] (t,y,x), tracks: [N,T,2] (x,y)
    # heuristic: if max coord <= 2 => likely normalized [0,1]
    max_xy = float(np.nanmax(tracks)) if tracks.size else 0.0
    max_q = float(np.nanmax(q[:, 1:3])) if q.size else 0.0
    if max(max_xy, max_q) <= 2.0:
        tracks = tracks * 256.0
        q2 = q.copy()
        q2[:, 1:3] *= 256.0
        return q2, tracks
    return q, tracks


def load_tapvid_points(pkl_path: Union[str, Path], resize_to_256: bool) -> List[TapVidExample]:
    raw = _load_pickle(pkl_path)
    out: List[TapVidExample] = []

    for vid, ex in _iter_examples_from_pickle(raw):
        if "query_points" not in ex:
            raise KeyError(f"[tapvid:{vid}] missing 'query_points'")
        q = np.asarray(ex["query_points"])

        if "target_points" in ex:
            gt_tracks = np.asarray(ex["target_points"])
        elif "gt_tracks" in ex:
            gt_tracks = np.asarray(ex["gt_tracks"])
        else:
            raise KeyError(f"[tapvid:{vid}] missing 'target_points'/'gt_tracks'")

        if "occluded" in ex:
            gt_occ = np.asarray(ex["occluded"])
        elif "gt_occluded" in ex:
            gt_occ = np.asarray(ex["gt_occluded"])
        else:
            raise KeyError(f"[tapvid:{vid}] missing 'occluded'/'gt_occluded'")

        # squeeze optional batch dim [1,...]
        if q.ndim == 3 and q.shape[0] == 1:
            q = q[0]
        if gt_tracks.ndim == 4 and gt_tracks.shape[0] == 1:
            gt_tracks = gt_tracks[0]
        if gt_occ.ndim == 3 and gt_occ.shape[0] == 1:
            gt_occ = gt_occ[0]

        if q.ndim != 2 or q.shape[1] != 3:
            raise ValueError(f"[tapvid:{vid}] query_points must be [N,3], got {q.shape}")
        if gt_tracks.ndim != 3 or gt_tracks.shape[2] != 2:
            raise ValueError(f"[tapvid:{vid}] gt_tracks must be [N,T,2], got {gt_tracks.shape}")

        gt_occ = _as_bool(gt_occ)
        if gt_occ.shape != gt_tracks.shape[:2]:
            raise ValueError(
                f"[tapvid:{vid}] occluded shape {gt_occ.shape} != (N,T) {gt_tracks.shape[:2]}"
            )

        q = q.astype(np.float32)
        gt_tracks = gt_tracks.astype(np.float32)

        if resize_to_256:
            q, gt_tracks = _auto_scale_to_256_if_normalized(q, gt_tracks)

        out.append(
            TapVidExample(
                video_id=str(vid),
                query_points=q,
                gt_tracks=gt_tracks,
                gt_occluded=gt_occ,
            )
        )

    if not out:
        raise ValueError("[tapvid] loaded 0 examples from pickle")
    return out


def run_tapvid_pred_oracle(
    pkl_path: str,
    pred_dir: str,
    resize_to_256: bool,
    overwrite: bool,
    config_path: str,
    config_sha1: str,
    tracker_name: str = "oracle",
) -> Dict[str, Any]:
    pred_root = Path(pred_dir)
    pred_root.mkdir(parents=True, exist_ok=True)

    examples = load_tapvid_points(pkl_path, resize_to_256=resize_to_256)

    wrote = 0
    skipped = 0
    for ex in examples:
        out_npz = pred_root / f"{ex.video_id}.npz"
        if out_npz.exists() and not overwrite:
            skipped += 1
            continue

        # oracle prediction = GT
        pred_tracks = ex.gt_tracks.astype(np.float32)  # [N,T,2]
        pred_occluded = _as_bool(ex.gt_occluded)

        np.savez_compressed(
            out_npz,
            pred_tracks=pred_tracks,
            pred_occluded=pred_occluded,
        )
        wrote += 1

    meta = {
        "tracker": tracker_name,
        "num_videos": len(examples),
        "wrote": wrote,
        "skipped": skipped,
        "pkl_path": str(pkl_path),
        "pred_dir": str(pred_root),
        "resize_to_256": bool(resize_to_256),
        "overwrite": bool(overwrite),
        "config_path": str(config_path),
        "config_sha1": str(config_sha1),
        "schema": {
            "npz_keys": ["pred_tracks", "pred_occluded"],
            "pred_tracks": "float32 [N,T,2] (x,y)",
            "pred_occluded": "bool [N,T]",
        },
    }
    (pred_root / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
    return metavv