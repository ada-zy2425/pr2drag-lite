# pr2drag/tapvid_eval.py
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

import numpy as np

from pr2drag.tier1.contracts import RootConfig
from pr2drag.tier0.runner_tapvid import run_tapvid_eval


def tapvid_eval_from_config(cfg_path: str) -> Dict[str, Any]:
    cfg = RootConfig.from_yaml(cfg_path)
    if cfg.dataset != "tapvid" or cfg.tapvid is None:
        raise ValueError(f"[tapvid_eval] config dataset must be 'tapvid'. got: {cfg.dataset}")

    if cfg.davis_root is None:
        raise KeyError("[tapvid_eval] missing davis_root in config (needed to resolve frames)")
    if cfg.res is None:
        # default consistent with your DAVIS run
        res = "480p"
    else:
        res = cfg.res

    tcfg = cfg.tapvid
    return run_tapvid_eval(
        davis_root=cfg.davis_root,
        res=res,
        split=tcfg.split,
        pkl_path=tcfg.pkl_path,
        query_mode=tcfg.query_mode,
        stride=tcfg.stride,
        pred_dir=tcfg.pred_dir,
        out_dir=tcfg.out_dir,
        resize_to_256=tcfg.resize_to_256,
        config_path=cfg.config_path,
        config_sha1=cfg.config_sha1,
    )




# -----------------------------
# TAP-Vid metrics (official-style)
# -----------------------------
def compute_tapvid_metrics(
    query_points: np.ndarray,
    gt_occluded: np.ndarray,
    gt_tracks: np.ndarray,
    pred_occluded: np.ndarray,
    pred_tracks: np.ndarray,
    query_mode: str,
    get_trackwise_metrics: bool = False,
) -> Mapping[str, np.ndarray]:
    """
    Official-style TAP-Vid metrics: OA, PtsWithin@{1,2,4,8,16}, Jac@{1,2,4,8,16}, AJ, <δ_avg.

    Shapes (with batch):
      query_points : [B, N, 3] in [t, y, x]
      gt_occluded  : [B, N, T] bool
      gt_tracks    : [B, N, T, 2] in [x, y]
      pred_occluded: [B, N, T] bool
      pred_tracks  : [B, N, T, 2] in [x, y]
    """
    if query_mode not in ("first", "strided"):
        raise ValueError(f"Unknown query_mode={query_mode!r}, expected 'first' or 'strided'.")

    # Basic validation
    for name, arr in [
        ("query_points", query_points),
        ("gt_occluded", gt_occluded),
        ("gt_tracks", gt_tracks),
        ("pred_occluded", pred_occluded),
        ("pred_tracks", pred_tracks),
    ]:
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"{name} must be np.ndarray, got {type(arr)}")
        if not np.isfinite(arr).all() and arr.dtype != np.bool_:
            raise ValueError(f"{name} contains non-finite values.")

    B = gt_tracks.shape[0]
    if query_points.shape[0] != B:
        raise ValueError("Batch mismatch: query_points vs gt_tracks")
    if gt_occluded.shape[:2] != gt_tracks.shape[:2]:
        raise ValueError("Shape mismatch: gt_occluded vs gt_tracks (B,N)")
    if pred_tracks.shape != gt_tracks.shape:
        raise ValueError("Shape mismatch: pred_tracks must match gt_tracks")
    if pred_occluded.shape != gt_occluded.shape:
        raise ValueError("Shape mismatch: pred_occluded must match gt_occluded")

    summing_axis = (2,) if get_trackwise_metrics else (1, 2)
    metrics: Dict[str, np.ndarray] = {}

    # Build evaluation mask (exclude query frame; for 'first' exclude pre-query frames)
    T = gt_tracks.shape[2]
    eye = np.eye(T, dtype=np.int32)
    if query_mode == "first":
        # evaluate frames AFTER the query frame
        query_frame_to_eval_frames = np.cumsum(eye, axis=1) - eye
    else:
        # evaluate ALL frames EXCEPT the query frame
        query_frame_to_eval_frames = 1 - eye

    query_frame = np.round(query_points[..., 0]).astype(np.int32)
    if (query_frame < 0).any() or (query_frame >= T).any():
        raise ValueError(f"query_frame out of range [0,{T-1}]")

    evaluation_points = query_frame_to_eval_frames[query_frame] > 0  # [B,N,T] bool

    # Occlusion accuracy: match gt occlusion on evaluation frames
    occ_acc = (
        np.sum((pred_occluded == gt_occluded) & evaluation_points, axis=summing_axis)
        / np.sum(evaluation_points, axis=summing_axis)
    )
    metrics["occlusion_accuracy"] = occ_acc

    visible = ~gt_occluded
    pred_visible = ~pred_occluded

    all_frac_within: List[np.ndarray] = []
    all_jaccard: List[np.ndarray] = []

    for thresh in [1, 2, 4, 8, 16]:
        within_dist = np.sum((pred_tracks - gt_tracks) ** 2, axis=-1) < (thresh**2)  # [B,N,T]
        is_correct = within_dist & visible

        count_correct = np.sum(is_correct & evaluation_points, axis=summing_axis)
        count_visible = np.sum(visible & evaluation_points, axis=summing_axis)
        # Avoid divide-by-zero
        frac_correct = np.where(count_visible > 0, count_correct / count_visible, 0.0)
        metrics[f"pts_within_{thresh}"] = frac_correct
        all_frac_within.append(frac_correct)

        true_pos = np.sum(is_correct & pred_visible & evaluation_points, axis=summing_axis)
        gt_pos = np.sum(visible & evaluation_points, axis=summing_axis)

        false_pos = (~visible) & pred_visible
        false_pos = false_pos | ((~within_dist) & pred_visible)
        false_pos = np.sum(false_pos & evaluation_points, axis=summing_axis)

        denom = gt_pos + false_pos
        jacc = np.where(denom > 0, true_pos / denom, 0.0)
        metrics[f"jaccard_{thresh}"] = jacc
        all_jaccard.append(jacc)

    metrics["average_jaccard"] = np.mean(np.stack(all_jaccard, axis=1), axis=1)
    metrics["average_pts_within_thresh"] = np.mean(np.stack(all_frac_within, axis=1), axis=1)
    return metrics


def latex_table(mean_scalars: Mapping[str, float]) -> str:
    """
    Produce a 2-line LaTeX table row header + body, aligned with the TAP-Vid codebase.
    """
    if "average_jaccard" in mean_scalars:
        fields = [
            "average_jaccard",
            "average_pts_within_thresh",
            "occlusion_accuracy",
            "jaccard_1", "jaccard_2", "jaccard_4", "jaccard_8", "jaccard_16",
            "pts_within_1", "pts_within_2", "pts_within_4", "pts_within_8", "pts_within_16",
        ]
        header = (
            "AJ & $<\\delta^{x}_{avg}$ & OA & "
            "Jac. $\\delta^{0}$ & Jac. $\\delta^{1}$ & Jac. $\\delta^{2}$ & Jac. $\\delta^{3}$ & Jac. $\\delta^{4}$ & "
            "$<\\delta^{0}$ & $<\\delta^{1}$ & $<\\delta^{2}$ & $<\\delta^{3}$ & $<\\delta^{4}$"
        )
    else:
        fields = ["PCK@0.1", "PCK@0.2", "PCK@0.3", "PCK@0.4", "PCK@0.5"]
        header = " & ".join(fields)

    body = " & ".join([f"{float(mean_scalars[k] * 100.0):.3f}" for k in fields])
    return "\n".join([header, body])


# -----------------------------
# Robust IO helpers
# -----------------------------
@dataclass(frozen=True)
class TapVidExample:
    video_id: str
    query_points: np.ndarray   # [N,3] [t,y,x]
    gt_tracks: np.ndarray      # [N,T,2] [x,y]
    gt_occluded: np.ndarray    # [N,T] bool


def _load_pickle(path: Union[str, Path]) -> Any:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Pickle not found: {path}")
    with path.open("rb") as f:
        return pickle.load(f)


def _as_bool(a: np.ndarray) -> np.ndarray:
    if a.dtype == np.bool_:
        return a
    # common: {0,1} int
    return (a.astype(np.int32) != 0)


def _ensure_batch(arr: np.ndarray) -> np.ndarray:
    return arr[np.newaxis, ...]


def _auto_scale_coords_to_256(
    query_points: np.ndarray,
    tracks: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    TAP-Vid metrics assume coords are in 256x256-scaled raster space (pixel-ish).
    But some dumps store coords in [0,1]. We auto-detect and scale if needed.
    """
    # tracks: [N,T,2] ; query_points: [N,3] [t,y,x]
    max_xy = float(np.nanmax(tracks[..., :2]))
    max_q = float(np.nanmax(query_points[..., 1:3])) if query_points.size else 0.0

    # Heuristic: if values look like normalized coords
    if max(max_xy, max_q) <= 2.0:
        tracks_256 = tracks * 256.0
        q_256 = query_points.copy()
        q_256[:, 1:3] *= 256.0
        return q_256, tracks_256
    return query_points, tracks


def _read_predictions(pred_npz: Path) -> Tuple[np.ndarray, np.ndarray]:
    if not pred_npz.exists():
        raise FileNotFoundError(f"Prediction npz missing: {pred_npz}")
    arr = np.load(pred_npz, allow_pickle=False)
    if "pred_tracks" not in arr or "pred_occluded" not in arr:
        raise KeyError(f"{pred_npz} must contain keys: pred_tracks, pred_occluded")
    pred_tracks = np.asarray(arr["pred_tracks"], dtype=np.float32)
    pred_occluded = _as_bool(np.asarray(arr["pred_occluded"]))
    return pred_tracks, pred_occluded


def _iter_examples_from_pickle(obj: Any) -> Iterable[Tuple[str, Mapping[str, Any]]]:
    """
    Support multiple common pickle layouts:
      - list[dict]
      - dict[str, dict]
      - dict with 'examples'
    Each example dict should include: query_points, target_points/gt_tracks, occluded
    """
    if isinstance(obj, dict) and "examples" in obj and isinstance(obj["examples"], (list, tuple)):
        for i, ex in enumerate(obj["examples"]):
            yield str(ex.get("video_id", i)), ex
        return

    if isinstance(obj, dict):
        # maybe dict[str, exdict]
        all_vals = list(obj.values())
        if all_vals and isinstance(all_vals[0], dict):
            for k, v in obj.items():
                yield str(k), v
            return

    if isinstance(obj, (list, tuple)):
        for i, ex in enumerate(obj):
            if not isinstance(ex, dict):
                raise TypeError(f"Example[{i}] must be dict, got {type(ex)}")
            yield str(ex.get("video_id", i)), ex
        return

    raise TypeError(f"Unsupported pickle root type: {type(obj)}")


def load_tapvid_points_pickle(points_path: Union[str, Path]) -> List[TapVidExample]:
    """
    Load a TAP-Vid points pickle into standardized TapVidExample list.

    Required per-example keys (any alias accepted):
      - query_points: [N,3]  (or build from 'query_points' only; we don't resample)
      - target_points / gt_tracks: [N,T,2]
      - occluded / gt_occluded: [N,T]
    """
    raw = _load_pickle(points_path)

    examples: List[TapVidExample] = []
    for vid, ex in _iter_examples_from_pickle(raw):
        # key aliases
        if "query_points" in ex:
            q = np.asarray(ex["query_points"])
        else:
            raise KeyError(f"[{vid}] missing 'query_points'")

        if "target_points" in ex:
            gt_tracks = np.asarray(ex["target_points"])
        elif "gt_tracks" in ex:
            gt_tracks = np.asarray(ex["gt_tracks"])
        else:
            raise KeyError(f"[{vid}] missing 'target_points'/'gt_tracks'")

        if "occluded" in ex:
            gt_occ = _as_bool(np.asarray(ex["occluded"]))
        elif "gt_occluded" in ex:
            gt_occ = _as_bool(np.asarray(ex["gt_occluded"]))
        else:
            raise KeyError(f"[{vid}] missing 'occluded'/'gt_occluded'")

        # Many dumps store batch dim [1,N,...]; squeeze it.
        q = np.asarray(q)
        gt_tracks = np.asarray(gt_tracks)
        gt_occ = np.asarray(gt_occ)

        if q.ndim == 3 and q.shape[0] == 1:
            q = q[0]
        if gt_tracks.ndim == 4 and gt_tracks.shape[0] == 1:
            gt_tracks = gt_tracks[0]
        if gt_occ.ndim == 3 and gt_occ.shape[0] == 1:
            gt_occ = gt_occ[0]

        if q.ndim != 2 or q.shape[1] != 3:
            raise ValueError(f"[{vid}] query_points must be [N,3], got {q.shape}")
        if gt_tracks.ndim != 3 or gt_tracks.shape[2] != 2:
            raise ValueError(f"[{vid}] gt_tracks must be [N,T,2], got {gt_tracks.shape}")
        if gt_occ.shape != gt_tracks.shape[:2]:
            raise ValueError(f"[{vid}] gt_occluded shape {gt_occ.shape} != (N,T) {gt_tracks.shape[:2]}")

        # Auto scale coords to 256 if needed (handles [0,1] dumps)
        q, gt_tracks = _auto_scale_coords_to_256(q.astype(np.float32), gt_tracks.astype(np.float32))

        examples.append(
            TapVidExample(
                video_id=str(vid),
                query_points=q.astype(np.float32),
                gt_tracks=gt_tracks.astype(np.float32),
                gt_occluded=_as_bool(gt_occ),
            )
        )
    return examples


# -----------------------------
# Eval runner
# -----------------------------
def evaluate_tapvid(
    points_path: Union[str, Path],
    pred_dir: Union[str, Path],
    query_mode: str,
    out_dir: Union[str, Path],
) -> Dict[str, float]:
    pred_dir = Path(pred_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    examples = load_tapvid_points_pickle(points_path)
    if not examples:
        raise ValueError("No examples loaded from points pickle.")

    per_video_rows: List[Dict[str, Any]] = []
    all_metrics: Dict[str, List[float]] = {}

    for ex in examples:
        pred_npz = pred_dir / f"{ex.video_id}.npz"
        pred_tracks, pred_occ = _read_predictions(pred_npz)

        # squeeze possible batch
        if pred_tracks.ndim == 4 and pred_tracks.shape[0] == 1:
            pred_tracks = pred_tracks[0]
        if pred_occ.ndim == 3 and pred_occ.shape[0] == 1:
            pred_occ = pred_occ[0]

        if pred_tracks.shape != ex.gt_tracks.shape:
            raise ValueError(
                f"[{ex.video_id}] pred_tracks shape {pred_tracks.shape} != gt_tracks {ex.gt_tracks.shape}"
            )
        if pred_occ.shape != ex.gt_occluded.shape:
            raise ValueError(
                f"[{ex.video_id}] pred_occluded shape {pred_occ.shape} != gt_occluded {ex.gt_occluded.shape}"
            )

        # Auto-scale pred coords too (in case your tracker outputs [0,1])
        q = ex.query_points
        _, pred_tracks_256 = _auto_scale_coords_to_256(q, pred_tracks.astype(np.float32))

        m = compute_tapvid_metrics(
            query_points=_ensure_batch(ex.query_points),
            gt_occluded=_ensure_batch(ex.gt_occluded),
            gt_tracks=_ensure_batch(ex.gt_tracks),
            pred_occluded=_ensure_batch(_as_bool(pred_occ)),
            pred_tracks=_ensure_batch(pred_tracks_256),
            query_mode=query_mode,
            get_trackwise_metrics=False,
        )
        row = {"video_id": ex.video_id}
        for k, v in m.items():
            row[k] = float(np.asarray(v).reshape(-1)[0])
            all_metrics.setdefault(k, []).append(row[k])
        per_video_rows.append(row)

    # aggregate mean across videos
    mean_scalars = {k: float(np.mean(v)) for k, v in all_metrics.items()}

    # write csv
    csv_path = out_dir / "tapvid_metrics_per_video.csv"
    keys = ["video_id"] + sorted([k for k in mean_scalars.keys()])
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in per_video_rows:
            w.writerow({k: r.get(k, "") for k in keys})

    mean_path = out_dir / "tapvid_metrics_mean.json"
    mean_path.write_text(json.dumps(mean_scalars, indent=2, sort_keys=True))

    latex_path = out_dir / "tapvid_metrics_latex.txt"
    latex_path.write_text(latex_table(mean_scalars) + "\n")

    return mean_scalars


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--points_path", type=str, required=True, help="TAP-Vid points pickle (e.g., tapvid_davis.pkl)")
    ap.add_argument("--pred_dir", type=str, required=True, help="Directory of per-video prediction npz files")
    ap.add_argument("--query_mode", type=str, default="first", choices=["first", "strided"])
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    mean = evaluate_tapvid(
        points_path=args.points_path,
        pred_dir=args.pred_dir,
        query_mode=args.query_mode,
        out_dir=args.out_dir,
    )
    print("[TAP-Vid] mean metrics:")
    for k in sorted(mean.keys()):
        print(f"  {k:>24s}: {mean[k]*100.0:7.3f}")


if __name__ == "__main__":
    main()