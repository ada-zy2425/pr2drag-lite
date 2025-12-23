# pr2drag/tier0/runner_tapvid.py
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from pr2drag.tier1.contracts import RootConfig
from pr2drag.datasets.tapvid import build_tapvid_dataset, TapVidSeq
from pr2drag.tier0.metrics_tapvid import compute_tapvid_metrics, TapVidMetrics


# ---------------------------
# small utils
# ---------------------------
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_pred_dir(base_out: str, tracker: str, query_mode: str, split: str) -> Path:
    return Path(base_out) / "tapvid_preds" / f"{tracker}_{query_mode}_{split}"


def _default_out_dir(base_out: str, tracker: str, query_mode: str, split: str) -> Path:
    return Path(base_out) / "tapvid_reports" / f"{tracker}_{query_mode}_{split}"


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _require(cond: bool, msg: str, exc: type[Exception] = ValueError) -> None:
    if not cond:
        raise exc(msg)


def _as_bool(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.dtype == np.bool_:
        return a
    return (a.astype(np.int32) != 0)


def _tyx_to_txy(q_tyx: np.ndarray) -> np.ndarray:
    """
    metrics_tapvid.py uses queries_txy where cols 1:2 are (x,y).
    We store queries as (t,y,x) for TAP-Vid, so convert.
    """
    q = np.asarray(q_tyx, dtype=np.float32)
    _require(q.ndim == 2 and q.shape[1] == 3, f"[tapvid] queries must be [Q,3], got {q.shape}")
    out = q.copy()
    out[:, 1] = q[:, 2]  # x
    out[:, 2] = q[:, 1]  # y
    return out


# ---------------------------
# video IO + resize to 256 (optional)
# ---------------------------
def _glob_davis_frames(davis_root: str, res: str, seq_name: str) -> List[str]:
    root = Path(davis_root) / "JPEGImages" / res / seq_name
    if not root.exists():
        return []
    # accept jpg/png
    files = sorted([p for p in root.glob("*.jpg")]) + sorted([p for p in root.glob("*.png")])
    return [str(p) for p in files]


def _load_video_from_frames(frame_paths: List[str]) -> np.ndarray:
    try:
        import cv2  # type: ignore
    except Exception as e:
        raise RuntimeError("OpenCV required to load frames: pip install opencv-python") from e

    _require(len(frame_paths) > 0, "[tapvid] cannot load video: empty frame list")

    frames = []
    for p in frame_paths:
        im = cv2.imread(str(p), cv2.IMREAD_COLOR)
        _require(im is not None, f"[tapvid] failed to read frame: {p}")
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        frames.append(im)
    return np.stack(frames, axis=0).astype(np.uint8)


def _resize_video_to_256(video_uint8: np.ndarray, keep_aspect: bool, interp: str) -> np.ndarray:
    """
    Resize/pad video to [T,256,256,3] with the SAME geometry as datasets/tapvid.py::_resize_coords_to_256.
    """
    try:
        import cv2  # type: ignore
    except Exception as e:
        raise RuntimeError("OpenCV required to resize video: pip install opencv-python") from e

    _require(interp in ("bilinear", "nearest"), f"[tapvid] interp must be bilinear/nearest, got {interp!r}")
    inter = cv2.INTER_LINEAR if interp == "bilinear" else cv2.INTER_NEAREST

    v = np.asarray(video_uint8)
    _require(v.ndim == 4 and v.shape[-1] == 3, f"[tapvid] video must be [T,H,W,3], got {v.shape}")
    _require(v.dtype == np.uint8, f"[tapvid] video must be uint8, got {v.dtype}")

    T, H, W, _ = v.shape
    if not keep_aspect:
        out = np.stack([cv2.resize(v[t], (256, 256), interpolation=inter) for t in range(T)], axis=0)
        return out.astype(np.uint8)

    scale = min(256.0 / float(W), 256.0 / float(H))
    newW = int(round(float(W) * scale))
    newH = int(round(float(H) * scale))
    newW = max(1, min(256, newW))
    newH = max(1, min(256, newH))

    pad_x = int(np.floor((256.0 - float(newW)) * 0.5))
    pad_y = int(np.floor((256.0 - float(newH)) * 0.5))

    resized = np.stack([cv2.resize(v[t], (newW, newH), interpolation=inter) for t in range(T)], axis=0)

    out = np.zeros((T, 256, 256, 3), dtype=np.uint8)
    out[:, pad_y:pad_y + newH, pad_x:pad_x + newW, :] = resized
    return out


def _get_seq_video_uint8(cfg: RootConfig, seq: TapVidSeq) -> np.ndarray:
    """
    Prefer seq.video if present in pkl (optional). Else use frame_paths; if empty, glob from davis_root.
    """
    v = getattr(seq, "video", None)
    if isinstance(v, np.ndarray):
        _require(v.ndim == 4 and v.shape[-1] == 3, f"[tapvid] seq.video must be [T,H,W,3], got {v.shape}")
        if v.dtype != np.uint8:
            if np.issubdtype(v.dtype, np.floating):
                vv = v.astype(np.float32)
                vmax = float(np.nanmax(vv)) if vv.size else 0.0
                if vmax <= 1.5:
                    vv = vv * 255.0
                vv = np.clip(vv, 0.0, 255.0)
                v = vv.astype(np.uint8)
            else:
                v = np.clip(v, 0, 255).astype(np.uint8)
        return v

    fps = list(seq.frame_paths)
    if len(fps) == 0:
        # fallback: glob from DAVIS folder
        res = cfg.res or "480p"
        _require(bool(cfg.davis_root), "[tapvid] davis_root is required to resolve frames")
        fps = _glob_davis_frames(cfg.davis_root, res, seq.name)
    return _load_video_from_frames(fps)


# ---------------------------
# NPZ IO (canonical + legacy compat)
# ---------------------------
def _save_pred_npz(
    npz_path: Path,
    tracks_xy_TQ2: np.ndarray,   # [T,Q,2]
    vis_TQ: np.ndarray,          # [T,Q] bool/0-1
    queries_tyx_Q3: np.ndarray,  # [Q,3] (t,y,x)
    video_hw: Tuple[int, int],
) -> None:
    tracks = np.asarray(tracks_xy_TQ2, dtype=np.float32)
    vis = _as_bool(np.asarray(vis_TQ))
    q = np.asarray(queries_tyx_Q3, dtype=np.float32)

    _require(tracks.ndim == 3 and tracks.shape[-1] == 2, f"[tapvid_npz] tracks_xy must be [T,Q,2], got {tracks.shape}")
    _require(vis.ndim == 2 and vis.shape == tracks.shape[:2], f"[tapvid_npz] vis must match [T,Q]={tracks.shape[:2]}, got {vis.shape}")
    _require(q.ndim == 2 and q.shape[1] == 3 and q.shape[0] == tracks.shape[1],
             f"[tapvid_npz] queries_tyx must be [Q,3] with Q={tracks.shape[1]}, got {q.shape}")

    np.savez_compressed(
        npz_path,
        tracks_xy=tracks.astype(np.float32),           # [T,Q,2]
        vis=vis.astype(np.uint8),                      # [T,Q] 1=visible
        queries_tyx=q.astype(np.float32),              # [Q,3] (t,y,x)
        video_hw=np.asarray(video_hw, dtype=np.int32), # [2] (H,W)
    )


def _load_pred_npz_compat(npz_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, int]]:
    """
    Canonical:
      - tracks_xy: [T,Q,2], vis: [T,Q], queries_tyx: [Q,3], video_hw

    Legacy-A:
      - pred_tracks: [Q,T,2], pred_occluded: [Q,T], query_points: [Q,3] (t,y,x), video_hw
    """
    arr = np.load(npz_path, allow_pickle=False)
    keys = set(arr.files)

    _require("video_hw" in keys, f"[tapvid_eval] {npz_path} missing video_hw")
    hw = tuple(int(x) for x in np.asarray(arr["video_hw"]).reshape(-1).tolist())
    _require(len(hw) == 2, f"[tapvid_eval] bad video_hw in {npz_path}: {hw}")
    video_hw = (hw[0], hw[1])

    if "tracks_xy" in keys and "vis" in keys:
        tracks = np.asarray(arr["tracks_xy"], dtype=np.float32)
        vis = _as_bool(np.asarray(arr["vis"]))
        if "queries_tyx" in keys:
            q = np.asarray(arr["queries_tyx"], dtype=np.float32)
        elif "query_points" in keys:
            q = np.asarray(arr["query_points"], dtype=np.float32)
        else:
            raise KeyError(f"[tapvid_eval] {npz_path} missing queries_tyx/query_points")
        return tracks, vis, q, video_hw

    if "pred_tracks" in keys and "pred_occluded" in keys:
        pred_QT2 = np.asarray(arr["pred_tracks"], dtype=np.float32)   # [Q,T,2]
        occ_QT = _as_bool(np.asarray(arr["pred_occluded"]))           # [Q,T]
        _require(pred_QT2.ndim == 3 and pred_QT2.shape[-1] == 2, f"[tapvid_eval] pred_tracks must be [Q,T,2], got {pred_QT2.shape}")
        _require(occ_QT.shape == pred_QT2.shape[:2], f"[tapvid_eval] pred_occluded must be [Q,T], got {occ_QT.shape}")

        tracks_TQ2 = np.transpose(pred_QT2, (1, 0, 2)).astype(np.float32)  # [T,Q,2]
        vis_TQ = (~np.transpose(occ_QT, (1, 0))).astype(bool)              # [T,Q]

        if "query_points" in keys:
            q = np.asarray(arr["query_points"], dtype=np.float32)
        else:
            raise KeyError(f"[tapvid_eval] {npz_path} missing query_points")
        return tracks_TQ2, vis_TQ, q, video_hw

    raise KeyError(f"[tapvid_eval] unknown npz schema in {npz_path.name}, keys={sorted(keys)}")


# ---------------------------
# CoTracker output normalization
# ---------------------------
def _extract_tracks_occ_from_any(out: Any) -> Tuple[np.ndarray, np.ndarray]:
    """
    Accept common outputs:
      - dict with keys tracks_xy/occluded or tracks/occ
      - object with attributes tracks_xy/occluded or tracks/occ
    """
    if isinstance(out, dict):
        tracks = out.get("tracks_xy", out.get("tracks", None))
        occ = out.get("occluded", out.get("occ", None))
        if tracks is None or occ is None:
            raise KeyError("[cotracker_v2] output dict must have tracks_xy(+occluded) or tracks(+occ)")
        return np.asarray(tracks), _as_bool(np.asarray(occ))

    tracks = getattr(out, "tracks_xy", None)
    if tracks is None:
        tracks = getattr(out, "tracks", None)
    occ = getattr(out, "occluded", None)
    if occ is None:
        occ = getattr(out, "occ", None)
    if tracks is None or occ is None:
        raise KeyError("[cotracker_v2] output object must have tracks_xy(+occluded) or tracks(+occ)")
    return np.asarray(tracks), _as_bool(np.asarray(occ))


def _force_TQ(tracks: np.ndarray, occ: np.ndarray, T: int, Q: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    CoTracker may return:
      - tracks [Q,T,2] + occ [Q,T]
      - tracks [T,Q,2] + occ [T,Q]
    Convert to tracks_TQ2 [T,Q,2] and vis_TQ [T,Q].
    """
    tracks = np.asarray(tracks, dtype=np.float32)
    occ = _as_bool(np.asarray(occ))
    _require(tracks.ndim == 3 and tracks.shape[-1] == 2, f"[cotracker_v2] tracks must be rank-3 [...,2], got {tracks.shape}")
    _require(occ.ndim == 2, f"[cotracker_v2] occ must be rank-2, got {occ.shape}")
    _require(occ.shape == tracks.shape[:2], f"[cotracker_v2] occ shape {occ.shape} != tracks[:2] {tracks.shape[:2]}")

    # case A: [T,Q,2]
    if tracks.shape[0] == T and tracks.shape[1] == Q:
        vis = (~occ).astype(bool)
        return tracks, vis

    # case B: [Q,T,2]
    if tracks.shape[0] == Q and tracks.shape[1] == T:
        tracks_TQ2 = np.transpose(tracks, (1, 0, 2)).astype(np.float32)
        vis_TQ = (~np.transpose(occ, (1, 0))).astype(bool)
        return tracks_TQ2, vis_TQ

    raise ValueError(f"[cotracker_v2] cannot align shapes: tracks={tracks.shape}, expected (T,Q,2)=({T},{Q},2) or (Q,T,2)=({Q},{T},2)")


# ---------------------------
# main entrypoints
# ---------------------------
def run_tapvid_pred(cfg: RootConfig, config_path: str, config_sha1: str) -> Dict[str, Any]:
    if cfg.dataset != "tapvid" or cfg.tapvid is None:
        raise ValueError("[tapvid_pred] RootConfig must be dataset=tapvid")

    tcfg = cfg.tapvid
    tracker = str(tcfg.tracker).strip().lower()

    if tracker not in ("oracle", "tapir", "cotracker_v2"):
        raise ValueError(f"[tapvid_pred] tracker must be one of oracle/tapir/cotracker_v2; got {tracker!r}")

    pred_dir = Path(tcfg.pred_dir).expanduser() if tcfg.pred_dir else _default_pred_dir(cfg.base_out, tracker, tcfg.query_mode, tcfg.split)
    out_dir = Path(tcfg.out_dir).expanduser() if tcfg.out_dir else _default_out_dir(cfg.base_out, tracker, tcfg.query_mode, tcfg.split)
    _ensure_dir(pred_dir)
    _ensure_dir(out_dir)

    res = cfg.res or "480p"

    # IMPORTANT: dataset coords follow cfg.tapvid.resize_to_256
    seqs = build_tapvid_dataset(
        davis_root=cfg.davis_root or "",
        pkl_path=tcfg.pkl_path,
        res=res,
        split=tcfg.split,
        query_mode=tcfg.query_mode,
        stride=tcfg.stride,
        max_queries=tcfg.max_queries,
        resize_to_256=bool(tcfg.resize_to_256),
        keep_aspect=bool(tcfg.keep_aspect),
        seed=tcfg.seed,
    )
    if not seqs:
        raise ValueError("[tapvid_pred] No sequences produced.")

    ok, skipped = 0, 0
    failed: List[Dict[str, str]] = []

    # lazy init tapir
    tapir_tracker = None
    if tracker == "tapir":
        from pr2drag.trackers.tapir import TapirTracker, TapirTrackerConfig  # noqa: F401
        tapir_tracker = TapirTracker(
            TapirTrackerConfig(
                resize_to_256=True,  # TAPIR backend convention
                keep_aspect=bool(tcfg.keep_aspect),
                interp=str(tcfg.interp),
            )
        )

    for s in seqs:
        npz_path = pred_dir / f"{s.name}.npz"
        if npz_path.exists() and (not bool(tcfg.overwrite_preds)):
            skipped += 1
            continue

        try:
            q_tyx = np.asarray(s.query_points_tyx, dtype=np.float32)  # [Q,3]
            q_ids = np.asarray(s.query_track_ids, dtype=np.int64)     # [Q]
            _require(q_tyx.ndim == 2 and q_tyx.shape[1] == 3, f"[tapvid_pred] {s.name} bad query_points_tyx {q_tyx.shape}")
            _require(q_ids.ndim == 1 and q_ids.shape[0] == q_tyx.shape[0], f"[tapvid_pred] {s.name} bad query_track_ids {q_ids.shape}")

            gt_NT2 = np.asarray(s.gt_tracks_xy, dtype=np.float32)     # [N,T,2]
            gt_occ_NT = _as_bool(np.asarray(s.gt_occluded))           # [N,T]
            _require(gt_NT2.ndim == 3 and gt_NT2.shape[-1] == 2, f"[tapvid_pred] {s.name} bad gt_tracks {gt_NT2.shape}")
            _require(gt_occ_NT.shape == gt_NT2.shape[:2], f"[tapvid_pred] {s.name} bad gt_occluded {gt_occ_NT.shape}")

            N, T = int(gt_NT2.shape[0]), int(gt_NT2.shape[1])
            Q = int(q_tyx.shape[0])
            _require(np.all((q_ids >= 0) & (q_ids < N)), f"[tapvid_pred] {s.name} query_track_ids out of range (N={N})")

            if tracker == "oracle":
                gt_QT2 = gt_NT2[q_ids, :, :]                # [Q,T,2]
                gt_occ_QT = gt_occ_NT[q_ids, :]             # [Q,T]
                tracks_TQ2 = np.transpose(gt_QT2, (1, 0, 2)).astype(np.float32)
                vis_TQ = (~np.transpose(gt_occ_QT, (1, 0))).astype(bool)
                _save_pred_npz(npz_path, tracks_TQ2, vis_TQ, q_tyx, s.video_hw)
                ok += 1
                continue

            elif tracker == "cotracker_v2":
                video = _get_seq_video_uint8(cfg, s)  # original frames
                _require(video.shape[0] == T, f"[tapvid_pred] {s.name} video T mismatch: video={video.shape[0]} gt={T}")

                # If coords were resized to 256 in dataset, we must feed a 256 video too.
                if bool(tcfg.resize_to_256):
                    video = _resize_video_to_256(video, keep_aspect=bool(tcfg.keep_aspect), interp=str(tcfg.interp))

                from pr2drag.trackers.cotracker_v2 import run_cotracker_v2  # local import
                out = run_cotracker_v2(
                    video_uint8=video,
                    queries_tyx=q_tyx.astype(np.float32),   # [Q,3] (t,y,x)
                    checkpoint=(tcfg.tracker_ckpt or None),
                    device=None,
                )
                tracks_any, occ_any = _extract_tracks_occ_from_any(out)
                tracks_TQ2, vis_TQ = _force_TQ(tracks_any, occ_any, T=T, Q=Q)

                _save_pred_npz(npz_path, tracks_TQ2, vis_TQ, q_tyx, s.video_hw)
                ok += 1
                continue

            elif tracker == "tapir":
                _require(tapir_tracker is not None, "[tapvid_pred] internal error: tapir_tracker is None")
                # TapirTracker in your codebase is expected to implement .predict(seq)->TapVidPred
                pred = tapir_tracker.predict(s)  # type: ignore[attr-defined]
                _save_pred_npz(npz_path, pred.tracks_xy, pred.vis, q_tyx, s.video_hw)
                ok += 1
                continue

            else:
                raise ValueError(f"[tapvid_pred] unknown tracker={tracker!r}")

        except Exception as e:
            if bool(tcfg.fail_fast):
                raise
            failed.append({"video_id": s.name, "error": repr(e)})

    manifest = {
        "created_utc": _now_iso(),
        "cmd": "tapvid_pred",
        "dataset": "tapvid",
        "config_path": config_path,
        "config_sha1": config_sha1,
        "davis_root": cfg.davis_root,
        "res": res,
        "tapvid": asdict(tcfg),
        "tracker_used": tracker,
        "pred_dir": str(pred_dir),
        "out_dir": str(out_dir),
        "num_seqs_total": int(len(seqs)),
        "num_ok": int(ok),
        "num_skipped": int(skipped),
        "num_failed": int(len(failed)),
        "failed": failed,
        "npz_contract": {
            "tracks_xy": "float32 [T,Q,2] (x,y) in SAME coord space as dataset GT (original or resized-to-256 depending on cfg.tapvid.resize_to_256)",
            "vis": "uint8 [T,Q] (1=visible,0=not)",
            "queries_tyx": "float32 [Q,3] (t,y,x) in same coord space as tracks",
            "video_hw": "int32 [2] (H,W) original",
        },
        "npz_legacy_compat": [
            "pred_tracks float32 [Q,T,2] + pred_occluded uint8 [Q,T] + query_points [Q,3]",
        ],
    }
    (pred_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))

    print(f"[tapvid_pred] pred_dir: {pred_dir}")
    print(f"[tapvid_pred] wrote manifest: {pred_dir/'manifest.json'}")
    print(f"[tapvid_pred] ok={ok} skipped={skipped} failed={len(failed)}")
    return manifest


def run_tapvid_eval(cfg: RootConfig, config_path: str, config_sha1: str) -> Dict[str, Any]:
    if cfg.dataset != "tapvid" or cfg.tapvid is None:
        raise ValueError("[tapvid_eval] RootConfig must be dataset=tapvid")

    tcfg = cfg.tapvid
    tracker = str(tcfg.tracker).strip().lower()

    # eval can allow tracker=none only if pred_dir explicitly given
    if tracker == "none" and not str(tcfg.pred_dir).strip():
        raise ValueError("[tapvid_eval] tracker=none requires explicit tapvid.pred_dir (cannot infer).")

    pred_dir = Path(tcfg.pred_dir).expanduser() if tcfg.pred_dir else _default_pred_dir(cfg.base_out, tracker, tcfg.query_mode, tcfg.split)
    out_dir = Path(tcfg.out_dir).expanduser() if tcfg.out_dir else _default_out_dir(cfg.base_out, tracker, tcfg.query_mode, tcfg.split)
    _ensure_dir(out_dir)

    res = cfg.res or "480p"

    seqs = build_tapvid_dataset(
        davis_root=cfg.davis_root or "",
        pkl_path=tcfg.pkl_path,
        res=res,
        split=tcfg.split,
        query_mode=tcfg.query_mode,
        stride=tcfg.stride,
        max_queries=tcfg.max_queries,
        resize_to_256=bool(tcfg.resize_to_256),
        keep_aspect=bool(tcfg.keep_aspect),
        seed=tcfg.seed,
    )
    if not seqs:
        raise ValueError("[tapvid_eval] No sequences produced.")

    evaluated, missing, failed = 0, 0, 0
    per_video: List[Dict[str, Any]] = []

    for s in seqs:
        npz_path = pred_dir / f"{s.name}.npz"
        if not npz_path.exists():
            missing += 1
            if (not bool(tcfg.allow_missing_preds)) and bool(tcfg.fail_fast):
                raise FileNotFoundError(f"[tapvid_eval] missing pred npz: {npz_path}")
            continue

        try:
            pred_tracks_TQ2, pred_vis_TQ, _pred_q_tyx, _pred_hw = _load_pred_npz_compat(npz_path)

            q_tyx = np.asarray(s.query_points_tyx, dtype=np.float32)  # [Q,3]
            q_ids = np.asarray(s.query_track_ids, dtype=np.int64)     # [Q]

            gt_NT2 = np.asarray(s.gt_tracks_xy, dtype=np.float32)     # [N,T,2]
            gt_occ_NT = _as_bool(np.asarray(s.gt_occluded))           # [N,T]
            N, T = int(gt_NT2.shape[0]), int(gt_NT2.shape[1])
            Q = int(q_tyx.shape[0])

            _require(pred_tracks_TQ2.shape == (T, Q, 2), f"[tapvid_eval] {s.name} pred tracks shape {pred_tracks_TQ2.shape} != {(T,Q,2)}")
            _require(pred_vis_TQ.shape == (T, Q), f"[tapvid_eval] {s.name} pred vis shape {pred_vis_TQ.shape} != {(T,Q)}")
            _require(np.all((q_ids >= 0) & (q_ids < N)), f"[tapvid_eval] {s.name} query ids out of range (N={N})")

            gt_QT2 = gt_NT2[q_ids, :, :]                    # [Q,T,2]
            gt_occ_QT = gt_occ_NT[q_ids, :]                 # [Q,T]
            gt_tracks_TQ2 = np.transpose(gt_QT2, (1, 0, 2)).astype(np.float32)
            gt_vis_TQ = (~np.transpose(gt_occ_QT, (1, 0))).astype(bool)

            # IMPORTANT: compute_tapvid_metrics expects queries_txy
            queries_txy = _tyx_to_txy(q_tyx)

            # Since dataset GT and preds are ALREADY in the chosen coord space (original or 256),
            # DO NOT rescale inside metrics again.
            m: TapVidMetrics = compute_tapvid_metrics(
                gt_tracks_xy=gt_tracks_TQ2,           # [T,Q,2]
                gt_vis=gt_vis_TQ,                     # [T,Q]
                pred_tracks_xy=pred_tracks_TQ2,        # [T,Q,2]
                pred_vis=pred_vis_TQ.astype(bool),     # [T,Q]
                queries_txy=queries_txy,               # [Q,3] (t,x,y)
                resize_to_256=False,                   # <-- critical: avoid double scaling
                video_hw=s.video_hw,
            )

            row: Dict[str, Any] = {
                "video_id": s.name,
                "aj": float(m.aj),
                "oa": float(m.oa),
                "delta_x": float(m.delta_x) if np.isfinite(m.delta_x) else None,
                "q": int(m.q),
                "t": int(m.t),
            }
            for thr, val in m.jaccard_by_thr.items():
                row[f"jaccard_{int(thr)}"] = float(val)
            for thr, val in m.pck_by_thr.items():
                row[f"pck_{int(thr)}"] = float(val)

            per_video.append(row)
            evaluated += 1

        except Exception as e:
            failed += 1
            if bool(tcfg.fail_fast):
                raise
            per_video.append({"video_id": s.name, "error": repr(e)})

    # mean over numeric fields only
    mean: Dict[str, float] = {}
    if per_video:
        keys: set[str] = set()
        for r in per_video:
            for k, v in r.items():
                if k in ("video_id", "error"):
                    continue
                if isinstance(v, (int, float)) and v is not None:
                    keys.add(k)
        for k in sorted(keys):
            vals = [float(r[k]) for r in per_video if (k in r and isinstance(r[k], (int, float)) and r[k] is not None)]
            if vals:
                mean[k] = float(np.mean(vals))

    (out_dir / "tapvid_metrics_per_video.json").write_text(json.dumps(per_video, indent=2, sort_keys=True))
    (out_dir / "tapvid_metrics_mean.json").write_text(json.dumps(mean, indent=2, sort_keys=True))

    print(f"[tapvid_eval] pred_dir: {pred_dir}")
    print(f"[tapvid_eval] out_dir : {out_dir}")
    print(f"[tapvid_eval] evaluated={evaluated} missing={missing} failed={failed}")
    print("[tapvid_eval] mean:")
    for k in sorted(mean.keys()):
        v = mean[k]
        if k in ("aj", "oa") or k.startswith("jaccard_") or k.startswith("pck_"):
            print(f"  {k:>24s}: {v*100.0:7.3f}")
        else:
            print(f"  {k:>24s}: {v:7.4f}")

    return {
        "created_utc": _now_iso(),
        "cmd": "tapvid_eval",
        "dataset": "tapvid",
        "config_path": config_path,
        "config_sha1": config_sha1,
        "tracker_used": tracker,
        "pred_dir": str(pred_dir),
        "out_dir": str(out_dir),
        "evaluated": int(evaluated),
        "missing": int(missing),
        "failed": int(failed),
        "mean": mean,
    }