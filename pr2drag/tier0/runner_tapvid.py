# pr2drag/tier0/runner_tapvid.py
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from pr2drag.tier1.contracts import RootConfig
from pr2drag.datasets.tapvid import build_tapvid_dataset, TapVidSeq
from pr2drag.tier0.metrics_tapvid import compute_tapvid_metrics, TapVidMetrics


# ---------------------------
# helpers
# ---------------------------
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_pred_dir(base_out: str, tracker: str, query_mode: str, split: str) -> Path:
    return Path(base_out) / "tapvid_preds" / f"{tracker}_{query_mode}_{split}"


def _default_out_dir(base_out: str, tracker: str, query_mode: str, split: str) -> Path:
    return Path(base_out) / "tapvid_reports" / f"{tracker}_{query_mode}_{split}"


def _as_bool(a: np.ndarray) -> np.ndarray:
    if a.dtype == np.bool_:
        return a
    return (a.astype(np.int32) != 0)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _require(cond: bool, msg: str, exc: type[Exception] = ValueError) -> None:
    if not cond:
        raise exc(msg)


def _tyx_to_txy(q_tyx: np.ndarray) -> np.ndarray:
    """
    Convert queries [Q,3] (t,y,x) -> (t,x,y) for metrics_tapvid.py (it treats cols 1: as (x,y)).
    """
    q = np.asarray(q_tyx, dtype=np.float32)
    _require(q.ndim == 2 and q.shape[1] == 3, f"[tapvid] queries must be [Q,3], got {q.shape}")
    out = q.copy()
    out[:, 1] = q[:, 2]  # x
    out[:, 2] = q[:, 1]  # y
    return out


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
    _require(vis.ndim == 2 and vis.shape == tracks.shape[:2], f"[tapvid_npz] vis must be [T,Q]={tracks.shape[:2]}, got {vis.shape}")
    _require(q.ndim == 2 and q.shape[1] == 3 and q.shape[0] == tracks.shape[1],
             f"[tapvid_npz] queries_tyx must be [Q,3] with Q={tracks.shape[1]}, got {q.shape}")

    np.savez_compressed(
        npz_path,
        tracks_xy=tracks.astype(np.float32),              # [T,Q,2]
        vis=vis.astype(np.uint8),                         # [T,Q] (0/1)
        queries_tyx=q.astype(np.float32),                 # [Q,3] (t,y,x)
        video_hw=np.asarray(video_hw, dtype=np.int32),    # [2] (H,W)
    )


def _load_pred_npz_compat(npz_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, int]]:
    """
    Canonical:
      - tracks_xy: [T,Q,2], vis: [T,Q], queries_tyx: [Q,3], video_hw

    Legacy-A (your first knife):
      - pred_tracks: [Q,T,2], pred_occluded: [Q,T], query_points: [Q,3] (t,y,x), video_hw

    Legacy-B (older base.py style):
      - tracks_xy: [T,Q,2], vis: [T,Q], queries_txy or queries_tyx may exist
    """
    arr = np.load(npz_path, allow_pickle=False)
    keys = set(arr.files)

    # video_hw (required)
    _require("video_hw" in keys, f"[tapvid_eval] {npz_path} missing video_hw")
    hw = tuple(int(x) for x in np.asarray(arr["video_hw"]).reshape(-1).tolist())
    _require(len(hw) == 2, f"[tapvid_eval] bad video_hw in {npz_path}: {hw}")
    video_hw = (hw[0], hw[1])

    # canonical
    if "tracks_xy" in keys and "vis" in keys:
        tracks = np.asarray(arr["tracks_xy"], dtype=np.float32)
        vis = _as_bool(np.asarray(arr["vis"]))
        if "queries_tyx" in keys:
            q = np.asarray(arr["queries_tyx"], dtype=np.float32)
        elif "query_points" in keys:
            # accept legacy naming but treat as tyx
            q = np.asarray(arr["query_points"], dtype=np.float32)
        elif "queries_txy" in keys:
            # if someone saved txy, convert to tyx
            qt = np.asarray(arr["queries_txy"], dtype=np.float32)
            _require(qt.ndim == 2 and qt.shape[1] == 3, f"[tapvid_eval] queries_txy must be [Q,3], got {qt.shape}")
            q = qt.copy()
            q[:, 1] = qt[:, 2]  # y
            q[:, 2] = qt[:, 1]  # x
        else:
            raise KeyError(f"[tapvid_eval] {npz_path} missing queries_tyx/query_points")
        return tracks, vis, q, video_hw

    # legacy-A
    if "pred_tracks" in keys and "pred_occluded" in keys:
        pred_QT2 = np.asarray(arr["pred_tracks"], dtype=np.float32)           # [Q,T,2]
        occ_QT = _as_bool(np.asarray(arr["pred_occluded"]))                    # [Q,T]
        _require(pred_QT2.ndim == 3 and pred_QT2.shape[-1] == 2, f"[tapvid_eval] pred_tracks must be [Q,T,2], got {pred_QT2.shape}")
        _require(occ_QT.shape == pred_QT2.shape[:2], f"[tapvid_eval] pred_occluded must be [Q,T], got {occ_QT.shape}")

        tracks_TQ2 = np.transpose(pred_QT2, (1, 0, 2)).astype(np.float32)      # [T,Q,2]
        vis_TQ = (~np.transpose(occ_QT, (1, 0))).astype(bool)                  # [T,Q]

        if "query_points" in keys:
            q = np.asarray(arr["query_points"], dtype=np.float32)              # assume tyx
        elif "queries_tyx" in keys:
            q = np.asarray(arr["queries_tyx"], dtype=np.float32)
        else:
            raise KeyError(f"[tapvid_eval] {npz_path} missing query_points/queries_tyx")
        return tracks_TQ2, vis_TQ, q, video_hw

    raise KeyError(f"[tapvid_eval] unknown npz schema in {npz_path.name}, keys={sorted(keys)}")


def _load_video_from_frames(frame_paths: List[str]) -> np.ndarray:
    """
    Fallback when seq.video is not present.
    Requires opencv-python.
    """
    try:
        import cv2  # type: ignore
    except Exception as e:
        raise RuntimeError("OpenCV required to load frames: pip install opencv-python") from e

    _require(len(frame_paths) > 0, "[tapvid] cannot load video: empty frame_paths")
    frames = []
    for p in frame_paths:
        im = cv2.imread(str(p), cv2.IMREAD_COLOR)
        _require(im is not None, f"[tapvid] failed to read frame: {p}")
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        frames.append(im)
    v = np.stack(frames, axis=0).astype(np.uint8)
    return v


def _get_seq_video_uint8(seq: TapVidSeq) -> np.ndarray:
    """
    Prefer seq.video (from pkl). Fallback to reading seq.frame_paths.
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
    return _load_video_from_frames(list(seq.frame_paths))


def _extract_cotracker_output(out: Any) -> Tuple[np.ndarray, np.ndarray]:
    """
    Accept common outputs:
      - dict with keys {tracks_xy, occluded} or {tracks, occ}
      - dataclass/object with attributes tracks_xy/occluded
    Returns:
      tracks_xy: [Q,T,2] or [T,Q,2]
      occluded : [Q,T] or [T,Q]
    """
    if isinstance(out, dict):
        if "tracks_xy" in out:
            tracks = out["tracks_xy"]
        elif "tracks" in out:
            tracks = out["tracks"]
        else:
            raise KeyError("[cotracker_v2] output missing tracks_xy/tracks")

        if "occluded" in out:
            occ = out["occluded"]
        elif "occ" in out:
            occ = out["occ"]
        else:
            raise KeyError("[cotracker_v2] output missing occluded/occ")

        return np.asarray(tracks), _as_bool(np.asarray(occ))

    # object-like
    tracks = getattr(out, "tracks_xy", None)
    if tracks is None:
        tracks = getattr(out, "tracks", None)
    occ = getattr(out, "occluded", None)
    if occ is None:
        occ = getattr(out, "occ", None)

    if tracks is None or occ is None:
        raise KeyError("[cotracker_v2] unrecognized output type; need tracks_xy + occluded")

    return np.asarray(tracks), _as_bool(np.asarray(occ))


def _to_TQ(tracks: np.ndarray, occ_or_vis: np.ndarray, want_vis: bool) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize to:
      tracks_TQ2: [T,Q,2]
      vis_TQ    : [T,Q] bool
    input may be [Q,T,2] or [T,Q,2], and occ/vis aligned accordingly.
    """
    tracks = np.asarray(tracks, dtype=np.float32)
    _require(tracks.ndim == 3 and tracks.shape[-1] == 2, f"[tapvid] tracks must be rank-3 [...,2], got {tracks.shape}")
    a = np.asarray(occ_or_vis)
    _require(a.ndim == 2, f"[tapvid] occ/vis must be [?,?], got {a.shape}")
    _require(a.shape == tracks.shape[:2], f"[tapvid] occ/vis shape {a.shape} != tracks[:2] {tracks.shape[:2]}")

    # decide whether first dim is T or Q by matching with "second dim likely equals T?"
    # We cannot know T from outside here; caller should already ensure alignment.
    # Use heuristic: if a.shape[0] is "large" and a.shape[1] is "small" could be Q,T; but not reliable.
    # Instead: accept both; caller sets it by checking seq length later. We handle both by trying both.
    # Here we just allow both; caller can validate with T later.
    # We'll pick: if tracks.shape[0] == a.shape[0] and tracks.shape[1] == a.shape[1], no info.
    # Return as-is and let caller validate; but we also provide a transpose option.
    if want_vis:
        vis = _as_bool(a)
    else:
        vis = (~_as_bool(a)).astype(bool)  # if input is occluded

    # default interpret as [T,Q] already
    return tracks, vis


def _maybe_transpose_QT_to_TQ(tracks: np.ndarray, vis_or_occ: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    If tracks appears to be [Q,T,2], transpose to [T,Q,2].
    Heuristic: if second dim looks like time (>=10) and first dim looks like queries (<=200).
    It's still heuristic, but works for TAP-Vid DAVIS (T=~90, Q usually <= points count).
    """
    tracks = np.asarray(tracks, dtype=np.float32)
    a = np.asarray(vis_or_occ)
    if tracks.ndim != 3:
        return tracks, a
    Q, T = tracks.shape[0], tracks.shape[1]
    if (T >= 10) and (Q <= 512) and (tracks.shape[2] == 2):
        # assume [Q,T,2]
        tracks_TQ2 = np.transpose(tracks, (1, 0, 2))
        a_TQ = np.transpose(a, (1, 0))
        return tracks_TQ2, a_TQ
    return tracks, a


# ---------------------------
# main entrypoints
# ---------------------------
def run_tapvid_pred(cfg: RootConfig, config_path: str, config_sha1: str) -> Dict[str, Any]:
    if cfg.dataset != "tapvid" or cfg.tapvid is None:
        raise ValueError("[tapvid_pred] RootConfig must be dataset=tapvid")
    tcfg = cfg.tapvid

    tracker = str(tcfg.tracker).strip().lower()
    if tracker == "none":
        raise ValueError("[tapvid_pred] tracker=none means 'do not run prediction'. Use tracker=oracle/tapir/cotracker_v2.")

    if tracker not in ("oracle", "tapir", "cotracker_v2"):
        raise ValueError(f"[tapvid_pred] unsupported tracker={tracker!r}. Allowed: oracle/tapir/cotracker_v2.")

    pred_dir = Path(tcfg.pred_dir).expanduser() if tcfg.pred_dir else _default_pred_dir(cfg.base_out, tracker, tcfg.query_mode, tcfg.split)
    out_dir = Path(tcfg.out_dir).expanduser() if tcfg.out_dir else _default_out_dir(cfg.base_out, tracker, tcfg.query_mode, tcfg.split)
    _ensure_dir(pred_dir)
    _ensure_dir(out_dir)

    res = cfg.res or "480p"

    # IMPORTANT: build dataset in ORIGINAL coords; metrics will do resize_to_256 if requested.
    seqs = build_tapvid_dataset(
        davis_root=cfg.davis_root or "",
        pkl_path=tcfg.pkl_path,
        res=res,
        split=tcfg.split,
        query_mode=tcfg.query_mode,
        stride=tcfg.stride,
        max_queries=tcfg.max_queries,
        resize_to_256=False,              # <--- key: keep original for tracker correctness
        keep_aspect=tcfg.keep_aspect,
        seed=tcfg.seed,
    )
    if not seqs:
        raise ValueError("[tapvid_pred] No sequences produced.")

    ok, skipped = 0, 0
    failed: List[Dict[str, str]] = []

    # lazy imports to avoid hard deps
    tapir_tracker = None
    if tracker == "tapir":
        from pr2drag.trackers.tapir import TapirTracker, TapirTrackerConfig  # noqa
        tapir_tracker = TapirTracker(
            TapirTrackerConfig(
                resize_to_256=True,
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
            # queries
            q_tyx = np.asarray(s.query_points_tyx, dtype=np.float32)
            q_ids = np.asarray(s.query_track_ids, dtype=np.int64)
            _require(q_tyx.ndim == 2 and q_tyx.shape[1] == 3, f"[tapvid_pred] {s.name} bad queries_tyx {q_tyx.shape}")
            _require(q_ids.ndim == 1 and q_ids.shape[0] == q_tyx.shape[0], f"[tapvid_pred] {s.name} bad query_track_ids {q_ids.shape}")

            # GT: [N,T,2] + [N,T]
            gt_NT2 = np.asarray(s.gt_tracks_xy, dtype=np.float32)
            gt_occ_NT = _as_bool(np.asarray(s.gt_occluded))
            _require(gt_NT2.ndim == 3 and gt_NT2.shape[-1] == 2, f"[tapvid_pred] {s.name} bad gt_tracks {gt_NT2.shape}")
            _require(gt_occ_NT.shape == gt_NT2.shape[:2], f"[tapvid_pred] {s.name} bad gt_occluded {gt_occ_NT.shape}")

            N, T = gt_NT2.shape[0], gt_NT2.shape[1]
            _require(np.all((q_ids >= 0) & (q_ids < N)), f"[tapvid_pred] {s.name} query_track_ids out of range (N={N})")

            if tracker == "oracle":
                # gather queried tracks -> [Q,T,2] then transpose to [T,Q,2]
                gt_QT2 = gt_NT2[q_ids, :, :]                          # [Q,T,2]
                gt_occ_QT = gt_occ_NT[q_ids, :]                        # [Q,T]
                tracks_TQ2 = np.transpose(gt_QT2, (1, 0, 2)).astype(np.float32)
                vis_TQ = (~np.transpose(gt_occ_QT, (1, 0))).astype(bool)

                _save_pred_npz(npz_path, tracks_TQ2, vis_TQ, q_tyx, s.video_hw)
                ok += 1
                continue

            if tracker == "cotracker_v2":
                # requires video
                video = _get_seq_video_uint8(s)  # [T,H,W,3]
                _require(video.shape[0] == T, f"[tapvid_pred] {s.name} video T mismatch: video={video.shape[0]} gt={T}")

                from pr2drag.trackers.cotracker_v2 import run_cotracker_v2  # noqa

                out = run_cotracker_v2(
                    video_uint8=video,
                    queries_tyx=q_tyx.astype(np.float32),   # [Q,3] (t,y,x)
                    checkpoint=(tcfg.tracker_ckpt or None),
                    device=None,
                )
                tracks, occ = _extract_cotracker_output(out)  # could be [Q,T,2] or [T,Q,2]

                tracks, occ = _maybe_transpose_QT_to_TQ(tracks, occ)
                tracks_TQ2, vis_TQ = _to_TQ(tracks, occ, want_vis=False)  # occ->vis

                # validate T,Q
                _require(tracks_TQ2.shape[0] == T, f"[tapvid_pred] {s.name} CoTracker T mismatch: {tracks_TQ2.shape[0]} vs {T}")
                _require(tracks_TQ2.shape[1] == q_tyx.shape[0], f"[tapvid_pred] {s.name} CoTracker Q mismatch: {tracks_TQ2.shape[1]} vs {q_tyx.shape[0]}")

                _save_pred_npz(npz_path, tracks_TQ2, vis_TQ, q_tyx, s.video_hw)
                ok += 1
                continue

            if tracker == "tapir":
                _require(tapir_tracker is not None, "[tapvid_pred] internal error: tapir_tracker is None")
                # TapirTracker expects seq.video present; we ensure by attaching on the fly if needed
                if not isinstance(getattr(s, "video", None), np.ndarray):
                    video = _get_seq_video_uint8(s)
                    # TapVidSeq is frozen dataclass in your code; cannot set attribute.
                    # So we rely on datasets/tapvid.py to include video. If not, fail with clear msg:
                    raise RuntimeError(
                        "[tapvid_pred] tracker=tapir requires TapVidSeq.video to exist (pkl must include video and dataset must pass it through)."
                    )

                pred = tapir_tracker.predict(s)  # TapVidPred: tracks_xy [T,Q,2] in ORIGINAL coords, vis [T,Q]
                _save_pred_npz(npz_path, pred.tracks_xy, pred.vis, q_tyx, s.video_hw)
                ok += 1
                continue

            raise ValueError(f"[tapvid_pred] unknown tracker={tracker!r}")

        except Exception as e:
            if tcfg.fail_fast:
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
        "npz_contract_primary": {
            "tracks_xy": "float32 [T,Q,2] (x,y) original coords",
            "vis": "uint8 [T,Q] (1=visible,0=not)",
            "queries_tyx": "float32 [Q,3] (t,y,x) original coords",
            "video_hw": "int32 [2] (H,W) original",
        },
        "npz_contract_compat": [
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

    # eval: allow tracker=none, but then pred_dir must be given explicitly
    if tracker == "none":
        if not str(tcfg.pred_dir).strip():
            raise ValueError("[tapvid_eval] tracker=none requires explicit tapvid.pred_dir (cannot infer).")

    pred_dir = Path(tcfg.pred_dir).expanduser() if tcfg.pred_dir else _default_pred_dir(cfg.base_out, tracker, tcfg.query_mode, tcfg.split)
    out_dir = Path(tcfg.out_dir).expanduser() if tcfg.out_dir else _default_out_dir(cfg.base_out, tracker, tcfg.query_mode, tcfg.split)
    _ensure_dir(out_dir)

    res = cfg.res or "480p"

    # IMPORTANT: build dataset in ORIGINAL coords; metrics handles resize_to_256 option.
    seqs = build_tapvid_dataset(
        davis_root=cfg.davis_root or "",
        pkl_path=tcfg.pkl_path,
        res=res,
        split=tcfg.split,
        query_mode=tcfg.query_mode,
        stride=tcfg.stride,
        max_queries=tcfg.max_queries,
        resize_to_256=False,
        keep_aspect=tcfg.keep_aspect,
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
            if not bool(tcfg.allow_missing_preds) and tcfg.fail_fast:
                raise FileNotFoundError(f"[tapvid_eval] missing pred npz: {npz_path}")
            continue

        try:
            pred_tracks_TQ2, pred_vis_TQ, pred_q_tyx, pred_hw = _load_pred_npz_compat(npz_path)

            # dataset GT
            q_tyx = np.asarray(s.query_points_tyx, dtype=np.float32)
            q_ids = np.asarray(s.query_track_ids, dtype=np.int64)
            gt_NT2 = np.asarray(s.gt_tracks_xy, dtype=np.float32)
            gt_occ_NT = _as_bool(np.asarray(s.gt_occluded))
            N, T = gt_NT2.shape[0], gt_NT2.shape[1]

            _require(pred_tracks_TQ2.shape[0] == T, f"[tapvid_eval] {s.name} pred T mismatch: pred={pred_tracks_TQ2.shape[0]} gt={T}")
            _require(pred_tracks_TQ2.shape[1] == q_tyx.shape[0], f"[tapvid_eval] {s.name} pred Q mismatch: pred={pred_tracks_TQ2.shape[1]} gtQ={q_tyx.shape[0]}")
            _require(pred_vis_TQ.shape == pred_tracks_TQ2.shape[:2], f"[tapvid_eval] {s.name} pred vis shape mismatch")

            _require(np.all((q_ids >= 0) & (q_ids < N)), f"[tapvid_eval] {s.name} query ids out of range (N={N})")

            gt_QT2 = gt_NT2[q_ids, :, :]                           # [Q,T,2]
            gt_occ_QT = gt_occ_NT[q_ids, :]                         # [Q,T]
            gt_tracks_TQ2 = np.transpose(gt_QT2, (1, 0, 2)).astype(np.float32)
            gt_vis_TQ = (~np.transpose(gt_occ_QT, (1, 0))).astype(bool)

            # metrics_tapvid.py expects queries_txy (t,x,y)
            queries_txy = _tyx_to_txy(q_tyx)

            m: TapVidMetrics = compute_tapvid_metrics(
                gt_tracks_xy=gt_tracks_TQ2,            # [T,Q,2]
                gt_vis=gt_vis_TQ,                      # [T,Q]
                pred_tracks_xy=pred_tracks_TQ2,         # [T,Q,2]
                pred_vis=pred_vis_TQ.astype(bool),      # [T,Q]
                queries_txy=queries_txy,                # [Q,3] (t,x,y)
                resize_to_256=bool(tcfg.resize_to_256),
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
            if tcfg.fail_fast:
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