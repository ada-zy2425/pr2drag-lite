# pr2drag/tier0/runner_tapvid.py
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from pr2drag.trackers.cotracker_v2 import run_cotracker_v2
from pr2drag.tier1.contracts import RootConfig
from pr2drag.datasets.tapvid import build_tapvid_dataset, TapVidSeq
from pr2drag.tier0.metrics_tapvid import compute_tapvid_metrics, TapVidMetrics


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_pred_dir(base_out: str, tracker: str, query_mode: str, split: str) -> Path:
    return Path(base_out) / "tapvid_preds" / f"{tracker}_{query_mode}_{split}"


def _default_out_dir(base_out: str, tracker: str, query_mode: str, split: str) -> Path:
    return Path(base_out) / "tapvid_reports" / f"{tracker}_{query_mode}_{split}"


def _save_npz_contract(
    npz_path: Path,
    pred_tracks: np.ndarray,       # [Q,T,2] float32
    pred_occ: np.ndarray,          # [Q,T] bool or {0,1}
    query_points_txy: np.ndarray,  # [Q,3] (t,x,y) float32
    video_hw: Tuple[int, int],     # (H,W)
) -> None:
    pred_tracks = np.asarray(pred_tracks, dtype=np.float32)
    pred_occ = (np.asarray(pred_occ).astype(np.int32) != 0)

    if pred_tracks.ndim != 3 or pred_tracks.shape[-1] != 2:
        raise ValueError(f"[tapvid_npz] pred_tracks must be [Q,T,2], got {pred_tracks.shape}")
    if pred_occ.ndim != 2 or pred_occ.shape != pred_tracks.shape[:2]:
        raise ValueError(f"[tapvid_npz] pred_occluded must be [Q,T], got {pred_occ.shape} vs {pred_tracks.shape[:2]}")
    if np.asarray(query_points_txy).ndim != 2 or np.asarray(query_points_txy).shape[1] != 3:
        raise ValueError(f"[tapvid_npz] query_points must be [Q,3], got {np.asarray(query_points_txy).shape}")

    np.savez_compressed(
        npz_path,
        pred_tracks=pred_tracks.astype(np.float32),          # [Q,T,2]
        pred_occluded=pred_occ.astype(np.uint8),             # [Q,T] 1=occ
        query_points=np.asarray(query_points_txy, dtype=np.float32),  # [Q,3]
        video_hw=np.asarray(video_hw, dtype=np.int32),       # [2] (H,W)
    )


def _load_pred_npz(npz_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, int]]:
    arr = np.load(npz_path, allow_pickle=False)
    if "pred_tracks" not in arr or "pred_occluded" not in arr:
        raise KeyError(f"[tapvid_eval] {npz_path} missing pred_tracks/pred_occluded")

    pred_tracks = np.asarray(arr["pred_tracks"], dtype=np.float32)
    pred_occ = (np.asarray(arr["pred_occluded"]).astype(np.int32) != 0)

    qpts = np.asarray(arr["query_points"], dtype=np.float32) if "query_points" in arr else None
    if qpts is None:
        raise KeyError(f"[tapvid_eval] {npz_path} missing query_points")

    hw = arr["video_hw"] if "video_hw" in arr else None
    if hw is None:
        raise KeyError(f"[tapvid_eval] {npz_path} missing video_hw")
    hw = tuple(int(x) for x in np.asarray(hw).reshape(-1).tolist())
    if len(hw) != 2:
        raise ValueError(f"[tapvid_eval] bad video_hw in {npz_path}: {hw}")

    return pred_tracks, pred_occ, qpts, (hw[0], hw[1])


def run_tapvid_pred(cfg: RootConfig, config_path: str, config_sha1: str) -> Dict[str, Any]:
    if cfg.dataset != "tapvid" or cfg.tapvid is None:
        raise ValueError("[tapvid_pred] RootConfig must be dataset=tapvid")
    tcfg = cfg.tapvid

    tracker = tcfg.tracker
    if tracker == "none":
        raise ValueError("[tapvid_pred] tracker=none means 'do not run prediction'. Use tracker=oracle/tapir/cotracker_v2.")

    pred_dir = Path(tcfg.pred_dir).expanduser() if tcfg.pred_dir else _default_pred_dir(cfg.base_out, tracker, tcfg.query_mode, tcfg.split)
    pred_dir.mkdir(parents=True, exist_ok=True)

    # out_dir here is optional (for logs); keep symmetry
    out_dir = Path(tcfg.out_dir).expanduser() if tcfg.out_dir else _default_out_dir(cfg.base_out, tracker, tcfg.query_mode, tcfg.split)
    out_dir.mkdir(parents=True, exist_ok=True)

    res = cfg.res or "480p"
    seqs = build_tapvid_dataset(
        davis_root=cfg.davis_root or "",
        pkl_path=tcfg.pkl_path,
        split=tcfg.split,
        query_mode=tcfg.query_mode,
        stride=tcfg.stride,
        max_queries=tcfg.max_queries,
        resize_to_256=tcfg.resize_to_256,
        keep_aspect=tcfg.keep_aspect,
        seed=tcfg.seed,
    )
    if not seqs:
        raise ValueError("[tapvid_pred] No sequences produced.")

    ok, skipped = 0, 0
    failed: List[Dict[str, str]] = []

    for s in seqs:
        npz_path = pred_dir / f"{s.name}.npz"
        if npz_path.exists() and (not bool(tcfg.overwrite_preds)):
            skipped += 1
            continue

        try:
            if tracker == "oracle":
                # TapVidSeq contract: gt_tracks_xy [T,N,2], gt_vis [T,N], queries_txy [Q,3] (t,x,y), query_track_ids [Q]
                ids = np.asarray(s.query_track_ids, dtype=np.int64)
                if ids.ndim != 1:
                    raise ValueError(f"[tapvid_pred] {s.name} query_track_ids must be [Q], got {ids.shape}")
                # gather tracks/vis for the queried track IDs
                gt_tracks_QT2 = np.transpose(s.gt_tracks_xy[:, ids, :], (1, 0, 2)).astype(np.float32)  # [Q,T,2]
                gt_occ_QT = np.transpose(~s.gt_vis[:, ids], (1, 0)).astype(bool)                          # [Q,T]
                _save_npz_contract(npz_path, gt_tracks_QT2, gt_occ_QT, s.queries_txy, s.video_hw)
                ok += 1
                continue

            if tracker == "cotracker_v2":
                out = run_cotracker_v2(
                    video_uint8=s.video,                          # [T,H,W,3] uint8, 需确保 dataset 填了 video
                    queries_tyx=s.query_points_tyx.astype(np.float32),  # [Q,3] (t,y,x)
                    checkpoint=(tcfg.tracker_ckpt or None),
                    device=None,
                )
                tracks_TQ2 = np.transpose(out.tracks_xy, (1, 0, 2))     # out.tracks_xy 是 [Q,T,2] -> [T,Q,2]
                vis_TQ = np.transpose(~out.occluded, (1, 0))            # [Q,T] -> [T,Q]

                np.savez_compressed(
                    npz_path,
                    tracks_xy=tracks_TQ2.astype(np.float32),
                    vis=vis_TQ.astype(np.uint8),            # 用 0/1 更稳
                    queries_tyx=s.query_points_tyx.astype(np.float32),
                    video_hw=np.asarray(s.video_hw, dtype=np.int32),
                )
            if tracker == "tapir":
                raise NotImplementedError("[tapvid_pred] tapir wrapper not implemented in this knife.")

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
        "pred_dir": str(pred_dir),
        "out_dir": str(out_dir),
        "num_seqs_total": int(len(seqs)),
        "num_ok": int(ok),
        "num_skipped": int(skipped),
        "num_failed": int(len(failed)),
        "failed": failed,
        "npz_contract": {
            "pred_tracks": "float32 [Q,T,2] (x,y)",
            "pred_occluded": "uint8 [Q,T] (1=occluded,0=visible)",
            "query_points": "float32 [Q,3] (t,x,y)",
            "video_hw": "int32 [2] (H,W)",
        },
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

    tracker = tcfg.tracker
    pred_dir = Path(tcfg.pred_dir).expanduser() if tcfg.pred_dir else _default_pred_dir(cfg.base_out, tracker, tcfg.query_mode, tcfg.split)
    out_dir = Path(tcfg.out_dir).expanduser() if tcfg.out_dir else _default_out_dir(cfg.base_out, tracker, tcfg.query_mode, tcfg.split)
    out_dir.mkdir(parents=True, exist_ok=True)

    res = cfg.res or "480p"
    seqs = build_tapvid_dataset(
        davis_root=cfg.davis_root or "",
        pkl_path=tcfg.pkl_path,
        split=tcfg.split,
        query_mode=tcfg.query_mode,
        stride=tcfg.stride,
        max_queries=tcfg.max_queries,
        resize_to_256=tcfg.resize_to_256,
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
            if not bool(tcfg.allow_missing_preds):
                if tcfg.fail_fast:
                    raise FileNotFoundError(f"[tapvid_eval] missing pred npz: {npz_path}")
            continue

        try:
            pred_tracks_QT2, pred_occ_QT, qpts_Q3, hw = _load_pred_npz(npz_path)

            # sanity: Q must match for strict evaluation
            if qpts_Q3.shape[0] != s.queries_txy.shape[0]:
                raise ValueError(f"[tapvid_eval] {s.name} Q mismatch: npz={qpts_Q3.shape[0]} dataset={s.queries_txy.shape[0]}")

            # GT for the same queried track ids
            ids = np.asarray(s.query_track_ids, dtype=np.int64)
            gt_tracks_QT2 = np.transpose(s.gt_tracks_xy[:, ids, :], (1, 0, 2)).astype(np.float32)  # [Q,T,2]
            gt_vis_QT = np.transpose(s.gt_vis[:, ids], (1, 0)).astype(bool)                           # [Q,T]
            pred_vis_QT = (~pred_occ_QT).astype(bool)

            # metrics (returns TapVidMetrics)
            m: TapVidMetrics = compute_tapvid_metrics(
                gt_tracks_xy=gt_tracks_QT2,
                gt_vis=gt_vis_QT,
                pred_tracks_xy=pred_tracks_QT2,
                pred_vis=pred_vis_QT,
                queries_txy=s.queries_txy.astype(np.float32),
                query_mode=tcfg.query_mode,          # "tapvid" or "strided"
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
            # flatten per-threshold
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
            # log failure row for audit
            per_video.append({"video_id": s.name, "error": repr(e)})

    # aggregate mean on numeric fields only
    mean: Dict[str, float] = {}
    if per_video:
        keys = set()
        for r in per_video:
            for k, v in r.items():
                if k == "video_id" or k == "error":
                    continue
                if isinstance(v, (int, float)) and v is not None:
                    keys.add(k)
        for k in sorted(keys):
            vals = [float(r[k]) for r in per_video if k in r and isinstance(r[k], (int, float))]
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
        # AJ/OA/Jaccard/PCK are fractions -> print in %
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
        "pred_dir": str(pred_dir),
        "out_dir": str(out_dir),
        "evaluated": int(evaluated),
        "missing": int(missing),
        "failed": int(failed),
        "mean": mean,
    }