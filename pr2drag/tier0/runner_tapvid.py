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
from pr2drag.tier0.metrics_tapvid import compute_tapvid_metrics_mean  # 你已有/或你需要实现（见下）


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_pred_dir(base_out: str, tracker: str, query_mode: str, split: str) -> Path:
    return Path(base_out) / "tapvid_preds" / f"{tracker}_{query_mode}_{split}"


def _default_out_dir(base_out: str, tracker: str, query_mode: str, split: str) -> Path:
    return Path(base_out) / "tapvid_reports" / f"{tracker}_{query_mode}_{split}"


def _save_npz_contract(npz_path: Path, pred_tracks: np.ndarray, pred_occ: np.ndarray, query_points_txy: np.ndarray, video_hw: Tuple[int, int]) -> None:
    np.savez_compressed(
        npz_path,
        pred_tracks=pred_tracks.astype(np.float32),              # [Q,T,2]
        pred_occluded=pred_occ.astype(np.uint8),                 # [Q,T] 1=occ
        query_points=query_points_txy.astype(np.float32),        # [Q,3] (t,x,y)
        video_hw=np.asarray(video_hw, dtype=np.int32),           # [2]
    )


def run_tapvid_pred(cfg: RootConfig, config_path: str, config_sha1: str) -> Dict[str, Any]:
    if cfg.dataset != "tapvid" or cfg.tapvid is None:
        raise ValueError("[tapvid_pred] RootConfig must be dataset=tapvid")

    tcfg = cfg.tapvid
    base_out = Path(cfg.base_out)
    base_out.mkdir(parents=True, exist_ok=True)

    tracker = tcfg.tracker
    if tracker == "none":
        raise ValueError("[tapvid_pred] tracker=none means 'do not run prediction'. Choose oracle/cotracker_v2/tapir.")

    pred_dir = Path(tcfg.pred_dir).expanduser() if tcfg.pred_dir else _default_pred_dir(cfg.base_out, tracker, tcfg.query_mode, tcfg.split)
    pred_dir.mkdir(parents=True, exist_ok=True)

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
        raise ValueError("[tapvid_pred] No sequences produced (maybe all queries invalid).")

    ok, skipped = 0, 0
    failed: List[Dict[str, str]] = []

    for s in seqs:
        npz_path = pred_dir / f"{s.name}.npz"
        if npz_path.exists() and (not bool(tcfg.overwrite_preds)):
            skipped += 1
            continue

        try:
            if tracker == "oracle":
                # oracle: directly use GT
                # NOTE: your TapVidSeq is [T,N,2] + queries [Q,3] (t,x,y) and query_track_ids [Q]
                ids = s.query_track_ids
                pred_tracks = np.transpose(s.gt_tracks_xy[:, ids, :], (1, 0, 2))  # [Q,T,2]
                pred_occ = np.transpose(~s.gt_vis[:, ids], (1, 0))                # [Q,T]
                _save_npz_contract(npz_path, pred_tracks, pred_occ, s.queries_txy, s.video_hw)
                ok += 1
                continue

            if tracker == "cotracker_v2":
                # IMPORTANT: need video frames; we rely on TapVidSeq.video (or load via frame_paths)
                video = s.video  # [T,H,W,3] uint8 (你 datasets/tapvid.py 里要保证有这个字段；如果没有就按 frame_paths 读)
                from pr2drag.trackers.cotracker_v2 import run_cotracker_v2

                out = run_cotracker_v2(
                    video_uint8=video,
                    queries_txy=s.queries_txy.astype(np.float32),
                    checkpoint=(tcfg.tracker_ckpt or None),
                    device=None,
                )
                pred_tracks = out.tracks_xy
                pred_occ = out.occluded

                _save_npz_contract(npz_path, pred_tracks, pred_occ, s.queries_txy, s.video_hw)
                ok += 1
                continue

            if tracker == "tapir":
                raise NotImplementedError("[tapvid_pred] tapir not wired yet (next knife).")

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
            else:
                continue

        try:
            arr = np.load(npz_path, allow_pickle=False)
            if "pred_tracks" not in arr or "pred_occluded" not in arr:
                raise KeyError(f"[tapvid_eval] {npz_path} missing pred_tracks/pred_occluded")

            pred_tracks = np.asarray(arr["pred_tracks"], dtype=np.float32)      # [Q,T,2]
            pred_occ = np.asarray(arr["pred_occluded"])
            pred_occ = (pred_occ.astype(np.int32) != 0)                         # bool [Q,T]

            # GT: you have [T,N,2] and ids [Q]
            ids = s.query_track_ids
            gt_tracks = np.transpose(s.gt_tracks_xy[:, ids, :], (1, 0, 2)).astype(np.float32)   # [Q,T,2]
            gt_occ = np.transpose(~s.gt_vis[:, ids], (1, 0)).astype(bool)                         # [Q,T]

            if pred_tracks.shape != gt_tracks.shape:
                raise ValueError(f"[tapvid_eval] {s.name} pred_tracks {pred_tracks.shape} != gt_tracks {gt_tracks.shape}")
            if pred_occ.shape != gt_occ.shape:
                raise ValueError(f"[tapvid_eval] {s.name} pred_occ {pred_occ.shape} != gt_occ {gt_occ.shape}")

            # metrics (你 metrics_tapvid.py 里实现 mean 版即可)
            m = compute_tapvid_metrics_mean(
                queries_txy=s.queries_txy.astype(np.float32),  # [Q,3] (t,x,y)
                gt_tracks_xy=gt_tracks,                        # [Q,T,2]
                gt_occluded=gt_occ,                            # [Q,T]
                pred_tracks_xy=pred_tracks,                    # [Q,T,2]
                pred_occluded=pred_occ,                        # [Q,T]
                query_mode=tcfg.query_mode,
            )
            m_row = {"video_id": s.name, **m}
            per_video.append(m_row)
            evaluated += 1

        except Exception as e:
            failed += 1
            if tcfg.fail_fast:
                raise

    # aggregate
    mean = {}
    if per_video:
        keys = [k for k in per_video[0].keys() if k != "video_id"]
        for k in keys:
            mean[k] = float(np.mean([float(r[k]) for r in per_video]))

    (out_dir / "tapvid_metrics_per_video.json").write_text(json.dumps(per_video, indent=2, sort_keys=True))
    (out_dir / "tapvid_metrics_mean.json").write_text(json.dumps(mean, indent=2, sort_keys=True))

    print(f"[tapvid_eval] pred_dir: {pred_dir}")
    print(f"[tapvid_eval] out_dir : {out_dir}")
    print(f"[tapvid_eval] evaluated={evaluated} missing={missing} failed={failed}")
    print("[tapvid_eval] mean:")
    for k in sorted(mean.keys()):
        print(f"  {k:>24s}: {mean[k]*100.0:7.3f}")

    return {
        "created_utc": _now_iso(),
        "cmd": "tapvid_eval",
        "dataset": "tapvid",
        "config_path": config_path,
        "config_sha1": config_sha1,
        "pred_dir": str(pred_dir),
        "out_dir": str(out_dir),
        "evaluated": evaluated,
        "missing": missing,
        "failed": failed,
        "mean": mean,
    }