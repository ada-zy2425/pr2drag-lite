# pr2drag/tier0/runner_tapvid.py
from __future__ import annotations

import csv
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from pr2drag.tier1.contracts import RootConfig
from pr2drag.datasets.tapvid import build_tapvid_dataset, TapVidSeq


# -------------------------
# utils
# -------------------------
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


def _load_npz_pred(npz_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Contract (strict):
      pred_tracks   float32 [Q,T,2] (x,y)
      pred_occluded uint8/bool [Q,T] (1=occluded,0=visible)
      query_points  float32 [Q,3] (t,y,x)
    """
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing prediction npz: {npz_path}")
    arr = np.load(npz_path, allow_pickle=False)
    for k in ("pred_tracks", "pred_occluded", "query_points"):
        if k not in arr:
            raise KeyError(f"{npz_path} missing key {k!r}. Required: pred_tracks, pred_occluded, query_points")

    pred_tracks = np.asarray(arr["pred_tracks"], dtype=np.float32)
    pred_occ = _as_bool(np.asarray(arr["pred_occluded"]))
    q_tyx = np.asarray(arr["query_points"], dtype=np.float32)

    if pred_tracks.ndim != 3 or pred_tracks.shape[-1] != 2:
        raise ValueError(f"{npz_path} pred_tracks must be [Q,T,2], got {pred_tracks.shape}")
    if pred_occ.shape != pred_tracks.shape[:2]:
        raise ValueError(f"{npz_path} pred_occluded shape {pred_occ.shape} != (Q,T) {pred_tracks.shape[:2]}")
    if q_tyx.ndim != 2 or q_tyx.shape[1] != 3 or q_tyx.shape[0] != pred_tracks.shape[0]:
        raise ValueError(f"{npz_path} query_points must be [Q,3] and Q must match pred_tracks. "
                         f"got q={q_tyx.shape}, pred_tracks={pred_tracks.shape}")

    return pred_tracks, pred_occ, q_tyx


# -------------------------
# metrics (solid but not overkill)
# -------------------------
def _eval_mask_from_query_points(q_tyx: np.ndarray, T: int, query_mode: str) -> np.ndarray:
    """
    Returns mask [Q,T] where True means evaluate this frame.

    - query_mode == "first": evaluate frames strictly AFTER query frame
    - query_mode in {"tapvid","strided"}: evaluate ALL frames EXCEPT query frame
    """
    qf = np.round(q_tyx[:, 0]).astype(np.int32)
    if (qf < 0).any() or (qf >= T).any():
        raise ValueError(f"query frame out of range [0,{T-1}] from query_points_tyx")

    mask = np.ones((q_tyx.shape[0], T), dtype=np.bool_)
    if query_mode == "first":
        for i, t0 in enumerate(qf):
            mask[i, : t0 + 1] = False
    else:
        # tapvid/strided: exclude query frame only
        mask[np.arange(q_tyx.shape[0]), qf] = False
    return mask


def _compute_metrics(
    gt_tracks: np.ndarray,      # [Q,T,2]
    gt_occ: np.ndarray,         # [Q,T] bool (True=occluded)
    pred_tracks: np.ndarray,    # [Q,T,2]
    pred_occ: np.ndarray,       # [Q,T] bool
    q_tyx: np.ndarray,          # [Q,3]
    video_hw: Tuple[int, int],  # (H,W) original
    resize_to_256: bool,
    query_mode: str,
) -> Dict[str, float]:
    """
    Solid, TAP-Vid-like metrics:
      - occlusion_accuracy
      - pts_within_{1,2,4,8,16}
      - jaccard_{1,2,4,8,16}
      - average_pts_within_thresh
      - average_jaccard
    Thresholds are defined in 256x256 space. If resize_to_256 is False, we scale thresholds to original space.
    """
    Q, T, _ = gt_tracks.shape
    if pred_tracks.shape != gt_tracks.shape:
        raise ValueError(f"pred_tracks {pred_tracks.shape} != gt_tracks {gt_tracks.shape}")
    if pred_occ.shape != gt_occ.shape:
        raise ValueError(f"pred_occluded {pred_occ.shape} != gt_occluded {gt_occ.shape}")

    eval_mask = _eval_mask_from_query_points(q_tyx, T=T, query_mode=query_mode)

    # threshold scaling
    if resize_to_256:
        thr_scale = 1.0
    else:
        H, W = int(video_hw[0]), int(video_hw[1])
        # scale pixel thresholds from 256-space to original-space (anisotropic -> use mean scale)
        thr_scale = 0.5 * ((W / 256.0) + (H / 256.0))

    # occlusion accuracy on eval frames
    denom = np.maximum(eval_mask.sum(), 1)
    occ_acc = float(((pred_occ == gt_occ) & eval_mask).sum() / denom)

    visible = ~gt_occ
    pred_visible = ~pred_occ

    metrics: Dict[str, float] = {"occlusion_accuracy": occ_acc}
    pts_within_list: List[float] = []
    jacc_list: List[float] = []

    # squared distances
    d2 = np.sum((pred_tracks - gt_tracks) ** 2, axis=-1)  # [Q,T]

    for thr in (1, 2, 4, 8, 16):
        thr_eff = float(thr) * thr_scale
        within = d2 < (thr_eff ** 2)

        # PtsWithin: among GT-visible eval frames
        vis_eval = visible & eval_mask
        vis_cnt = vis_eval.sum()
        if vis_cnt == 0:
            pts_within = 0.0
        else:
            pts_within = float((within & vis_eval).sum() / vis_cnt)

        # Jaccard: correct visible pred vs GT-visible, penalize false visible
        correct_visible = within & visible
        tp = (correct_visible & pred_visible & eval_mask).sum()
        gt_pos = (visible & eval_mask).sum()

        # false positives: predict visible where GT is occluded OR far from GT
        fp = (((~visible) & pred_visible) | ((~within) & pred_visible)) & eval_mask
        fp = fp.sum()
        denom_j = gt_pos + fp
        jacc = float(tp / denom_j) if denom_j > 0 else 0.0

        metrics[f"pts_within_{thr}"] = float(pts_within)
        metrics[f"jaccard_{thr}"] = float(jacc)
        pts_within_list.append(float(pts_within))
        jacc_list.append(float(jacc))

    metrics["average_pts_within_thresh"] = float(np.mean(pts_within_list)) if pts_within_list else 0.0
    metrics["average_jaccard"] = float(np.mean(jacc_list)) if jacc_list else 0.0
    return metrics


# -------------------------
# Tier0 entrypoints
# -------------------------
def run_tapvid_pred(cfg: RootConfig, config_path: str, config_sha1: str) -> Dict[str, Any]:
    if cfg.dataset != "tapvid" or cfg.tapvid is None:
        raise ValueError("[tapvid_pred] RootConfig must be dataset=tapvid")

    tcfg = cfg.tapvid
    base_out = Path(cfg.base_out)
    base_out.mkdir(parents=True, exist_ok=True)

    pred_dir = Path(tcfg.pred_dir).expanduser() if tcfg.pred_dir else _default_pred_dir(
        cfg.base_out, tcfg.tracker, tcfg.query_mode, tcfg.split
    )
    pred_dir.mkdir(parents=True, exist_ok=True)

    # We keep res default "480p" for DAVIS frames unless cfg.res provided
    res = cfg.res or "480p"

    # Load sequences (validates davis_root/res/T matches)
    seqs: List[TapVidSeq] = build_tapvid_dataset(
        davis_root=cfg.davis_root or "",
        pkl_path=tcfg.pkl_path,
        res=res,
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

    # First knife: oracle only (GT preds to validate plumbing)
    if tcfg.tracker not in ("oracle",):
        raise NotImplementedError(
            f"[tapvid_pred] tracker={tcfg.tracker!r} not implemented yet. "
            f"First knife only supports tracker='oracle' to validate pred_dir plumbing."
        )

    ok, skipped = 0, 0
    failed: List[Dict[str, str]] = []

    for s in seqs:
        npz_path = pred_dir / f"{s.name}.npz"
        if npz_path.exists() and not bool(tcfg.overwrite_preds):
            skipped += 1
            continue

        try:
            ids = np.asarray(s.query_track_ids, dtype=np.int64)  # [Q]
            if ids.ndim != 1 or ids.size == 0:
                raise ValueError("query_track_ids must be [Q] with Q>0")
            if ids.min() < 0 or ids.max() >= s.gt_tracks_xy.shape[0]:
                raise ValueError(f"query_track_ids out of range: min={ids.min()} max={ids.max()} "
                                 f"num_tracks={s.gt_tracks_xy.shape[0]}")

            pred_tracks = s.gt_tracks_xy[ids, :, :]   # [Q,T,2]
            pred_occ = s.gt_occluded[ids, :]          # [Q,T]
            q_tyx = np.asarray(s.query_points_tyx, dtype=np.float32)  # [Q,3]

            np.savez_compressed(
                npz_path,
                pred_tracks=pred_tracks.astype(np.float32),
                pred_occluded=pred_occ.astype(np.uint8),  # store 0/1
                query_points=q_tyx.astype(np.float32),
                video_hw=np.asarray(s.video_hw, dtype=np.int32),
            )
            ok += 1
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
        "num_seqs_total": int(len(seqs)),
        "num_ok": int(ok),
        "num_skipped": int(skipped),
        "num_failed": int(len(failed)),
        "failed": failed,
        "npz_contract": {
            "pred_tracks": "float32 [Q,T,2] (x,y)",
            "pred_occluded": "uint8 [Q,T] (1=occluded,0=visible)",
            "query_points": "float32 [Q,3] (t,y,x)",
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
    base_out = Path(cfg.base_out)
    base_out.mkdir(parents=True, exist_ok=True)

    pred_dir = Path(tcfg.pred_dir).expanduser() if tcfg.pred_dir else _default_pred_dir(
        cfg.base_out, tcfg.tracker, tcfg.query_mode, tcfg.split
    )
    out_dir = Path(tcfg.out_dir).expanduser() if tcfg.out_dir else _default_out_dir(
        cfg.base_out, tcfg.tracker, tcfg.query_mode, tcfg.split
    )
    pred_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    res = cfg.res or "480p"

    seqs: List[TapVidSeq] = build_tapvid_dataset(
        davis_root=cfg.davis_root or "",
        pkl_path=tcfg.pkl_path,
        res=res,
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

    per_video: List[Dict[str, Any]] = []
    missing: List[str] = []
    failed: List[Dict[str, str]] = []

    # aggregate
    agg: Dict[str, List[float]] = {}

    for s in seqs:
        npz_path = pred_dir / f"{s.name}.npz"
        if not npz_path.exists():
            missing.append(s.name)
            if not bool(tcfg.allow_missing_preds):
                if tcfg.fail_fast:
                    raise FileNotFoundError(f"[tapvid_eval] missing pred npz: {npz_path}")
            continue

        try:
            pred_tracks, pred_occ, q_tyx = _load_npz_pred(npz_path)

            ids = np.asarray(s.query_track_ids, dtype=np.int64)
            gt_tracks = s.gt_tracks_xy[ids, :, :].astype(np.float32)   # [Q,T,2]
            gt_occ = _as_bool(s.gt_occluded[ids, :])                   # [Q,T]

            if pred_tracks.shape != gt_tracks.shape:
                raise ValueError(f"pred_tracks {pred_tracks.shape} != gt_tracks {gt_tracks.shape}")
            if pred_occ.shape != gt_occ.shape:
                raise ValueError(f"pred_occluded {pred_occ.shape} != gt_occluded {gt_occ.shape}")

            m = _compute_metrics(
                gt_tracks=gt_tracks,
                gt_occ=gt_occ,
                pred_tracks=pred_tracks.astype(np.float32),
                pred_occ=_as_bool(pred_occ),
                q_tyx=q_tyx.astype(np.float32),
                video_hw=s.video_hw,
                resize_to_256=bool(tcfg.resize_to_256),
                query_mode=tcfg.query_mode,
            )

            row = {"video_id": s.name, "Q": int(gt_tracks.shape[0]), "T": int(gt_tracks.shape[1])}
            row.update({k: float(v) for k, v in m.items()})
            per_video.append(row)
            for k, v in m.items():
                agg.setdefault(k, []).append(float(v))
        except Exception as e:
            if tcfg.fail_fast:
                raise
            failed.append({"video_id": s.name, "error": repr(e)})

    if (missing and (not bool(tcfg.allow_missing_preds))) and (len(per_video) == 0):
        raise RuntimeError(
            f"[tapvid_eval] No videos evaluated and missing preds exist (allow_missing_preds=false). "
            f"First missing: {missing[0]}"
        )

    mean = {k: float(np.mean(v)) for k, v in agg.items()} if agg else {}

    # write outputs
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "_analysis").mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "tapvid_metrics_per_video.csv"
    if per_video:
        keys = ["video_id", "Q", "T"] + sorted([k for k in per_video[0].keys() if k not in ("video_id", "Q", "T")])
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in per_video:
                w.writerow({k: r.get(k, "") for k in keys})

    (out_dir / "tapvid_metrics_mean.json").write_text(json.dumps(mean, indent=2, sort_keys=True))

    report = {
        "created_utc": _now_iso(),
        "cmd": "tapvid_eval",
        "dataset": "tapvid",
        "config_path": config_path,
        "config_sha1": config_sha1,
        "davis_root": cfg.davis_root,
        "res": res,
        "tapvid": asdict(tcfg),
        "pred_dir": str(pred_dir),
        "out_dir": str(out_dir),
        "num_seqs_total": int(len(seqs)),
        "num_evaluated": int(len(per_video)),
        "num_missing": int(len(missing)),
        "num_failed": int(len(failed)),
        "missing": missing[:50],  # avoid huge logs
        "failed": failed,
        "mean": mean,
    }
    (out_dir / "manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True))

    print(f"[tapvid_eval] pred_dir: {pred_dir}")
    print(f"[tapvid_eval] out_dir : {out_dir}")
    print(f"[tapvid_eval] evaluated={len(per_video)} missing={len(missing)} failed={len(failed)}")
    if mean:
        print("[tapvid_eval] mean:")
        for k in sorted(mean.keys()):
            print(f"  {k:>24s}: {mean[k]*100.0:7.3f}")

    return report