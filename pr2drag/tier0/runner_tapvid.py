# pr2drag/tier0/runner_tapvid.py
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np

from pr2drag.tier1.contracts import RootConfig
from ..datasets.tapvid import build_tapvid_dataset, load_tapvid_pkl, TapVidSeq

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_pred_dir(base_out: str, tracker: str, query_mode: str, split: str) -> Path:
    return Path(base_out) / "tapvid_preds" / f"{tracker}_{query_mode}_{split}"


def _default_out_dir(base_out: str, tracker: str, query_mode: str, split: str) -> Path:
    return Path(base_out) / "tapvid_reports" / f"{tracker}_{query_mode}_{split}"


def run_tapvid_pred(cfg: RootConfig, config_path: str, config_sha1: str) -> Dict[str, Any]:
    if cfg.dataset != "tapvid" or cfg.tapvid is None:
        raise ValueError("[tapvid_pred] RootConfig must be dataset=tapvid")

    tcfg = cfg.tapvid
    base_out = Path(cfg.base_out)
    base_out.mkdir(parents=True, exist_ok=True)

    pred_dir = Path(tcfg.pred_dir).expanduser() if tcfg.pred_dir else _default_pred_dir(cfg.base_out, tcfg.tracker, tcfg.query_mode, tcfg.split)
    pred_dir.mkdir(parents=True, exist_ok=True)

    out_dir = Path(tcfg.out_dir).expanduser() if tcfg.out_dir else _default_out_dir(cfg.base_out, tcfg.tracker, tcfg.query_mode, tcfg.split)
    out_dir.mkdir(parents=True, exist_ok=True)

    # We keep res default "480p" for DAVIS frames unless cfg.res provided
    res = cfg.res or "480p"

    # Load sequences (this also validates davis_root/res/T matches)
    seqs = build_tapvid_dataset(
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
        raise ValueError("[tapvid_pred] No sequences produced (maybe all queries invalid).")

    # tracker handling (first knife: oracle only)
    if tcfg.tracker != "none":
        raise NotImplementedError(
            f"[tapvid_pred] tracker={tcfg.tracker!r} not implemented yet. "
            f"First knife supports tracker='none' (oracle/GT) to validate pred_dir plumbing."
        )

    ok, skipped = 0, 0
    failed: list[dict[str, str]] = []

    for s in seqs:
        npz_path = pred_dir / f"{s.name}.npz"
        if npz_path.exists() and not bool(tcfg.overwrite_preds):
            skipped += 1
            continue

        try:
            # materialize oracle predictions for the chosen queries
            # each query refers to a track id in [0,N)
            ids = s.query_track_ids
            pred_tracks = s.gt_tracks_xy[ids, :, :]     # [Q,T,2]
            pred_occ = s.gt_occluded[ids, :]            # [Q,T]

            # Write minimal, strict contract
            np.savez_compressed(
                npz_path,
                pred_tracks=pred_tracks.astype(np.float32),
                pred_occluded=pred_occ.astype(np.uint8),   # store as 0/1 to be safe
                query_points=s.query_points_tyx.astype(np.float32),
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
        "out_dir": str(out_dir),
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