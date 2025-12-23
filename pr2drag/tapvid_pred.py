# pr2drag/tapvid_pred.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np

from pr2drag.tier1.contracts import RootConfig
from pr2drag.datasets.tapvid import build_tapvid_dataset


def _safe_mkdir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _write_pred_npz(path: Path, tracks_xy: np.ndarray, vis: np.ndarray, queries_txy: np.ndarray) -> None:
    if tracks_xy.ndim != 3 or tracks_xy.shape[-1] != 2:
        raise ValueError(f"[tapvid_pred] tracks_xy must be [T,Q,2], got {tracks_xy.shape}")
    if vis.ndim != 2 or vis.shape[:2] != tracks_xy.shape[:2]:
        raise ValueError(f"[tapvid_pred] vis must be [T,Q], got {vis.shape} vs {tracks_xy.shape[:2]}")
    if queries_txy.ndim != 2 or queries_txy.shape[1] != 3 or queries_txy.shape[0] != tracks_xy.shape[1]:
        raise ValueError(f"[tapvid_pred] queries_txy must be [Q,3], got {queries_txy.shape}")

    if not np.isfinite(tracks_xy).all():
        raise ValueError("[tapvid_pred] tracks_xy contains NaN/Inf")

    np.savez_compressed(str(path), tracks_xy=tracks_xy.astype(np.float32), vis=vis.astype(bool), queries_txy=queries_txy.astype(np.float32))


def tapvid_pred_from_config(cfg_path: str, tracker: str = "oracle", overwrite: bool = False) -> Dict[str, Any]:
    cfg = RootConfig.from_yaml(cfg_path)
    if cfg.dataset != "tapvid" or cfg.tapvid is None:
        raise ValueError(f"[tapvid_pred] config dataset must be 'tapvid'. got: {cfg.dataset}")
    if cfg.davis_root is None:
        raise KeyError("[tapvid_pred] missing davis_root in config")

    res = cfg.res or "480p"
    tcfg = cfg.tapvid
    pred_dir = _safe_mkdir(tcfg.pred_dir)

    seqs = build_tapvid_dataset(
        davis_root=cfg.davis_root,
        pkl_path=tcfg.pkl_path,
        split=tcfg.split,
        res=res,
        query_mode=tcfg.query_mode,
        stride=tcfg.stride,
    )

    if tracker != "oracle":
        raise NotImplementedError("[tapvid_pred] currently only supports tracker='oracle' (sanity baseline).")

    wrote = 0
    skipped = 0
    for seq in seqs:
        out = pred_dir / f"{seq.name}.npz"
        if out.exists() and not overwrite:
            skipped += 1
            continue
        # ORACLE: pred == GT
        _write_pred_npz(out, tracks_xy=seq.gt_tracks_xy, vis=seq.gt_vis, queries_txy=seq.queries_txy)
        wrote += 1

    return {"pred_dir": str(pred_dir), "wrote": wrote, "skipped": skipped, "num_seqs": len(seqs)}