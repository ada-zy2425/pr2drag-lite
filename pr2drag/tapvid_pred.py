# pr2drag/tapvid_pred.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np

from pr2drag.tier1.contracts import RootConfig
from pr2drag.tier0.runner_tapvid import run_tapvid_pred


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



def tapvid_pred_from_config(cfg_path: str) -> Dict[str, Any]:
    cfg = RootConfig.from_yaml(cfg_path)
    if cfg.dataset != "tapvid" or cfg.tapvid is None:
        raise ValueError(f"[tapvid_pred] config dataset must be 'tapvid'. got: {cfg.dataset}")
    return run_tapvid_pred(cfg, config_path=cfg.config_path, config_sha1=cfg.config_sha1)