# pr2drag/tapvid_eval.py
from __future__ import annotations

from typing import Any, Dict

from pr2drag.tier1.contracts import RootConfig
from pr2drag.tier0.runner_tapvid import run_tapvid_eval


def tapvid_eval_from_config(cfg_path: str) -> Dict[str, Any]:
    cfg = RootConfig.from_yaml(cfg_path)
    if cfg.dataset != "tapvid" or cfg.tapvid is None:
        raise ValueError(f"[tapvid_eval] config dataset must be 'tapvid'. got: {cfg.dataset}")

    if cfg.davis_root is None:
        raise KeyError("[tapvid_eval] missing davis_root in config (needed to resolve frames)")

    res = cfg.res or "480p"
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