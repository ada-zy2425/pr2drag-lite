# pr2drag/tapvid_eval.py
from __future__ import annotations

from typing import Any, Dict

from pr2drag.tier1.contracts import RootConfig
from pr2drag.tier0.runner_tapvid import run_tapvid_eval


def tapvid_eval_from_config(cfg_path: str) -> Dict[str, Any]:
    cfg = RootConfig.from_yaml(cfg_path)
    if cfg.dataset != "tapvid" or cfg.tapvid is None:
        raise ValueError(f"[tapvid_eval] config dataset must be 'tapvid'. got: {cfg.dataset}")
    return run_tapvid_eval(cfg, config_path=cfg.config_path, config_sha1=cfg.config_sha1)