from __future__ import annotations

import argparse
import os
from typing import Any, Dict

from .utils import load_yaml, deep_update, get_logger
from .pipeline import run_all


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True, help="Path to yaml config")
    p.add_argument("--cmd", type=str, default="run_all", choices=["run_all"], help="Command")
    p.add_argument("--preset", type=str, default="", help="Optional preset name from configs/presets.yaml")
    p.add_argument("--presets", type=str, default="configs/presets.yaml", help="Path to presets yaml")
    return p.parse_args()


def apply_preset(cfg: Dict[str, Any], preset_name: str, presets_path: str) -> Dict[str, Any]:
    if not preset_name:
        return cfg
    if not os.path.exists(presets_path):
        raise FileNotFoundError(f"presets.yaml not found: {presets_path}")
    presets = load_yaml(presets_path) or {}
    if preset_name not in presets:
        raise KeyError(f"Preset '{preset_name}' not found in {presets_path}. Available: {list(presets.keys())}")
    patch = presets[preset_name]
    return deep_update(cfg, patch)


def main() -> None:
    args = parse_args()
    logger = get_logger()

    cfg = load_yaml(args.config)
    cfg = apply_preset(cfg, args.preset, args.presets)

    if args.cmd == "run_all":
        run_all(cfg)
    else:
        raise ValueError(f"Unsupported cmd: {args.cmd}")


if __name__ == "__main__":
    main()
