from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from .pipeline import run_all, stage1_precompute_split, stage2_compute_split, stage3_train_eval
from .utils import pretty_header, read_yaml, resolve_davis_root, set_seed


def _validate_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    need = ["davis_root", "res", "base_out", "splits"]
    for k in need:
        if k not in cfg:
            raise KeyError(f"Missing config key: {k}")

    cfg["davis_root"] = str(resolve_davis_root(cfg["davis_root"]))
    cfg.setdefault("seed", 0)

    if "train" not in cfg["splits"] or "val" not in cfg["splits"]:
        raise KeyError("cfg.splits must contain train and val entries")

    return cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="Path to yaml config")
    ap.add_argument("--cmd", type=str, default="run_all", choices=["run_all", "stage1", "stage2", "stage3"])
    args = ap.parse_args()

    cfg = read_yaml(args.config)
    cfg = _validate_cfg(cfg)
    set_seed(int(cfg.get("seed", 0)))

    print(pretty_header("PR2-Drag Lite", {
        "cmd": args.cmd,
        "config": args.config,
        "davis_root": cfg["davis_root"],
        "res": cfg["res"],
        "base_out": cfg["base_out"],
    }))

    base_out = Path(cfg["base_out"])
    out_train_s1 = base_out / "davis2016_train_precompute"
    out_val_s1 = base_out / "davis2016_val_precompute"
    out_train_s2 = base_out / "davis2016_stage2_fixed"
    out_val_s2 = base_out / "davis2016_val_stage2"
    out_dir_s3 = base_out / cfg.get("stage3", {}).get("out_dir", "davis2016_stage3_fixed")

    if args.cmd == "run_all":
        run_all(cfg)
        return

    if args.cmd == "stage1":
        stage1_precompute_split(cfg, split="train", out_dir=out_train_s1)
        stage1_precompute_split(cfg, split="val", out_dir=out_val_s1)
        return

    if args.cmd == "stage2":
        stage2_compute_split(cfg, split="train", stage1_dir=out_train_s1, out_dir=out_train_s2)
        stage2_compute_split(cfg, split="val", stage1_dir=out_val_s1, out_dir=out_val_s2)
        return

    if args.cmd == "stage3":
        stage3_train_eval(cfg, stage2_train=out_train_s2, stage2_val=out_val_s2, out_dir=out_dir_s3)
        return


if __name__ == "__main__":
    main()
