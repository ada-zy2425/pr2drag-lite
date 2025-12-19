from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

from .config import load_config
from .stage1 import stage1_precompute
from .stage2_flow import stage2_one_split
from .stage3 import run_stage3

def main():
    ap = argparse.ArgumentParser("pr2drag-lite")
    ap.add_argument("--config", type=str, required=True, help="path to yaml config")
    ap.add_argument("cmd", type=str, choices=["stage1", "stage2", "stage3", "run_all"], help="command")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    np.random.seed(cfg.seed)

    davis_root = cfg.davis_root
    res = cfg.res
    base_out = cfg.base_out

    stage1_train = base_out / "davis2016_train_precompute"
    stage1_val   = base_out / "davis2016_val_precompute"
    stage2_base  = base_out / "davis2016_stage2_fixed"
    stage2_train = stage2_base / "train"
    stage2_val   = stage2_base / "val"
    stage3_out   = base_out / "davis2016_stage3_fixed"

    train_txt = davis_root / f"ImageSets/{res}/train.txt"
    val_txt   = davis_root / f"ImageSets/{res}/val.txt"
    if not val_txt.exists():
        raise FileNotFoundError(f"Missing val.txt: {val_txt}")
    if not train_txt.exists():
        print(f"[WARN] train.txt not found at {train_txt}. Fallback to val.txt for train (NOT recommended).")
        train_txt = val_txt

    overwrite = cfg.overwrite
    skip_if_exists = cfg.skip_if_exists
    verbose_skip = cfg.verbose_skip

    if args.cmd in ["stage1", "run_all"]:
        stage1_precompute(
            davis_root=davis_root, res=res,
            split_txt=train_txt, out_dir=stage1_train, split_name="train",
            overwrite=overwrite, skip_if_exists=skip_if_exists, verbose_skip=verbose_skip
        )
        stage1_precompute(
            davis_root=davis_root, res=res,
            split_txt=val_txt, out_dir=stage1_val, split_name="val",
            overwrite=overwrite, skip_if_exists=skip_if_exists, verbose_skip=verbose_skip
        )

    if args.cmd in ["stage2", "run_all"]:
        label = cfg.raw["label"]
        flow = cfg.raw["flow"]
        stage2_one_split(
            stage1_dir=stage1_train, out_dir=stage2_train, split_name="train",
            iou_thr=float(label["iou_thr"]),
            empty_ok=bool(label["empty_ok"]),
            flow_downscale=float(flow["downscale"]),
            farneback_params=flow["farneback"],
            overwrite=overwrite, skip_if_exists=skip_if_exists, verbose_skip=verbose_skip
        )
        stage2_one_split(
            stage1_dir=stage1_val, out_dir=stage2_val, split_name="val",
            iou_thr=float(label["iou_thr"]),
            empty_ok=bool(label["empty_ok"]),
            flow_downscale=float(flow["downscale"]),
            farneback_params=flow["farneback"],
            overwrite=overwrite, skip_if_exists=skip_if_exists, verbose_skip=verbose_skip
        )
        print("[OK] Stage2 done:",
              "train_npz=", len(list(stage2_train.glob("*.npz"))),
              "val_npz=", len(list(stage2_val.glob("*.npz"))))

    if args.cmd in ["stage3", "run_all"]:
        sp = cfg.raw["sp"]
        aob = cfg.raw["aob"]
        metrics = cfg.raw["metrics"]
        run_stage3(
            train_dir=stage2_train,
            val_dir=stage2_val,
            out_dir=stage3_out,
            sp_conf=sp,
            aob_conf=aob,
            metrics_conf=metrics,
        )

if __name__ == "__main__":
    main()
