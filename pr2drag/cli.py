from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

from .archive import (
    ArchiveOptions,
    archive_on_exception,
    archive_run,
    parse_archive_options,
    tee_stdout_stderr,
)
from .pipeline import stage1_precompute_split, stage2_compute_split, stage3_train_eval
from .utils import pretty_header, read_yaml, resolve_davis_root, set_seed


# ---------------------------
# Dataset-aware cfg utilities
# ---------------------------
def _dataset_name(cfg: Dict[str, Any]) -> str:
    ds = cfg.get("dataset", None)
    if isinstance(ds, dict):
        name = str(ds.get("name", "")).strip().lower()
        return name if name else "davis"
    # legacy configs have no dataset section => davis
    return "davis"


def _normalize_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Backward compatible:
      - legacy: no cfg.dataset => assume davis
      - new: require cfg.dataset.name
    """
    if "dataset" not in cfg or not isinstance(cfg.get("dataset", {}), dict):
        cfg = dict(cfg)
        cfg["dataset"] = {"name": "davis"}
    else:
        cfg = dict(cfg)
        cfg["dataset"] = dict(cfg["dataset"])
        cfg["dataset"].setdefault("name", "davis")
    return cfg


def _validate_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = _normalize_cfg(cfg)
    cfg.setdefault("seed", 0)

    ds = _dataset_name(cfg)
    if ds == "davis":
        need = ["davis_root", "res", "base_out", "splits"]
        for k in need:
            if k not in cfg:
                raise KeyError(f"Missing config key: {k} (dataset=davis)")

        cfg["davis_root"] = str(resolve_davis_root(cfg["davis_root"]))

        splits = cfg.get("splits", {})
        if not isinstance(splits, dict) or "train" not in splits or "val" not in splits:
            raise KeyError("cfg.splits must contain train and val entries (dataset=davis)")
        return cfg

    if ds == "tapvid":
        # We keep it strict but minimal here; tier0 runner will do deeper checks.
        if "base_out" not in cfg:
            raise KeyError("Missing config key: base_out (dataset=tapvid)")
        # optional: cfg.tapvid.out_dir
        cfg.setdefault("tapvid", {})
        if not isinstance(cfg["tapvid"], dict):
            raise TypeError("cfg.tapvid must be a dict (dataset=tapvid)")
        return cfg

    raise ValueError(f"Unknown dataset name: {ds}. Expected 'davis' or 'tapvid'.")


def _compute_paths(cfg: Dict[str, Any]) -> Dict[str, Path]:
    """
    Path contract:
      - davis keeps legacy folder names exactly.
      - tapvid uses base_out / tapvid.out_dir (default: tapvid_tier0)
    """
    base_out = Path(cfg["base_out"])
    ds = _dataset_name(cfg)

    if ds == "davis":
        out_train_s1 = base_out / "davis2016_train_precompute"
        out_val_s1 = base_out / "davis2016_val_precompute"
        out_train_s2 = base_out / "davis2016_stage2_fixed"
        out_val_s2 = base_out / "davis2016_val_stage2"
        out_dir_s3 = base_out / cfg.get("stage3", {}).get("out_dir", "davis2016_stage3_fixed")
        return {
            "base_out": base_out,
            "out_train_s1": out_train_s1,
            "out_val_s1": out_val_s1,
            "out_train_s2": out_train_s2,
            "out_val_s2": out_val_s2,
            "out_dir_s3": out_dir_s3,
        }

    if ds == "tapvid":
        out_dir = base_out / cfg.get("tapvid", {}).get("out_dir", "tapvid_tier0")
        return {"base_out": base_out, "out_dir_tapvid": out_dir}

    # should never hit
    raise ValueError(f"Unknown dataset: {ds}")


def _run_all_explicit(cfg: Dict[str, Any]) -> None:
    """
    Legacy DAVIS run_all (exactly as before).
    For other datasets, run their dedicated command.
    """
    ds = _dataset_name(cfg)
    if ds != "davis":
        raise ValueError(f"cmd=run_all only supports dataset=davis (got {ds}). Use cmd=run_tapvid_tier0 etc.")

    paths = _compute_paths(cfg)
    base_out = paths["base_out"]
    out_train_s1 = paths["out_train_s1"]
    out_val_s1 = paths["out_val_s1"]
    out_train_s2 = paths["out_train_s2"]
    out_val_s2 = paths["out_val_s2"]
    out_dir_s3 = paths["out_dir_s3"]

    meta_tr = stage1_precompute_split(cfg, split="train", out_dir=out_train_s1)
    print(f"[OK] Stage1(train) meta: {out_train_s1/'meta.json'}")
    print(f"[OK] Stage1(train) npz_dir: {out_train_s1}  num_npz={len(meta_tr.get('seqs', []))}")

    meta_va = stage1_precompute_split(cfg, split="val", out_dir=out_val_s1)
    print(f"[OK] Stage1(val) meta: {out_val_s1/'meta.json'}")
    print(f"[OK] Stage1(val) npz_dir: {out_val_s1}  num_npz={len(meta_va.get('seqs', []))}")

    stage2_compute_split(cfg, split="train", stage1_dir=out_train_s1, out_dir=out_train_s2)
    stage2_compute_split(cfg, split="val", stage1_dir=out_val_s1, out_dir=out_val_s2)
    print(
        f"[OK] Stage2 done: train_npz= {len(list(out_train_s2.glob('*.npz')))} "
        f"val_npz= {len(list(out_val_s2.glob('*.npz')))}"
    )

    stage3_train_eval(cfg, stage2_train=out_train_s2, stage2_val=out_val_s2, out_dir=out_dir_s3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="Path to yaml config")
    ap.add_argument(
        "--cmd",
        type=str,
        default="run_all",
        choices=["run_all", "stage1", "stage2", "stage3", "run_tapvid_tier0"],
    )
    ap.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional tag name for paperpack folder/zip. Default uses output folder name.",
    )
    ap.add_argument(
        "--no_archive",
        action="store_true",
        help="Disable archiving even if enabled in config.",
    )
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = read_yaml(args.config)
    cfg = _validate_cfg(cfg)
    set_seed(int(cfg.get("seed", 0)))

    ds = _dataset_name(cfg)
    paths = _compute_paths(cfg)

    base_out = paths["base_out"]
    out_dir_for_tag: Path
    if ds == "davis":
        out_dir_for_tag = paths["out_dir_s3"]
    else:
        out_dir_for_tag = paths["out_dir_tapvid"]

    print(pretty_header("PR2-Drag Lite", {
        "cmd": args.cmd,
        "dataset": ds,
        "config": args.config,
        "base_out": str(cfg["base_out"]),
        **({"davis_root": cfg["davis_root"], "res": cfg["res"]} if ds == "davis" else {}),
    }))

    # archive options
    opts: ArchiveOptions = parse_archive_options(cfg)
    if args.no_archive:
        opts.enable = False

    # decide tag for paperpack
    tag = args.tag if args.tag else out_dir_for_tag.name

    # log path (auto)
    # keep your legacy behavior: stage3 analysis/logging
    if ds == "davis":
        s3_ana = dict(cfg.get("stage3", {}).get("analysis", {}))
        log_to_file = bool(s3_ana.get("log_to_file", False)) or opts.enable
        log_path = paths["out_dir_s3"] / "_analysis" / "run.log"
    else:
        # tapvid tier0 also logs under its out_dir/_analysis
        t_ana = dict(cfg.get("tapvid", {}).get("analysis", {}))
        log_to_file = bool(t_ana.get("log_to_file", False)) or opts.enable
        log_path = paths["out_dir_tapvid"] / "_analysis" / "run.log"

    def _maybe_archive(cmd_name: str) -> None:
        # Only archive for commands that produce final outputs
        if cmd_name not in ("stage3", "run_all", "run_tapvid_tier0"):
            return
        if not opts.enable:
            return

        # archive_run expects out_dir; use dataset output dir
        out_dir = paths["out_dir_s3"] if ds == "davis" else paths["out_dir_tapvid"]

        z = archive_run(
            base_out=base_out,
            out_dir=out_dir,
            tag=tag,
            cmd=cmd_name,
            cfg_path=cfg_path,
            cfg=cfg,
            opts=opts,
        )
        if z is not None:
            print(f"[ARCHIVE] wrote: {z}")

    try:
        if log_to_file:
            with tee_stdout_stderr(log_path):
                _dispatch(args.cmd, cfg, paths)
        else:
            _dispatch(args.cmd, cfg, paths)

        _maybe_archive(args.cmd)

    except Exception as e:
        # try to archive traceback for post-mortem
        if opts.enable:
            out_dir = paths["out_dir_s3"] if ds == "davis" else paths["out_dir_tapvid"]
            archive_on_exception(
                base_out=base_out,
                out_dir=out_dir,
                tag=tag,
                cmd=args.cmd,
                cfg_path=cfg_path,
                cfg=cfg,
                opts=opts,
                exc=e,
            )
            print(f"[ARCHIVE] error manifest/traceback attempted under: {base_out}/{opts.pack_root}/{tag}")
        raise


def _dispatch(cmd: str, cfg: Dict[str, Any], paths: Dict[str, Path]) -> None:
    ds = _dataset_name(cfg)

    if cmd == "run_all":
        _run_all_explicit(cfg)
        return

    if ds == "davis":
        out_train_s1 = paths["out_train_s1"]
        out_val_s1 = paths["out_val_s1"]
        out_train_s2 = paths["out_train_s2"]
        out_val_s2 = paths["out_val_s2"]
        out_dir_s3 = paths["out_dir_s3"]

        if cmd == "stage1":
            stage1_precompute_split(cfg, split="train", out_dir=out_train_s1)
            stage1_precompute_split(cfg, split="val", out_dir=out_val_s1)
            return

        if cmd == "stage2":
            stage2_compute_split(cfg, split="train", stage1_dir=out_train_s1, out_dir=out_train_s2)
            stage2_compute_split(cfg, split="val", stage1_dir=out_val_s1, out_dir=out_val_s2)
            return

        if cmd == "stage3":
            stage3_train_eval(cfg, stage2_train=out_train_s2, stage2_val=out_val_s2, out_dir=out_dir_s3)
            return
        
        if cmd == "tapvid_eval":
            from pr2drag.tapvid_eval import tapvid_eval_from_config
            tapvid_eval_from_config(args.config)
            return  

        if cmd == "run_tapvid_tier0":
            raise ValueError("cmd=run_tapvid_tier0 requires dataset=tapvid (current config is dataset=davis).")

        raise ValueError(f"Unknown cmd for dataset=davis: {cmd}")

    if ds == "tapvid":
        if cmd != "run_tapvid_tier0":
            raise ValueError(f"dataset=tapvid only supports cmd=run_tapvid_tier0 (got {cmd}).")

        # Lazy import so current DAVIS baseline never depends on tapvid code.
        from .pipeline import run_tapvid_tier0  # local import on purpose

        out_dir = paths["out_dir_tapvid"]
        run_tapvid_tier0(cfg, out_dir=out_dir)
        return

    raise ValueError(f"Unknown dataset: {ds}")


if __name__ == "__main__":
    main()