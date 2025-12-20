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


def _compute_paths(cfg: Dict[str, Any]) -> Dict[str, Path]:
    base_out = Path(cfg["base_out"])
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


def _run_all_explicit(cfg: Dict[str, Any]) -> None:
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
    ap.add_argument("--cmd", type=str, default="run_all", choices=["run_all", "stage1", "stage2", "stage3"])
    ap.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional tag name for paperpack folder/zip. Default uses stage3 out_dir name.",
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

    paths = _compute_paths(cfg)
    base_out = paths["base_out"]
    out_train_s1 = paths["out_train_s1"]
    out_val_s1 = paths["out_val_s1"]
    out_train_s2 = paths["out_train_s2"]
    out_val_s2 = paths["out_val_s2"]
    out_dir_s3 = paths["out_dir_s3"]

    print(pretty_header("PR2-Drag Lite", {
        "cmd": args.cmd,
        "config": args.config,
        "davis_root": cfg["davis_root"],
        "res": cfg["res"],
        "base_out": cfg["base_out"],
    }))

    # archive options
    opts: ArchiveOptions = parse_archive_options(cfg)
    if args.no_archive:
        opts.enable = False

    # decide tag for paperpack
    tag = args.tag if args.tag else out_dir_s3.name

    # log path (auto)
    s3_ana = dict(cfg.get("stage3", {}).get("analysis", {}))
    log_to_file = bool(s3_ana.get("log_to_file", False)) or opts.enable
    log_path = out_dir_s3 / "_analysis" / "run.log"

    def _maybe_archive(cmd_name: str) -> None:
        # Only archive for commands that produce stage3 outputs
        if cmd_name not in ("stage3", "run_all"):
            return
        if not opts.enable:
            return
        z = archive_run(
            base_out=base_out,
            out_dir=out_dir_s3,
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
                _dispatch(args.cmd, cfg, out_train_s1, out_val_s1, out_train_s2, out_val_s2, out_dir_s3)
        else:
            _dispatch(args.cmd, cfg, out_train_s1, out_val_s1, out_train_s2, out_val_s2, out_dir_s3)

        _maybe_archive(args.cmd)

    except Exception as e:
        # try to archive traceback for post-mortem
        if opts.enable:
            archive_on_exception(
                base_out=base_out,
                out_dir=out_dir_s3,
                tag=tag,
                cmd=args.cmd,
                cfg_path=cfg_path,
                cfg=cfg,
                opts=opts,
                exc=e,
            )
            print(f"[ARCHIVE] error manifest/traceback attempted under: {base_out}/{opts.pack_root}/{tag}")
        raise


def _dispatch(
    cmd: str,
    cfg: Dict[str, Any],
    out_train_s1: Path,
    out_val_s1: Path,
    out_train_s2: Path,
    out_val_s2: Path,
    out_dir_s3: Path,
) -> None:
    if cmd == "run_all":
        _run_all_explicit(cfg)
        return

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

    raise ValueError(f"Unknown cmd: {cmd}")


if __name__ == "__main__":
    main()
