from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Callable

from .archive import (
    ArchiveOptions,
    archive_on_exception,
    archive_run,
    parse_archive_options,
    tee_stdout_stderr,
)
from .utils import pretty_header, read_yaml, resolve_davis_root, set_seed


# ---------------------------
# Dataset-aware cfg utilities
# ---------------------------
def _dataset_name(cfg: Dict[str, Any]) -> str:
    """
    Support both:
      - dataset: tapvid          (string)
      - dataset: {name: tapvid}  (dict)
    """
    ds = cfg.get("dataset", None)

    if isinstance(ds, str):
        name = ds.strip().lower()
        return name if name else "davis"

    if isinstance(ds, dict):
        name = str(ds.get("name", "")).strip().lower()
        return name if name else "davis"

    # legacy configs have no dataset section => davis
    return "davis"

def _normalize_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Backward compatible:
      - legacy: no cfg.dataset => assume davis
      - new: cfg.dataset can be string or dict
    Normalize to dict form: cfg["dataset"] = {"name": "..."}.
    """
    cfg = dict(cfg)

    if "dataset" not in cfg:
        cfg["dataset"] = {"name": "davis"}
        return cfg

    ds = cfg["dataset"]
    if isinstance(ds, str):
        cfg["dataset"] = {"name": ds.strip().lower() or "davis"}
        return cfg

    if isinstance(ds, dict):
        dd = dict(ds)
        dd.setdefault("name", "davis")
        dd["name"] = str(dd["name"]).strip().lower() or "davis"
        cfg["dataset"] = dd
        return cfg

    raise TypeError(f"cfg.dataset must be str or dict, got {type(ds)}")



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
        # Strict but relevant checks for tapvid
        if "base_out" not in cfg:
            raise KeyError("Missing config key: base_out (dataset=tapvid)")
        if "davis_root" not in cfg:
            raise KeyError("Missing config key: davis_root (dataset=tapvid)")

        cfg["davis_root"] = str(resolve_davis_root(cfg["davis_root"]))

        cfg.setdefault("tapvid", {})
        if not isinstance(cfg["tapvid"], dict):
            raise TypeError("cfg.tapvid must be a dict (dataset=tapvid)")

        # optional deeper key checks (runner will still validate)
        if "pkl_path" not in cfg["tapvid"]:
            raise KeyError("Missing config key: tapvid.pkl_path (dataset=tapvid)")
        if "split" not in cfg["tapvid"]:
            cfg["tapvid"]["split"] = "davis"

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

    raise ValueError(f"Unknown dataset: {ds}")


# ---------------------------
# Lazy pipeline resolver (robust to renames)
# ---------------------------
def _resolve_davis_pipeline_fns() -> Tuple[Callable[..., Any], Callable[..., Any], Callable[..., Any]]:
    """
    Import pr2drag.pipeline lazily and resolve function names robustly.
    This avoids tapvid_* commands being blocked by pipeline import/name issues.
    """
    try:
        import pr2drag.pipeline as pipeline  # lazy import
    except Exception as e:
        raise ImportError(
            "[cli] Failed to import pr2drag.pipeline. "
            "This does NOT affect tapvid_* commands, but DAVIS commands need pipeline to import."
        ) from e

    def pick(candidates: Tuple[str, ...]) -> Callable[..., Any]:
        for name in candidates:
            if hasattr(pipeline, name):
                fn = getattr(pipeline, name)
                if callable(fn):
                    return fn
        avail = [k for k in dir(pipeline) if k.startswith("stage") or k.startswith("run_")]
        raise AttributeError(
            "[cli] Cannot find expected DAVIS pipeline entrypoints in pr2drag.pipeline.\n"
            f"  Tried: {candidates}\n"
            f"  Available (filtered): {avail}\n"
            "Fix: either export these functions in pipeline.py or update cli.py resolver."
        )

    stage1 = pick(("stage1_precompute_split", "stage1_precompute", "stage1"))
    stage2 = pick(("stage2_compute_split", "stage2_compute", "stage2"))
    stage3 = pick(("stage3_train_eval", "stage3_eval", "stage3"))
    return stage1, stage2, stage3


def _run_all_explicit(cfg: Dict[str, Any], paths: Dict[str, Path]) -> None:
    """
    Legacy DAVIS run_all (exactly as before), but with lazy pipeline import.
    """
    ds = _dataset_name(cfg)
    if ds != "davis":
        raise ValueError(f"cmd=run_all only supports dataset=davis (got {ds}). Use tapvid_pred/tapvid_eval etc.")

    stage1_precompute_split, stage2_compute_split, stage3_train_eval = _resolve_davis_pipeline_fns()

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
        choices=[
            "run_all",
            "stage1",
            "stage2",
            "stage3",
            "tapvid_pred",
            "tapvid_eval",
            "run_tapvid_tier0",
        ],
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
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="For tapvid_pred: overwrite existing prediction npz files",
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

    print(
        pretty_header(
            "PR2-Drag Lite",
            {
                "cmd": args.cmd,
                "dataset": ds,
                "config": args.config,
                "base_out": str(cfg["base_out"]),
                **({"davis_root": cfg["davis_root"], "res": cfg["res"]} if ds == "davis" else {}),
            },
        )
    )

    # archive options
    opts: ArchiveOptions = parse_archive_options(cfg)
    if args.no_archive:
        opts.enable = False

    # decide tag for paperpack
    tag = args.tag if args.tag else out_dir_for_tag.name

    # log path (auto)
    if ds == "davis":
        s3_ana = dict(cfg.get("stage3", {}).get("analysis", {}))
        log_to_file = bool(s3_ana.get("log_to_file", False)) or opts.enable
        log_path = paths["out_dir_s3"] / "_analysis" / "run.log"
    else:
        t_ana = dict(cfg.get("tapvid", {}).get("analysis", {}))
        log_to_file = bool(t_ana.get("log_to_file", False)) or opts.enable
        log_path = paths["out_dir_tapvid"] / "_analysis" / "run.log"

    def _maybe_archive(cmd_name: str) -> None:
        if cmd_name not in ("stage3", "run_all", "run_tapvid_tier0", "tapvid_eval", "tapvid_pred"):
            return
        if not opts.enable:
            return

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
                _dispatch(args.cmd, args.config, cfg, paths, overwrite=bool(args.overwrite))
        else:
            _dispatch(args.cmd, args.config, cfg, paths, overwrite=bool(args.overwrite))

        _maybe_archive(args.cmd)

    except Exception as e:
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


def _dispatch(cmd: str, cfg_path_str: str, cfg: Dict[str, Any], paths: Dict[str, Path], overwrite: bool) -> None:
    ds = _dataset_name(cfg)

    # -------------------------
    # TAP-Vid commands (independent, no pipeline)
    # -------------------------
    if cmd == "tapvid_pred":
        if ds != "tapvid":
            raise ValueError(f"cmd=tapvid_pred requires dataset=tapvid (got {ds}).")
        from pr2drag.tapvid_pred import tapvid_pred_from_config

        tapvid_pred_from_config(cfg_path_str, tracker="oracle", overwrite=overwrite)
        return

    if cmd == "tapvid_eval":
        if ds != "tapvid":
            raise ValueError(f"cmd=tapvid_eval requires dataset=tapvid (got {ds}).")
        from pr2drag.tapvid_eval import tapvid_eval_from_config

        tapvid_eval_from_config(cfg_path_str)
        return

    # -------------------------
    # DAVIS legacy commands
    # -------------------------
    if cmd == "run_all":
        _run_all_explicit(cfg, paths)
        return

    if ds == "davis":
        stage1_precompute_split, stage2_compute_split, stage3_train_eval = _resolve_davis_pipeline_fns()

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

        if cmd == "run_tapvid_tier0":
            raise ValueError("cmd=run_tapvid_tier0 requires dataset=tapvid (current config is dataset=davis).")

        raise ValueError(f"Unknown cmd for dataset=davis: {cmd}")

    if ds == "tapvid":
        if cmd != "run_tapvid_tier0":
            raise ValueError(f"dataset=tapvid only supports cmd=tapvid_pred/tapvid_eval/run_tapvid_tier0 (got {cmd}).")

        # keep your previous behavior: lazy import inside
        from .pipeline import run_tapvid_tier0  # type: ignore

        out_dir = paths["out_dir_tapvid"]
        run_tapvid_tier0(cfg, out_dir=out_dir)
        return

    raise ValueError(f"Unknown dataset: {ds}")


if __name__ == "__main__":
    main()