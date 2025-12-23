# pr2drag/tier1/contracts.py
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def _sha1_text(s: str) -> str:
    h = hashlib.sha1()
    h.update(s.encode("utf-8", errors="ignore"))
    return h.hexdigest()


def _as_str(x: Any, default: str = "") -> str:
    return default if x is None else str(x)


def _require_enum(name: str, v: str, allowed: set[str]) -> str:
    v2 = v.strip()
    if v2 not in allowed:
        raise ValueError(f"[contracts] {name} must be one of {sorted(allowed)}; got {v!r}")
    return v2


@dataclass(frozen=True)
class TapVidConfig:
    split: str
    pkl_path: str

    query_mode: str        # "tapvid" or "strided"
    stride: int
    max_queries: int

    tracker: str           # "tapir" / "cotracker_v2" / "none"
    tracker_ckpt: str
    pred_dir: str
    overwrite_preds: bool

    resize_to_256: bool
    keep_aspect: bool
    interp: str            # "bilinear" / "nearest"

    out_dir: str
    allow_missing_preds: bool
    fail_fast: bool
    seed: int


@dataclass(frozen=True)
class RootConfig:
    # normalized string name: "davis" / "tapvid"
    dataset: str

    davis_root: Optional[str]
    base_out: str
    res: Optional[str]  # keep for compatibility; not required for tapvid

    tapvid: Optional[TapVidConfig]

    # book-keeping (for audit/manifests)
    config_path: str
    config_sha1: str

    @staticmethod
    def from_yaml(path: str) -> "RootConfig":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"[contracts] yaml not found: {p}")
        raw_text = p.read_text(encoding="utf-8")
        obj = yaml.safe_load(raw_text)
        if not isinstance(obj, dict):
            raise TypeError(f"[contracts] yaml root must be dict, got {type(obj)}")

        # dataset supports BOTH string and dict, but your yaml uses string
        ds_raw = obj.get("dataset", "davis")
        if isinstance(ds_raw, dict):
            ds = _as_str(ds_raw.get("name", "davis"), "davis").strip().lower()
        else:
            ds = _as_str(ds_raw, "davis").strip().lower()
        ds = ds or "davis"
        if ds not in ("davis", "tapvid"):
            raise ValueError(f"[contracts] dataset must be 'davis' or 'tapvid', got {ds!r}")

        base_out = _as_str(obj.get("base_out", "")).strip()
        if not base_out:
            raise KeyError("[contracts] missing base_out")

        davis_root = obj.get("davis_root", None)
        davis_root = None if davis_root is None else _as_str(davis_root).strip()
        res = obj.get("res", None)
        res = None if res is None else _as_str(res).strip()

        tapvid_cfg = None
        if ds == "tapvid":
            t = obj.get("tapvid", None)
            if not isinstance(t, dict):
                raise KeyError("[contracts] dataset=tapvid requires a tapvid: {...} dict")

            split = _as_str(t.get("split", "davis")).strip() or "davis"
            pkl_path = _as_str(t.get("pkl_path", "")).strip()
            if not pkl_path:
                raise KeyError("[contracts] tapvid.pkl_path is required")
            if not Path(pkl_path).exists():
                raise FileNotFoundError(f"[contracts] tapvid.pkl_path not found: {pkl_path}")

            query_mode = _require_enum("tapvid.query_mode", _as_str(t.get("query_mode", "tapvid")).strip(),
                                       {"tapvid", "strided"})
            stride = int(t.get("stride", 5))
            if stride < 1:
                raise ValueError("[contracts] tapvid.stride must be >=1")

            max_queries = int(t.get("max_queries", 0))
            if max_queries < 0:
                raise ValueError("[contracts] tapvid.max_queries must be >=0")

            tracker_raw = _as_str(t.get("tracker", "none")).strip().lower()

            # 可选：alias
            if tracker_raw in ("gt", "ground_truth"):
                tracker_raw = "oracle"

            tracker = _require_enum(
                "tapvid.tracker",
                tracker_raw,
                allowed={"tapir", "cotracker_v2", "none", "oracle"},
            )
            tracker_ckpt = _as_str(t.get("tracker_ckpt", "")).strip()
            pred_dir = _as_str(t.get("pred_dir", "")).strip()
            overwrite_preds = bool(t.get("overwrite_preds", False))

            resize_to_256 = bool(t.get("resize_to_256", True))
            keep_aspect = bool(t.get("keep_aspect", True))
            interp = _require_enum("tapvid.interp", _as_str(t.get("interp", "bilinear")).strip(),
                                   {"bilinear", "nearest"})

            out_dir = _as_str(t.get("out_dir", "")).strip()
            allow_missing_preds = bool(t.get("allow_missing_preds", False))
            fail_fast = bool(t.get("fail_fast", False))
            seed = int(t.get("seed", 0))

            if not davis_root:
                raise KeyError("[contracts] dataset=tapvid requires davis_root to resolve frames")

            tapvid_cfg = TapVidConfig(
                split=split,
                pkl_path=pkl_path,
                query_mode=query_mode,
                stride=stride,
                max_queries=max_queries,
                tracker=tracker,
                tracker_ckpt=tracker_ckpt,
                pred_dir=pred_dir,
                overwrite_preds=overwrite_preds,
                resize_to_256=resize_to_256,
                keep_aspect=keep_aspect,
                interp=interp,
                out_dir=out_dir,
                allow_missing_preds=allow_missing_preds,
                fail_fast=fail_fast,
                seed=seed,
            )

        return RootConfig(
            dataset=ds,
            davis_root=davis_root,
            base_out=base_out,
            res=res,
            tapvid=tapvid_cfg,
            config_path=str(p),
            config_sha1=_sha1_text(raw_text),
        )