# pr2drag/tier1/contracts.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Mapping
import hashlib
import json

try:
    import yaml  # PyYAML
except Exception as e:  # pragma: no cover
    yaml = None


def _sha1_of_dict(d: Mapping[str, Any]) -> str:
    # stable json encoding
    s = json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def _as_path_str(x: Any, field: str) -> str:
    _require(isinstance(x, str) and x.strip() != "", f"[config] {field} must be a non-empty string")
    return x


@dataclass(frozen=True)
class TapVidConfig:
    split: str
    pkl_path: str
    query_mode: str
    stride: int
    pred_dir: str
    out_dir: str
    resize_to_256: bool

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "TapVidConfig":
        allowed = {
            "split", "pkl_path", "query_mode", "stride", "pred_dir", "out_dir", "resize_to_256"
        }
        unknown = set(d.keys()) - allowed
        _require(len(unknown) == 0, f"[config.tapvid] unknown keys: {sorted(list(unknown))}")

        split = _as_path_str(d.get("split", "davis"), "tapvid.split")
        pkl_path = _as_path_str(d.get("pkl_path"), "tapvid.pkl_path")

        query_mode = d.get("query_mode", "first")
        _require(query_mode in ("first", "strided"), "[config.tapvid] query_mode must be 'first' or 'strided'")

        stride = int(d.get("stride", 5))
        _require(stride > 0, "[config.tapvid] stride must be positive")

        pred_dir = _as_path_str(d.get("pred_dir"), "tapvid.pred_dir")
        out_dir = _as_path_str(d.get("out_dir"), "tapvid.out_dir")

        resize_to_256 = bool(d.get("resize_to_256", True))

        return TapVidConfig(
            split=split,
            pkl_path=pkl_path,
            query_mode=query_mode,
            stride=stride,
            pred_dir=pred_dir,
            out_dir=out_dir,
            resize_to_256=resize_to_256,
        )


@dataclass(frozen=True)
class RootConfig:
    # keep the names aligned with your existing davis configs
    dataset: str
    davis_root: Optional[str]
    res: Optional[str]
    tapvid: Optional[TapVidConfig]

    # bookkeeping
    config_path: str
    config_sha1: str
    raw: Dict[str, Any]

    @staticmethod
    def from_yaml(cfg_path: str) -> "RootConfig":
        if yaml is None:
            raise RuntimeError("PyYAML not installed. `pip install pyyaml`")

        p = Path(cfg_path)
        if not p.exists():
            raise FileNotFoundError(f"[config] file not found: {p}")

        raw = yaml.safe_load(p.read_text())
        if not isinstance(raw, dict):
            raise TypeError("[config] YAML root must be a dict")

        allowed = {"dataset", "davis_root", "res", "tapvid"}
        unknown = set(raw.keys()) - allowed
        _require(len(unknown) == 0, f"[config] unknown keys: {sorted(list(unknown))}")

        dataset = raw.get("dataset", None)
        _require(dataset in ("davis", "tapvid"), "[config] dataset must be 'davis' or 'tapvid'")

        davis_root = raw.get("davis_root", None)
        if davis_root is not None:
            _require(isinstance(davis_root, str), "[config] davis_root must be string or null")

        res = raw.get("res", None)
        if res is not None:
            _require(isinstance(res, str), "[config] res must be string or null")

        tapvid_cfg = None
        if raw.get("tapvid", None) is not None:
            _require(isinstance(raw["tapvid"], dict), "[config.tapvid] must be a dict")
            tapvid_cfg = TapVidConfig.from_dict(raw["tapvid"])

        sha1 = _sha1_of_dict(raw)

        return RootConfig(
            dataset=dataset,
            davis_root=davis_root,
            res=res,
            tapvid=tapvid_cfg,
            config_path=str(p),
            config_sha1=sha1,
            raw=raw,
        )