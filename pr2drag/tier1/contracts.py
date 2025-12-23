# pr2drag/tier1/contracts.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Literal

import json
import hashlib

try:
    import yaml  # type: ignore
except Exception as e:  # pragma: no cover
    yaml = None
    _YAML_IMPORT_ERROR = e


TapVidQueryMode = Literal["first", "strided"]


def _sha1_jsonable(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()


def load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"[Config] YAML not found: {p}")
    if yaml is None:
        raise RuntimeError(f"[Config] PyYAML import failed: {_YAML_IMPORT_ERROR}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"[Config] YAML root must be a dict, got {type(data)}")
    data["_config_path"] = str(p.resolve())
    data["_config_sha1"] = _sha1_jsonable(data)
    return data


@dataclass(frozen=True)
class TapVidConfig:
    split: str
    pkl_path: str
    query_mode: TapVidQueryMode
    stride: int
    pred_dir: str
    out_dir: str
    resize_to_256: bool

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "TapVidConfig":
        required = ["split", "pkl_path", "query_mode", "stride", "pred_dir", "out_dir", "resize_to_256"]
        missing = [k for k in required if k not in d]
        if missing:
            raise KeyError(f"[TapVidConfig] missing keys: {missing}")

        qm = str(d["query_mode"]).strip().lower()
        if qm not in ("first", "strided"):
            raise ValueError(f"[TapVidConfig] query_mode must be 'first' or 'strided', got: {d['query_mode']}")

        stride = int(d["stride"])
        if stride <= 0:
            raise ValueError(f"[TapVidConfig] stride must be positive, got: {stride}")
        if qm == "first":
            # still allow stride present (fixed schema), but it won't be used
            pass

        return TapVidConfig(
            split=str(d["split"]),
            pkl_path=str(d["pkl_path"]),
            query_mode=qm,  # type: ignore
            stride=stride,
            pred_dir=str(d["pred_dir"]),
            out_dir=str(d["out_dir"]),
            resize_to_256=bool(d["resize_to_256"]),
        )


@dataclass(frozen=True)
class RootConfig:
    dataset: str
    tapvid: Optional[TapVidConfig]
    davis_root: Optional[str]
    res: Optional[str]

    raw: Dict[str, Any]
    config_path: Optional[str]
    config_sha1: Optional[str]

    @staticmethod
    def from_yaml(path: str | Path) -> "RootConfig":
        raw = load_yaml(path)
        dataset = str(raw.get("dataset", "")).strip().lower()
        davis_root = raw.get("davis_root", None)
        res = raw.get("res", None)

        tapvid_cfg = None
        if dataset == "tapvid":
            tapvid_cfg = TapVidConfig.from_dict(raw.get("tapvid", {}))

        return RootConfig(
            dataset=dataset,
            tapvid=tapvid_cfg,
            davis_root=str(davis_root) if davis_root is not None else None,
            res=str(res) if res is not None else None,
            raw=raw,
            config_path=raw.get("_config_path", None),
            config_sha1=raw.get("_config_sha1", None),
        )