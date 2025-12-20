from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def safe_mkdir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        cfg = {}
    return cfg


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    safe_mkdir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def read_txt_lines(path: str | Path) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines()]
    return [ln for ln in lines if ln]


def sha1_of_dict(d: Dict[str, Any]) -> str:
    """
    Stable hash for cache compatibility checks.
    """
    s = json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def npz_write(path: str | Path, arrays: Dict[str, np.ndarray], meta: Dict[str, Any]) -> None:
    path = Path(path)
    safe_mkdir(path.parent)
    meta_json = json.dumps(meta, sort_keys=True).encode("utf-8")
    payload = dict(arrays)
    payload["_meta_json"] = np.frombuffer(meta_json, dtype=np.uint8)
    np.savez_compressed(path, **payload)


def npz_read(path: str | Path) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    data = np.load(path, allow_pickle=False)
    arrays: Dict[str, np.ndarray] = {}
    meta: Dict[str, Any] = {}
    for k in data.files:
        if k == "_meta_json":
            meta_bytes = data[k].tobytes()
            meta = json.loads(meta_bytes.decode("utf-8"))
        else:
            arrays[k] = data[k]
    return arrays, meta


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def resolve_davis_root(davis_root: str | Path) -> Path:
    """
    Accept either:
      - .../DAVIS (contains JPEGImages, Annotations, ImageSets)
      - ... (parent containing DAVIS/)
    Also tolerate 'DAVIS unzipped' vs 'DAVIS_unzipped' style path drift if user passed wrong.
    """
    p = Path(davis_root)

    # Tolerate trivial drive path issues: spaces <-> underscores
    if not p.exists():
        alt = Path(str(p).replace(" ", "_"))
        if alt.exists():
            p = alt

    # If they passed parent that contains DAVIS/
    if (p / "DAVIS").exists() and not (p / "JPEGImages").exists():
        p = p / "DAVIS"

    # Validate
    need = ["JPEGImages", "Annotations", "ImageSets"]
    missing = [x for x in need if not (p / x).exists()]
    if missing:
        raise FileNotFoundError(
            f"[DAVIS] Invalid davis_root={p}. Missing subfolders: {missing}. "
            f"Expected structure: {p}/JPEGImages/{p}/Annotations/{p}/ImageSets"
        )
    return p


def davis_split_path(davis_root: Path, rel_path: str) -> Path:
    # rel_path like "ImageSets/480p/train.txt"
    rel = Path(rel_path)
    if rel.is_absolute():
        # avoid silent bugs: absolute paths override davis_root
        return rel
    return davis_root / rel


def pretty_header(title: str, kv: Dict[str, Any]) -> str:
    lines = []
    lines.append(f"========== {title} ==========")
    for k, v in kv.items():
        lines.append(f"{k:<10}: {v}")
    lines.append("=" * (len(lines[0])))
    return "\n".join(lines)
