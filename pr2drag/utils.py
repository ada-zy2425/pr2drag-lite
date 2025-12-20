from __future__ import annotations

import os
import io
import json
import time
import hashlib
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import yaml


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def get_logger(name: str = "pr2drag") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    h = logging.StreamHandler()
    fmt = logging.Formatter("[%(levelname)s] %(message)s")
    h.setFormatter(fmt)
    logger.addHandler(h)
    return logger


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def deep_update(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively update dict."""
    out = dict(base)
    for k, v in (patch or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def save_json(path: str, obj: Any) -> None:
    ensure_dir(os.path.dirname(path))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_npz_atomic(path: str, **arrays: Any) -> None:
    ensure_dir(os.path.dirname(path))
    tmp = path + ".tmp"
    np.savez_compressed(tmp, **arrays)
    # numpy adds .npz automatically only if path ends without .npz; we force:
    if not tmp.endswith(".npz"):
        tmp2 = tmp + ".npz"
        if os.path.exists(tmp2):
            os.replace(tmp2, path)
            return
    os.replace(tmp, path)


def load_npz(path: str) -> Dict[str, Any]:
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def sha1_bytes(b: bytes) -> str:
    h = hashlib.sha1()
    h.update(b)
    return h.hexdigest()


def stable_hash_dict(d: Dict[str, Any]) -> str:
    """Stable hash for nested dict with JSON canonicalization."""
    b = json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return sha1_bytes(b)


def signature_for_stage(stage: str, cfg: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> str:
    """Compute a cache signature string for stage outputs."""
    # Only include relevant keys to avoid accidental invalidations.
    base = {
        "stage": stage,
        "res": cfg.get("res"),
        "feat_set": cfg.get("stage1", {}).get("feat_set"),
        "phasecorr": cfg.get("stage1", {}).get("phasecorr", {}),
        "label": cfg.get("stage2", {}).get("label", {}),
        "emission": cfg.get("stage3", {}).get("emission", {}),
        "hmm": cfg.get("stage3", {}).get("hmm", {}),
        "tau": cfg.get("stage3", {}).get("tau", {}),
        "aob": cfg.get("stage3", {}).get("aob", {}),
        "eval": cfg.get("stage3", {}).get("eval", {}),
    }
    if extra:
        base["extra"] = extra
    return stable_hash_dict(base)


def is_compatible_npz(npz_path: str, sig: str) -> bool:
    if not os.path.exists(npz_path):
        return False
    try:
        z = load_npz(npz_path)
        if "signature" not in z:
            return False
        old = str(z["signature"])
        return old == sig
    except Exception:
        return False


def read_split_list(davis_root: str, rel_path: str) -> list[str]:
    path = os.path.join(davis_root, rel_path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Split file not found: {path}")
    seqs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                seqs.append(s)
    return seqs


def safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return default
