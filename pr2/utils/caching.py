from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io import ensure_dir


@dataclass
class DiskCache:
    """
    简单的磁盘缓存（以 key -> pickle 文件 存储）。

    - 适合缓存：特征、tracker 输出、中间证据等。
    - 设计目标：稳、可断点恢复、跨进程可共享（无强锁；多进程写同 key 请自行规避）。
    """
    root: Path

    def __post_init__(self) -> None:
        self.root = ensure_dir(self.root)

    def key_to_path(self, key: str) -> Path:
        safe = key.replace("/", "_")
        return self.root / f"{safe}.pkl"

    def has(self, key: str) -> bool:
        return self.key_to_path(key).exists()

    def get(self, key: str) -> Any:
        p = self.key_to_path(key)
        with open(p, "rb") as f:
            return pickle.load(f)

    def set(self, key: str, obj: Any) -> None:
        p = self.key_to_path(key)
        ensure_dir(p.parent)
        tmp = p.with_suffix(".tmp.pkl")
        with open(tmp, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, p)
