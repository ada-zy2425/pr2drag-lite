from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from pr2.utils.io import read_json, atomic_write_json


@dataclass
class TrackerOutput:
    """
    统一 tracker 输出格式的建议：
    - tracks: (T, N, 2)
    - conf: (T, N) or (T,)
    - visibility: (T, N) or (T,)
    """
    tracks: Any
    conf: Any = None
    visibility: Any = None
    meta: Optional[Dict[str, Any]] = None


def load_tracker_json(path: str | Path) -> Dict[str, Any]:
    return read_json(path)


def save_tracker_json(path: str | Path, obj: Dict[str, Any]) -> None:
    atomic_write_json(path, obj, indent=2)
