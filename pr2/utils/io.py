from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Union

Jsonable = Union[Dict[str, Any], List[Any], str, int, float, bool, None]


def ensure_dir(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _to_jsonable(obj: Any) -> Jsonable:
    if is_dataclass(obj):
        return asdict(obj)
    return obj


def atomic_write_text(path: Union[str, Path], text: str, encoding: str = "utf-8") -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path.parent), encoding=encoding) as tf:
        tf.write(text)
        tmp = Path(tf.name)
    os.replace(tmp, path)


def atomic_write_json(path: Union[str, Path], obj: Any, indent: int = 2) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    payload = json.dumps(_to_jsonable(obj), ensure_ascii=False, indent=indent)
    atomic_write_text(path, payload)


def read_json(path: Union[str, Path]) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: Union[str, Path], rows: Iterable[Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(_to_jsonable(r), ensure_ascii=False) + "\n")


def append_jsonl(path: Union[str, Path], row: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(_to_jsonable(row), ensure_ascii=False) + "\n")


def iter_jsonl(path: Union[str, Path]) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def pretty(obj: Any) -> str:
    return json.dumps(_to_jsonable(obj), ensure_ascii=False, indent=2)
