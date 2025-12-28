from __future__ import annotations

import hashlib
import json
from typing import Any, Dict


def stable_json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def hash_dict(d: Dict[str, Any], algo: str = "sha256") -> str:
    s = stable_json_dumps(d).encode("utf-8")
    h = hashlib.new(algo)
    h.update(s)
    return h.hexdigest()
