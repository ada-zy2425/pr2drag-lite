from __future__ import annotations

from typing import Dict


def compute_catastrophic(result: Dict) -> int:
    return int(result.get("catastrophic", 0))
