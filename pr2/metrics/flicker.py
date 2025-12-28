from __future__ import annotations

from typing import Dict


def compute_flicker_p95(result: Dict) -> float:
    return float(result.get("flicker_p95", 0.0))
