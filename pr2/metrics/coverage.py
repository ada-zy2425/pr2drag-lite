from __future__ import annotations

from typing import Dict


def compute_coverage(result: Dict) -> float:
    return float(result.get("coverage", 0.0))
