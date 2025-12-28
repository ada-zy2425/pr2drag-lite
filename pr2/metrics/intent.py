from __future__ import annotations

from typing import Dict


def compute_intent_success(result: Dict) -> int:
    return int(result.get("intent_success", 0))
