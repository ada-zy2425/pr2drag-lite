from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class CompletionCertificate:
    boundary_ok: bool
    length_ok: bool
    speed_ok: bool
    disp_ok: bool
    feasible: bool
    decision: str  # "abstain" | "linear" | "minaccel"
    reason: str = ""


def check_feasibility(
    z_prev: Optional[np.ndarray],
    z_next: Optional[np.ndarray],
    seg_len: int,
    max_seg_len: int,
    v_max: float,
) -> CompletionCertificate:
    boundary_ok = (z_prev is not None) and (z_next is not None)
    length_ok = seg_len <= max_seg_len

    if not boundary_ok:
        return CompletionCertificate(boundary_ok=False, length_ok=length_ok, speed_ok=False, disp_ok=False,
                                     feasible=False, decision="abstain", reason="missing_boundary")

    if seg_len <= 0:
        return CompletionCertificate(boundary_ok=True, length_ok=length_ok, speed_ok=True, disp_ok=True,
                                     feasible=True, decision="linear", reason="empty_segment")

    delta = np.linalg.norm(z_next - z_prev)
    avg_step = delta / float(seg_len + 1)
    speed_ok = avg_step <= v_max + 1e-6

    disp_ok = True  # 可扩展 non-overshoot 等
    feasible = boundary_ok and length_ok and speed_ok and disp_ok
    decision = "minaccel" if feasible else "abstain"
    reason = "ok" if feasible else ("too_fast" if not speed_ok else "too_long" if not length_ok else "infeasible")

    return CompletionCertificate(
        boundary_ok=boundary_ok,
        length_ok=length_ok,
        speed_ok=speed_ok,
        disp_ok=disp_ok,
        feasible=feasible,
        decision=decision,
        reason=reason,
    )
