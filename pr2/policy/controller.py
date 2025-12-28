from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from pr2.policy.schedules import RadiusSchedule, GMaxSchedule, LambdaSafeSchedule
from pr2.policy.completion.feasibility import CompletionCertificate, check_feasibility
from pr2.policy.completion.linear import linear_completion
from pr2.policy.completion.minaccel import minaccel_completion, MinAccelConfig


@dataclass
class FramePolicy:
    t: int
    w_tilde: float
    update_mask: bool
    radius: float
    gmax: float
    lambda_safe: float
    completion: Optional[Dict[str, Any]] = None


@dataclass
class SegmentDecision:
    start: int
    end: int
    certificate: CompletionCertificate
    z_filled: Optional[np.ndarray] = None
    solver: str = "abstain"


def project_l2(delta: np.ndarray, radius: float, eps: float = 1e-8) -> np.ndarray:
    d = delta.reshape(-1).astype(np.float32)
    norm = float(np.linalg.norm(d))
    if norm <= radius:
        return delta
    scale = radius / (norm + eps)
    return (delta * scale).astype(delta.dtype)


class PR2Controller:
    def __init__(
        self,
        tau: float,
        radius_schedule: RadiusSchedule,
        gmax_schedule: GMaxSchedule,
        lambda_schedule: LambdaSafeSchedule,
        completion_enabled: bool = True,
        completion_solver: str = "minaccel",
        completion_fallback: str = "abstain",
        max_seg_len: int = 12,
        v_max: float = 40.0,
    ) -> None:
        self.tau = float(tau)
        self.radius_schedule = radius_schedule
        self.gmax_schedule = gmax_schedule
        self.lambda_schedule = lambda_schedule

        self.completion_enabled = bool(completion_enabled)
        self.completion_solver = str(completion_solver)
        self.completion_fallback = str(completion_fallback)
        self.max_seg_len = int(max_seg_len)
        self.v_max = float(v_max)

    def make_frame_policies(self, w_tilde: Sequence[float]) -> List[FramePolicy]:
        w = np.asarray(w_tilde, dtype=np.float32)
        T = int(w.shape[0])
        policies: List[FramePolicy] = []
        for t in range(T):
            wt = float(w[t])
            policies.append(
                FramePolicy(
                    t=t,
                    w_tilde=wt,
                    update_mask=bool(wt >= self.tau),
                    radius=float(self.radius_schedule(wt)),
                    gmax=float(self.gmax_schedule(wt)),
                    lambda_safe=float(self.lambda_schedule(wt)),
                    completion=None,
                )
            )
        return policies

    def _find_low_segments(self, w: np.ndarray) -> List[Tuple[int, int]]:
        low = w < self.tau
        segs: List[Tuple[int, int]] = []
        i = 0
        T = len(w)
        while i < T:
            if not low[i]:
                i += 1
                continue
            a = i
            while i < T and low[i]:
                i += 1
            b = i - 1
            segs.append((a, b))
        return segs

    def decide_completion(
        self,
        w_tilde: Sequence[float],
        z_obs: Optional[np.ndarray] = None,
    ) -> List[SegmentDecision]:
        w = np.asarray(w_tilde, dtype=np.float32)
        T = int(w.shape[0])
        segs = self._find_low_segments(w)

        decisions: List[SegmentDecision] = []
        if not self.completion_enabled:
            for a, b in segs:
                cert = CompletionCertificate(
                    boundary_ok=False, length_ok=True, speed_ok=False, disp_ok=False,
                    feasible=False, decision="abstain", reason="completion_disabled"
                )
                decisions.append(SegmentDecision(start=a, end=b, certificate=cert, z_filled=None, solver="abstain"))
            return decisions

        for a, b in segs:
            L = b - a + 1
            z_prev = None
            z_next = None
            if z_obs is not None:
                if a - 1 >= 0:
                    z_prev = z_obs[a - 1]
                if b + 1 < T:
                    z_next = z_obs[b + 1]

            cert = check_feasibility(z_prev, z_next, seg_len=L, max_seg_len=self.max_seg_len, v_max=self.v_max)

            if not cert.feasible or z_obs is None:
                decisions.append(SegmentDecision(start=a, end=b, certificate=cert, z_filled=None, solver="abstain"))
                continue

            if self.completion_solver == "linear":
                z_fill = linear_completion(z_prev, z_next, seg_len=L)
                solver = "linear"
                cert.decision = "linear"
            else:
                z_fill = minaccel_completion(
                    z_prev=z_prev.astype(np.float32),
                    z_next=z_next.astype(np.float32),
                    seg_len=L,
                    cfg=MinAccelConfig(v_max=self.v_max),
                )
                solver = "minaccel"
                cert.decision = "minaccel"

            decisions.append(SegmentDecision(start=a, end=b, certificate=cert, z_filled=z_fill, solver=solver))

        return decisions

    def attach_completion_to_policies(
        self,
        policies: List[FramePolicy],
        seg_decisions: List[SegmentDecision],
    ) -> List[FramePolicy]:
        for sd in seg_decisions:
            for t in range(sd.start, sd.end + 1):
                if 0 <= t < len(policies):
                    policies[t].completion = {
                        "seg_start": sd.start,
                        "seg_end": sd.end,
                        "solver": sd.solver,
                        "feasible": sd.certificate.feasible,
                        "decision": sd.certificate.decision,
                        "reason": sd.certificate.reason,
                    }
        return policies
