from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pr2.editors.base import Task, EditorOutput
from pr2.policy.controller import PR2Controller, project_l2


@dataclass
class DummyEditor:
    """
    Dummy editor：用于跑通实验框架，不做真实编辑。
    """
    name: str = "dummy"

    def run_task(self, task: Task, controller: PR2Controller, seed: int, out_dir: Path) -> EditorOutput:
        rng = np.random.default_rng(seed + (hash(task.task_id) % 10000))
        T = int(task.T)

        # demo posterior: 高低交替 + 噪声
        w = np.clip(0.7 + 0.25 * np.sin(np.linspace(0, 3.14 * 2, T)) + rng.normal(scale=0.05, size=T), 0.0, 1.0)
        policies = controller.make_frame_policies(w_tilde=w)
        segs = controller.decide_completion(w_tilde=w, z_obs=None)
        policies = controller.attach_completion_to_policies(policies, segs)

        frame_logs = []
        catastrophic = 0
        flicker_proxy = []

        state = rng.normal(size=(32,)).astype(np.float32)
        for t in range(T):
            pol = policies[t]
            delta = rng.normal(size=state.shape).astype(np.float32)
            if rng.random() < 0.1:
                delta *= 20.0  # spike

            delta_proj = project_l2(delta, radius=pol.radius)

            if pol.update_mask:
                state = state + delta_proj

            update_norm = float(np.linalg.norm(delta))
            proj_norm = float(np.linalg.norm(delta_proj))
            flicker_proxy.append(proj_norm)

            if (pol.w_tilde < controller.tau) and (proj_norm > 0.8 * controller.radius_schedule(controller.tau)):
                catastrophic = 1

            frame_logs.append({
                "t": t,
                "w_tilde": float(pol.w_tilde),
                "update_mask": bool(pol.update_mask),
                "radius": float(pol.radius),
                "gmax": float(pol.gmax),
                "lambda_safe": float(pol.lambda_safe),
                "completion": pol.completion,
                "update_norm_raw": update_norm,
                "update_norm_proj": proj_norm,
            })

        success = 1 if float(np.mean(w)) > 0.55 else 0

        result = {
            "task_id": task.task_id,
            "difficulty": task.meta.get("difficulty", "unknown"),
            "intent_success": int(success),
            "catastrophic": int(catastrophic),
            "flicker_p95": float(np.percentile(np.array(flicker_proxy, dtype=np.float32), 95)),
            "coverage": float(np.mean([1.0 if p.update_mask else 0.0 for p in policies])),
        }

        return EditorOutput(result=result, frame_logs=frame_logs, artifacts_dir=None)
