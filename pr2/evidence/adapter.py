from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np


@dataclass
class EvidenceOutput:
    """
    E: (T, D) 证据向量
    missing: (T, M) 缺失指示（0/1），用于让 posterior 在缺失时仍可工作
    meta: 任何调试信息（可选）
    """
    E: np.ndarray
    missing: np.ndarray
    meta: Dict[str, Any]


class EvidenceAdapter:
    """
    证据适配器：把 (video, tracker, mask...) 映射为固定维度 evidence E_t.

    注意：这里不强行绑定 DINO/CLIP/光流等，避免把方法写死。
    你可以在接入真实系统后在 pr2/evidence/ 下扩展实现。
    """

    def __init__(self, D: int = 8, M: int = 4) -> None:
        self.D = int(D)
        self.M = int(M)

    def build(
        self,
        frames: Optional[Sequence[np.ndarray]],
        tracker: Optional[Dict[str, Any]],
        masks: Optional[Sequence[np.ndarray]],
    ) -> EvidenceOutput:
        """
        返回一个可工作的默认实现（用于 demo 跑通）：
        - 真实实验中你应替换为：FB cycle error、patch feature drift、visibility flags、mask stability 等。
        """
        T = 0
        if frames is not None:
            T = len(frames)
        elif tracker is not None and "T" in tracker:
            T = int(tracker["T"])
        elif masks is not None:
            T = len(masks)
        else:
            raise ValueError("EvidenceAdapter.build: 无法推断 T（frames/tracker/masks 都是 None 或缺少长度信息）")

        # demo: 生成一些稳定的“伪证据”，保证可复现实验框架
        rng = np.random.default_rng(0)
        E = rng.normal(size=(T, self.D)).astype(np.float32)

        # missing flags：假设某些帧可见性缺失（演示用）
        missing = np.zeros((T, self.M), dtype=np.float32)
        if T > 0:
            missing[:, 0] = 0.0  # tracker ok
            missing[:, 1] = 0.0  # features ok
            missing[:, 2] = 1.0 * (np.arange(T) % 11 == 0)  # occasional occlusion flag
            missing[:, 3] = 1.0 if masks is None else 0.0

        meta = {"note": "demo evidence; replace with real evidence in your integration"}
        return EvidenceOutput(E=E, missing=missing, meta=meta)
