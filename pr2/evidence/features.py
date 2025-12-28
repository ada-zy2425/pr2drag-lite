from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from pr2.utils.caching import DiskCache


@dataclass
class FeatureConfig:
    """
    你接入 DINO/CLIP patch embedding 后，可以把参数放这里并做 cache。
    """
    model_name: str = "demo"
    patch_size: int = 16


class FeatureExtractor:
    """
    目前提供一个“可跑通”的 demo extractor：
    - 输入 frame（H,W,3 uint8），输出一个固定维度向量（例如 128 维）。
    - 真实系统中你可以替换为 DINO/CLIP patch embeddings + pool。
    """

    def __init__(self, cfg: FeatureConfig, cache: Optional[DiskCache] = None, out_dim: int = 128) -> None:
        self.cfg = cfg
        self.cache = cache
        self.out_dim = int(out_dim)

    def _key(self, frame_id: str) -> str:
        return f"feat_{self.cfg.model_name}_{self.cfg.patch_size}_{frame_id}"

    def extract(self, frame: np.ndarray, frame_id: str) -> np.ndarray:
        if self.cache is not None and self.cache.has(self._key(frame_id)):
            return self.cache.get(self._key(frame_id))

        # demo: 做一个简单的、可复现的 hash pooling
        x = frame.astype(np.float32).reshape(-1)
        if x.size == 0:
            feat = np.zeros((self.out_dim,), dtype=np.float32)
        else:
            idx = np.linspace(0, x.size - 1, num=self.out_dim).astype(np.int64)
            feat = x[idx]
            feat = (feat - feat.mean()) / (feat.std() + 1e-6)

        if self.cache is not None:
            self.cache.set(self._key(frame_id), feat)
        return feat
