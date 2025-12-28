from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .linear import linear_completion


@dataclass
class MinAccelConfig:
    v_max: float = 40.0
    iters: int = 200
    step: float = 0.1


def minaccel_completion(
    z_prev: np.ndarray,
    z_next: np.ndarray,
    seg_len: int,
    cfg: MinAccelConfig,
) -> np.ndarray:
    """
    “稳健优先”的 min-accel 近似求解器：
    - 目标：最小化二阶差分平方和（离散加速度）
    - 约束：每步速度 <= v_max（通过投影式迭代逼近）
    - 若你环境有 cvxpy/osqp，可后续替换为严格 QP 求解。
    """
    if seg_len <= 0:
        return np.zeros((0, z_prev.shape[-1]), dtype=np.float32)

    z = linear_completion(z_prev, z_next, seg_len).astype(np.float32)

    def full(z_inner: np.ndarray) -> np.ndarray:
        return np.concatenate([z_prev[None, :], z_inner, z_next[None, :]], axis=0)

    for _ in range(int(cfg.iters)):
        Z = full(z)
        acc = Z[2:] - 2 * Z[1:-1] + Z[:-2]  # (L, D)
        grad = np.zeros_like(z)

        for i in range(seg_len):
            a_i = acc[i]
            grad[i] += 2 * a_i
            if i - 1 >= 0:
                grad[i - 1] += -1 * a_i
            if i + 1 < seg_len:
                grad[i + 1] += -1 * a_i

        z = z - cfg.step * grad

        # project velocity constraints on full sequence
        Z = full(z)
        d = Z[1:] - Z[:-1]
        norms = np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
        scale = np.minimum(1.0, cfg.v_max / norms)
        d = d * scale

        Zp = [Z[0]]
        for i in range(d.shape[0]):
            Zp.append(Zp[-1] + d[i])
        Zp = np.stack(Zp, axis=0)
        z = Zp[1:-1].astype(np.float32)

    return z.astype(np.float32)
