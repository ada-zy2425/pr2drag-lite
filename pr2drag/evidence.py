from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import imageio.v2 as imageio
import numpy as np
from scipy.ndimage import laplace


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img.astype(np.float32)
    if img.ndim == 3 and img.shape[2] >= 3:
        r, g, b = img[..., 0], img[..., 1], img[..., 2]
        return (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float32)
    return img.astype(np.float32)


def _read_mask(path: Path) -> np.ndarray:
    m = imageio.imread(path)
    if m.ndim == 3:
        m = m[..., 0]
    # DAVIS mask: 0 background, 255 foreground (or 1)
    return (m > 0).astype(np.uint8)


def _mask_centroid(mask: np.ndarray) -> Tuple[float, float, bool]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return 0.0, 0.0, False
    return float(xs.mean()), float(ys.mean()), True


def _shift_mask(mask: np.ndarray, dx: int, dy: int) -> np.ndarray:
    # zero-padded shift (not circular roll)
    h, w = mask.shape[:2]
    out = np.zeros_like(mask)
    x0_src = max(0, -dx)
    x1_src = min(w, w - dx)
    y0_src = max(0, -dy)
    y1_src = min(h, h - dy)
    x0_dst = max(0, dx)
    x1_dst = min(w, w + dx)
    y0_dst = max(0, dy)
    y1_dst = min(h, h + dy)
    if x1_src <= x0_src or y1_src <= y0_src:
        return out
    out[y0_dst:y1_dst, x0_dst:x1_dst] = mask[y0_src:y1_src, x0_src:x1_src]
    return out


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(inter / union)


def blur_inv_lapvar(img: np.ndarray, eps: float = 1.0) -> float:
    g = _to_gray(img)
    lv = laplace(g).var()
    # invert: larger means blurrier (matches your dim5 ~ 10..60 style)
    return float(1000.0 / (lv + eps))


@dataclass
class SeqEvidence:
    seq: str
    frames: List[str]          # absolute frame paths
    annos: List[str]           # absolute mask paths
    z_gt: np.ndarray           # (T,2)
    has_gt: np.ndarray         # (T,)
    img_h: int
    img_w: int

    # evidence features (per-frame)
    iou_shift: np.ndarray      # (T,)
    cycle_err: np.ndarray      # (T,) in px
    area_change: np.ndarray    # (T,)
    occl_flag: np.ndarray      # (T,) {0,1}
    blur_flag: np.ndarray      # (T,) {0,1}
    blur_inv: np.ndarray       # (T,)
    motion: np.ndarray         # (T,) px

    # synthetic tracker
    z_obs: np.ndarray          # (T,2)
    err_obs: np.ndarray        # (T,) px
    y: np.ndarray              # (T,) {0,1}


def build_sequence_evidence(
    seq: str,
    frame_paths: List[Path],
    anno_paths: List[Path],
    *,
    label_err_thresh: float,
    noise_cfg: Dict,
    rng: np.random.RandomState,
) -> SeqEvidence:
    assert len(frame_paths) == len(anno_paths)
    T = len(frame_paths)

    # load first to get shape
    img0 = imageio.imread(frame_paths[0])
    h, w = img0.shape[0], img0.shape[1]

    z_gt = np.zeros((T, 2), dtype=np.float32)
    has_gt = np.zeros((T,), dtype=np.uint8)
    masks: List[np.ndarray] = []

    blur_inv = np.zeros((T,), dtype=np.float32)

    for t in range(T):
        m = _read_mask(anno_paths[t])
        masks.append(m)
        cx, cy, ok = _mask_centroid(m)
        z_gt[t, 0] = cx
        z_gt[t, 1] = cy
        has_gt[t] = 1 if ok else 0

        img = imageio.imread(frame_paths[t])
        blur_inv[t] = blur_inv_lapvar(img)

    # derived signals
    motion = np.zeros((T,), dtype=np.float32)
    area = np.array([m.sum() for m in masks], dtype=np.float32) / float(h * w + 1e-6)

    for t in range(1, T):
        if has_gt[t] and has_gt[t - 1]:
            dx = z_gt[t, 0] - z_gt[t - 1, 0]
            dy = z_gt[t, 1] - z_gt[t - 1, 1]
            motion[t] = float(np.sqrt(dx * dx + dy * dy))
        else:
            motion[t] = 0.0

    area_change = np.zeros((T,), dtype=np.float32)
    area_change[1:] = np.abs(area[1:] - area[:-1]) / (np.maximum(area[:-1], 1e-6))

    # occlusion proxy: tiny mask
    occl_flag = (area < 0.002).astype(np.uint8)

    # blur proxy: top tail of blur_inv => blurrier
    blur_thr = float(np.quantile(blur_inv, 0.90)) if T >= 10 else float(blur_inv.max())
    blur_flag = (blur_inv >= blur_thr).astype(np.uint8)

    # iou_shift: align previous mask by centroid delta (translation only)
    iou_shift = np.zeros((T,), dtype=np.float32)
    iou_shift[0] = 1.0
    for t in range(1, T):
        if has_gt[t] and has_gt[t - 1]:
            dx = int(round(z_gt[t, 0] - z_gt[t - 1, 0]))
            dy = int(round(z_gt[t, 1] - z_gt[t - 1, 1]))
            m_prev_shift = _shift_mask(masks[t - 1], dx=dx, dy=dy)
            iou_shift[t] = mask_iou(m_prev_shift, masks[t])
        else:
            iou_shift[t] = 0.0

    # cycle_err: constant velocity prediction error (pixel)
    cycle_err = np.zeros((T,), dtype=np.float32)
    for t in range(2, T):
        if has_gt[t] and has_gt[t - 1] and has_gt[t - 2]:
            vx = z_gt[t - 1, 0] - z_gt[t - 2, 0]
            vy = z_gt[t - 1, 1] - z_gt[t - 2, 1]
            predx = z_gt[t - 1, 0] + vx
            predy = z_gt[t - 1, 1] + vy
            ex = z_gt[t, 0] - predx
            ey = z_gt[t, 1] - predy
            cycle_err[t] = float(np.sqrt(ex * ex + ey * ey))
        else:
            cycle_err[t] = 0.0

    # synthetic tracker: noise depends on occl/blur/motion
    base_sigma = float(noise_cfg.get("base_sigma", 4.0))
    occl_sigma = float(noise_cfg.get("occl_sigma", 220.0))
    blur_sigma = float(noise_cfg.get("blur_sigma", 120.0))
    motion_sigma = float(noise_cfg.get("motion_sigma", 0.35))

    z_obs = np.zeros_like(z_gt, dtype=np.float32)
    err_obs = np.zeros((T,), dtype=np.float32)
    y = np.zeros((T,), dtype=np.uint8)

    for t in range(T):
        sigma = base_sigma
        sigma += occl_sigma * float(occl_flag[t])
        sigma += blur_sigma * float(blur_flag[t])
        sigma += motion_sigma * float(motion[t])

        noise = rng.normal(loc=0.0, scale=sigma, size=(2,)).astype(np.float32)
        z_obs[t] = z_gt[t] + noise

        # clip inside frame
        z_obs[t, 0] = float(np.clip(z_obs[t, 0], 0.0, w - 1.0))
        z_obs[t, 1] = float(np.clip(z_obs[t, 1], 0.0, h - 1.0))

        if has_gt[t]:
            e = z_obs[t] - z_gt[t]
            err = float(np.sqrt(float(e[0] * e[0] + e[1] * e[1])))
        else:
            err = float("inf")
        err_obs[t] = err
        y[t] = 1 if (has_gt[t] and err <= label_err_thresh) else 0

    return SeqEvidence(
        seq=seq,
        frames=[str(p) for p in frame_paths],
        annos=[str(p) for p in anno_paths],
        z_gt=z_gt,
        has_gt=has_gt,
        img_h=h,
        img_w=w,
        iou_shift=iou_shift,
        cycle_err=cycle_err,
        area_change=area_change,
        occl_flag=occl_flag,
        blur_flag=blur_flag,
        blur_inv=blur_inv,
        motion=motion,
        z_obs=z_obs,
        err_obs=err_obs,
        y=y,
    )
