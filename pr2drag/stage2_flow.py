from __future__ import annotations
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2

from .core import (
    load_mask_uint8, choose_target_id, mask_to_binary, centroid_from_binary,
    bbox_from_binary, border_margin_from_bbox, iou_binary
)

def read_rgb(path: Path):
    return np.array(Image.open(path).convert("RGB"))

def to_gray_u8(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

def resize_gray(g, scale):
    if scale == 1.0:
        return g
    H, W = g.shape[:2]
    nh, nw = int(round(H * scale)), int(round(W * scale))
    nh = max(nh, 16); nw = max(nw, 16)
    return cv2.resize(g, (nw, nh), interpolation=cv2.INTER_AREA)

def upsample_flow(flow, out_hw):
    H, W = out_hw
    h, w = flow.shape[:2]
    if (h, w) == (H, W):
        return flow
    flow_up = cv2.resize(flow, (W, H), interpolation=cv2.INTER_LINEAR)
    sx = W / max(w, 1)
    sy = H / max(h, 1)
    flow_up[..., 0] *= sx
    flow_up[..., 1] *= sy
    return flow_up

def warp_mask_backward(prev_mask01, flow_back, grid_x, grid_y):
    H, W = prev_mask01.shape
    map_x = (grid_x + flow_back[..., 0]).astype(np.float32)
    map_y = (grid_y + flow_back[..., 1]).astype(np.float32)

    warped = cv2.remap(
        prev_mask01.astype(np.float32),
        map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0.0
    )
    pred01 = (warped >= 0.5).astype(np.uint8)

    fg = pred01 > 0
    if fg.any():
        mx = map_x[fg]; my = map_y[fg]
        oob = float(np.mean((mx < 0) | (mx > W - 1) | (my < 0) | (my > H - 1)))
    else:
        oob = 1.0
    return pred01, oob

def stage2_one_split(
    stage1_dir: Path,
    out_dir: Path,
    split_name: str,
    iou_thr: float,
    empty_ok: bool,
    flow_downscale: float,
    farneback_params: dict,
    overwrite: bool = False,
    skip_if_exists: bool = True,
    verbose_skip: bool = True,
):
    npz_paths = sorted(stage1_dir.glob("*.npz"))
    if len(npz_paths) == 0:
        raise RuntimeError(f"Stage1 empty: {stage1_dir}")
    print(f"[Stage2:{split_name}] stage1_dir={stage1_dir} num_seq={len(npz_paths)}")

    out_dir.mkdir(parents=True, exist_ok=True)

    for p in tqdm(npz_paths, desc=f"Stage2({split_name})"):
        out_path = out_dir / p.name
        if skip_if_exists and out_path.exists() and (not overwrite):
            if verbose_skip:
                print(f"[skip] Stage2 exists: {out_path.name}")
            continue

        a = np.load(p, allow_pickle=True)
        seq = str(a.get("seq", p.stem))

        frames = a["frames"]
        masks  = a["masks"]
        z_gt   = a["z_gt"].astype(np.float32)
        H = int(a["H"]); W = int(a["W"])
        tid = int(a["target_id"]) if "target_id" in a else -1

        if len(masks) == 0:
            print(f"[warn] {seq}: empty masks list, skip.")
            continue

        tid2 = None
        if tid == -1:
            m0 = load_mask_uint8(Path(str(masks[0])))
            tid2 = choose_target_id(m0)
        else:
            tid2 = int(tid)

        # load GT masks as binary
        gt_bin = []
        ok = True
        for mp in masks:
            mp = Path(str(mp))
            if not mp.exists():
                print(f"[warn] {seq}: missing mask {mp}, skip seq.")
                ok = False; break
            m = load_mask_uint8(mp)
            gt_bin.append(mask_to_binary(m, tid2))
        if not ok:
            continue

        gt_bin = np.stack(gt_bin, axis=0).astype(np.uint8)
        T = len(frames)
        if gt_bin.shape[0] != T:
            T2 = min(T, gt_bin.shape[0])
            frames = frames[:T2]
            z_gt = z_gt[:T2]
            gt_bin = gt_bin[:T2]
            T = T2

        pred_prev = gt_bin[0].copy()

        z_obs = np.zeros((T, 2), dtype=np.float32)
        E = np.zeros((T, 5), dtype=np.float32)
        chi = np.zeros((T,), dtype=np.int64)
        iou_to_gt = np.zeros((T,), dtype=np.float32)

        # precompute grid for remap
        grid_x, grid_y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))

        # t=0
        z0 = centroid_from_binary(pred_prev)
        if not np.all(np.isfinite(z0)):
            z0 = np.array([np.nan, np.nan], dtype=np.float32)
        z_obs[0] = z0

        iou0 = iou_binary(pred_prev, gt_bin[0])
        iou_to_gt[0] = iou0
        c0 = 1 if iou0 >= iou_thr else 0
        if (not empty_ok) and (gt_bin[0].sum() == 0):
            c0 = 0
        chi[0] = c0

        bm0 = border_margin_from_bbox(bbox_from_binary(pred_prev), H, W)
        pred_empty0 = 1.0 if pred_prev.sum() == 0 else 0.0
        E[0] = np.array([1.0, 0.0, bm0, 0.0, pred_empty0], dtype=np.float32)

        # iterate
        cur_T = T
        for t in range(1, T):
            f_prev = Path(str(frames[t - 1]))
            f_cur  = Path(str(frames[t]))
            if (not f_prev.exists()) or (not f_cur.exists()):
                print(f"[warn] {seq}: missing frame at t={t}, truncate.")
                cur_T = t
                break

            rgb0 = read_rgb(f_prev)
            rgb1 = read_rgb(f_cur)
            g0 = to_gray_u8(rgb0)
            g1 = to_gray_u8(rgb1)

            g0s = resize_gray(g0, flow_downscale)
            g1s = resize_gray(g1, flow_downscale)

            # BACKWARD flow: cur -> prev
            flow_back_s = cv2.calcOpticalFlowFarneback(g1s, g0s, None, **farneback_params).astype(np.float32)
            flow_back = upsample_flow(flow_back_s, (H, W))

            pred_t, oob_ratio = warp_mask_backward(pred_prev, flow_back, grid_x, grid_y)

            iou_pred_prev = iou_binary(pred_t, pred_prev)
            area_t = float(pred_t.sum())
            area_prev = float(pred_prev.sum())
            lac = float(abs(np.log((area_t + 1.0) / (area_prev + 1.0))))
            bm = border_margin_from_bbox(bbox_from_binary(pred_t), H, W)
            pred_empty = 1.0 if area_t == 0 else 0.0

            E[t] = np.array([iou_pred_prev, lac, bm, oob_ratio, pred_empty], dtype=np.float32)

            zt = centroid_from_binary(pred_t)
            if not np.all(np.isfinite(zt)):
                zt = z_obs[t - 1].copy()
            z_obs[t] = zt

            iou_gt = iou_binary(pred_t, gt_bin[t])
            iou_to_gt[t] = iou_gt
            ct = 1 if iou_gt >= iou_thr else 0
            if (not empty_ok) and (gt_bin[t].sum() == 0):
                ct = 0
            chi[t] = ct

            pred_prev = pred_t

        cur_T = int(cur_T)
        np.savez_compressed(
            out_path,
            seq=seq,
            z_gt=z_gt[:cur_T].astype(np.float32),
            z_obs=z_obs[:cur_T].astype(np.float32),
            E_min=E[:cur_T].astype(np.float32),
            chi=chi[:cur_T].astype(np.int64),
            iou_to_gt=iou_to_gt[:cur_T].astype(np.float32),
            H=H, W=W,
        )
