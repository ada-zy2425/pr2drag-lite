from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import imageio.v2 as imageio
from tqdm import tqdm

from .utils import (
    ensure_dir,
    get_logger,
    read_split_list,
    save_json,
    load_npz,
    save_npz_atomic,
    signature_for_stage,
    is_compatible_npz,
    stable_hash_dict,
)
from .evidence import build_features
from .sp import fit_emission, build_hmm_params, hmm_smooth_binary
from .aob import apply_aob
from .eval import compute_emission_metrics, per_seq_summary, write_analysis


def _davis_frame_paths(davis_root: str, res: str, seq: str) -> Tuple[List[str], List[str]]:
    img_dir = os.path.join(davis_root, "JPEGImages", res, seq)
    ann_dir = os.path.join(davis_root, "Annotations", res, seq)
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"Missing DAVIS images dir: {img_dir}")
    if not os.path.isdir(ann_dir):
        raise FileNotFoundError(f"Missing DAVIS annotations dir: {ann_dir}")

    # DAVIS uses 5-digit names
    imgs = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    anns = sorted([os.path.join(ann_dir, f) for f in os.listdir(ann_dir) if f.lower().endswith((".png", ".jpg"))])

    if len(imgs) == 0 or len(anns) == 0:
        raise RuntimeError(f"Empty sequence folders: {seq}")
    if len(imgs) != len(anns):
        # Still proceed but align by basename
        ann_map = {os.path.splitext(os.path.basename(p))[0]: p for p in anns}
        aligned_anns = []
        aligned_imgs = []
        for ip in imgs:
            k = os.path.splitext(os.path.basename(ip))[0]
            if k in ann_map:
                aligned_imgs.append(ip)
                aligned_anns.append(ann_map[k])
        imgs, anns = aligned_imgs, aligned_anns
    return imgs, anns


def _load_rgb(path: str) -> np.ndarray:
    try:
        im = imageio.imread(path)
        if im.ndim == 2:
            im = np.stack([im, im, im], axis=-1)
        if im.shape[-1] == 4:
            im = im[..., :3]
        return im.astype(np.uint8)
    except Exception:
        im = Image.open(path).convert("RGB")
        return np.array(im, dtype=np.uint8)


def _load_mask(path: str) -> np.ndarray:
    try:
        m = imageio.imread(path)
    except Exception:
        m = np.array(Image.open(path), dtype=np.uint8)
    if m.ndim == 3:
        m = m[..., 0]
    return (m > 0).astype(np.uint8)


def _centroid_from_mask(mask: np.ndarray) -> Tuple[float, float, bool]:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return float("nan"), float("nan"), True
    return float(xs.mean()), float(ys.mean()), False


def _to_gray(img: np.ndarray) -> np.ndarray:
    # img uint8 HxWx3
    img = img.astype(np.float32) / 255.0
    return 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]


def _hann2d(H: int, W: int) -> np.ndarray:
    wy = np.hanning(H)
    wx = np.hanning(W)
    return np.outer(wy, wx).astype(np.float32)


def _phase_correlation_shift(a: np.ndarray, b: np.ndarray, use_hann: bool = True, eps: float = 1e-9) -> Tuple[float, float, float]:
    """
    Estimate global translation shift from a->b using phase correlation.
    Returns (dx, dy, peak) where dx,dy are in pixels.
    """
    # a,b: gray float32
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    H, W = a.shape
    if use_hann:
        win = _hann2d(H, W)
        a = a * win
        b = b * win
    Fa = np.fft.fft2(a)
    Fb = np.fft.fft2(b)
    R = Fa * np.conj(Fb)
    denom = np.abs(R)
    R = R / (denom + eps)
    r = np.fft.ifft2(R)
    r = np.abs(r)
    peak = float(np.max(r))
    maxpos = np.unravel_index(np.argmax(r), r.shape)
    dy, dx = int(maxpos[0]), int(maxpos[1])
    # wrap-around
    if dy > H // 2:
        dy -= H
    if dx > W // 2:
        dx -= W
    return float(dx), float(dy), peak


def stage1_precompute_split(cfg: Dict[str, Any], split_name: str, seqs: List[str], out_dir: str) -> None:
    logger = get_logger()
    davis_root = cfg["paths"]["davis_root"]
    base_out = cfg["paths"]["base_out"]
    res = cfg["res"]
    feat_set = cfg["stage1"]["feat_set"]
    pcfg = cfg["stage1"].get("phasecorr", {})
    use_hann = bool(pcfg.get("use_hann", True))
    eps = float(pcfg.get("eps", 1e-9))

    out_abs = os.path.join(base_out, out_dir)
    ensure_dir(out_abs)

    sig = signature_for_stage("stage1", cfg, extra={"split": split_name, "out_dir": out_dir})

    meta = {
        "split": split_name,
        "num_seqs": len(seqs),
        "res": res,
        "feat_set": feat_set,
        "signature": sig,
    }
    save_json(os.path.join(out_abs, "meta.json"), meta)

    pbar = tqdm(seqs, desc=f"Stage1({split_name})")
    for seq in pbar:
        npz_path = os.path.join(out_abs, f"{seq}.npz")
        if is_compatible_npz(npz_path, sig):
            logger.info(f"[skip] Stage1 exists: {seq}.npz")
            continue

        img_paths, ann_paths = _davis_frame_paths(davis_root, res, seq)
        T = len(img_paths)

        z_gt = np.zeros((T, 2), dtype=np.float32)
        mask_missing = np.zeros((T,), dtype=np.float32)
        frames_gray = []

        H = W = None
        for t in range(T):
            img = _load_rgb(img_paths[t])
            m = _load_mask(ann_paths[t])
            if H is None:
                H, W = img.shape[0], img.shape[1]
            x, y, miss = _centroid_from_mask(m)
            z_gt[t] = [x if np.isfinite(x) else 0.0, y if np.isfinite(y) else 0.0]
            mask_missing[t] = 1.0 if miss else 0.0
            frames_gray.append(_to_gray(img))

        frames_gray = np.stack(frames_gray, axis=0)  # (T,H,W)

        # Propagate observed state using consecutive global shifts
        z_obs = np.zeros((T, 2), dtype=np.float32)
        z_obs[0] = z_gt[0]
        peak = np.zeros((T,), dtype=np.float32)
        cycle = np.zeros((T,), dtype=np.float32)

        cum_dx = 0.0
        cum_dy = 0.0
        prev_dx = 0.0
        prev_dy = 0.0

        peak[0] = 1.0
        cycle[0] = 0.0

        for t in range(1, T):
            dx, dy, pk = _phase_correlation_shift(frames_gray[t - 1], frames_gray[t], use_hann=use_hann, eps=eps)
            # backward consistency
            bdx, bdy, _ = _phase_correlation_shift(frames_gray[t], frames_gray[t - 1], use_hann=use_hann, eps=eps)
            cyc = np.sqrt((dx + bdx) ** 2 + (dy + bdy) ** 2)

            cum_dx += dx
            cum_dy += dy
            z_obs[t] = z_obs[t - 1] + np.array([dx, dy], dtype=np.float32)

            peak[t] = float(pk)
            cycle[t] = float(cyc)

            prev_dx, prev_dy = dx, dy

        # out_of_frame flag
        out_of_frame = np.zeros((T,), dtype=np.float32)
        out_of_frame[(z_obs[:, 0] < 0) | (z_obs[:, 0] >= W) | (z_obs[:, 1] < 0) | (z_obs[:, 1] >= H)] = 1.0

        # motion stats
        speed = np.zeros((T,), dtype=np.float32)
        accel = np.zeros((T,), dtype=np.float32)
        for t in range(1, T):
            d = z_obs[t] - z_obs[t - 1]
            speed[t] = float(np.sqrt((d ** 2).sum()))
        for t in range(2, T):
            dd = z_obs[t] - 2 * z_obs[t - 1] + z_obs[t - 2]
            accel[t] = float(np.sqrt((dd ** 2).sum()))

        # normalize peak into [0,1] (robust)
        pk = peak.copy()
        pk = np.clip(pk, 0.0, np.quantile(pk, 0.99) + 1e-6)
        pk = pk / (pk.max() + 1e-6)

        E, feat_names = build_features(
            feat_set=feat_set,
            peak=pk,
            cycle=cycle,
            out_of_frame=out_of_frame,
            mask_missing=mask_missing,
            speed=speed,
            accel=accel,
        )

        # Observed error (proxy task): how far z_obs from z_gt
        err_obs = np.sqrt(((z_obs - z_gt) ** 2).sum(axis=1)).astype(np.float32)

        save_npz_atomic(
            npz_path,
            signature=sig,
            seq=seq,
            res=res,
            H=int(H),
            W=int(W),
            T=int(T),
            z_gt=z_gt,
            z_obs=z_obs,
            err_obs=err_obs,
            E=E.astype(np.float32),
            feat_names=np.array(feat_names, dtype=object),
            peak=pk.astype(np.float32),
            cycle=cycle.astype(np.float32),
            out_of_frame=out_of_frame.astype(np.float32),
            mask_missing=mask_missing.astype(np.float32),
            speed=speed.astype(np.float32),
            accel=accel.astype(np.float32),
        )

    logger.info(f"[OK] Stage1({split_name}) meta: {os.path.join(out_abs,'meta.json')}")
    logger.info(f"[OK] Stage1({split_name}) npz_dir: {out_abs}  num_npz={len(seqs)}")


def stage2_build(cfg: Dict[str, Any], train_seqs: List[str], val_seqs: List[str]) -> Tuple[str, str, Dict[str, Any]]:
    """
    Stage2 writes per-seq npz with labels y and stores a train-derived label threshold in meta.
    Returns (train_dir, val_dir, meta)
    """
    logger = get_logger()
    base_out = cfg["paths"]["base_out"]

    s1_train = os.path.join(base_out, cfg["stage1"]["out_train"])
    s1_val = os.path.join(base_out, cfg["stage1"]["out_val"])
    s2_train = os.path.join(base_out, cfg["stage2"]["out_train"])
    s2_val = os.path.join(base_out, cfg["stage2"]["out_val"])
    ensure_dir(s2_train)
    ensure_dir(s2_val)

    sig_train = signature_for_stage("stage2", cfg, extra={"split": "train"})
    sig_val = signature_for_stage("stage2", cfg, extra={"split": "val"})

    label_cfg = cfg["stage2"]["label"]
    mode = str(label_cfg.get("mode", "quantile")).lower()
    q_pos = float(label_cfg.get("q_pos", 0.455))
    fixed_thr = float(label_cfg.get("fixed_err_thresh", 30.0))

    # Determine threshold from train distribution
    train_errs = []
    for seq in train_seqs:
        z = load_npz(os.path.join(s1_train, f"{seq}.npz"))
        train_errs.append(z["err_obs"].astype(np.float64))
    all_train_err = np.concatenate(train_errs, axis=0) if train_errs else np.array([], dtype=np.float64)

    if all_train_err.size == 0:
        raise RuntimeError("Stage2: empty train_errs; check DAVIS paths and Stage1 outputs.")

    if mode == "fixed":
        err_thr = fixed_thr
    elif mode == "quantile":
        # positive = "usable" = small error; choose threshold such that pos_rate ~= q_pos
        err_thr = float(np.quantile(all_train_err, q_pos))
    else:
        raise ValueError(f"Unknown label.mode={mode}. Use fixed|quantile.")

    meta = {
        "label_mode": mode,
        "q_pos": q_pos,
        "fixed_err_thresh": fixed_thr,
        "err_thresh": float(err_thr),
        "train_sig": sig_train,
        "val_sig": sig_val,
    }

    # Write train per-seq
    logger.info(f"[Stage2:train] stage1_dir={s1_train} num_seq={len(train_seqs)}")
    for seq in tqdm(train_seqs, desc="Stage2(train)"):
        in_npz = os.path.join(s1_train, f"{seq}.npz")
        out_npz = os.path.join(s2_train, f"{seq}.npz")
        if is_compatible_npz(out_npz, sig_train):
            logger.info(f"[skip] Stage2 exists (compatible): {seq}.npz")
            continue
        z = load_npz(in_npz)
        E = z["E"].astype(np.float32)
        err_obs = z["err_obs"].astype(np.float32)
        y = (err_obs <= err_thr).astype(np.int64)
        save_npz_atomic(
            out_npz,
            signature=sig_train,
            seq=seq,
            E=E,
            y=y,
            err_obs=err_obs,
            z_gt=z["z_gt"].astype(np.float32),
            z_obs=z["z_obs"].astype(np.float32),
            feat_names=z["feat_names"],
        )

    # Write val per-seq (use same err_thr)
    logger.info(f"[Stage2:val] stage1_dir={s1_val} num_seq={len(val_seqs)}")
    for seq in tqdm(val_seqs, desc="Stage2(val)"):
        in_npz = os.path.join(s1_val, f"{seq}.npz")
        out_npz = os.path.join(s2_val, f"{seq}.npz")
        if is_compatible_npz(out_npz, sig_val):
            logger.info(f"[skip] Stage2 exists (compatible): {seq}.npz")
            continue
        z = load_npz(in_npz)
        E = z["E"].astype(np.float32)
        err_obs = z["err_obs"].astype(np.float32)
        y = (err_obs <= err_thr).astype(np.int64)
        save_npz_atomic(
            out_npz,
            signature=sig_val,
            seq=seq,
            E=E,
            y=y,
            err_obs=err_obs,
            z_gt=z["z_gt"].astype(np.float32),
            z_obs=z["z_obs"].astype(np.float32),
            feat_names=z["feat_names"],
        )

    save_json(os.path.join(s2_train, "meta.json"), meta)
    logger.info(f"[OK] Stage2 done: train_npz={len(train_seqs)} val_npz={len(val_seqs)}")
    return s2_train, s2_val, meta


def _load_stage2_split(stage2_dir: str, seqs: List[str]) -> Dict[str, Dict[str, Any]]:
    data = {}
    for seq in seqs:
        data[seq] = load_npz(os.path.join(stage2_dir, f"{seq}.npz"))
    return data


def stage3_run(cfg: Dict[str, Any], train_seqs: List[str], val_seqs: List[str]) -> None:
    logger = get_logger()
    base_out = cfg["paths"]["base_out"]

    s2_train = os.path.join(base_out, cfg["stage2"]["out_train"])
    s2_val = os.path.join(base_out, cfg["stage2"]["out_val"])
    meta2 = None
    try:
        import json
        with open(os.path.join(s2_train, "meta.json"), "r", encoding="utf-8") as f:
            meta2 = json.load(f)
    except Exception:
        meta2 = {}

    out_dir = os.path.join(base_out, cfg["stage3"]["out_dir"])
    ensure_dir(out_dir)

    # Load per-seq stage2 data
    tr = _load_stage2_split(s2_train, train_seqs)
    va = _load_stage2_split(s2_val, val_seqs)

    logger.info(f"[Stage3] train seq: {len(tr)} val seq: {len(va)}")

    # Build frame-level datasets
    X_tr = np.concatenate([tr[s]["E"] for s in train_seqs], axis=0)
    y_tr = np.concatenate([tr[s]["y"] for s in train_seqs], axis=0)

    X_va = np.concatenate([va[s]["E"] for s in val_seqs], axis=0)
    y_va = np.concatenate([va[s]["y"] for s in val_seqs], axis=0)

    logger.info(f"[Data] X_tr {X_tr.shape} pos_rate {float(y_tr.mean())}")
    logger.info(f"[Data] X_va {X_va.shape} pos_rate {float(y_va.mean())}")

    # Keep dims (drop all-zero columns if any)
    keep = np.any(np.abs(X_tr) > 1e-12, axis=0)
    X_tr2 = X_tr[:, keep]
    X_va2 = X_va[:, keep]
    logger.info(f"[Feat] keep dims: {keep} num_keep: {int(keep.sum())}")

    # Fit emission
    em_cfg = cfg["stage3"]["emission"]
    model = fit_emission(X_tr2, y_tr, em_cfg)

    p_tr = model.predict_proba(X_tr2)
    p_va = model.predict_proba(X_va2)

    # Emission metrics on val (frame-level)
    m = compute_emission_metrics(p_va, y_va, bins=int(cfg["stage3"]["analysis"].get("bins", 10)))
    logger.info(f"[Emission] AUROC={m['AUROC']:.4f}  ECE={m['ECE']:.4f}  risk@50%={m['risk@50%']:.4f}")

    # HMM params
    A, pi = build_hmm_params(cfg["stage3"]["hmm"])

    # Per-seq smoothed posterior
    w_tr_seqs = {}
    w_va_seqs = {}

    def smooth_split(split: Dict[str, Dict[str, Any]], seqs: List[str], keep_mask: np.ndarray) -> Dict[str, np.ndarray]:
        out = {}
        for s in seqs:
            E = split[s]["E"][:, keep_mask].astype(np.float64)
            p = model.predict_proba(E)
            w = hmm_smooth_binary(p, A=A, pi=pi)
            out[s] = w.astype(np.float64)
        return out

    w_tr_seqs = smooth_split(tr, train_seqs, keep)
    w_va_seqs = smooth_split(va, val_seqs, keep)

    # Choose tau by train global quantile
    tau_cfg = cfg["stage3"]["tau"]
    mode = str(tau_cfg.get("mode", "global_quantile")).lower()
    target_frac = float(tau_cfg.get("target_frac", 0.25))

    all_w_tr = np.concatenate([w_tr_seqs[s] for s in train_seqs], axis=0)
    if mode == "global_quantile":
        tau = float(np.quantile(all_w_tr, target_frac))
    else:
        raise ValueError(f"Unknown tau.mode={mode}")

    logger.info(f"[Tau] global_quantile from train: tau_global={tau:.6f} (target_frac={target_frac})")

    # AoB + eval
    fail_thr = float(cfg["stage3"]["eval"].get("fail_err_thresh", 50.0))
    aob_cfg = cfg["stage3"]["aob"]
    # If eps_gate>=1.0, enforce pure abstain (bridge disabled)
    if float(aob_cfg.get("eps_gate", 1.0)) >= 1.0:
        aob_cfg = dict(aob_cfg)
        bridge = dict(aob_cfg.get("bridge") or {})
        bridge["enabled"] = False
        aob_cfg["bridge"] = bridge

    rows = []
    w_all_val = []
    err_all_val = []
    probs_all_val = []
    y_all_val = []

    # For feat stats on train
    feat_stats = {}
    Xtr_kept = X_tr[:, keep]
    for d in range(Xtr_kept.shape[1]):
        col = Xtr_kept[:, d].astype(np.float64)
        feat_stats[f"dim{d}"] = {
            "mean": float(np.mean(col)),
            "std": float(np.std(col)),
            "q1": float(np.quantile(col, 0.25)),
            "q3": float(np.quantile(col, 0.75)),
            "iqr": float(np.quantile(col, 0.75) - np.quantile(col, 0.25)),
            "min": float(np.min(col)),
            "max": float(np.max(col)),
            "frac_zero": float(np.mean(col == 0.0)),
        }

    for seq in tqdm(val_seqs, desc="Stage3(val)"):
        dat = va[seq]
        w = w_va_seqs[seq]
        z_gt = dat["z_gt"].astype(np.float64)
        z_obs = dat["z_obs"].astype(np.float64)
        err_obs = dat["err_obs"].astype(np.float64)

        # Build z_bar (proxy) as z_obs; AoB operates on z_bar to "fix" unreliable segments
        z_bar = z_obs.copy()

        z_fin, stats = apply_aob(w=w, z_bar=z_bar, tau=tau, cfg_aob=aob_cfg)
        z_fin = z_fin.astype(np.float64)
        err_fin = np.sqrt(((z_fin - z_gt) ** 2).sum(axis=1)).astype(np.float64)

        low_mask = (w < tau)
        # bridged/abstained masks from decisions
        bridged_mask = np.zeros_like(low_mask, dtype=bool)
        abstained_mask = np.zeros_like(low_mask, dtype=bool)
        for seg in stats["segments"]:
            s0, e0 = int(seg["start"]), int(seg["end"])
            if seg["kind"] == "bridge":
                bridged_mask[s0:e0 + 1] = True
            else:
                abstained_mask[s0:e0 + 1] = True

        row = per_seq_summary(
            seq=seq,
            tau=tau,
            w=w,
            err_obs=err_obs,
            err_fin=err_fin,
            low_mask=low_mask,
            bridged_mask=bridged_mask,
            abstained_mask=abstained_mask,
            fail_thr=fail_thr,
        )
        rows.append(row)

        # Collect for global analysis
        w_all_val.append(w)
        err_all_val.append(err_obs)

        # emission probs for this seq (calibrated)
        E = dat["E"][:, keep].astype(np.float64)
        p = model.predict_proba(E)
        probs_all_val.append(p)
        y_all_val.append(dat["y"].astype(np.int64))

    table = pd.DataFrame(rows)
    w_all_val = np.concatenate(w_all_val, axis=0) if w_all_val else np.array([], dtype=np.float64)
    err_all_val = np.concatenate(err_all_val, axis=0) if err_all_val else np.array([], dtype=np.float64)
    probs_all_val = np.concatenate(probs_all_val, axis=0) if probs_all_val else np.array([], dtype=np.float64)
    y_all_val = np.concatenate(y_all_val, axis=0) if y_all_val else np.array([], dtype=np.int64)

    # Aggregates (match your log style)
    logger.info("\n[Val aggregate]")
    logger.info(f"mean obs_p95 {float(table['all_obs_p95'].mean())} mean fin_p95 {float(table['all_fin_p95'].mean())}")
    logger.info(f"mean obs_fail {float(table['all_obs_fail'].mean())} mean fin_fail {float(table['all_fin_fail'].mean())}")
    logger.info(f"mean low_frac {float(table['low_frac'].mean())} mean bridged_frames {float(table['bridge_n'].mean())}")

    # Write analysis outputs
    write_analysis(
        out_dir=out_dir,
        table=table,
        w_all=w_all_val,
        err_all=err_all_val,
        y_all=y_all_val,
        probs_all=probs_all_val,
        tau=tau,
        feat_stats=feat_stats,
        cfg_analysis=cfg["stage3"]["analysis"],
    )

    logger.info(f"[OK] wrote stage3 to: {out_dir}")


def run_all(cfg: Dict[str, Any]) -> None:
    logger = get_logger()

    davis_root = cfg["paths"]["davis_root"]
    base_out = cfg["paths"]["base_out"]
    res = cfg["res"]

    print("\n========== PR2-Drag Lite ==========")
    print(f"cmd       : run_all")
    print(f"config    : (loaded yaml)")
    print(f"davis_root : {davis_root}")
    print(f"res       : {res}")
    print(f"base_out  : {base_out}")
    print("===================================\n")

    train_seqs = read_split_list(davis_root, cfg["splits"]["train"])
    val_seqs = read_split_list(davis_root, cfg["splits"]["val"])

    # Stage1
    stage1_precompute_split(cfg, "train", train_seqs, cfg["stage1"]["out_train"])
    stage1_precompute_split(cfg, "val", val_seqs, cfg["stage1"]["out_val"])

    # Stage2
    stage2_build(cfg, train_seqs, val_seqs)

    # Stage3
    stage3_run(cfg, train_seqs, val_seqs)
