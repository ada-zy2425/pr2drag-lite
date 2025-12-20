# pr2drag/pipeline.py
from __future__ import annotations

import os

debug_aob = os.environ.get("PR2DRAG_DEBUG_AOB", "0") == "1"
debug_seqs = set(
    s.strip()
    for s in os.environ.get(
        "PR2DRAG_DEBUG_AOB_SEQS",
        "breakdance,horsejump-high,soapbox",
    ).split(",")
    if s.strip()
)

import math
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, Optional, Callable

import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt

from .aob import AoBParams, aob_fill
from .evidence import build_sequence_evidence
from .eval import compute_seq_metrics, metrics_to_row
from .sp import (
    HMMParams,
    emission_metrics,
    fit_emission_model,
    forward_backward_binary,
    fit_temperature_grid,
    apply_temperature_scaling,
)
from .utils import (
    clamp,
    npz_read,
    npz_write,
    pretty_header,
    read_txt_lines,
    resolve_davis_root,
    safe_mkdir,
    sha1_of_dict,
    write_json,
)


# -----------------------------
# DAVIS helpers
# -----------------------------
def _resolve_davis_root(davis_root: Union[str, Path]) -> Path:
    p = Path(davis_root).expanduser()
    if str(p).strip() == "":
        raise ValueError("[DAVIS] davis_root is empty. Please set cfg['davis_root'] correctly.")
    if (p / "JPEGImages").is_dir() and (p / "Annotations").is_dir() and (p / "ImageSets").is_dir():
        return p
    cand = p / "DAVIS"
    if (cand / "JPEGImages").is_dir() and (cand / "Annotations").is_dir() and (cand / "ImageSets").is_dir():
        return cand
    raise FileNotFoundError(
        "[DAVIS] Cannot resolve davis_root.\n"
        f"  given: {p}\n"
        "  expected structure:\n"
        "    davis_root/\n"
        "      JPEGImages/\n"
        "      Annotations/\n"
        "      ImageSets/\n"
        "  or parent containing davis_root/DAVIS/...\n"
    )


def _parse_seq_from_split_line(line: str) -> str | None:
    s = line.strip()
    if not s or s.startswith("#"):
        return None
    tok = s.split()[0]
    p = Path(tok)
    parts = p.parts
    if "JPEGImages" in parts:
        i = parts.index("JPEGImages")
        if len(parts) >= i + 3:
            return parts[i + 2]
    low = tok.lower()
    if low.endswith((".jpg", ".jpeg", ".png")):
        if p.parent.name:
            return p.parent.name
    if "/" in tok or "\\" in tok:
        return Path(tok).parts[0]
    return tok


def _list_seq_from_split(davis_root: Path, split_rel: str) -> List[str]:
    split_path = Path(split_rel)
    if not split_path.is_absolute():
        split_path = davis_root / split_path
    if not split_path.exists():
        raise FileNotFoundError(f"[DAVIS] split file not found: {split_path}")

    lines = read_txt_lines(split_path)
    seqs: List[str] = []
    seen: set[str] = set()
    bad: int = 0

    for ln in lines:
        seq = _parse_seq_from_split_line(ln)
        if seq is None:
            continue
        if " " in seq or seq.lower().endswith((".jpg", ".png")):
            bad += 1
            continue
        if seq not in seen:
            seen.add(seq)
            seqs.append(seq)

    print(f"[SplitParse] split={split_path} num_lines={len(lines)} num_seqs(dedup)={len(seqs)} bad_lines={bad}")
    if len(seqs) == 0:
        raise RuntimeError(f"[DAVIS] No valid sequences parsed from split file: {split_path}")
    return seqs


def davis_frame_paths(davis_root: Union[str, Path], res: str, seq: str) -> Tuple[List[Path], List[Path]]:
    root = _resolve_davis_root(davis_root)
    res = str(res)
    seq = str(seq)

    img_dir = root / "JPEGImages" / res / seq
    ann_dir = root / "Annotations" / res / seq

    if not img_dir.is_dir():
        avail_res = [p.name for p in (root / "JPEGImages").glob("*") if p.is_dir()]
        raise FileNotFoundError(
            f"[DAVIS] Missing image directory:\n  {img_dir}\n"
            f"[hint] available resolutions under {root/'JPEGImages'}: {avail_res}\n"
            f"[hint] check cfg.res and cfg.davis_root"
        )
    if not ann_dir.is_dir():
        raise FileNotFoundError(f"[DAVIS] Missing annotation directory:\n  {ann_dir}")

    frames = sorted(img_dir.glob("*.jpg"))
    if not frames:
        frames = sorted(img_dir.glob("*.png"))
    if not frames:
        raise FileNotFoundError(f"[DAVIS] No frames found in: {img_dir}")

    annos_all = sorted(ann_dir.glob("*.png"))
    if not annos_all:
        raise FileNotFoundError(f"[DAVIS] No annotations found in: {ann_dir}")

    anno_map = {p.stem: p for p in annos_all}
    annos: List[Path] = []
    missing = []
    for fp in frames:
        ap = anno_map.get(fp.stem)
        if ap is None:
            missing.append(fp.name)
        else:
            annos.append(ap)

    if missing:
        sample = missing[:5]
        raise FileNotFoundError(
            f"[DAVIS] Missing {len(missing)} annotation(s) in {ann_dir}.\n"
            f"  examples: {sample}\n"
            f"[hint] verify DAVIS extraction is complete and uses the same frame naming."
        )

    return frames, annos


# -----------------------------
# stage signatures
# -----------------------------
def _stage_signature(cfg: Dict[str, Any], stage: str) -> str:
    pick = {
        "stage": stage,
        "res": cfg.get("res"),
        "stage1": cfg.get("stage1", {}),
        "stage2": cfg.get("stage2", {}),
        "stage3": cfg.get("stage3", {}),
        "version": "0.3.1-lite-risk-calib-scale",
    }
    return sha1_of_dict(pick)


# -----------------------------
# Stage 1
# -----------------------------
def stage1_precompute_split(cfg: Dict[str, Any], *, split: str, out_dir: Path) -> Dict[str, Any]:
    davis_root = resolve_davis_root(cfg["davis_root"])
    res = cfg["res"]
    seqs = _list_seq_from_split(davis_root, cfg["splits"][split])

    out_dir = safe_mkdir(out_dir)
    sig = _stage_signature(cfg, stage=f"stage1-{split}")

    s1cfg = cfg.get("stage1", {})
    cache = bool(s1cfg.get("cache", True))
    label_err_thresh = float(s1cfg.get("label_err_thresh", 50.0))
    noise_cfg = dict(s1cfg.get("noise", {}))

    meta = {
        "split": split,
        "num_seqs": len(seqs),
        "res": res,
        "davis_root": str(davis_root),
        "signature": sig,
        "label_err_thresh": label_err_thresh,
        "noise_cfg": noise_cfg,
        "seqs": [],
    }

    pbar = tqdm(seqs, desc=f"Stage1({split})")
    for seq in pbar:
        out_npz = out_dir / f"{seq}.npz"
        if cache and out_npz.exists():
            pbar.write(f"[skip] Stage1 exists: {out_npz.name}")
            meta["seqs"].append(seq)
            continue

        frames, annos = davis_frame_paths(davis_root, res, seq)

        seed = int(cfg.get("seed", 0))
        seed_offset = int(noise_cfg.get("seed_offset", 0))
        rng = np.random.RandomState(seed + seed_offset + (abs(hash(seq)) % 10_000))

        ev = build_sequence_evidence(
            seq=seq,
            frame_paths=frames,
            anno_paths=annos,
            label_err_thresh=label_err_thresh,
            noise_cfg=noise_cfg,
            rng=rng,
        )

        arrays = {
            "z_gt": ev.z_gt.astype(np.float32),
            "has_gt": ev.has_gt.astype(np.uint8),
            "z_obs": ev.z_obs.astype(np.float32),
            "err_obs": ev.err_obs.astype(np.float32),
            "y": ev.y.astype(np.uint8),
            "iou_shift": ev.iou_shift.astype(np.float32),
            "cycle_err": ev.cycle_err.astype(np.float32),
            "area_change": ev.area_change.astype(np.float32),
            "occl_flag": ev.occl_flag.astype(np.uint8),
            "blur_flag": ev.blur_flag.astype(np.uint8),
            "blur_inv": ev.blur_inv.astype(np.float32),
            "motion": ev.motion.astype(np.float32),
            "img_hw": np.array([ev.img_h, ev.img_w], dtype=np.int32),
        }
        meta_local = {
            "seq": seq,
            "T": len(frames),
            "signature": sig,
            "frames_rel0": str(Path(frames[0]).name),
            "img_h": ev.img_h,
            "img_w": ev.img_w,
        }
        npz_write(out_npz, arrays=arrays, meta=meta_local)
        meta["seqs"].append(seq)

    write_json(out_dir / "meta.json", meta)
    return meta


# -----------------------------
# Stage 2
# -----------------------------
def stage2_build_features_for_seq(arr: Dict[str, np.ndarray], *, feat_mode: str) -> np.ndarray:
    iou_shift = arr["iou_shift"].astype(np.float32)
    cycle_err = arr["cycle_err"].astype(np.float32)
    area_change = arr["area_change"].astype(np.float32)
    occl = arr["occl_flag"].astype(np.float32)
    blur = arr["blur_flag"].astype(np.float32)
    blur_inv = arr["blur_inv"].astype(np.float32)
    motion = arr["motion"].astype(np.float32)

    hw = arr["img_hw"].astype(np.float32)
    h, w = float(hw[0]), float(hw[1])
    diag = float(math.sqrt(h * h + w * w) + 1e-6)

    cycle_norm = cycle_err / diag
    motion_norm = motion / 10.0

    if feat_mode.lower() == "a":
        X = np.stack([iou_shift, cycle_norm, area_change, occl, blur], axis=1)
    elif feat_mode.lower() == "b":
        X = np.stack([iou_shift, cycle_norm, area_change, occl, blur, blur_inv, motion_norm], axis=1)
    else:
        raise ValueError(f"Unknown feat_mode={feat_mode}. Use 'a' or 'b'.")
    return X.astype(np.float32)


def stage2_compute_split(cfg: Dict[str, Any], *, split: str, stage1_dir: Path, out_dir: Path) -> Dict[str, Any]:
    stage1_dir = Path(stage1_dir)
    out_dir = safe_mkdir(out_dir)

    s2cfg = cfg.get("stage2", {})
    cache = bool(s2cfg.get("cache", True))
    feat_mode = str(s2cfg.get("feat_mode", "b")).lower()

    sig = _stage_signature(cfg, stage=f"stage2-{split}-feat{feat_mode}")

    meta_path = stage1_dir / "meta.json"
    if meta_path.exists():
        meta = pd.read_json(meta_path, typ="series").to_dict()  # type: ignore
        seqs = list(meta.get("seqs", []))
    else:
        seqs = sorted([p.stem for p in stage1_dir.glob("*.npz") if p.name != "meta.npz"])

    pbar = tqdm(seqs, desc=f"Stage2({split})")
    for seq in pbar:
        in_npz = stage1_dir / f"{seq}.npz"
        out_npz = out_dir / f"{seq}.npz"
        if not in_npz.exists():
            raise FileNotFoundError(f"[Stage2] Missing Stage1 npz: {in_npz}")

        if cache and out_npz.exists():
            _, m = npz_read(out_npz)
            if m.get("signature", "") == sig and m.get("feat_mode", "") == feat_mode:
                pbar.write(f"[skip] Stage2 exists (compatible): {out_npz.name}")
                continue
            else:
                pbar.write(f"[recompute] Stage2 stale/mismatch -> {out_npz.name}")

        arr, _m1 = npz_read(in_npz)
        X = stage2_build_features_for_seq(arr, feat_mode=feat_mode)
        y = arr["y"].astype(np.uint8)
        err_obs = arr["err_obs"].astype(np.float32)

        arrays = {
            "X": X,
            "y": y,
            "err_obs": err_obs,
            "z_gt": arr["z_gt"].astype(np.float32),
            "z_obs": arr["z_obs"].astype(np.float32),
            "img_hw": arr["img_hw"].astype(np.int32),
        }
        meta2 = {"seq": seq, "T": int(X.shape[0]), "feat_mode": feat_mode, "signature": sig}
        npz_write(out_npz, arrays=arrays, meta=meta2)

    out_meta = {"split": split, "signature": sig, "feat_mode": feat_mode, "num_seqs": len(seqs)}
    return out_meta


def _concat_stage2(npz_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    Xs, ys, errs = [], [], []
    for p in sorted(npz_dir.glob("*.npz")):
        if p.name == "meta.json":
            continue
        arr, _ = npz_read(p)
        Xs.append(arr["X"])
        ys.append(arr["y"])
        errs.append(arr["err_obs"])
    X = np.concatenate(Xs, axis=0) if Xs else np.zeros((0, 1), dtype=np.float32)
    y = np.concatenate(ys, axis=0) if ys else np.zeros((0,), dtype=np.uint8)
    err = np.concatenate(errs, axis=0) if errs else np.zeros((0,), dtype=np.float32)
    return X, y, err


def _feature_keep_mask(X: np.ndarray) -> np.ndarray:
    if X.size == 0:
        return np.ones((X.shape[1],), dtype=bool) if X.ndim == 2 else np.ones((1,), dtype=bool)
    std = X.std(axis=0)
    keep = std > 1e-12
    if not bool(keep.any()):
        keep[:] = True
    return keep


# -----------------------------
# Scaling / calibration helpers
# -----------------------------
def _clip_probs(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.asarray(p, dtype=np.float32)
    return np.clip(p, eps, 1.0 - eps)


def _fit_robust_scaler(
    X: np.ndarray,
    *,
    eps: float = 1e-6,
    clip: Optional[float] = 8.0,
) -> Dict[str, Any]:
    """
    Robust scaling: (x - median) / IQR, with optional symmetric clip.
    Returns a dict usable by _apply_robust_scaler.
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2 or X.shape[0] == 0:
        # degenerate
        d = int(X.shape[1]) if X.ndim == 2 else 1
        return {"center": np.zeros((d,), np.float32), "scale": np.ones((d,), np.float32), "clip": clip, "eps": eps}

    center = np.median(X, axis=0).astype(np.float32)
    q25 = np.percentile(X, 25.0, axis=0).astype(np.float32)
    q75 = np.percentile(X, 75.0, axis=0).astype(np.float32)
    scale = (q75 - q25).astype(np.float32)
    scale = np.where(scale < eps, 1.0, scale).astype(np.float32)

    return {"center": center, "scale": scale, "clip": clip, "eps": eps}


def _apply_robust_scaler(X: np.ndarray, scaler: Dict[str, Any]) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    center = np.asarray(scaler["center"], dtype=np.float32)
    scale = np.asarray(scaler["scale"], dtype=np.float32)
    clipv = scaler.get("clip", None)

    Z = (X - center[None, :]) / scale[None, :]
    if clipv is not None:
        Z = np.clip(Z, -float(clipv), float(clipv))
    return Z.astype(np.float32)


class _PiecewiseMonotoneCalibrator:
    """
    A lightweight monotone calibrator fallback (no sklearn):
    - bin w into quantile bins
    - compute empirical P(y=1) per bin
    - apply PAV-like monotone correction (pool-adjacent-violators)
    """
    def __init__(self, edges: np.ndarray, values: np.ndarray):
        self.edges = np.asarray(edges, dtype=np.float64)
        self.values = np.asarray(values, dtype=np.float64)

    def __call__(self, w: np.ndarray) -> np.ndarray:
        w = np.asarray(w, dtype=np.float64)
        # bins: [edges[i], edges[i+1])
        idx = np.searchsorted(self.edges[1:-1], w, side="right")
        out = self.values[idx]
        return np.clip(out, 0.0, 1.0).astype(np.float32)


def _fit_w_calibrator(
    w_train: np.ndarray,
    y_train: np.ndarray,
    *,
    method: str = "isotonic",
    n_bins: int = 50,
) -> Callable[[np.ndarray], np.ndarray]:
    w_train = np.asarray(w_train, dtype=np.float64).reshape(-1)
    y_train = np.asarray(y_train, dtype=np.float64).reshape(-1)
    m = np.isfinite(w_train) & np.isfinite(y_train)
    w_train = np.clip(w_train[m], 0.0, 1.0)
    y_train = np.clip(y_train[m], 0.0, 1.0)

    if w_train.size < 100:
        # too small -> identity
        return lambda w: np.clip(np.asarray(w, dtype=np.float32), 0.0, 1.0)

    method = str(method).lower()
    if method == "isotonic":
        try:
            from sklearn.isotonic import IsotonicRegression  # type: ignore
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            iso.fit(w_train, y_train)
            return lambda w: iso.predict(np.clip(np.asarray(w, dtype=np.float64), 0.0, 1.0)).astype(np.float32)
        except Exception:
            method = "piecewise"

    # piecewise fallback
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(w_train, qs)
    # ensure strictly increasing edges (avoid zero-width bins)
    edges[0] = 0.0
    edges[-1] = 1.0
    for i in range(1, len(edges)):
        if edges[i] < edges[i - 1]:
            edges[i] = edges[i - 1]

    # compute bin means
    vals = []
    counts = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i < n_bins - 1:
            mask = (w_train >= lo) & (w_train < hi)
        else:
            mask = (w_train >= lo) & (w_train <= hi)
        if mask.sum() == 0:
            vals.append(0.0)
            counts.append(0)
        else:
            vals.append(float(y_train[mask].mean()))
            counts.append(int(mask.sum()))
    vals = np.asarray(vals, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.int64)

    # PAV monotone correction
    v = vals.copy()
    c = counts.astype(np.float64).copy()
    i = 0
    while i < len(v) - 1:
        if v[i] <= v[i + 1] or (c[i] == 0 and c[i + 1] == 0):
            i += 1
            continue
        # pool i and i+1
        tot = c[i] + c[i + 1]
        if tot <= 0:
            newv = max(v[i], v[i + 1])
        else:
            newv = (v[i] * c[i] + v[i + 1] * c[i + 1]) / tot
        v[i] = newv
        c[i] = tot
        v = np.delete(v, i + 1)
        c = np.delete(c, i + 1)
        # also merge edges by removing one boundary (keep length consistent by rebuilding later)
        # simplest: break and rebuild a calibrator with uniform bin indexing
        # -> we skip complex edge merging; instead enforce monotone by cumulative maximum as a safe fallback
        # (still monotone, less sharp but stable)
        break

    if len(v) != len(vals):
        v = np.maximum.accumulate(vals)

    return _PiecewiseMonotoneCalibrator(edges=np.asarray(edges, dtype=np.float64), values=v)


# -----------------------------
# Risk-constrained tau (tie-aware)
# -----------------------------
def _select_tau_risk_constrained(
    w: np.ndarray,
    y: np.ndarray,
    *,
    r0: float,
    min_keep_frac: float = 0.05,
    tau_min: float = 1e-6,
    tau_max: float = 0.999999,
) -> Tuple[float, Dict[str, float], pd.DataFrame]:
    """
    Pick tau to maximize coverage subject to risk<=r0 on kept set {w>=tau}.
    risk = P(y=0 | kept), coverage = kept_frac

    Tie-aware: evaluates only thresholds at unique w values (group boundaries),
    so the achieved mask {w>=tau} matches the computed (coverage, risk).

    Returns:
      tau, summary dict (ACHIEVED under clamped tau), curve df (coverage, risk vs threshold)
    """
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    n = int(w.size)
    if n == 0 or y.size != n:
        tau = clamp(0.5, tau_min, tau_max)
        df = pd.DataFrame({"k": [], "tau": [], "coverage": [], "risk": []})
        return tau, {"coverage": float("nan"), "risk": float("nan")}, df

    w = np.clip(w, 0.0, 1.0)
    idx = np.argsort(-w)  # descending
    ws = w[idx]
    ys = y[idx]
    bad = 1 - ys  # bad=1 when y=0
    cum_bad = np.cumsum(bad, dtype=np.float64)

    # group ends where w changes
    change = np.where(np.diff(ws) != 0.0)[0] + 1
    ends = np.concatenate([change, np.array([n], dtype=np.int64)], axis=0)  # 1..n
    ks = ends.astype(np.int64)
    bad_k = cum_bad[ks - 1]
    risk = bad_k / ks.astype(np.float64)
    coverage = ks.astype(np.float64) / float(n)
    taus = ws[ks - 1]  # threshold value yielding keep all >= tau

    min_keep = max(1, int(math.ceil(float(min_keep_frac) * n)))
    feasible = (risk <= float(r0)) & (ks >= int(min_keep))

    if feasible.any():
        i_star = int(np.where(feasible)[0].max())
    else:
        # among ks>=min_keep, pick minimal risk; else k=1
        valid = ks >= int(min_keep)
        if valid.any():
            i_star = int(np.where(valid)[0][np.argmin(risk[valid])])
        else:
            i_star = 0

    tau_raw = float(taus[i_star])
    tau = float(clamp(tau_raw, tau_min, tau_max))

    # achieved under clamped tau
    kept = w >= tau
    k_ach = int(kept.sum())
    if k_ach <= 0:
        cov_ach = 0.0
        risk_ach = float("nan")
    else:
        cov_ach = k_ach / float(n)
        risk_ach = float((1 - y[kept]).mean())

    df = pd.DataFrame(
        {
            "k": ks.astype(int),
            "tau": taus.astype(np.float64),
            "coverage": coverage.astype(np.float64),
            "risk": risk.astype(np.float64),
        }
    )

    summary = {
        "coverage": float(cov_ach),
        "risk": float(risk_ach),
        "k": float(k_ach),
        "n": float(n),
        "min_keep": float(min_keep),
        "tau_raw": float(tau_raw),
        "tau_clamped": float(tau),
    }
    return tau, summary, df


# -----------------------------
# Stage 3
# -----------------------------
def stage3_train_eval(cfg: Dict[str, Any], *, stage2_train: Path, stage2_val: Path, out_dir: Path) -> None:
    out_dir = safe_mkdir(out_dir)
    analysis_dir = safe_mkdir(out_dir / "_analysis")

    s3 = cfg.get("stage3", {})
    hmm_cfg = s3.get("hmm", {})
    em_cfg = s3.get("emission", {})
    aob_cfg = s3.get("aob", {})
    met_cfg = s3.get("metrics", {})
    ana_cfg = s3.get("analysis", {})

    tau_mode = str(s3.get("tau_mode", "global")).lower()
    target_frac = float(s3.get("tau_target_frac", 0.25))
    tau_min = float(s3.get("tau_min", 1e-6))
    tau_max = float(s3.get("tau_max", 0.999999))

    # risk-constrained tau config
    r0 = float(s3.get("tau_risk", 0.05))
    min_keep_frac = float(s3.get("tau_min_keep_frac", 0.05))

    # feature scaling
    sc_cfg = dict(em_cfg.get("scaler", {}))
    sc_enable = bool(sc_cfg.get("enable", True))
    sc_clip = sc_cfg.get("clip", 8.0)
    sc_clip = None if sc_clip is None else float(sc_clip)

    # w calibration
    wcal_cfg = dict(s3.get("w_calibration", {}))
    wcal_enable = bool(wcal_cfg.get("enable", True))
    wcal_method = str(wcal_cfg.get("method", "isotonic")).lower()
    wcal_bins = int(wcal_cfg.get("bins", 50))

    hmm = HMMParams(
        prior_good=float(hmm_cfg.get("prior_good", 0.85)),
        p_stay_good=float(hmm_cfg.get("p_stay_good", 0.92)),
        p_stay_bad=float(hmm_cfg.get("p_stay_bad", 0.92)),
        eps=float(hmm_cfg.get("eps", 1e-12)),
    )
    fail_px = float(met_cfg.get("fail_px", 50.0))

    aobp = AoBParams(
        eps_gate=float(aob_cfg.get("eps_gate", 0.5)),
        abstain_mode=str(aob_cfg.get("abstain_mode", "hold")).lower(),
        eta_L=float(aob_cfg.get("eta_L", 0.18)),
        eta_u=float(aob_cfg.get("eta_u", 0.65)),
        max_bridge_len=int(aob_cfg.get("max_bridge_len", 20)),
        bridge_mode=str(aob_cfg.get("bridge_mode", "hermite")).lower(),
        hermite_scan=int(aob_cfg.get("hermite_scan", 6)),
        clamp_hermite=bool(aob_cfg.get("clamp_hermite", True)),
        clamp_margin_px=float(aob_cfg.get("clamp_margin_px", 0.0)),
    )

    # -----------------
    # load stage2 arrays
    # -----------------
    X_tr, y_tr, _err_tr = _concat_stage2(stage2_train)
    X_va, y_va, _err_va = _concat_stage2(stage2_val)

    print(f"[Data] X_tr {tuple(X_tr.shape)} pos_rate {float(y_tr.mean()) if y_tr.size else float('nan')}")
    print(f"[Data] X_va {tuple(X_va.shape)} pos_rate {float(y_va.mean()) if y_va.size else float('nan')}")

    keep = _feature_keep_mask(X_tr)
    X_tr_k = X_tr[:, keep]
    X_va_k = X_va[:, keep]
    print(f"[Feat] keep dims: {keep} num_keep: {int(keep.sum())}")

    # -----------------
    # feature scaling (robust + clip)
    # -----------------
    scaler: Optional[Dict[str, Any]] = None
    if sc_enable:
        scaler = _fit_robust_scaler(X_tr_k, clip=sc_clip)
        X_tr_s = _apply_robust_scaler(X_tr_k, scaler)
        X_va_s = _apply_robust_scaler(X_va_k, scaler)
        write_json(analysis_dir / "feature_scaler.json", {
            "type": "robust_iqr",
            "clip": scaler.get("clip", None),
            "eps": scaler.get("eps", None),
            "center": scaler["center"].tolist(),
            "scale": scaler["scale"].tolist(),
            "keep_mask": keep.astype(int).tolist(),
        })
        print(f"[Feat] robust scaling enabled (clip={sc_clip}) -> wrote scaler to _analysis/feature_scaler.json")
    else:
        X_tr_s, X_va_s = X_tr_k, X_va_k
        print("[Feat] scaling disabled")

    # -----------------
    # emission model
    # -----------------
    clf = fit_emission_model(
        X_tr_s,
        y_tr,
        c=float(em_cfg.get("c", 1.0)),
        max_iter=int(em_cfg.get("max_iter", 2000)),
    )

    p_tr_raw = clf.predict_proba(X_tr_s)[:, 1].astype(np.float32)
    p_va_raw = clf.predict_proba(X_va_s)[:, 1].astype(np.float32)

    # Optional temperature scaling
    cal_cfg = dict(em_cfg.get("calibration", {}))
    cal_enable = bool(cal_cfg.get("enable", True))
    cal_eps = float(cal_cfg.get("eps", 1e-6))

    T = 1.0
    if cal_enable and p_tr_raw.size == y_tr.size and y_tr.size > 0:
        T, diag = fit_temperature_grid(p_tr_raw, y_tr, eps=cal_eps)
        p_tr = apply_temperature_scaling(p_tr_raw, T, eps=cal_eps)
        p_va = apply_temperature_scaling(p_va_raw, T, eps=cal_eps)
        print(f"[Calib] temperature T={T:.4f}  nll_raw={diag['nll_raw']:.6f}  nll_cal={diag['nll_cal']:.6f}")
    else:
        p_tr, p_va = p_tr_raw, p_va_raw
        print("[Calib] disabled -> using raw probabilities")

    # clamp probs before any downstream metrics / HMM usage
    p_va_c = _clip_probs(p_va, eps=cal_eps)
    m = emission_metrics(p_va_c, y_va)
    print(f"[Emission] AUROC={m['auroc']:.4f}  ECE={m['ece']:.4f}  risk@50%={m['risk@50%']:.4f}")

    # -----------------
    # helper: compute w per seq (with SAME scaler + calib)
    # -----------------
    def _transform_X(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        Xk = X[:, keep].astype(np.float32)
        if scaler is not None:
            return _apply_robust_scaler(Xk, scaler)
        return Xk

    def _seq_post_w(npz_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        arr, _ = npz_read(npz_path)
        Xs = _transform_X(arr["X"])
        p_raw = clf.predict_proba(Xs)[:, 1].astype(np.float32)
        if cal_enable:
            p = apply_temperature_scaling(p_raw, T, eps=cal_eps)
        else:
            p = p_raw
        p = _clip_probs(p, eps=cal_eps)
        w = forward_backward_binary(p, hmm)
        y = arr.get("y", np.zeros((w.shape[0],), dtype=np.uint8)).astype(np.uint8)
        return w.astype(np.float32), p.astype(np.float32), y

    # -----------------
    # (A) collect train w,y once (also reused for tau + w-calibration)
    # -----------------
    ws_tr, ys_tr = [], []
    for pth in sorted(stage2_train.glob("*.npz")):
        w_post, _p, y = _seq_post_w(pth)
        ws_tr.append(w_post)
        ys_tr.append(y.astype(np.uint8))
    w_post_all = np.concatenate(ws_tr, axis=0) if ws_tr else np.zeros((0,), dtype=np.float32)
    y_all = np.concatenate(ys_tr, axis=0) if ys_tr else np.zeros((0,), dtype=np.uint8)

    # -----------------
    # w calibration (train only) -> wtil used for gating/semantics
    # -----------------
    w_cal: Callable[[np.ndarray], np.ndarray]
    if wcal_enable and w_post_all.size > 0 and y_all.size == w_post_all.size:
        w_cal = _fit_w_calibrator(w_post_all, y_all, method=wcal_method, n_bins=wcal_bins)
        wtil_all = w_cal(w_post_all)
        print(f"[W-Calib] enabled method={wcal_method} bins={wcal_bins}")
        # save a simple reliability curve (train) for debugging
        if bool(ana_cfg.get("write_figs", True)):
            bins = np.linspace(0.0, 1.0, 11)
            xs, accs, cnts = [], [], []
            for i in range(10):
                lo, hi = bins[i], bins[i + 1]
                msk = (wtil_all >= lo) & (wtil_all < hi) if i < 9 else (wtil_all >= lo) & (wtil_all <= hi)
                if msk.sum() == 0:
                    continue
                xs.append(0.5 * (lo + hi))
                accs.append(float(y_all[msk].mean()))
                cnts.append(int(msk.sum()))
            plt.figure()
            plt.plot(xs, accs, marker="o")
            plt.xlabel("wtil bin center (calibrated)")
            plt.ylabel("empirical P(y=1)")
            plt.title("Reliability curve on train (after w calibration)")
            plt.tight_layout()
            plt.savefig(analysis_dir / "fig_wcal_reliability_train.png", dpi=160)
            plt.close()
    else:
        w_cal = lambda w: np.clip(np.asarray(w, dtype=np.float32), 0.0, 1.0)
        wtil_all = w_post_all.astype(np.float32)
        print("[W-Calib] disabled/insufficient data -> using raw HMM posterior")

    # -----------------
    # tau selection (global / risk)
    # -----------------
    tau_global: float = 0.5

    if tau_mode == "global":
        if wtil_all.size == 0:
            tau_global = 0.5
        else:
            tau_global = float(np.quantile(wtil_all.astype(np.float64), target_frac))
            tau_global = clamp(tau_global, tau_min, tau_max)
        print(f"[Tau] global_quantile from train: tau_global={tau_global:.6f} (target_frac={target_frac})")

    elif tau_mode == "risk":
        tau_global, summ, curve = _select_tau_risk_constrained(
            wtil_all, y_all, r0=r0, min_keep_frac=min_keep_frac, tau_min=tau_min, tau_max=tau_max
        )

        print(
            f"[Tau] risk-constrained: tau_global={tau_global:.6f}  "
            f"coverage={summ['coverage']:.3f}  risk={summ['risk']:.3f}  r0={r0} "
            f"(tau_raw={summ['tau_raw']:.6f} -> tau_clamped={summ['tau_clamped']:.6f})"
        )

        curve_path = analysis_dir / "tau_risk_curve_train.csv"
        curve.to_csv(curve_path, index=False)
        write_json(analysis_dir / "tau_risk_summary_train.json", {**summ, "r0": r0, "min_keep_frac": min_keep_frac})

        if bool(ana_cfg.get("write_figs", True)) and len(curve) > 0:
            plt.figure()
            plt.plot(curve["coverage"].values, curve["risk"].values)
            plt.axhline(r0, linestyle="--")
            plt.xlabel("coverage (kept fraction)")
            plt.ylabel("risk (P(y=0 | kept))")
            plt.title("Coverage–Risk curve (train, tie-aware)")
            plt.tight_layout()
            plt.savefig(analysis_dir / "fig_coverage_risk_train.png", dpi=160)
            plt.close()

    else:
        raise ValueError("Unknown tau_mode. Use: global | risk | per_video")

    # -----------------
    # eval on val seq-wise
    # -----------------
    rows = []
    all_w_val = []
    all_err_obs_val = []
    all_y_val = []

    pbar = tqdm(sorted(stage2_val.glob("*.npz")), desc="Stage3(val)")
    for npz_path in pbar:
        arr, meta = npz_read(npz_path)
        seq = meta.get("seq", npz_path.stem)

        Xs = _transform_X(arr["X"])
        z_gt = arr["z_gt"].astype(np.float32)
        z_obs = arr["z_obs"].astype(np.float32)

        p_raw = clf.predict_proba(Xs)[:, 1].astype(np.float32)
        p = apply_temperature_scaling(p_raw, T, eps=cal_eps) if cal_enable else p_raw
        p = _clip_probs(p, eps=cal_eps)

        w_post = forward_backward_binary(p, hmm).astype(np.float32)
        y = arr.get("y", np.zeros((w_post.shape[0],), dtype=np.uint8)).astype(np.uint8)

        # calibrated wtil for gating/semantics
        w = w_cal(w_post)

        if tau_mode in ("global", "risk"):
            tau = float(tau_global)
        elif tau_mode == "per_video":
            tau = float(np.quantile(w.astype(np.float64), target_frac)) if w.size else 0.5
            tau = clamp(tau, tau_min, tau_max)
        else:
            raise ValueError(f"Unknown tau_mode={tau_mode}")

        seg_debug = [] if (debug_aob and seq in debug_seqs) else None
        z_fin, is_bridge, is_abst = aob_fill(z_base=z_obs, w=w, tau=tau, params=aobp, debug=seg_debug)

        if debug_aob and seq in debug_seqs:
            Tseq = int(len(w))
            q = np.quantile(w.astype(np.float64), [0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0]).tolist()
            low_mask = (is_bridge | is_abst)
            low_n = int(low_mask.sum())

            err_obs = np.linalg.norm((z_obs - z_gt).astype(np.float64), axis=1)
            err_fin = np.linalg.norm((z_fin - z_gt).astype(np.float64), axis=1)

            def _q95(x):
                return float(np.quantile(x, 0.95)) if x.size else float("nan")

            pbar.write(
                f"[AoB-Debug] seq={seq} tau={tau:.6f} "
                f"wtil[q0,q1,q5,q50,q95,q99,q100]={['%.4f'%v for v in q]} "
                f"low_frac={low_n/max(Tseq,1):.4f} low_n={low_n} segs={0 if seg_debug is None else len(seg_debug)}"
            )
            if low_n > 0:
                pbar.write(
                    f"[AoB-Debug] seq={seq} err_obs_p95_all={_q95(err_obs):.3f} err_fin_p95_all={_q95(err_fin):.3f} "
                    f"err_obs_p95_low={_q95(err_obs[low_mask]):.3f} err_fin_p95_low={_q95(err_fin[low_mask]):.3f}"
                )

            write_json(
                analysis_dir / f"debug_aob_{seq}.json",
                {
                    "seq": seq,
                    "tau": float(tau),
                    "aobp": {
                        "eps_gate": aobp.eps_gate,
                        "abstain_mode": aobp.abstain_mode,
                        "eta_L": aobp.eta_L,
                        "eta_u": aobp.eta_u,
                        "max_bridge_len": aobp.max_bridge_len,
                    },
                    "w_quantiles": q,
                    "segments": seg_debug if seg_debug is not None else [],
                },
            )

        sm = compute_seq_metrics(
            seq=seq,
            z_gt=z_gt,
            z_obs=z_obs,
            z_fin=z_fin,
            w=w,  # NOTE: using calibrated wtil
            tau=tau,
            is_bridge=is_bridge,
            is_abst=is_abst,
            fail_px=fail_px,
        )
        rows.append(metrics_to_row(sm))

        all_w_val.append(w.astype(np.float32))
        all_err_obs_val.append(arr["err_obs"].astype(np.float32))
        all_y_val.append(y.astype(np.float32))

    df = pd.DataFrame(rows)
    if len(df) > 0:
        print("\n[Val aggregate]")
        print("mean obs_p95", float(df["all_obs_p95"].mean()), "mean fin_p95", float(df["all_fin_p95"].mean()))
        print("mean obs_fail", float(df["all_obs_fail"].mean()), "mean fin_fail", float(df["all_fin_fail"].mean()))
        print("mean low_frac", float(df["low_frac"].mean()), "mean bridged_frames", float(df["bridge_n"].mean()))
    else:
        print("[Val aggregate] empty")

    if bool(ana_cfg.get("write_csv", True)):
        csv_path = analysis_dir / "table_v1.csv"
        df.to_csv(csv_path, index=False)
        print(f"[OK] wrote: {csv_path}")

    # correlation spearman(w, -err)
    try:
        from scipy.stats import spearmanr
        wcat = np.concatenate(all_w_val, axis=0) if all_w_val else np.zeros((0,), dtype=np.float32)
        ecat = np.concatenate(all_err_obs_val, axis=0) if all_err_obs_val else np.zeros((0,), dtype=np.float32)
        if wcat.size > 10 and ecat.size == wcat.size:
            rho = float(spearmanr(wcat, -ecat).correlation)
        else:
            rho = float("nan")
    except Exception:
        rho = float("nan")

    (analysis_dir / "corr_wtil_err.txt").write_text(f"spearman(wtil, -err_obs) = {rho}\n", encoding="utf-8")
    print(f"[OK] wrote stage3 to: {out_dir}")
    print(f"  spearman(wtil, -err_obs) = {rho}")

    # figs
    if bool(ana_cfg.get("write_figs", True)) and len(df) > 0:
        # gate bins: reliability vs empirical accuracy (VAL, calibrated)
        wcat = np.concatenate(all_w_val, axis=0) if all_w_val else np.zeros((0,), dtype=np.float32)
        ycat = np.concatenate(all_y_val, axis=0) if all_y_val else np.zeros((0,), dtype=np.float32)

        if wcat.size == ycat.size and wcat.size > 0:
            bins = np.linspace(0.0, 1.0, 11)
            xs, accs = [], []
            for i in range(10):
                lo, hi = bins[i], bins[i + 1]
                msk = (wcat >= lo) & (wcat < hi) if i < 9 else (wcat >= lo) & (wcat <= hi)
                if msk.sum() == 0:
                    continue
                xs.append(0.5 * (lo + hi))
                accs.append(float(ycat[msk].mean()))
            plt.figure()
            plt.plot(xs, accs, marker="o")
            plt.xlabel("wtil bin center")
            plt.ylabel("empirical P(y=1)")
            plt.title("Gate bins (reliability semantics, val)")
            plt.tight_layout()
            plt.savefig(analysis_dir / "fig_gate_bins.png", dpi=160)
            plt.close()

        plt.figure()
        plt.hist(df["tau"].values, bins=20)
        plt.xlabel("tau")
        plt.ylabel("count")
        plt.title("Tau distribution")
        plt.tight_layout()
        plt.savefig(analysis_dir / "fig_tau_hist.png", dpi=160)
        plt.close()

        plt.figure()
        plt.hist(df["low_frac"].values, bins=20)
        plt.xlabel("low_frac")
        plt.ylabel("count")
        plt.title("Low fraction distribution")
        plt.tight_layout()
        plt.savefig(analysis_dir / "fig_lowfrac_hist.png", dpi=160)
        plt.close()

        # feature hist: plot clipped ranges to avoid one dim dominating x-axis
        plt.figure()
        for j in range(X_tr.shape[1]):
            v = X_tr[:, j].astype(np.float64)
            if v.size == 0:
                continue
            lo = float(np.quantile(v, 0.01))
            hi = float(np.quantile(v, 0.99))
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                lo, hi = float(v.min()), float(v.max())
            vv = np.clip(v, lo, hi)
            plt.hist(vv, bins=30, alpha=0.5, label=f"dim{j}")
        plt.legend()
        plt.title("Feature hist (train, per-dim clipped 1%-99%)")
        plt.tight_layout()
        plt.savefig(analysis_dir / "fig_feat_hist.png", dpi=160)
        plt.close()

        print(f"[OK] wrote figs into: {analysis_dir}")


def run_all(cfg: Dict[str, Any]) -> None:
    davis_root = resolve_davis_root(cfg["davis_root"])
    base_out = Path(cfg["base_out"])
    res = cfg["res"]

    out_train_s1 = base_out / "davis2016_train_precompute"
    out_val_s1 = base_out / "davis2016_val_precompute"

    out_train_s2 = base_out / "davis2016_stage2_fixed"
    out_val_s2 = base_out / "davis2016_val_stage2"

    out_dir_s3 = base_out / cfg.get("stage3", {}).get("out_dir", "davis2016_stage3_fixed")

    print(
        pretty_header(
            "PR2-Drag Lite",
            {
                "cmd": "run_all",
                "davis_root": str(davis_root),
                "res": res,
                "base_out": str(base_out),
            },
        )
    )

    meta_tr = stage1_precompute_split(cfg, split="train", out_dir=out_train_s1)
    print(f"[OK] Stage1(train) meta: {out_train_s1/'meta.json'}")
    print(f"[OK] Stage1(train) npz_dir: {out_train_s1}  num_npz={len(meta_tr.get('seqs', []))}")

    meta_va = stage1_precompute_split(cfg, split="val", out_dir=out_val_s1)
    print(f"[OK] Stage1(val) meta: {out_val_s1/'meta.json'}")
    print(f"[OK] Stage1(val) npz_dir: {out_val_s1}  num_npz={len(meta_va.get('seqs', []))}")

    stage2_compute_split(cfg, split="train", stage1_dir=out_train_s1, out_dir=out_train_s2)
    stage2_compute_split(cfg, split="val", stage1_dir=out_val_s1, out_dir=out_val_s2)
    print(
        f"[OK] Stage2 done: train_npz= {len(list(out_train_s2.glob('*.npz')))} val_npz= {len(list(out_val_s2.glob('*.npz')))}"
    )

    stage3_train_eval(cfg, stage2_train=out_train_s2, stage2_val=out_val_s2, out_dir=out_dir_s3)
