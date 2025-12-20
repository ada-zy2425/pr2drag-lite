# pr2drag/pipeline.py
from __future__ import annotations

import os
import math
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, Optional

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

# -----------------------
# Optional debug controls
# -----------------------
debug_aob = os.environ.get("PR2DRAG_DEBUG_AOB", "0") == "1"
debug_seqs = set(
    s.strip()
    for s in os.environ.get(
        "PR2DRAG_DEBUG_AOB_SEQS",
        "breakdance,horsejump-high,soapbox",
    ).split(",")
    if s.strip()
)


# ============================================================
# DAVIS helpers
# ============================================================
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


# ============================================================
# Stage signature
# ============================================================
def _stage_signature(cfg: Dict[str, Any], stage: str) -> str:
    pick = {
        "stage": stage,
        "res": cfg.get("res"),
        "stage1": cfg.get("stage1", {}),
        "stage2": cfg.get("stage2", {}),
        "stage3": cfg.get("stage3", {}),
        "version": "0.3.2-lite-risk-wcalib-tieaware-policy-seedfix",
    }
    return sha1_of_dict(pick)


# ============================================================
# Stage1
# ============================================================
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

        # IMPORTANT: do NOT use Python's built-in hash(seq) for seeding.
        # hash() is randomized per process unless PYTHONHASHSEED is fixed, breaking reproducibility.
        seq_hash = int(hashlib.sha1(seq.encode("utf-8")).hexdigest()[:8], 16) % 10_000
        rng = np.random.RandomState(seed + seed_offset + seq_hash)

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


# ============================================================
# Stage2
# ============================================================
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
        seqs = sorted([p.stem for p in stage1_dir.glob("*.npz")])

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
    for p in sorted(Path(npz_dir).glob("*.npz")):
        arr, _ = npz_read(p)
        if "X" not in arr:
            continue
        Xs.append(arr["X"])
        ys.append(arr.get("y", np.zeros((arr["X"].shape[0],), dtype=np.uint8)))
        errs.append(arr.get("err_obs", np.zeros((arr["X"].shape[0],), dtype=np.float32)))
    X = np.concatenate(Xs, axis=0) if Xs else np.zeros((0, 1), dtype=np.float32)
    y = np.concatenate(ys, axis=0) if ys else np.zeros((0,), dtype=np.uint8)
    err = np.concatenate(errs, axis=0) if errs else np.zeros((0,), dtype=np.float32)
    return X, y, err


def _feature_keep_mask(X: np.ndarray) -> np.ndarray:
    if X.ndim != 2:
        return np.ones((1,), dtype=bool)
    if X.size == 0:
        return np.ones((X.shape[1],), dtype=bool)
    std = X.std(axis=0)
    keep = std > 1e-12
    if not bool(keep.any()):
        keep[:] = True
    return keep


# ============================================================
# Feature scaler (robust IQR)
# ============================================================
def _robust_iqr_fit(X: np.ndarray, *, eps: float = 1e-6) -> Dict[str, Any]:
    if X.ndim != 2 or X.size == 0:
        return {"type": "robust_iqr", "center": [], "scale": [], "eps": float(eps)}
    q1 = np.quantile(X, 0.25, axis=0)
    q3 = np.quantile(X, 0.75, axis=0)
    center = np.median(X, axis=0)
    scale = (q3 - q1).astype(np.float64)
    scale = np.maximum(scale, float(eps))
    return {
        "type": "robust_iqr",
        "eps": float(eps),
        "center": center.astype(float).tolist(),
        "scale": scale.astype(float).tolist(),
    }


def _robust_iqr_apply(
    X: np.ndarray,
    scaler: Dict[str, Any],
    *,
    clip: float = 8.0,
) -> np.ndarray:
    if X.ndim != 2 or X.size == 0:
        return X.astype(np.float32, copy=False)
    center = np.array(scaler.get("center", []), dtype=np.float64)
    scale = np.array(scaler.get("scale", []), dtype=np.float64)
    if center.size != X.shape[1] or scale.size != X.shape[1]:
        raise ValueError(
            f"[Scaler] shape mismatch: center/scale dims={center.size}/{scale.size} but X has {X.shape[1]} dims"
        )
    Z = (X.astype(np.float64) - center[None, :]) / scale[None, :]
    if clip is not None:
        Z = np.clip(Z, -float(clip), float(clip))
    return Z.astype(np.float32)


def _feature_stats(X: np.ndarray) -> Dict[str, Any]:
    if X.ndim != 2 or X.size == 0:
        return {}
    out: Dict[str, Any] = {}
    for j in range(X.shape[1]):
        x = X[:, j].astype(np.float64)
        out[f"dim{j}"] = {
            "mean": float(np.mean(x)),
            "std": float(np.std(x)),
            "q1": float(np.quantile(x, 0.25)),
            "q3": float(np.quantile(x, 0.75)),
            "iqr": float(np.quantile(x, 0.75) - np.quantile(x, 0.25)),
            "min": float(np.min(x)),
            "max": float(np.max(x)),
            "frac_zero": float(np.mean(x == 0.0)),
        }
    return out


# ============================================================
# w calibration (isotonic preferred, fallback to binning)
# ============================================================
class _WCalibrator:
    def __init__(self, method: str = "isotonic", bins: int = 50, eps: float = 1e-8) -> None:
        self.method = str(method).lower()
        self.bins = int(bins)
        self.eps = float(eps)

        self._iso = None  # sklearn model
        self._bin_edges: Optional[np.ndarray] = None
        self._bin_vals: Optional[np.ndarray] = None

    def fit(self, w: np.ndarray, y: np.ndarray) -> None:
        w = np.asarray(w, dtype=np.float64).reshape(-1)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        if w.size == 0 or y.size != w.size:
            raise ValueError("[W-Calib] invalid inputs")
        if not (np.min(w) >= -1e-6 and np.max(w) <= 1.0 + 1e-6):
            raise ValueError(f"[W-Calib] w outside [0,1]: min={w.min()} max={w.max()}")

        if self.method == "isotonic":
            try:
                from sklearn.isotonic import IsotonicRegression  # type: ignore

                iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
                iso.fit(w, y)
                self._iso = iso
                self._bin_edges = None
                self._bin_vals = None
                return
            except Exception as e:
                print(f"[W-Calib] isotonic unavailable/failed ({e}) -> fallback to binning")
                self.method = "binning"

        # Binning calibration (monotone by cumulative max)
        bins = max(2, int(self.bins))
        qs = np.linspace(0.0, 1.0, bins + 1)
        edges = np.quantile(w, qs)
        edges = np.unique(edges)
        if edges.size < 3:
            self._bin_edges = np.array([0.0, 1.0], dtype=np.float64)
            self._bin_vals = np.array([float(np.mean(y))], dtype=np.float64)
            return

        vals = []
        for i in range(edges.size - 1):
            lo, hi = edges[i], edges[i + 1]
            if i < edges.size - 2:
                m = (w >= lo) & (w < hi)
            else:
                m = (w >= lo) & (w <= hi)
            if np.sum(m) == 0:
                vals.append(np.nan)
            else:
                vals.append(float(np.mean(y[m])))

        v = np.array(vals, dtype=np.float64)
        if np.any(np.isnan(v)):
            good = np.where(~np.isnan(v))[0]
            if good.size == 0:
                v[:] = float(np.mean(y))
            else:
                for i in range(v.size):
                    if np.isnan(v[i]):
                        j = good[np.argmin(np.abs(good - i))]
                        v[i] = v[j]

        v = np.maximum.accumulate(v)
        v = np.clip(v, 0.0, 1.0)

        self._bin_edges = edges
        self._bin_vals = v
        self._iso = None

    def transform(self, w: np.ndarray) -> np.ndarray:
        w = np.asarray(w, dtype=np.float64).reshape(-1)
        if self._iso is not None:
            out = self._iso.predict(w)
            return np.clip(out, 0.0, 1.0).astype(np.float32)
        if self._bin_edges is not None and self._bin_vals is not None:
            edges = self._bin_edges
            vals = self._bin_vals
            idx = np.searchsorted(edges[1:-1], w, side="right")
            idx = np.clip(idx, 0, vals.size - 1)
            out = vals[idx]
            return np.clip(out, 0.0, 1.0).astype(np.float32)
        return np.clip(w, 0.0, 1.0).astype(np.float32)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"method": self.method, "bins": int(self.bins), "eps": float(self.eps)}
        if self._iso is not None:
            d["type"] = "isotonic"
            d["x"] = np.asarray(self._iso.X_thresholds_, dtype=float).tolist()
            d["y"] = np.asarray(self._iso.y_thresholds_, dtype=float).tolist()
        elif self._bin_edges is not None and self._bin_vals is not None:
            d["type"] = "binning"
            d["edges"] = self._bin_edges.astype(float).tolist()
            d["vals"] = self._bin_vals.astype(float).tolist()
        else:
            d["type"] = "identity"
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "_WCalibrator":
        method = str(d.get("method", d.get("type", "identity"))).lower()
        bins = int(d.get("bins", 50))
        eps = float(d.get("eps", 1e-8))
        cal = _WCalibrator(method=method, bins=bins, eps=eps)
        typ = str(d.get("type", method)).lower()
        if typ == "isotonic":
            try:
                from sklearn.isotonic import IsotonicRegression  # type: ignore

                iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
                iso.X_thresholds_ = np.asarray(d.get("x", []), dtype=np.float64)
                iso.y_thresholds_ = np.asarray(d.get("y", []), dtype=np.float64)
                cal._iso = iso
            except Exception:
                cal._iso = None
        elif typ == "binning":
            cal._bin_edges = np.asarray(d.get("edges", [0.0, 1.0]), dtype=np.float64)
            cal._bin_vals = np.asarray(d.get("vals", [0.5]), dtype=np.float64)
        return cal


# ============================================================
# tau selection (tie-aware, kept={w>=tau}, with policy options)
# ============================================================
def _select_tau_risk_constrained_tieaware(
    w: np.ndarray,
    y: np.ndarray,
    *,
    r0: float,
    min_keep_frac: float = 0.05,
    tau_min: float = 1e-6,
    tau_max: float = 0.999999,
    policy: str = "max_coverage",
    risk_margin: float = 0.0,
) -> Tuple[float, Dict[str, float], pd.DataFrame]:
    """
    Tie-aware tau selection on kept set {w>=tau}.

    Definitions:
      risk     = P(y=0 | kept)
      coverage = |kept| / n

    Many calibrators (e.g., isotonic) produce many ties in w, so we only evaluate thresholds
    at the end of each tie group to ensure kept={w>=tau} is consistent.

    policy:
      - "max_coverage": among feasible (risk<=r0), pick the one with max coverage (lowest tau).
      - "max_tau": among feasible, pick the one with max tau (most conservative).
      - "min_risk": among feasible, pick the minimal risk (break ties by higher coverage).
      - "balanced": maximize coverage - alpha*risk (alpha ~ 1/r0).

    risk_margin:
      use effective constraint r_eff = max(0, r0 - risk_margin) to be slightly conservative.
    """
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    if w.size == 0 or y.size != w.size:
        tau = clamp(0.5, tau_min, tau_max)
        df = pd.DataFrame({"tau": [], "k": [], "coverage": [], "risk": []})
        return tau, {"coverage": float("nan"), "risk": float("nan")}, df

    # sanitize
    m = np.isfinite(w)
    w = w[m]
    y = y[m]
    n = int(w.size)
    if n == 0:
        tau = clamp(0.5, tau_min, tau_max)
        df = pd.DataFrame({"tau": [], "k": [], "coverage": [], "risk": []})
        return tau, {"coverage": float("nan"), "risk": float("nan")}, df
    w = np.clip(w, 0.0, 1.0)

    # stable sort to make ties deterministic
    idx = np.argsort(-w, kind="mergesort")  # desc
    ws = w[idx]
    ys = y[idx]
    bad = (ys == 0).astype(np.float64)
    cum_bad = np.cumsum(bad, dtype=np.float64)

    # group ends for ties in ws: end positions where value changes OR last element
    if n == 1:
        ends = np.array([0], dtype=np.int64)
    else:
        ends = np.concatenate([np.nonzero(np.diff(ws))[0], np.array([n - 1])]).astype(np.int64)

    ks = (ends + 1).astype(np.int64)
    risks = (cum_bad[ends] / ks.astype(np.float64))
    covs = (ks.astype(np.float64) / float(n))
    taus = ws[ends]

    min_keep = max(1, int(math.ceil(float(min_keep_frac) * n)))
    r_eff = max(0.0, float(r0) - float(risk_margin))
    feasible = (ks >= min_keep) & (risks <= r_eff)

    pol = str(policy).lower()
    i_star: int

    if np.any(feasible):
        cand = np.where(feasible)[0]
        if pol in ("max_coverage", "maxcov"):
            i_star = int(cand[np.argmax(covs[cand])])  # largest coverage (lowest tau)
        elif pol in ("max_tau", "maxtau", "conservative"):
            i_star = int(cand.min())  # highest tau among feasible
        elif pol in ("min_risk", "minrisk"):
            rr = risks[cand]
            cc = covs[cand]
            j = int(np.lexsort((-cc, rr))[0])  # min risk, then max coverage
            i_star = int(cand[j])
        elif pol in ("balanced", "pareto"):
            alpha = 1.0 / max(float(r0), 1e-12)
            score = covs[cand] - alpha * risks[cand]
            i_star = int(cand[np.argmax(score)])
        else:
            raise ValueError(f"Unknown tau policy: {policy}. Use max_coverage|max_tau|min_risk|balanced.")
    else:
        valid = ks >= min_keep
        if np.any(valid):
            cand = np.where(valid)[0]
            rr = risks[cand]
            cc = covs[cand]
            j = int(np.lexsort((-cc, rr))[0])
            i_star = int(cand[j])
        else:
            i_star = 0

    tau_raw = float(taus[i_star])
    tau = clamp(tau_raw, tau_min, tau_max)

    df = pd.DataFrame({"tau": taus, "k": ks, "coverage": covs, "risk": risks})
    summary = {
        "coverage": float(covs[i_star]),
        "risk": float(risks[i_star]),
        "k": float(ks[i_star]),
        "n": float(n),
        "min_keep": float(min_keep),
        "r0": float(r0),
        "r_eff": float(r_eff),
        "policy": str(pol),
        "tau_raw": float(tau_raw),
        "tau_clamped": float(tau),
        "risk_margin": float(risk_margin),
    }
    return tau, summary, df


# ============================================================
# Stage3
# ============================================================
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
    tau_policy = str(s3.get("tau_policy", "max_coverage")).lower()
    tau_risk_margin = float(s3.get("tau_risk_margin", 0.0))

    # feature scaling
    sc_cfg = dict(s3.get("feature_scaler", {}))
    sc_enable = bool(sc_cfg.get("enable", True))
    sc_type = str(sc_cfg.get("type", "robust_iqr")).lower()
    sc_clip = float(sc_cfg.get("clip", 8.0))
    sc_eps = float(sc_cfg.get("eps", 1e-6))

    # w calibration
    wc_cfg = dict(s3.get("w_calibration", {}))
    wc_enable = bool(wc_cfg.get("enable", False))
    wc_method = str(wc_cfg.get("method", "isotonic")).lower()
    wc_bins = int(wc_cfg.get("bins", 50))

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

    if y_tr.size == 0 or X_tr.size == 0:
        raise RuntimeError("[Stage3] empty training data. Check stage2_train path and split.")
    if float(np.unique(y_tr).size) < 2:
        raise RuntimeError(f"[Stage3] y_tr has <2 classes. pos_rate={float(y_tr.mean())}")

    keep = _feature_keep_mask(X_tr)
    X_tr_k_raw = X_tr[:, keep].astype(np.float32)
    X_va_k_raw = X_va[:, keep].astype(np.float32)
    print(f"[Feat] keep dims: {keep} num_keep: {int(keep.sum())}")

    # write feature stats (raw, before scaling)
    try:
        write_json(analysis_dir / "feature_stats_train.json", _feature_stats(X_tr_k_raw))
    except Exception as e:
        print(f"[Warn] failed to write feature_stats_train.json: {e}")

    # fit scaler and transform
    scaler: Optional[Dict[str, Any]] = None
    if sc_enable:
        if sc_type != "robust_iqr":
            raise ValueError(f"[Scaler] unknown type={sc_type}, expected robust_iqr")
        scaler = _robust_iqr_fit(X_tr_k_raw, eps=sc_eps)
        X_tr_k = _robust_iqr_apply(X_tr_k_raw, scaler, clip=sc_clip)
        X_va_k = _robust_iqr_apply(X_va_k_raw, scaler, clip=sc_clip)
        scaler_dump = {**scaler, "clip": float(sc_clip), "keep_mask": keep.astype(int).tolist()}
        try:
            write_json(analysis_dir / "feature_scaler.json", scaler_dump)
        except Exception as e:
            print(f"[Warn] failed to write feature_scaler.json: {e}")
        print(f"[Feat] robust scaling enabled (clip={sc_clip}) -> wrote scaler to _analysis/feature_scaler.json")
    else:
        X_tr_k, X_va_k = X_tr_k_raw, X_va_k_raw
        print("[Feat] scaling disabled")

    # -----------------
    # emission model
    # -----------------
    clf = fit_emission_model(
        X_tr_k,
        y_tr,
        c=float(em_cfg.get("c", 1.0)),
        max_iter=int(em_cfg.get("max_iter", 2000)),
    )

    p_tr_raw = clf.predict_proba(X_tr_k)[:, 1].astype(np.float32)
    p_va_raw = clf.predict_proba(X_va_k)[:, 1].astype(np.float32)

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

    m = emission_metrics(p_va, y_va)
    print(f"[Emission] AUROC={m['auroc']:.4f}  ECE={m['ece']:.4f}  risk@50%={m['risk@50%']:.4f}")

    # -----------------
    # helper: per-seq (scaled X -> p -> w_raw)
    # -----------------
    def _seq_post_w_raw(npz_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        arr, _ = npz_read(npz_path)
        X = arr["X"][:, keep].astype(np.float32)
        if sc_enable and scaler is not None:
            X = _robust_iqr_apply(X, scaler, clip=sc_clip)
        p_raw = clf.predict_proba(X)[:, 1].astype(np.float32)
        p = apply_temperature_scaling(p_raw, T, eps=cal_eps) if cal_enable else p_raw
        w_raw = forward_backward_binary(p, hmm)  # in [0,1]
        y = arr.get("y", np.zeros((w_raw.shape[0],), dtype=np.uint8)).astype(np.uint8)
        return w_raw.astype(np.float32), p.astype(np.float32), y

    # -----------------
    # w calibration: fit on TRAIN (w_raw -> wtil)
    # -----------------
    wcal: Optional[_WCalibrator] = None
    wtil_tr_all: Optional[np.ndarray] = None

    if wc_enable:
        ws, ys = [], []
        for pth in sorted(Path(stage2_train).glob("*.npz")):
            w_raw, _p, y = _seq_post_w_raw(pth)
            ws.append(w_raw)
            ys.append(y.astype(np.uint8))
        w_all = np.concatenate(ws, axis=0) if ws else np.zeros((0,), dtype=np.float32)
        y_all = np.concatenate(ys, axis=0) if ys else np.zeros((0,), dtype=np.uint8)

        if w_all.size == 0 or y_all.size != w_all.size:
            print("[W-Calib] train w empty/mismatch -> disabled")
            wc_enable = False
        else:
            wcal = _WCalibrator(method=wc_method, bins=wc_bins)
            wcal.fit(w_all, y_all)
            wtil_tr_all = wcal.transform(w_all)
            try:
                write_json(analysis_dir / "w_calibrator.json", wcal.to_dict())
            except Exception as e:
                print(f"[Warn] failed to write w_calibrator.json: {e}")
            print(f"[W-Calib] enabled method={wc_method} bins={wc_bins}")

            # reliability curve on TRAIN after w calibration
            if bool(ana_cfg.get("write_figs", True)):
                bins = np.linspace(0.0, 1.0, 11)
                xs, accs = [], []
                for i in range(10):
                    lo, hi = bins[i], bins[i + 1]
                    msk = (wtil_tr_all >= lo) & (wtil_tr_all < hi) if i < 9 else (wtil_tr_all >= lo) & (wtil_tr_all <= hi)
                    if int(msk.sum()) == 0:
                        continue
                    xs.append(0.5 * (lo + hi))
                    accs.append(float(y_all[msk].mean()))
                if len(xs) > 0:
                    plt.figure()
                    plt.plot(xs, accs, marker="o")
                    plt.xlabel("wtil bin center (calibrated)")
                    plt.ylabel("empirical P(y=1)")
                    plt.title("Reliability curve on train (after w calibration)")
                    plt.tight_layout()
                    plt.savefig(analysis_dir / "fig_reliability_train_wcalib.png", dpi=160)
                    plt.close()

    # apply (or identity) mapping
    def _w_to_wtil(w_raw: np.ndarray) -> np.ndarray:
        if wc_enable and wcal is not None:
            return wcal.transform(w_raw)
        return np.clip(w_raw, 0.0, 1.0).astype(np.float32)

    # -----------------
    # tau selection (global)
    # -----------------
    tau_global: float = 0.5

    if tau_mode == "global":
        # quantile on TRAIN wtil
        ws = []
        for pth in sorted(Path(stage2_train).glob("*.npz")):
            w_raw, _p, _y = _seq_post_w_raw(pth)
            ws.append(_w_to_wtil(w_raw))
        w_all = np.concatenate(ws, axis=0) if ws else np.zeros((0,), dtype=np.float32)
        if w_all.size == 0:
            tau_global = 0.5
        else:
            tau_global = float(np.quantile(w_all.astype(np.float64), target_frac))
            tau_global = clamp(tau_global, tau_min, tau_max)
        print(f"[Tau] global_quantile from train: tau_global={tau_global:.6f} (target_frac={target_frac})")

    elif tau_mode == "risk":
        # risk-constrained on TRAIN wtil (tie-aware + policy)
        ws, ys = [], []
        for pth in sorted(Path(stage2_train).glob("*.npz")):
            w_raw, _p, y = _seq_post_w_raw(pth)
            ws.append(_w_to_wtil(w_raw))
            ys.append(y.astype(np.uint8))
        w_all = np.concatenate(ws, axis=0) if ws else np.zeros((0,), dtype=np.float32)
        y_all = np.concatenate(ys, axis=0) if ys else np.zeros((0,), dtype=np.uint8)

        tau_global, summ, curve = _select_tau_risk_constrained_tieaware(
            w_all,
            y_all,
            r0=r0,
            min_keep_frac=min_keep_frac,
            tau_min=tau_min,
            tau_max=tau_max,
            policy=tau_policy,
            risk_margin=tau_risk_margin,
        )

        print(
            f"[Tau] risk-constrained({tau_policy}): tau_global={tau_global:.6f}  "
            f"coverage={summ.get('coverage', float('nan')):.3f}  risk={summ.get('risk', float('nan')):.3f}  "
            f"r0={r0} margin={tau_risk_margin}"
        )

        # save curve/summary
        try:
            curve.to_csv(analysis_dir / "tau_risk_curve_train.csv", index=False)
        except Exception as e:
            print(f"[Warn] failed to write tau_risk_curve_train.csv: {e}")
        try:
            write_json(
                analysis_dir / "tau_risk_summary_train.json",
                {**summ, "tau": float(tau_global), "min_keep_frac": float(min_keep_frac)},
            )
        except Exception as e:
            print(f"[Warn] failed to write tau_risk_summary_train.json: {e}")

        if bool(ana_cfg.get("write_figs", True)) and len(curve) > 0:
            plt.figure()
            plt.plot(curve["coverage"].values, curve["risk"].values)
            plt.axhline(float(r0), linestyle="--")
            plt.xlabel("coverage (kept fraction)")
            plt.ylabel("risk (P(y=0 | kept))")
            plt.title("Coverage–Risk curve (train, tie-aware)")
            plt.tight_layout()
            plt.savefig(analysis_dir / "fig_coverage_risk_train.png", dpi=160)
            plt.close()

    elif tau_mode == "per_video":
        print("[Tau] per_video mode enabled (legacy).")
        tau_global = float("nan")
    else:
        raise ValueError("Unknown tau_mode. Use: global | risk | per_video")

    # -----------------
    # eval on val seq-wise
    # -----------------
    rows = []
    all_wtil_val = []
    all_err_obs_val = []
    all_y_val = []

    pbar = tqdm(sorted(Path(stage2_val).glob("*.npz")), desc="Stage3(val)")
    for npz_path in pbar:
        arr, meta = npz_read(npz_path)
        seq = meta.get("seq", npz_path.stem)

        X = arr["X"][:, keep].astype(np.float32)
        if sc_enable and scaler is not None:
            X = _robust_iqr_apply(X, scaler, clip=sc_clip)

        z_gt = arr["z_gt"].astype(np.float32)
        z_obs = arr["z_obs"].astype(np.float32)
        y_seq = arr.get("y", np.zeros((X.shape[0],), dtype=np.uint8)).astype(np.uint8)

        p_raw = clf.predict_proba(X)[:, 1].astype(np.float32)
        p = apply_temperature_scaling(p_raw, T, eps=cal_eps) if cal_enable else p_raw

        w_raw = forward_backward_binary(p, hmm).astype(np.float32)
        wtil = _w_to_wtil(w_raw)

        if tau_mode == "global" or tau_mode == "risk":
            tau = float(tau_global)
        elif tau_mode == "per_video":
            tau = float(np.quantile(wtil.astype(np.float64), target_frac)) if wtil.size else 0.5
            tau = clamp(tau, tau_min, tau_max)
        else:
            raise ValueError(f"Unknown tau_mode={tau_mode}")

        seg_debug = [] if (debug_aob and seq in debug_seqs) else None
        z_fin, is_bridge, is_abst = aob_fill(z_base=z_obs, w=wtil, tau=tau, params=aobp, debug=seg_debug)

        if debug_aob and seq in debug_seqs:
            Tseq = int(len(wtil))
            q = np.quantile(wtil.astype(np.float64), [0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0]).tolist()
            low_mask = (is_bridge | is_abst)
            low_n = int(low_mask.sum())

            err_obs = np.linalg.norm((z_obs - z_gt).astype(np.float64), axis=1)
            err_fin = np.linalg.norm((z_fin - z_gt).astype(np.float64), axis=1)

            def _q95(x: np.ndarray) -> float:
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

            try:
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
                        "wtil_quantiles": q,
                        "segments": seg_debug if seg_debug is not None else [],
                    },
                )
            except Exception as e:
                print(f"[Warn] failed to write debug json for {seq}: {e}")

        sm = compute_seq_metrics(
            seq=seq,
            z_gt=z_gt,
            z_obs=z_obs,
            z_fin=z_fin,
            w=wtil,
            tau=tau,
            is_bridge=is_bridge,
            is_abst=is_abst,
            fail_px=fail_px,
        )
        rows.append(metrics_to_row(sm))

        all_wtil_val.append(wtil)
        all_err_obs_val.append(arr.get("err_obs", np.zeros((wtil.shape[0],), dtype=np.float32)).astype(np.float32))
        all_y_val.append(y_seq.astype(np.float32))

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

    # correlation spearman(wtil, -err)
    try:
        from scipy.stats import spearmanr  # type: ignore

        wcat = np.concatenate(all_wtil_val, axis=0) if all_wtil_val else np.zeros((0,), dtype=np.float32)
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
        wcat = np.concatenate(all_wtil_val, axis=0) if all_wtil_val else np.zeros((0,), dtype=np.float32)
        ycat = np.concatenate(all_y_val, axis=0) if all_y_val else np.zeros((0,), dtype=np.float32)

        # gate bins on VAL
        if wcat.size == ycat.size and wcat.size > 0:
            bins = np.linspace(0.0, 1.0, 11)
            xs, accs = [], []
            for i in range(10):
                lo, hi = bins[i], bins[i + 1]
                msk = (wcat >= lo) & (wcat < hi) if i < 9 else (wcat >= lo) & (wcat <= hi)
                if int(msk.sum()) == 0:
                    continue
                xs.append(0.5 * (lo + hi))
                accs.append(float(ycat[msk].mean()))
            if len(xs) > 0:
                plt.figure()
                plt.plot(xs, accs, marker="o")
                plt.xlabel("wtil bin center")
                plt.ylabel("empirical P(y=1)")
                plt.title("Gate bins (reliability semantics, val)")
                plt.tight_layout()
                plt.savefig(analysis_dir / "fig_gate_bins_val.png", dpi=160)
                plt.close()

        # tau / low_frac hist
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

        # feature hist (train, per-dim 1%-99% clipped) on RAW (before scaling)
        try:
            plt.figure()
            Xh = X_tr_k_raw  # raw kept features
            for j in range(Xh.shape[1]):
                x = Xh[:, j].astype(np.float64)
                lo = float(np.quantile(x, 0.01))
                hi = float(np.quantile(x, 0.99))
                x_clip = np.clip(x, lo, hi)
                plt.hist(x_clip, bins=30, alpha=0.5, label=f"dim{j}")
            plt.legend()
            plt.title("Feature hist (train, per-dim clipped 1%-99%)")
            plt.tight_layout()
            plt.savefig(analysis_dir / "fig_feat_hist.png", dpi=160)
            plt.close()
        except Exception as e:
            print(f"[Warn] failed to draw feature hist: {e}")

        print(f"[OK] wrote figs into: {analysis_dir}")


# ============================================================
# CLI entry
# ============================================================
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
        f"[OK] Stage2 done: train_npz= {len(list(out_train_s2.glob('*.npz')))} "
        f"val_npz= {len(list(out_val_s2.glob('*.npz')))}"
    )

    stage3_train_eval(cfg, stage2_train=out_train_s2, stage2_val=out_val_s2, out_dir=out_dir_s3)
