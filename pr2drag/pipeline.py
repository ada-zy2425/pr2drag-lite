
from __future__ import annotations

import os
import warnings
import sys
import math
import json
import platform
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

# --- robust headless plotting (CLI/Colab-safe when saving figs)
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402

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
    resolve_davis_root,  # keep import for compatibility; we use local fallback too.
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
    """
    More defensive resolver than utils.resolve_davis_root.
    Accepts:
      - exact DAVIS root containing JPEGImages/Annotations/ImageSets
      - parent that contains DAVIS/ with those subdirs
    """
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
        "version": "0.3.1-lite-risk-wcalib-tieaware+reliability_ci_fix+oracle_segments_audit_v1+segment_routing_v1",
    }
    return sha1_of_dict(pick)


# ============================================================
# Helpers (audit / robustness)
# ============================================================
def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _try_git_head() -> Optional[str]:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.STDOUT)
        s = out.decode("utf-8", errors="ignore").strip()
        return s if s else None
    except Exception:
        return None


def _safe_quantile(x: np.ndarray, q: float, *, default: float = float("nan")) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return float(default)
    return float(np.quantile(x, float(q)))


def _segments_from_mask(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Return inclusive segments [t0, t1] where mask is True."""
    m = np.asarray(mask, dtype=bool).reshape(-1)
    n = int(m.size)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0)] if bool(m[0]) else []
    dm = np.diff(m.astype(np.int8))
    starts = (np.where(dm == 1)[0] + 1).astype(int).tolist()
    ends = np.where(dm == -1)[0].astype(int).tolist()
    if bool(m[0]):
        starts = [0] + starts
    if bool(m[-1]):
        ends = ends + [n - 1]
    if len(starts) != len(ends):
        # Should not happen, but guard anyway.
        k = min(len(starts), len(ends))
        starts, ends = starts[:k], ends[:k]
    return list(zip(starts, ends))


def _read_meta_json(path: Path) -> Dict[str, Any]:
    """
    Robust meta.json reader.
    - avoids pandas JSON quirks
    - returns {} on missing
    """
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            return obj
        return {"_raw": obj}
    except Exception as e:
        raise RuntimeError(f"[Meta] failed to read json: {path} ({e})") from e


def _npz_read_safe(npz_path: Path, *, context: str = "") -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Wrap utils.npz_read with more actionable error messages for common corruption / mismatch cases.
    """
    try:
        arr, meta = npz_read(npz_path)
        if not isinstance(arr, dict) or not isinstance(meta, dict):
            raise ValueError("npz_read did not return (dict, dict)")
        return arr, meta
    except Exception as e:
        msg = (
            f"[NPZ] failed to read npz: {npz_path}\n"
            f"  context: {context}\n"
            f"  error: {type(e).__name__}: {e}\n"
            "  likely causes:\n"
            "    - Stage1 interrupted -> wrote a partial/corrupt .npz\n"
            "    - file size is 0 or upload/drive sync truncated it\n"
            "    - mixing old/new cache signature with incompatible npz schema\n"
            "  fixes:\n"
            "    1) delete the bad file and rerun Stage1 for that split\n"
            "    2) if many are bad, delete the whole stage1 output dir and rerun\n"
        )
        raise RuntimeError(msg) from e


# ============================================================
# Segment-level routing (safety filter: bridge -> abstain only)
# ============================================================
@dataclass
class _SegRoutingCfg:
    enable: bool = False
    mode: str = "train_quantile"          # fixed | train_quantile
    disp_norm_q: float = 0.95             # used when mode=train_quantile
    disp_norm: Optional[float] = None     # used when mode=fixed

    # hard constraints
    Lmax: int = 10
    require_both_sides: bool = True

    # optional speed gate
    use_speed: bool = False
    speed_norm_q: float = 0.97
    speed_norm: Optional[float] = None

    # ---- NEW: soft decision for disp_too_large ----
    # if disp_too_large but segment is short and has both boundaries, we can do safe-bridge instead of abstain
    safe_bridge_max_len: int = 8          # L <= this => eligible for safe-bridge
    disp_relax_max: Optional[float] = 0.40  # if not None, require dnorm <= this for safe-bridge
    # supported: linear_safe | hermite_safe
    bridge_mode_relaxed: str = "linear_safe"
    L_short: int = 2   # <=2 的低段默认走 safe-bridge(linear)

@dataclass
class _SegRoutingFit:
    disp_norm_thr: Optional[float] = None
    speed_norm_thr: Optional[float] = None
    n_segments: int = 0
    n_valid_boundary: int = 0


def _diag_from_hw(hw: np.ndarray) -> float:
    hw = np.asarray(hw, dtype=np.float64).reshape(-1)
    if hw.size >= 2:
        h, w = float(hw[0]), float(hw[1])
        return float(math.sqrt(h * h + w * w) + 1e-6)
    return float("nan")


def _fit_segment_routing_thresholds_train(
    *,
    npz_dir: Path,
    cfg: _SegRoutingCfg,
    tau_mode: str,
    tau_global: float,
    target_frac: float,
    tau_min: float,
    tau_max: float,
    seq_post_w_raw_fn,   # callable(npz_path)->(w_raw,p,y)
    w_to_wtil_fn,        # callable(w_raw)->wtil
    analysis_dir: Path,
) -> _SegRoutingFit:
    """
    Fit routing thresholds on TRAIN ONLY to avoid val leakage.
    We use low-segments induced by (wtil < tau) and collect boundary_disp_norm / speed_norm.
    """
    fit = _SegRoutingFit()
    if not cfg.enable:
        return fit

    dnorm_all: List[float] = []
    vnorm_all: List[float] = []
    nseg = 0
    nvalid = 0

    for npz_path in sorted(Path(npz_dir).glob("*.npz")):
        arr, _meta = _npz_read_safe(npz_path, context="fit_segment_routing_thresholds_train")

        w_raw, _p, _y = seq_post_w_raw_fn(npz_path)
        wtil = w_to_wtil_fn(w_raw)
        if wtil.size == 0:
            continue

        if tau_mode == "per_video":
            tau = float(np.quantile(wtil.astype(np.float64), target_frac))
            tau = clamp(tau, tau_min, tau_max)
        else:
            tau = float(tau_global)

        low = (wtil < tau)
        segs = _segments_from_mask(low)
        if len(segs) == 0:
            continue

        z_obs = arr.get("z_obs", None)
        hw = arr.get("img_hw", None)
        if z_obs is None or hw is None:
            continue
        z_obs = np.asarray(z_obs, dtype=np.float32)
        diag = _diag_from_hw(hw)
        if not np.isfinite(diag) or diag <= 0:
            continue

        for (t0, t1) in segs:
            nseg += 1
            L = int(t1 - t0 + 1)
            left = t0 - 1
            right = t1 + 1
            has_left = (left >= 0) and (not bool(low[left]))
            has_right = (right < int(wtil.size)) and (not bool(low[right]))

            if cfg.require_both_sides and not (has_left and has_right):
                continue
            if not (has_left and has_right):
                continue

            dpx = float(np.linalg.norm((z_obs[left] - z_obs[right]).astype(np.float64)))
            dnorm = float(dpx / diag)
            if not np.isfinite(dnorm):
                continue

            nvalid += 1
            dnorm_all.append(dnorm)

            if cfg.use_speed:
                vnorm_all.append(float(dnorm / max(L, 1)))

    fit.n_segments = int(nseg)
    fit.n_valid_boundary = int(nvalid)

    mode = str(cfg.mode).lower()
    if mode == "fixed":
        fit.disp_norm_thr = None if cfg.disp_norm is None else float(cfg.disp_norm)
        fit.speed_norm_thr = None if (not cfg.use_speed or cfg.speed_norm is None) else float(cfg.speed_norm)

    elif mode == "train_quantile":
        if len(dnorm_all) > 0:
            q = float(np.clip(cfg.disp_norm_q, 0.0, 1.0))
            fit.disp_norm_thr = float(np.quantile(np.asarray(dnorm_all, dtype=np.float64), q))
        else:
            fit.disp_norm_thr = None

        if cfg.use_speed:
            if len(vnorm_all) > 0:
                qv = float(np.clip(cfg.speed_norm_q, 0.0, 1.0))
                fit.speed_norm_thr = float(np.quantile(np.asarray(vnorm_all, dtype=np.float64), qv))
            else:
                fit.speed_norm_thr = None
    else:
        raise ValueError(f"[SegRouting] unknown mode={cfg.mode}, expected fixed|train_quantile")

    # write fit audit
    try:
        write_json(analysis_dir / "segment_routing_fit_v1.json", {
            "enable": bool(cfg.enable),
            "mode": cfg.mode,
            "disp_norm_q": float(cfg.disp_norm_q),
            "disp_norm_thr": None if fit.disp_norm_thr is None else float(fit.disp_norm_thr),
            "Lmax": int(cfg.Lmax),
            "require_both_sides": bool(cfg.require_both_sides),
            "use_speed": bool(cfg.use_speed),
            "speed_norm_q": float(cfg.speed_norm_q),
            "speed_norm_thr": None if fit.speed_norm_thr is None else float(fit.speed_norm_thr),
            "n_segments_total": int(fit.n_segments),
            "n_valid_boundary": int(fit.n_valid_boundary),
            "safe_bridge_max_len": int(getattr(cfg, "safe_bridge_max_len", 8)),
            "disp_relax_max": None if getattr(cfg, "disp_relax_max", None) is None else float(getattr(cfg, "disp_relax_max")),
            "bridge_mode_relaxed": str(getattr(cfg, "bridge_mode_relaxed", "linear_safe")).lower(),
            "L_short": int(getattr(cfg, "L_short", 0)),
        })
    except Exception as e:
        print(f"[Warn] failed to write segment_routing_fit_v1.json: {e}")

    if fit.disp_norm_thr is None and not cfg.require_both_sides:
        print("[SegRouting] no valid boundary segments on train -> routing likely no-op (disp_norm_thr=None)")
    else:
        if fit.disp_norm_thr is not None:
            print(f"[SegRouting] fitted disp_norm_thr={fit.disp_norm_thr:.6f} (train-only)")
        if cfg.use_speed and fit.speed_norm_thr is not None:
            print(f"[SegRouting] fitted speed_norm_thr={fit.speed_norm_thr:.6f} (train-only)")
        if cfg.require_both_sides:
            print("[SegRouting] require_both_sides=True -> missing-boundary segments will be forced abstain")

    return fit

def _segment_routing_decide_masks(
    *,
    low_by_tau: np.ndarray,
    z_obs: np.ndarray,
    img_hw: np.ndarray,
    cfg: _SegRoutingCfg,
    fit: _SegRoutingFit,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, Dict[str, Any]]]:
    low = np.asarray(low_by_tau, dtype=bool).reshape(-1)
    T = int(low.size)

    force_abst = np.zeros((T,), dtype=bool)
    force_safe = np.zeros((T,), dtype=bool)
    segs = _segments_from_mask(low)

    z_obs = np.asarray(z_obs, dtype=np.float32)
    diag = _diag_from_hw(img_hw)

    decisions: Dict[int, Dict[str, Any]] = {}

    # thresholds (may be None)
    disp_thr = None if fit.disp_norm_thr is None else float(fit.disp_norm_thr)
    speed_thr = None if fit.speed_norm_thr is None else float(fit.speed_norm_thr)

    safe_L = int(getattr(cfg, "safe_bridge_max_len", 8))
    L_short = int(getattr(cfg, "L_short", 0))
    disp_relax_max = getattr(cfg, "disp_relax_max", None)
    disp_relax_max = None if disp_relax_max is None else float(disp_relax_max)
    mode_relaxed = str(getattr(cfg, "bridge_mode_relaxed", "linear_safe")).lower()
    SUPPORTED_SAFE = {"linear_safe", "hermite_safe"}
    if mode_relaxed not in SUPPORTED_SAFE:
        mode_relaxed = "linear_safe"
    if not np.isfinite(diag) or diag <= 0:
        # no geometry => do nothing (auto), but still record
        for seg_id, (t0, t1) in enumerate(segs):
            decisions[int(seg_id)] = {
                "decision": "auto",
                "reason": "bad_diag",
                "boundary_disp_norm": float("nan"),
                "speed_norm": float("nan"),
                "has_left": 0,
                "has_right": 0,
                "L": int(t1 - t0 + 1),
            }
        return force_abst, force_safe, decisions

    for seg_id, (t0, t1) in enumerate(segs):
        L = int(t1 - t0 + 1)
        left = t0 - 1
        right = t1 + 1

        has_left = (left >= 0) and (not bool(low[left]))
        has_right = (right < T) and (not bool(low[right]))

        # compute boundary stats if possible
        dnorm = float("nan")
        vnorm = float("nan")
        if has_left and has_right:
            dpx = float(np.linalg.norm((z_obs[left] - z_obs[right]).astype(np.float64)))
            dnorm = float(dpx / diag)
            vnorm = float(dnorm / max(L, 1))

        # ---- Priority 1: missing boundary (if required) -> abstain
        if cfg.require_both_sides and not (has_left and has_right):
            force_abst[t0:t1 + 1] = True
            decisions[int(seg_id)] = {
                "decision": "force_abstain",
                "reason": "missing_boundary",
                "boundary_disp_norm": dnorm,
                "speed_norm": vnorm,
                "has_left": int(has_left),
                "has_right": int(has_right),
                "L": int(L),
            }
            continue

        # ---- Priority 2: too long -> abstain
        if L > int(cfg.Lmax):
            force_abst[t0:t1 + 1] = True
            decisions[int(seg_id)] = {
                "decision": "force_abstain",
                "reason": "too_long",
                "boundary_disp_norm": dnorm,
                "speed_norm": vnorm,
                "has_left": int(has_left),
                "has_right": int(has_right),
                "L": int(L),
            }
            continue

        # ---- Priority 3: very short segment -> safe-bridge (only if we have both boundaries)
        if (L_short > 0) and (L <= L_short) and (has_left and has_right):
            force_safe[t0:t1 + 1] = True
            decisions[int(seg_id)] = {
                "decision": "force_safe_bridge",
                "reason": "short_seg",
                "boundary_disp_norm": dnorm,
                "speed_norm": vnorm,
                "has_left": 1,
                "has_right": 1,
                "L": int(L),
                "bridge_mode_relaxed": mode_relaxed,
            }
            continue

        # helper: safe-bridge eligibility
        def _eligible_safe_bridge() -> bool:
            if not (has_left and has_right):
                return False
            if L > safe_L:
                return False
            if disp_relax_max is not None and np.isfinite(dnorm):
                if float(dnorm) > float(disp_relax_max):
                    return False
            return True

        # ---- Speed gate: prefer safe-bridge (not abstain)
        if cfg.use_speed and (speed_thr is not None) and np.isfinite(vnorm):
            if float(vnorm) > float(speed_thr):
                if _eligible_safe_bridge():
                    force_safe[t0:t1 + 1] = True
                    decisions[int(seg_id)] = {
                        "decision": "force_safe_bridge",
                        "reason": "speed_too_large",
                        "boundary_disp_norm": dnorm,
                        "speed_norm": vnorm,
                        "has_left": int(has_left),
                        "has_right": int(has_right),
                        "L": int(L),
                        "bridge_mode_relaxed": mode_relaxed,
                    }
                    continue
                else:
                    force_abst[t0:t1 + 1] = True
                    decisions[int(seg_id)] = {
                        "decision": "force_abstain",
                        "reason": "speed_too_large",
                        "boundary_disp_norm": dnorm,
                        "speed_norm": vnorm,
                        "has_left": int(has_left),
                        "has_right": int(has_right),
                        "L": int(L),
                    }
                    continue

        # ---- Disp gate: safe-bridge if eligible else abstain
        if (disp_thr is not None) and np.isfinite(dnorm):
            if float(dnorm) > float(disp_thr):
                if _eligible_safe_bridge():
                    force_safe[t0:t1 + 1] = True
                    decisions[int(seg_id)] = {
                        "decision": "force_safe_bridge",
                        "reason": "disp_relaxed",
                        "boundary_disp_norm": dnorm,
                        "speed_norm": vnorm,
                        "has_left": int(has_left),
                        "has_right": int(has_right),
                        "L": int(L),
                        "bridge_mode_relaxed": mode_relaxed,
                    }
                else:
                    force_abst[t0:t1 + 1] = True
                    decisions[int(seg_id)] = {
                        "decision": "force_abstain",
                        "reason": "disp_too_large",
                        "boundary_disp_norm": dnorm,
                        "speed_norm": vnorm,
                        "has_left": int(has_left),
                        "has_right": int(has_right),
                        "L": int(L),
                    }
                continue

        # ---- default: auto
        decisions[int(seg_id)] = {
            "decision": "auto",
            "reason": "",
            "boundary_disp_norm": dnorm,
            "speed_norm": vnorm,
            "has_left": int(has_left),
            "has_right": int(has_right),
            "L": int(L),
        }

    # safety: only apply inside low frames, disjoint
    force_abst = force_abst & low
    force_safe = force_safe & low
    force_safe = force_safe & (~force_abst)
    return force_abst, force_safe, decisions

# ============================================================
# Stage1
# ============================================================
def stage1_precompute_split(cfg: Dict[str, Any], *, split: str, out_dir: Path) -> Dict[str, Any]:
    # use local robust resolver; keep utils.resolve_davis_root compatibility as fallback
    try:
        davis_root = _resolve_davis_root(cfg["davis_root"])
    except Exception:
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


# ============================================================
# Stage2
# ============================================================
def stage2_build_features_for_seq(arr: Dict[str, np.ndarray], *, feat_mode: str) -> np.ndarray:
    # schema checks (better error than KeyError downstream)
    need = ["iou_shift", "cycle_err", "area_change", "occl_flag", "blur_flag", "blur_inv", "motion", "img_hw"]
    missing = [k for k in need if k not in arr]
    if missing:
        raise KeyError(
            f"[Stage2] missing keys in Stage1 npz: {missing}\n"
            "  likely you are mixing an older Stage1 cache with the new pipeline.\n"
            "  fix: delete stage1 output dir and rerun Stage1."
        )

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
    meta = _read_meta_json(meta_path) if meta_path.exists() else {}
    seqs = list(meta.get("seqs", [])) if isinstance(meta.get("seqs", []), list) else []
    if not seqs:
        seqs = sorted([p.stem for p in stage1_dir.glob("*.npz")])

    pbar = tqdm(seqs, desc=f"Stage2({split})")
    for seq in pbar:
        in_npz = stage1_dir / f"{seq}.npz"
        out_npz = out_dir / f"{seq}.npz"
        if not in_npz.exists():
            raise FileNotFoundError(f"[Stage2] Missing Stage1 npz: {in_npz}")

        if cache and out_npz.exists():
            _arr2, m2 = _npz_read_safe(out_npz, context="stage2 cache check")
            if m2.get("signature", "") == sig and m2.get("feat_mode", "") == feat_mode:
                pbar.write(f"[skip] Stage2 exists (compatible): {out_npz.name}")
                continue
            else:
                pbar.write(f"[recompute] Stage2 stale/mismatch -> {out_npz.name}")

        arr, _m1 = _npz_read_safe(in_npz, context="stage2 read stage1 npz")
        X = stage2_build_features_for_seq(arr, feat_mode=feat_mode)

        if "y" not in arr:
            raise KeyError(
                f"[Stage2] Stage1 npz missing y: {in_npz}\n"
                "  fix: delete stage1 cache and rerun Stage1."
            )

        y = arr["y"].astype(np.uint8)
        err_obs = arr.get("err_obs", np.zeros((y.shape[0],), dtype=np.float32)).astype(np.float32)

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

    out_meta = {"split": split, "signature": sig, "feat_mode": feat_mode, "num_seqs": len(seqs), "seqs": seqs}
    try:
        write_json(out_dir / "meta.json", out_meta)
    except Exception as e:
        print(f"[Warn] failed to write Stage2 meta.json: {e}")
    return out_meta


def _concat_stage2(npz_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    Xs, ys, errs = [], [], []
    for p in sorted(Path(npz_dir).glob("*.npz")):
        arr, _ = _npz_read_safe(p, context="_concat_stage2")
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
# Reliability plotting helpers (FIXED: yerr >= 0 always)
# ============================================================
def _z_value(alpha: float) -> float:
    a = float(alpha)
    if abs(a - 0.05) < 1e-12:
        return 1.959963984540054  # 95%
    try:
        from scipy.stats import norm  # type: ignore
        return float(norm.ppf(1.0 - a / 2.0))
    except Exception:
        return 1.959963984540054


def _binomial_ci_wilson(k: int, n: int, *, alpha: float = 0.05) -> Tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    k = int(k)
    n = int(n)
    p = float(k) / float(n)
    z = _z_value(alpha)
    z2 = z * z
    den = 1.0 + z2 / float(n)
    center = (p + z2 / (2.0 * float(n))) / den
    half = (z * math.sqrt((p * (1.0 - p) / float(n)) + (z2 / (4.0 * float(n) * float(n))))) / den
    lo = center - half
    hi = center + half
    lo = float(np.clip(lo, 0.0, 1.0))
    hi = float(np.clip(hi, 0.0, 1.0))
    lo = min(lo, p)
    hi = max(hi, p)
    return lo, hi


def _reliability_bins(
    w: np.ndarray,
    y: np.ndarray,
    *,
    nbins: int = 10,
    alpha: float = 0.05,
    min_count: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      x: bin centers
      p: empirical mean(y)
      lo, hi: Wilson CI
      cnt: bin counts
    """
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if w.size == 0 or y.size != w.size:
        return (
            np.zeros((0,), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
            np.zeros((0,), dtype=np.int64),
        )

    edges = np.linspace(0.0, 1.0, int(nbins) + 1)
    xs: List[float] = []
    ps: List[float] = []
    los: List[float] = []
    his: List[float] = []
    cnts: List[int] = []

    for i in range(int(nbins)):
        lo_e, hi_e = float(edges[i]), float(edges[i + 1])
        if i < int(nbins) - 1:
            m = (w >= lo_e) & (w < hi_e)
        else:
            m = (w >= lo_e) & (w <= hi_e)

        n = int(np.sum(m))
        if n < int(min_count):
            continue

        yy = y[m]
        k = int(np.sum(yy >= 0.5))
        p = float(k) / float(n)
        lo_ci, hi_ci = _binomial_ci_wilson(k, n, alpha=alpha)

        xs.append(0.5 * (lo_e + hi_e))
        ps.append(p)
        los.append(lo_ci)
        his.append(hi_ci)
        cnts.append(n)

    return (
        np.asarray(xs, dtype=np.float64),
        np.asarray(ps, dtype=np.float64),
        np.asarray(los, dtype=np.float64),
        np.asarray(his, dtype=np.float64),
        np.asarray(cnts, dtype=np.int64),
    )


def _plot_reliability(
    w: np.ndarray,
    y: np.ndarray,
    *,
    out_path: Union[str, Path],
    title: str,
    xlabel: str = "w bin center",
    ylabel: str = "empirical P(y=1)",
    nbins: int = 10,
    alpha: float = 0.05,
    min_count: int = 1,
) -> None:
    out_path = Path(out_path)
    x, p, lo, hi, cnt = _reliability_bins(w, y, nbins=nbins, alpha=alpha, min_count=min_count)
    if x.size == 0:
        return

    p = np.clip(p, 0.0, 1.0)
    lo = np.clip(lo, 0.0, 1.0)
    hi = np.clip(hi, 0.0, 1.0)

    lo = np.minimum(lo, p)
    hi = np.maximum(hi, p)

    yerr_low = np.clip(p - lo, 0.0, np.inf)
    yerr_high = np.clip(hi - p, 0.0, np.inf)
    yerr = np.vstack([yerr_low, yerr_high])

    m = np.isfinite(x) & np.isfinite(p) & np.isfinite(yerr_low) & np.isfinite(yerr_high)
    x, p, yerr = x[m], p[m], yerr[:, m]
    if x.size == 0:
        return

    plt.figure()
    plt.errorbar(x, p, yerr=yerr, fmt="o-")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.ylim(-0.02, 1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


# ============================================================
# w calibration (isotonic preferred, fallback to binning)
# ============================================================
class _WCalibrator:
    def __init__(self, method: str = "isotonic", bins: int = 50, eps: float = 1e-8) -> None:
        self.method = str(method).lower()
        self.bins = int(bins)
        self.eps = float(eps)

        self._iso = None
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


# ============================================================
# tau selection (tie-aware)
# ============================================================
def _select_tau_risk_constrained_tieaware(
    w: np.ndarray,
    y: np.ndarray,
    *,
    r0: float,
    min_keep_frac: float = 0.05,
    tau_min: float = 1e-6,
    tau_max: float = 0.999999,
) -> Tuple[float, Dict[str, float], pd.DataFrame]:
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    n = int(w.size)
    if n == 0 or y.size != n:
        tau = clamp(0.5, tau_min, tau_max)
        df = pd.DataFrame({"tau": [], "k": [], "coverage": [], "risk": []})
        return tau, {"coverage": float("nan"), "risk": float("nan")}, df

    idx = np.argsort(-w)
    ws = w[idx]
    ys = y[idx]
    bad = 1 - ys
    cum_bad = np.cumsum(bad, dtype=np.float64)

    change = np.r_[ws[1:] != ws[:-1], True]
    ends = np.where(change)[0]

    ks = (ends + 1).astype(np.int64)
    risks = (cum_bad[ends] / ks.astype(np.float64))
    covs = (ks.astype(np.float64) / float(n))
    taus = ws[ends]

    min_keep = max(1, int(math.ceil(float(min_keep_frac) * n)))
    feasible = (ks >= min_keep) & (risks <= float(r0))

    if np.any(feasible):
        i_star = int(np.where(feasible)[0].max())
    else:
        valid = ks >= min_keep
        if np.any(valid):
            i_star = int(np.where(valid)[0][np.argmin(risks[valid])])
        else:
            i_star = 0

    tau = float(taus[i_star])
    tau = clamp(tau, tau_min, tau_max)

    df = pd.DataFrame({"tau": taus, "k": ks, "coverage": covs, "risk": risks})
    summary = {
        "coverage": float(covs[i_star]),
        "risk": float(risks[i_star]),
        "k": float(ks[i_star]),
        "n": float(n),
        "min_keep": float(min_keep),
    }
    return tau, summary, df


# ============================================================
# Stage3


def _safe_bridge_linear(*, z_obs: np.ndarray, z_fin: np.ndarray, t0: int, t1: int, left: int, right: int) -> None:
    """Linear interpolation between z_obs[left] and z_obs[right], fill only t0..t1."""
    Tseq = int(z_obs.shape[0])
    if left < 0 or right >= Tseq or right <= left:
        return
    zL = z_obs[left].astype(np.float32)
    zR = z_obs[right].astype(np.float32)
    denom = float(right - left)
    if denom <= 0:
        return
    for t in range(int(t0), int(t1) + 1):
        a = float(t - left) / denom
        a = float(np.clip(a, 0.0, 1.0))
        z_fin[t] = (1.0 - a) * zL + a * zR


def _safe_bridge_hermite(*, z_obs: np.ndarray, z_fin: np.ndarray, t0: int, t1: int, left: int, right: int, clamp: bool = True) -> None:
    """Safe cubic Hermite between endpoints with conservative tangents; clamps to avoid overshoot."""
    Tseq = int(z_obs.shape[0])
    if left < 0 or right >= Tseq or right <= left:
        return
    zL = z_obs[left].astype(np.float32)
    zR = z_obs[right].astype(np.float32)

    # conservative tangents
    m0 = (zR - zL).astype(np.float32)
    m1 = (zR - zL).astype(np.float32)

    denom = float(right - left)
    if denom <= 0:
        return

    if clamp:
        lo = np.minimum(zL, zR)
        hi = np.maximum(zL, zR)

    for t in range(int(t0), int(t1) + 1):
        s = float(t - left) / denom
        s = float(np.clip(s, 0.0, 1.0))
        s2 = s * s
        s3 = s2 * s
        h00 = 2.0 * s3 - 3.0 * s2 + 1.0
        h10 = s3 - 2.0 * s2 + s
        h01 = -2.0 * s3 + 3.0 * s2
        h11 = s3 - s2
        z = h00 * zL + h10 * m0 + h01 * zR + h11 * m1
        if clamp:
            z = np.clip(z, lo, hi)
        z_fin[t] = z.astype(np.float32, copy=False)


def _apply_safe_bridge(*, mode: str, z_obs: np.ndarray, z_fin: np.ndarray, t0: int, t1: int, left: int, right: int) -> str:
    """Apply safe bridge override. Supported: linear_safe | hermite_safe. Returns effective mode."""
    m = str(mode).lower().strip()
    if m in ("linear", "linear_safe"):
        _safe_bridge_linear(z_obs=z_obs, z_fin=z_fin, t0=t0, t1=t1, left=left, right=right)
        return "linear_safe"
    if m in ("hermite", "hermite_safe"):
        _safe_bridge_hermite(z_obs=z_obs, z_fin=z_fin, t0=t0, t1=t1, left=left, right=right, clamp=True)
        return "hermite_safe"
    warnings.warn(f"[SegRouting] unknown bridge_mode_relaxed='{mode}', fallback to linear_safe")
    _safe_bridge_linear(z_obs=z_obs, z_fin=z_fin, t0=t0, t1=t1, left=left, right=right)
    return "linear_safe"
# ============================================================
def stage3_train_eval(cfg: Dict[str, Any], *, stage2_train: Path, stage2_val: Path, out_dir: Path) -> None:
    out_dir = safe_mkdir(out_dir)
    analysis_dir = safe_mkdir(out_dir / "_analysis")

    # --- run manifest (helps reproducibility / meeting)
    try:
        manifest = {
            "time_utc": _now_utc_iso(),
            "python": sys.version,
            "platform": platform.platform(),
            "argv": list(sys.argv),
            "git_head": _try_git_head(),
            "out_dir": str(out_dir),
            "stage2_train": str(stage2_train),
            "stage2_val": str(stage2_val),
        }
        write_json(analysis_dir / "run_manifest_v1.json", manifest)
        write_json(analysis_dir / "config_resolved_v1.json", cfg)
    except Exception as e:
        print(f"[Warn] failed to write run manifest/config: {e}")

    s3 = cfg.get("stage3", {})
    hmm_cfg = s3.get("hmm", {})
    em_cfg = s3.get("emission", {})
    aob_cfg = s3.get("aob", {})
    met_cfg = s3.get("metrics", {})
    ana_cfg = s3.get("analysis", {})

    # --- Segment routing config (safety filter)
    sr_raw = dict(s3.get("segment_routing", {}))
    seg_route_cfg = _SegRoutingCfg(
        enable=bool(sr_raw.get("enable", False)),
        mode=str(sr_raw.get("mode", "train_quantile")).lower(),
        disp_norm_q=float(sr_raw.get("disp_norm_q", 0.95)),
        disp_norm=sr_raw.get("disp_norm", None),
        Lmax=int(sr_raw.get("Lmax", 10)),
        require_both_sides=bool(sr_raw.get("require_both_sides", True)),
        use_speed=bool(sr_raw.get("use_speed", False)),
        speed_norm_q=float(sr_raw.get("speed_norm_q", 0.97)),
        speed_norm=sr_raw.get("speed_norm", None),
        L_short=int(sr_raw.get("L_short", 2)),

        # NEW
        safe_bridge_max_len=int(sr_raw.get("safe_bridge_max_len", 8)),
        disp_relax_max=sr_raw.get("disp_relax_max", 0.40),
        bridge_mode_relaxed=str(sr_raw.get("bridge_mode_relaxed", "linear_safe")).lower(),
    )
    seg_route_fit = _SegRoutingFit()

    tau_mode = str(s3.get("tau_mode", "global")).lower()
    target_frac = float(s3.get("tau_target_frac", 0.25))
    tau_min = float(s3.get("tau_min", 1e-6))
    tau_max = float(s3.get("tau_max", 0.999999))

    r0 = float(s3.get("tau_risk", 0.05))
    min_keep_frac = float(s3.get("tau_min_keep_frac", 0.05))

    sc_cfg = dict(s3.get("feature_scaler", {}))
    sc_enable = bool(sc_cfg.get("enable", True))
    sc_type = str(sc_cfg.get("type", "robust_iqr")).lower()
    sc_clip = float(sc_cfg.get("clip", 8.0))
    sc_eps = float(sc_cfg.get("eps", 1e-6))

    wc_cfg = dict(s3.get("w_calibration", {}))
    wc_enable = bool(wc_cfg.get("enable", False))
    wc_method = str(wc_cfg.get("method", "isotonic")).lower()
    wc_bins = int(wc_cfg.get("bins", 50))

    # --- NEW: w override for ablations ("" | "ones")
    w_override = str(s3.get("w_override", "")).lower().strip()
    if w_override not in ("", "none", "ones"):
        raise ValueError(f"[Stage3] unknown w_override={w_override}, expected ''|'ones'")
    w_override = "" if w_override in ("", "none") else w_override

    # --- NEW: oracle reliability switch (for mechanism validation ONLY)
    oracle_cfg = dict(s3.get("oracle", {}))
    oracle_enable = bool(oracle_cfg.get("enable", False))
    oracle_source = str(oracle_cfg.get("source", "y")).lower()  # y | err
    oracle_tau = float(oracle_cfg.get("tau", 0.5))
    oracle_err_px = float(oracle_cfg.get("err_px", met_cfg.get("fail_px", 50.0)))

    if oracle_enable:
        if tau_mode != "oracle":
            print(f"[Oracle] overriding tau_mode={tau_mode} -> tau fixed to oracle_tau={oracle_tau}")
        tau_mode = "oracle"
        if wc_enable:
            print("[Oracle] disabling w_calibration (oracle overrides wtil anyway).")
        wc_enable = False

    # --- NEW: allow disabling HMM smoothing (ablation)
    hmm_enable = bool(hmm_cfg.get("enable", True))
    if not hmm_enable:
        print("[HMM] disabled -> use framewise w=p (no forward-backward).")

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

    X_tr, y_tr, _ = _concat_stage2(stage2_train)
    X_va, y_va, _ = _concat_stage2(stage2_val)

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

    try:
        write_json(analysis_dir / "feature_stats_train.json", _feature_stats(X_tr_k_raw))
    except Exception as e:
        print(f"[Warn] failed to write feature_stats_train.json: {e}")

    scaler: Optional[Dict[str, Any]] = None
    if sc_enable:
        if sc_type != "robust_iqr":
            raise ValueError(f"[Scaler] unknown type={sc_type}, expected robust_iqr")
        scaler = _robust_iqr_fit(X_tr_k_raw, eps=sc_eps)
        X_tr_k = _robust_iqr_apply(X_tr_k_raw, scaler, clip=sc_clip)
        X_va_k = _robust_iqr_apply(X_va_k_raw, scaler, clip=sc_clip)
        scaler_dump = {**scaler, "clip": float(sc_clip), "keep_mask": keep.astype(int).tolist()}
        write_json(analysis_dir / "feature_scaler.json", scaler_dump)
        print(f"[Feat] robust scaling enabled (clip={sc_clip}) -> wrote scaler to _analysis/feature_scaler.json")
    else:
        X_tr_k, X_va_k = X_tr_k_raw, X_va_k_raw
        print("[Feat] scaling disabled")

    clf = fit_emission_model(
        X_tr_k,
        y_tr,
        c=float(em_cfg.get("c", 1.0)),
        max_iter=int(em_cfg.get("max_iter", 2000)),
    )

    p_tr_raw = clf.predict_proba(X_tr_k)[:, 1].astype(np.float32)
    p_va_raw = clf.predict_proba(X_va_k)[:, 1].astype(np.float32)

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


    def _infer_w_from_p(p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=np.float32).reshape(-1)
        if p.size == 0:
            return p.astype(np.float32)
        if hmm_enable:
            return forward_backward_binary(p, hmm).astype(np.float32)
        return np.clip(p, 0.0, 1.0).astype(np.float32)

    def _seq_post_w_raw(npz_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        arr, _meta = _npz_read_safe(npz_path, context="_seq_post_w_raw")
        X = arr["X"][:, keep].astype(np.float32)
        if sc_enable and scaler is not None:
            X = _robust_iqr_apply(X, scaler, clip=sc_clip)
        p_raw = clf.predict_proba(X)[:, 1].astype(np.float32)
        p = apply_temperature_scaling(p_raw, T, eps=cal_eps) if cal_enable else p_raw
        w_raw = _infer_w_from_p(p)
        y = arr.get("y", np.zeros((w_raw.shape[0],), dtype=np.uint8)).astype(np.uint8)
        return w_raw.astype(np.float32), p.astype(np.float32), y

    wcal: Optional[_WCalibrator] = None
    if wc_enable:
        ws, ys = [], []
        for pth in sorted(Path(stage2_train).glob("*.npz")):
            w_raw, _, y = _seq_post_w_raw(pth)
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
            write_json(analysis_dir / "w_calibrator.json", wcal.to_dict())
            print(f"[W-Calib] enabled method={wc_method} bins={wc_bins}")

            if bool(ana_cfg.get("write_figs", True)):
                wtil = wcal.transform(w_all)
                _plot_reliability(
                    wtil,
                    y_all,
                    out_path=analysis_dir / "fig_reliability_train_wcalib.png",
                    title="Reliability curve on train (after w calibration)",
                    xlabel="wtil bin center (calibrated)",
                    nbins=10,
                    alpha=0.05,
                    min_count=1,
                )

    def _w_to_wtil(w_raw: np.ndarray) -> np.ndarray:
        w = w_raw
        if wc_enable and wcal is not None:
            w = wcal.transform(w)
        else:
            w = np.clip(w, 0.0, 1.0).astype(np.float32)
        if w_override == "ones":
            return np.ones_like(w, dtype=np.float32)
        return w.astype(np.float32, copy=False)

    tau_global: float = 0.5

    if tau_mode == "oracle":
        tau_global = clamp(float(oracle_tau), tau_min, tau_max)
        print(f"[Tau] oracle-fixed: tau_global={tau_global:.6f}")

    elif tau_mode == "global":
        ws = []
        for pth in sorted(Path(stage2_train).glob("*.npz")):
            w_raw, _, _ = _seq_post_w_raw(pth)
            ws.append(_w_to_wtil(w_raw))
        w_all = np.concatenate(ws, axis=0) if ws else np.zeros((0,), dtype=np.float32)
        if w_all.size == 0:
            tau_global = 0.5
        else:
            tau_global = float(np.quantile(w_all.astype(np.float64), target_frac))
            tau_global = clamp(tau_global, tau_min, tau_max)
        print(f"[Tau] global_quantile from train: tau_global={tau_global:.6f} (target_frac={target_frac})")

    elif tau_mode == "risk":
        ws, ys = [], []
        for pth in sorted(Path(stage2_train).glob("*.npz")):
            w_raw, _, y = _seq_post_w_raw(pth)
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
        )
        print(
            f"[Tau] risk-constrained: tau_global={tau_global:.6f}  "
            f"coverage={summ['coverage']:.3f}  risk={summ['risk']:.3f}  r0={r0}"
        )

        curve.to_csv(analysis_dir / "tau_risk_curve_train.csv", index=False)
        write_json(
            analysis_dir / "tau_risk_summary_train.json",
            {**summ, "tau": tau_global, "r0": r0, "min_keep_frac": min_keep_frac},
        )

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

    elif tau_mode == "per_video":
        print("[Tau] per_video mode enabled (legacy).")
        tau_global = float("nan")
    else:
        raise ValueError("Unknown tau_mode. Use: oracle | global | risk | per_video")

    # ---- Fit segment-routing thresholds on TRAIN ONLY (after tau is decided)
    if seg_route_cfg.enable:
        seg_route_fit = _fit_segment_routing_thresholds_train(
            npz_dir=Path(stage2_train),
            cfg=seg_route_cfg,
            tau_mode=tau_mode,
            tau_global=float(tau_global) if np.isfinite(tau_global) else float("nan"),
            target_frac=float(target_frac),
            tau_min=float(tau_min),
            tau_max=float(tau_max),
            seq_post_w_raw_fn=_seq_post_w_raw,
            w_to_wtil_fn=_w_to_wtil,
            analysis_dir=analysis_dir,
        )

    rows = []
    seg_rows: List[Dict[str, Any]] = []
    all_wtil_val = []
    all_err_obs_val = []
    all_y_val = []

    # --- Audit (stricter + more defensible)
    # leak_high: (is_bridge|is_abst) but wtil>=tau  => definition inconsistency (should be 0 ideally)
    # uncovered_low: (wtil<tau) but NOT(is_bridge|is_abst) => can be valid but must be reported
    leak_high_total = 0
    uncovered_low_total = 0

    pbar = tqdm(sorted(Path(stage2_val).glob("*.npz")), desc="Stage3(val)")
    for npz_path in pbar:
        arr, meta = _npz_read_safe(npz_path, context="stage3 val read")
        seq = meta.get("seq", npz_path.stem)

        X = arr["X"][:, keep].astype(np.float32)
        if sc_enable and scaler is not None:
            X = _robust_iqr_apply(X, scaler, clip=sc_clip)

        z_gt = arr["z_gt"].astype(np.float32)
        z_obs = arr["z_obs"].astype(np.float32)
        y_seq = arr.get("y", np.zeros((X.shape[0],), dtype=np.uint8)).astype(np.uint8)

        p_raw = clf.predict_proba(X)[:, 1].astype(np.float32)
        p = apply_temperature_scaling(p_raw, T, eps=cal_eps) if cal_enable else p_raw

        w_raw = _infer_w_from_p(p).astype(np.float32)
        wtil = _w_to_wtil(w_raw)

        if tau_mode in ("oracle", "global", "risk"):
            tau = float(tau_global)
        elif tau_mode == "per_video":
            tau = float(np.quantile(wtil.astype(np.float64), target_frac)) if wtil.size else 0.5
            tau = clamp(tau, tau_min, tau_max)
        else:
            raise ValueError(f"Unknown tau_mode={tau_mode}")

        # --- NEW: oracle override (mechanism validation)
        if oracle_enable:
            if oracle_source == "y":
                if y_seq.size != wtil.size:
                    raise ValueError(f"[Oracle] y shape mismatch: y={y_seq.size} w={wtil.size} (seq={seq})")
                wtil = y_seq.astype(np.float32)
            elif oracle_source == "err":
                # use provided err_obs if present; otherwise compute from z_obs/z_gt
                if "err_obs" in arr and arr["err_obs"].size == wtil.size:
                    err_obs_1d = arr["err_obs"].astype(np.float32).reshape(-1)
                else:
                    err_obs_1d = np.linalg.norm((z_obs - z_gt).astype(np.float64), axis=1).astype(np.float32)
                wtil = (err_obs_1d < float(oracle_err_px)).astype(np.float32)
            else:
                raise ValueError(f"[Oracle] unknown source={oracle_source}, expected y|err")
            tau = float(oracle_tau)

        seg_debug = [] if (debug_aob and seq in debug_seqs) else None

        # --- Segment routing decisions computed BEFORE AoB (for auditing),
        # but applied AFTER base AoB as a safety override.
        low_by_tau = (wtil < float(tau))
        force_abst_mask = np.zeros_like(low_by_tau, dtype=bool)
        force_safe_bridge_mask = np.zeros_like(low_by_tau, dtype=bool)
        seg_route_decisions: Dict[int, Dict[str, Any]] = {}

        if seg_route_cfg.enable:
            try:
                force_abst_mask, force_safe_bridge_mask, seg_route_decisions = _segment_routing_decide_masks(
                    low_by_tau=low_by_tau,
                    z_obs=z_obs,
                    img_hw=arr.get("img_hw", np.array([0, 0], dtype=np.int32)),
                    cfg=seg_route_cfg,
                    fit=seg_route_fit,
                )
            except Exception as e:
                print(f"[SegRouting] failed on seq={seq}: {e} -> disabled for this seq")
                force_abst_mask = np.zeros_like(low_by_tau, dtype=bool)
                force_safe_bridge_mask = np.zeros_like(low_by_tau, dtype=bool)
                seg_route_decisions = {}
        else:
            force_safe_bridge_mask = np.zeros_like(low_by_tau, dtype=bool)

        # --- Base AoB run (original behavior)
        z_fin, is_bridge, is_abst = aob_fill(z_base=z_obs, w=wtil, tau=tau, params=aobp, debug=seg_debug)

        # --- Safety override: apply TWO overrides:
        #   (1) force_safe_bridge_mask -> linear-safe bridge
        #   (2) force_abst_mask        -> safe-hold (explicit, not relying on AoB abstain)
        if seg_route_cfg.enable and (bool(np.any(force_safe_bridge_mask)) or bool(np.any(force_abst_mask))):
            try:
                Tseq = int(wtil.size)
                low = low_by_tau  # segments are defined on low_by_tau
                segs = _segments_from_mask(low)

                def _safe_hold_segment(t0: int, t1: int, left: int, right: int, has_left: bool, has_right: bool) -> None:
                    # safest: hold boundary observation if exists; else hold local obs
                    if has_left and left >= 0:
                        anchor = z_obs[left].astype(np.float32)
                    elif has_right and right < Tseq:
                        anchor = z_obs[right].astype(np.float32)
                    else:
                        anchor = z_obs[t0].astype(np.float32)
                    z_fin[t0:t1 + 1] = anchor[None, :]

                for seg_id, (t0, t1) in enumerate(segs):
                    left = t0 - 1
                    right = t1 + 1
                    has_left = (left >= 0) and (not bool(low[left]))
                    has_right = (right < Tseq) and (not bool(low[right]))

                    # frame-level masks decide whether this segment needs override
                    seg_force_safe = bool(np.any(force_safe_bridge_mask[t0:t1 + 1]))
                    seg_force_abst = bool(np.any(force_abst_mask[t0:t1 + 1]))

                    # priority: abstain beats safe-bridge if overlap (shouldn't after disjointing, but be defensive)
                    if seg_force_abst:
                        _safe_hold_segment(t0, t1, left, right, has_left, has_right)
                        is_bridge[t0:t1 + 1] = False
                        is_abst[t0:t1 + 1] = True
                    elif seg_force_safe:
                        if has_left and has_right:
                            req = str(getattr(seg_route_cfg, "bridge_mode_relaxed", "linear_safe")).lower()
                            eff = _apply_safe_bridge(
                                mode=req,
                                z_obs=z_obs,
                                z_fin=z_fin,
                                t0=t0, t1=t1,
                                left=left, right=right,
                            )
                            is_bridge[t0:t1 + 1] = True
                            is_abst[t0:t1 + 1] = False

                            # 可审计：记录“请求模式/最终生效模式”
                            if int(seg_id) in seg_route_decisions:
                                seg_route_decisions[int(seg_id)]["bridge_mode_req"] = req
                                seg_route_decisions[int(seg_id)]["bridge_mode_eff"] = eff
                        else:
                            _safe_hold_segment(t0, t1, left, right, has_left, has_right)
                            is_bridge[t0:t1 + 1] = False
                            is_abst[t0:t1 + 1] = True
                            if int(seg_id) in seg_route_decisions:
                                seg_route_decisions[int(seg_id)]["bridge_mode_req"] = str(getattr(seg_route_cfg, "bridge_mode_relaxed", "linear_safe")).lower()
                                seg_route_decisions[int(seg_id)]["bridge_mode_eff"] = "hold_fallback_missing_boundary"

            except Exception as e:
                print(f"[SegRouting] override failed on seq={seq}: {e} -> keep base AoB output")

        # --- NEW: segment-level audit table (one row per low segment)
        low_by_aob = (is_bridge | is_abst)
        if low_by_tau.shape != low_by_aob.shape:
            raise ValueError(f"[Audit] mask shape mismatch for seq={seq}: {low_by_tau.shape} vs {low_by_aob.shape}")

        leak_high = int(np.sum(low_by_aob & (~low_by_tau)))
        uncovered_low = int(np.sum(low_by_tau & (~low_by_aob)))
        leak_high_total += leak_high
        uncovered_low_total += uncovered_low

        segs = _segments_from_mask(low_by_tau)
        if len(segs) > 0:
            err_obs = np.linalg.norm((z_obs - z_gt).astype(np.float64), axis=1)
            err_fin = np.linalg.norm((z_fin - z_gt).astype(np.float64), axis=1)
            diag = _diag_from_hw(arr.get("img_hw", np.array([0, 0], dtype=np.int32)))

            for seg_id, (t0, t1) in enumerate(segs):
                L = int(t1 - t0 + 1)
                b_n = int(np.sum(is_bridge[t0:t1 + 1]))
                a_n = int(np.sum(is_abst[t0:t1 + 1]))
                if b_n > 0 and a_n > 0:
                    seg_type = "mixed"
                elif b_n > 0:
                    seg_type = "bridge"
                elif a_n > 0:
                    seg_type = "abstain"
                else:
                    seg_type = "none"  # should not happen

                left = t0 - 1
                right = t1 + 1
                has_left = (left >= 0) and (not bool(low_by_tau[left]))
                has_right = (right < int(wtil.size)) and (not bool(low_by_tau[right]))

                if has_left and has_right:
                    bd_obs = float(np.linalg.norm((z_obs[left] - z_obs[right]).astype(np.float64)))
                    bd_fin = float(np.linalg.norm((z_fin[left] - z_fin[right]).astype(np.float64)))
                else:
                    bd_obs = float("nan")
                    bd_fin = float("nan")

                bd_obs_norm = float(bd_obs / diag) if (np.isfinite(bd_obs) and np.isfinite(diag) and diag > 0) else float("nan")
                v_norm = float(bd_obs_norm / max(L, 1)) if np.isfinite(bd_obs_norm) else float("nan")

                seg_err_obs = err_obs[t0:t1 + 1]
                seg_err_fin = err_fin[t0:t1 + 1]
                seg_fail_obs = float(np.mean(seg_err_obs > float(fail_px))) if seg_err_obs.size else float("nan")
                seg_fail_fin = float(np.mean(seg_err_fin > float(fail_px))) if seg_err_fin.size else float("nan")

                route_info = seg_route_decisions.get(int(seg_id), {})
                route_decision = str(route_info.get("decision", "auto"))
                route_reason = str(route_info.get("reason", ""))

                seg_rows.append({
                    "seq": seq,
                    "T": int(wtil.size),
                    "tau": float(tau),
                    "seg_id": int(seg_id),
                    "t0": int(t0),
                    "t1": int(t1),
                    "L": int(L),
                    "type": seg_type,
                    "bridge_n": int(b_n),
                    "bridge_mode_req": str(route_info.get("bridge_mode_req", "")),
                    "bridge_mode_eff": str(route_info.get("bridge_mode_eff", "")),
                    "abst_n": int(a_n),
                    "has_left": int(has_left),
                    "has_right": int(has_right),
                    "boundary_disp_obs": bd_obs,
                    "boundary_disp_fin": bd_fin,
                    "boundary_disp_obs_norm": bd_obs_norm,
                    "speed_norm": v_norm,
                    "route_decision": route_decision,
                    "route_reason": route_reason,
                    "route_disp_norm_thr": None if seg_route_fit.disp_norm_thr is None else float(seg_route_fit.disp_norm_thr),
                    "route_Lmax": int(seg_route_cfg.Lmax),
                    "w_min": float(np.min(wtil[t0:t1 + 1])) if L > 0 else float("nan"),
                    "w_mean": float(np.mean(wtil[t0:t1 + 1])) if L > 0 else float("nan"),
                    "p_mean": float(np.mean(p[t0:t1 + 1])) if L > 0 else float("nan"),
                    "seg_obs_p95": _safe_quantile(seg_err_obs, 0.95),
                    "seg_fin_p95": _safe_quantile(seg_err_fin, 0.95),
                    "seg_obs_fail": seg_fail_obs,
                    "seg_fin_fail": seg_fail_fin,
                })

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
                f"wtil[q0,q1,q5,q50,q95,q99,q100]={['%.4f' % v for v in q]} "
                f"low_frac={low_n / max(Tseq, 1):.4f} low_n={low_n} segs={0 if seg_debug is None else len(seg_debug)}"
            )
            if low_n > 0:
                pbar.write(
                    f"[AoB-Debug] seq={seq} err_obs_p95_all={_q95(err_obs):.3f} err_fin_p95_all={_q95(err_fin):.3f} "
                    f"err_obs_p95_low={_q95(err_obs[low_mask]):.3f} err_fin_p95_low={_q95(err_fin[low_mask]):.3f}"
                )

            write_json(analysis_dir / f"debug_aob_{seq}.json", {
                "seq": seq,
                "tau": float(tau),
                "oracle": {"enable": oracle_enable, "source": oracle_source, "tau": oracle_tau, "err_px": oracle_err_px},
                "aobp": {
                    "eps_gate": aobp.eps_gate,
                    "abstain_mode": aobp.abstain_mode,
                    "eta_L": aobp.eta_L,
                    "eta_u": aobp.eta_u,
                    "max_bridge_len": aobp.max_bridge_len,
                    "bridge_mode": aobp.bridge_mode,
                },
                "wtil_quantiles": q,
                "segments": seg_debug if seg_debug is not None else [],
            })

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

    # --- NEW: write segment audit csv
    if bool(ana_cfg.get("write_csv", True)):
        seg_df = pd.DataFrame(seg_rows)
        seg_path = analysis_dir / "segments_v1.csv"
        seg_df.to_csv(seg_path, index=False)
        print(f"[OK] wrote: {seg_path}  num_segments={len(seg_df)}")

        # small summary (useful for meeting)
        if len(seg_df) > 0:
            try:
                seg_df2 = seg_df.copy()
                # simple length buckets
                seg_df2["L_bucket"] = pd.cut(
                    seg_df2["L"].astype(float),
                    bins=[0, 5, 15, 10_000],
                    labels=["short(1-5)", "med(6-15)", "long(16+)"],
                    include_lowest=True,
                    right=True,
                )
                summ = (
                    seg_df2
                    .groupby(["type", "L_bucket"], dropna=False, observed=False)
                    .agg(
                        n=("L", "count"),
                        fin_p95=("seg_fin_p95", "mean"),
                        fin_fail=("seg_fin_fail", "mean"),
                        obs_p95=("seg_obs_p95", "mean"),
                        obs_fail=("seg_obs_fail", "mean"),
                        bd_obs=("boundary_disp_obs", "mean"),
                    )
                    .reset_index()
                )
                summ.to_csv(analysis_dir / "segments_summary_v1.csv", index=False)
                print(f"[OK] wrote: {analysis_dir/'segments_summary_v1.csv'}")
            except Exception as e:
                print(f"[Warn] failed to write segments_summary_v1.csv: {e}")

    try:
        write_json(analysis_dir / "audit_mask_mismatch_v2.json", {
            "leak_high_total": int(leak_high_total),
            "uncovered_low_total": int(uncovered_low_total),
        })
    except Exception as e:
        print(f"[Warn] failed to write audit_mask_mismatch_v2.json: {e}")

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

    if leak_high_total > 0:
        print(f"[Audit][WARN] leak_high_total={leak_high_total} (should be 0 ideally).")
    print(f"[Audit] uncovered_low_total={uncovered_low_total} (report for transparency).")
    print(f"  spearman(wtil, -err_obs) = {rho}")

    if bool(ana_cfg.get("write_figs", True)) and len(df) > 0:
        wcat = np.concatenate(all_wtil_val, axis=0) if all_wtil_val else np.zeros((0,), dtype=np.float32)
        ycat = np.concatenate(all_y_val, axis=0) if all_y_val else np.zeros((0,), dtype=np.float32)

        if wcat.size == ycat.size and wcat.size > 0:
            _plot_reliability(
                wcat,
                ycat,
                out_path=analysis_dir / "fig_gate_bins_val.png",
                title="Gate bins (reliability semantics, val)",
                xlabel="wtil bin center",
                nbins=10,
                alpha=0.05,
                min_count=1,
            )

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

        try:
            plt.figure()
            Xh = X_tr_k_raw
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
    # keep old behavior but prefer the robust resolver first
    try:
        davis_root = _resolve_davis_root(cfg["davis_root"])
    except Exception:
        davis_root = resolve_davis_root(cfg["davis_root"])

    base_out = Path(cfg["base_out"])
    res = cfg["res"]

    out_train_s1 = base_out / "davis2016_train_precompute"
    out_val_s1 = base_out / "davis2016_val_precompute"

    out_train_s2 = base_out / "davis2016_stage2_fixed"
    out_val_s2 = base_out / "davis2016_val_stage2"

    out_dir_s3 = base_out / cfg.get("stage3", {}).get("out_dir", "davis2016_stage3_fixed")

    print(pretty_header("PR2-Drag Lite", {
        "cmd": "run_all",
        "davis_root": str(davis_root),
        "res": res,
        "base_out": str(base_out),
    }))

    meta_tr = stage1_precompute_split(cfg, split="train", out_dir=out_train_s1)
    print(f"[OK] Stage1(train) meta: {out_train_s1/'meta.json'}")
    print(f"[OK] Stage1(train) npz_dir: {out_train_s1}  num_npz={len(meta_tr.get('seqs', []))}")

    meta_va = stage1_precompute_split(cfg, split="val", out_dir=out_val_s1)
    print(f"[OK] Stage1(val) meta: {out_val_s1/'meta.json'}")
    print(f"[OK] Stage1(val) npz_dir: {out_val_s1}  num_npz={len(meta_va.get('seqs', []))}")

    stage2_compute_split(cfg, split="train", stage1_dir=out_train_s1, out_dir=out_train_s2)
    stage2_compute_split(cfg, split="val", stage1_dir=out_val_s1, out_dir=out_val_s2)
    print(f"[OK] Stage2 done: train_npz= {len(list(out_train_s2.glob('*.npz')))} val_npz= {len(list(out_val_s2.glob('*.npz')))}")

    stage3_train_eval(cfg, stage2_train=out_train_s2, stage2_val=out_val_s2, out_dir=out_dir_s3)
