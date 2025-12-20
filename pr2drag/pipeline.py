# pr2drag/pipeline.py
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

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

    print(
        f"[SplitParse] split={split_path} num_lines={len(lines)} num_seqs(dedup)={len(seqs)} bad_lines={bad}"
    )
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


def _stage_signature(cfg: Dict[str, Any], stage: str) -> str:
    pick = {
        "stage": stage,
        "res": cfg.get("res"),
        "stage1": cfg.get("stage1", {}),
        "stage2": cfg.get("stage2", {}),
        "stage3": cfg.get("stage3", {}),
        "version": "0.3.0-lite-risk",
    }
    return sha1_of_dict(pick)


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
        # fallback: keep all (avoid empty feature vector)
        keep[:] = True
    return keep


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

    Returns:
      tau, summary dict, curve df (coverage, risk vs tau index)
    """
    w = np.asarray(w, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    n = int(w.size)
    if n == 0 or y.size != n:
        tau = clamp(0.5, tau_min, tau_max)
        df = pd.DataFrame({"k": [], "tau": [], "coverage": [], "risk": []})
        return tau, {"coverage": float("nan"), "risk": float("nan")}, df

    # sort by w descending: lowering threshold includes more samples
    idx = np.argsort(-w)
    ws = w[idx]
    ys = y[idx]
    bad = 1 - ys  # bad=1 when y=0

    cum_bad = np.cumsum(bad, dtype=np.float64)
    ks = np.arange(1, n + 1, dtype=np.float64)
    risk = cum_bad / ks
    coverage = ks / float(n)

    min_keep = max(1, int(math.ceil(float(min_keep_frac) * n)))

    # feasible = risk<=r0 and keep>=min_keep
    feasible = (risk <= float(r0)) & (ks >= float(min_keep))

    if feasible.any():
        # maximize coverage => choose largest k among feasible
        k_star = int(np.where(feasible)[0].max()) + 1
    else:
        # if nothing feasible, choose k that minimizes risk (with min_keep constraint)
        valid = ks >= float(min_keep)
        if valid.any():
            k_star = int(np.where(valid)[0][np.argmin(risk[valid])]) + 1
        else:
            k_star = 1

    # tau can be set to ws[k_star-1] (keep all >= tau)
    tau = float(ws[k_star - 1])
    tau = clamp(tau, tau_min, tau_max)

    df = pd.DataFrame(
        {
            "k": ks.astype(int),
            "tau": ws,
            "coverage": coverage,
            "risk": risk,
        }
    )

    summary = {
        "coverage": float(coverage[k_star - 1]),
        "risk": float(risk[k_star - 1]),
        "k": float(k_star),
        "n": float(n),
        "min_keep": float(min_keep),
    }
    return tau, summary, df


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

    # risk-constrained tau config (general)
    r0 = float(s3.get("tau_risk", 0.05))  # allow 5% bad frames among trusted frames
    min_keep_frac = float(s3.get("tau_min_keep_frac", 0.05))

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
    # emission model
    # -----------------
    clf = fit_emission_model(
        X_tr_k, y_tr,
        c=float(em_cfg.get("c", 1.0)),
        max_iter=int(em_cfg.get("max_iter", 2000))
    )

    p_tr_raw = clf.predict_proba(X_tr_k)[:, 1].astype(np.float32)
    p_va_raw = clf.predict_proba(X_va_k)[:, 1].astype(np.float32)

    # Optional temperature scaling (general, stabilizes transfer)
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
    # helper: compute w per seq
    # -----------------
    def _seq_post_w(npz_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        arr, _ = npz_read(npz_path)
        X = arr["X"][:, keep].astype(np.float32)
        p_raw = clf.predict_proba(X)[:, 1].astype(np.float32)
        if cal_enable:
            p = apply_temperature_scaling(p_raw, T, eps=cal_eps)
        else:
            p = p_raw
        w = forward_backward_binary(p, hmm)
        y = arr.get("y", np.zeros((w.shape[0],), dtype=np.uint8)).astype(np.uint8)
        return w, p, y

    # -----------------
    # tau selection (global)
    # -----------------
    tau_global: float = 0.5

    if tau_mode == "global":
        # old: quantile on train w
        ws = []
        for pth in sorted(stage2_train.glob("*.npz")):
            w, _, _ = _seq_post_w(pth)
            ws.append(w)
        w_all = np.concatenate(ws, axis=0) if ws else np.zeros((0,), dtype=np.float32)
        if w_all.size == 0:
            tau_global = 0.5
        else:
            tau_global = float(np.quantile(w_all, target_frac))
            tau_global = clamp(tau_global, tau_min, tau_max)
        print(f"[Tau] global_quantile from train: tau_global={tau_global:.6f} (target_frac={target_frac})")

    elif tau_mode == "risk":
        # new: risk-constrained tau on train (general)
        ws, ys = [], []
        for pth in sorted(stage2_train.glob("*.npz")):
            w, _, y = _seq_post_w(pth)
            ws.append(w.astype(np.float32))
            ys.append(y.astype(np.uint8))
        w_all = np.concatenate(ws, axis=0) if ws else np.zeros((0,), dtype=np.float32)
        y_all = np.concatenate(ys, axis=0) if ys else np.zeros((0,), dtype=np.uint8)

        tau_global, summ, curve = _select_tau_risk_constrained(
            w_all, y_all, r0=r0, min_keep_frac=min_keep_frac, tau_min=tau_min, tau_max=tau_max
        )

        print(
            f"[Tau] risk-constrained: tau_global={tau_global:.6f}  "
            f"coverage={summ['coverage']:.3f}  risk={summ['risk']:.3f}  r0={r0}"
        )

        # save curve
        curve_path = analysis_dir / "tau_risk_curve_train.csv"
        curve.to_csv(curve_path, index=False)
        (analysis_dir / "tau_risk_summary_train.json").write_text(
            pd.Series({**summ, "tau": tau_global, "r0": r0, "min_keep_frac": min_keep_frac}).to_json(),
            encoding="utf-8",
        )

        if bool(ana_cfg.get("write_figs", True)) and len(curve) > 0:
            plt.figure()
            plt.plot(curve["coverage"].values, curve["risk"].values)
            plt.axhline(r0, linestyle="--")
            plt.xlabel("coverage (kept fraction)")
            plt.ylabel("risk (P(y=0 | kept))")
            plt.title("Coverage–Risk curve (train)")
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

    pbar = tqdm(sorted(stage2_val.glob("*.npz")), desc="Stage3(val)")
    for npz_path in pbar:
        arr, meta = npz_read(npz_path)
        seq = meta.get("seq", npz_path.stem)

        X = arr["X"][:, keep].astype(np.float32)
        z_gt = arr["z_gt"].astype(np.float32)
        z_obs = arr["z_obs"].astype(np.float32)

        p_raw = clf.predict_proba(X)[:, 1].astype(np.float32)
        p = apply_temperature_scaling(p_raw, T, eps=cal_eps) if cal_enable else p_raw
        w = forward_backward_binary(p, hmm)

        if tau_mode == "global":
            tau = float(tau_global)
        elif tau_mode == "risk":
            tau = float(tau_global)
        elif tau_mode == "per_video":
            # keep legacy option (not recommended for transfer)
            tau = float(np.quantile(w, target_frac)) if w.size else 0.5
            tau = clamp(tau, tau_min, tau_max)
        else:
            raise ValueError(f"Unknown tau_mode={tau_mode}")

        z_fin, is_bridge, is_abst = aob_fill(z_base=z_obs, w=w, tau=tau, params=aobp)

        sm = compute_seq_metrics(
            seq=seq,
            z_gt=z_gt,
            z_obs=z_obs,
            z_fin=z_fin,
            w=w,
            tau=tau,
            is_bridge=is_bridge,
            is_abst=is_abst,
            fail_px=fail_px,
        )
        rows.append(metrics_to_row(sm))

        all_w_val.append(w)
        all_err_obs_val.append(arr["err_obs"].astype(np.float32))

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
        # gate bins: reliability vs empirical accuracy
        wcat = np.concatenate(all_w_val, axis=0) if all_w_val else np.zeros((0,), dtype=np.float32)
        ys = []
        for pth in sorted(stage2_val.glob("*.npz")):
            arr2, _ = npz_read(pth)
            ys.append(arr2["y"].astype(np.float32))
        ycat = np.concatenate(ys, axis=0) if ys else np.zeros((0,), dtype=np.float32)

        if wcat.size == ycat.size and wcat.size > 0:
            bins = np.linspace(0.0, 1.0, 11)
            xs, accs = [], []
            for i in range(10):
                lo, hi = bins[i], bins[i + 1]
                m = (wcat >= lo) & (wcat < hi) if i < 9 else (wcat >= lo) & (wcat <= hi)
                if m.sum() == 0:
                    continue
                xs.append(0.5 * (lo + hi))
                accs.append(float(ycat[m].mean()))
            plt.figure()
            plt.plot(xs, accs, marker="o")
            plt.xlabel("wtil bin center")
            plt.ylabel("empirical P(y=1)")
            plt.title("Gate bins (reliability semantics)")
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

        plt.figure()
        for j in range(X_tr.shape[1]):
            plt.hist(X_tr[:, j], bins=30, alpha=0.5, label=f"dim{j}")
        plt.legend()
        plt.title("Feature hist (train)")
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
