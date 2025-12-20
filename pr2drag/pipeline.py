from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

import imageio.v2 as imageio
import matplotlib.pyplot as plt

from .aob import AoBParams, aob_fill
from .evidence import build_sequence_evidence
from .eval import compute_seq_metrics, metrics_to_row
from .sp import HMMParams, emission_metrics, fit_emission_model, forward_backward_binary
from .utils import (
    clamp,
    davis_split_path,
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
    """
    Make davis_root robust:
    - accept either .../DAVIS  (contains JPEGImages/Annotations/ImageSets)
    - or accept its parent that contains DAVIS/ subfolder
    """
    p = Path(davis_root).expanduser()

    # if user passed something like "" or None accidentally
    if str(p).strip() == "":
        raise ValueError("[DAVIS] davis_root is empty. Please set cfg['davis_root'] correctly.")

    # Case A: already points to DAVIS root
    if (p / "JPEGImages").is_dir() and (p / "Annotations").is_dir() and (p / "ImageSets").is_dir():
        return p

    # Case B: user passed parent dir, and inside it there is DAVIS/
    cand = p / "DAVIS"
    if (cand / "JPEGImages").is_dir() and (cand / "Annotations").is_dir() and (cand / "ImageSets").is_dir():
        return cand

    # Helpful diagnostics
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

def _list_seq_from_split(davis_root: Path, split_rel: str) -> List[str]:
    split_path = davis_split_path(davis_root, split_rel)
    if not split_path.exists():
        raise FileNotFoundError(f"[DAVIS] split file not found: {split_path}")
    return read_txt_lines(split_path)


def davis_frame_paths(davis_root: Union[str, Path], res: str, seq: str) -> Tuple[List[Path], List[Path]]:
    """
    Return aligned (frames, annos) path lists for a DAVIS sequence.

    frames: JPEGImages/{res}/{seq}/*.jpg (or *.png fallback)
    annos : Annotations/{res}/{seq}/*.png
    Alignment is by filename stem (e.g. 00000).
    """
    root = _resolve_davis_root(davis_root)
    res = str(res)
    seq = str(seq)

    img_dir = root / "JPEGImages" / res / seq
    ann_dir = root / "Annotations" / res / seq

    if not img_dir.is_dir():
        # extra hint for common mistakes
        avail_res = (root / "JPEGImages").glob("*")
        avail_res = [p.name for p in avail_res if p.is_dir()]
        raise FileNotFoundError(
            f"[DAVIS] Missing image directory:\n  {img_dir}\n"
            f"[hint] available resolutions under {root/'JPEGImages'}: {avail_res}\n"
            f"[hint] check cfg.res and cfg.davis_root"
        )
    if not ann_dir.is_dir():
        raise FileNotFoundError(f"[DAVIS] Missing annotation directory:\n  {ann_dir}")

    # DAVIS is usually jpg, but keep fallback for safety
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
        # DAVIS should have dense masks; if missing, better fail loudly
        sample = missing[:5]
        raise FileNotFoundError(
            f"[DAVIS] Missing {len(missing)} annotation(s) in {ann_dir}.\n"
            f"  examples: {sample}\n"
            f"[hint] verify DAVIS extraction is complete and uses the same frame naming."
        )

    return frames, annos


def _stage_signature(cfg: Dict[str, Any], stage: str) -> str:
    """
    Hash only relevant sub-config so cache invalidation is stable.
    """
    pick = {
        "stage": stage,
        "res": cfg.get("res"),
        "stage1": cfg.get("stage1", {}),
        "stage2": cfg.get("stage2", {}),
        "stage3": cfg.get("stage3", {}),
        "version": "0.2.0-lite",
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
            # do not over-trust: but allow skip for stage1 (cheap)
            pbar.write(f"[skip] Stage1 exists: {out_npz.name}")
            meta["seqs"].append(seq)
            continue

        frames, annos = davis_frame_paths(davis_root, res, seq)

        # deterministic per-seq rng
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
    """
    Build per-frame features X.
    Mode 'a' => 5 dims:
      [iou_shift, cycle_err_norm, area_change, occl_flag, blur_flag]
    Mode 'b' => 7 dims adds:
      [blur_inv, motion_norm]
    """
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
    motion_norm = motion / 10.0  # scale to ~[0..30] on fast motion

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

    # read meta to get seqs (fallback: list npz)
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

        arr, m1 = npz_read(in_npz)
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
        meta2 = {
            "seq": seq,
            "T": int(X.shape[0]),
            "feat_mode": feat_mode,
            "signature": sig,
        }
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
    # drop constant dims
    std = X.std(axis=0)
    keep = std > 1e-12
    return keep


def stage3_train_eval(cfg: Dict[str, Any], *, stage2_train: Path, stage2_val: Path, out_dir: Path) -> None:
    out_dir = safe_mkdir(out_dir)
    analysis_dir = safe_mkdir(out_dir / "_analysis")

    s3 = cfg.get("stage3", {})
    hmm_cfg = s3.get("hmm", {})
    em_cfg = s3.get("emission", {})
    aob_cfg = s3.get("aob", {})
    met_cfg = s3.get("metrics", {})
    ana_cfg = s3.get("analysis", {})

    tau_mode = str(s3.get("tau_mode", "global"))
    target_frac = float(s3.get("tau_target_frac", 0.25))
    tau_min = float(s3.get("tau_min", 1e-6))
    tau_max = float(s3.get("tau_max", 0.999999))

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

    # train emission
    X_tr, y_tr, err_tr = _concat_stage2(stage2_train)
    X_va, y_va, err_va = _concat_stage2(stage2_val)

    print(f"[Data] X_tr {tuple(X_tr.shape)} pos_rate {float(y_tr.mean()) if y_tr.size else float('nan')}")
    print(f"[Data] X_va {tuple(X_va.shape)} pos_rate {float(y_va.mean()) if y_va.size else float('nan')}")

    keep = _feature_keep_mask(X_tr)
    X_tr_k = X_tr[:, keep]
    X_va_k = X_va[:, keep]
    print(f"[Feat] keep dims: {keep} num_keep: {int(keep.sum())}")

    clf = fit_emission_model(X_tr_k, y_tr, c=float(em_cfg.get("c", 1.0)), max_iter=int(em_cfg.get("max_iter", 2000)))

    p_tr = clf.predict_proba(X_tr_k)[:, 1].astype(np.float32)
    p_va = clf.predict_proba(X_va_k)[:, 1].astype(np.float32)
    m = emission_metrics(p_va, y_va)
    print(f"[Emission] AUROC={m['auroc']:.4f}  ECE={m['ece']:.4f}  risk@50%={m['risk@50%']:.4f}")

    # For tau_global we want threshold such that fraction(w<tau)=target_frac
    # => tau = quantile(w, target_frac). But w here should be smoothed per sequence.
    # We'll compute w_train by running HMM per seq and concatenating.

    def _seq_post_w(npz_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        arr, _ = npz_read(npz_path)
        X = arr["X"][:, keep].astype(np.float32)
        p = clf.predict_proba(X)[:, 1].astype(np.float32)
        w = forward_backward_binary(p, hmm)
        return w, p

    # compute global tau if needed
    tau_global = None
    if tau_mode.lower() == "global":
        ws = []
        for p in sorted(stage2_train.glob("*.npz")):
            w, _ = _seq_post_w(p)
            ws.append(w)
        w_all = np.concatenate(ws, axis=0) if ws else np.zeros((0,), dtype=np.float32)
        if w_all.size == 0:
            tau_global = 0.5
        else:
            tau_global = float(np.quantile(w_all, target_frac))
            tau_global = clamp(tau_global, tau_min, tau_max)
        print(f"[Tau] global_quantile from train: tau_global={tau_global:.6f} (target_frac={target_frac})")

    # eval on val seq-wise
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
        p = clf.predict_proba(X)[:, 1].astype(np.float32)
        w = forward_backward_binary(p, hmm)

        if tau_mode.lower() == "global":
            tau = float(tau_global)
        elif tau_mode.lower() == "per_video":
            tau = float(np.quantile(w, target_frac)) if w.size else 0.5
            tau = clamp(tau, tau_min, tau_max)
        else:
            raise ValueError(f"Unknown tau_mode={tau_mode}. Use global|per_video.")

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

    # write analysis
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
        # need y_val too -> approximate by concatenating from val dir
        ys = []
        for p in sorted(stage2_val.glob("*.npz")):
            arr, _ = npz_read(p)
            ys.append(arr["y"].astype(np.float32))
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
            fig1 = analysis_dir / "fig_gate_bins.png"
            plt.savefig(fig1, dpi=160)
            plt.close()

        # tau hist (if per_video)
        plt.figure()
        plt.hist(df["tau"].values, bins=20)
        plt.xlabel("tau")
        plt.ylabel("count")
        plt.title("Tau distribution")
        plt.tight_layout()
        fig2 = analysis_dir / "fig_tau_hist.png"
        plt.savefig(fig2, dpi=160)
        plt.close()

        # low_frac hist
        plt.figure()
        plt.hist(df["low_frac"].values, bins=20)
        plt.xlabel("low_frac")
        plt.ylabel("count")
        plt.title("Low fraction distribution")
        plt.tight_layout()
        fig3 = analysis_dir / "fig_lowfrac_hist.png"
        plt.savefig(fig3, dpi=160)
        plt.close()

        # feature histogram
        plt.figure()
        for j in range(X_tr.shape[1]):
            plt.hist(X_tr[:, j], bins=30, alpha=0.5, label=f"dim{j}")
        plt.legend()
        plt.title("Feature hist (train)")
        plt.tight_layout()
        fig4 = analysis_dir / "fig_feat_hist.png"
        plt.savefig(fig4, dpi=160)
        plt.close()

        print(f"[OK] wrote: {analysis_dir}/fig_gate_bins.png")
        print(f"[OK] wrote: {analysis_dir}/fig_tau_hist.png {analysis_dir}/fig_lowfrac_hist.png")
        print(f"[OK] wrote: {analysis_dir}/fig_feat_hist.png")


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
