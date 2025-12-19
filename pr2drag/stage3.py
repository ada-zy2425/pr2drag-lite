from __future__ import annotations
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .core import (
    hmm_smooth, apply_aob, traj_metrics,
    expected_calibration_error, coverage_risk
)

def load_split(stage2_dir: Path):
    items = []
    for p in sorted(stage2_dir.glob("*.npz")):
        a = np.load(p, allow_pickle=True)
        items.append(dict(
            seq=str(a.get("seq", p.stem)),
            z_gt=a["z_gt"].astype(np.float32),
            z_obs=a["z_obs"].astype(np.float32),
            E=a["E_min"].astype(np.float32),
            chi=a["chi"].astype(np.int64),
            path=str(p),
        ))
    if len(items) == 0:
        raise RuntimeError(f"empty split: {stage2_dir}")
    return items

def stack_frames(items):
    Xs, Ys = [], []
    for it in items:
        Xs.append(it["E"])
        Ys.append(it["chi"])
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(Ys, axis=0)
    return X, y

def robust_fit_transform(X_tr: np.ndarray):
    med = np.median(X_tr, axis=0)
    q1 = np.percentile(X_tr, 25, axis=0)
    q3 = np.percentile(X_tr, 75, axis=0)
    iqr = np.maximum(q3 - q1, 1e-6)
    keep = (iqr > 1e-6)

    def xform(X):
        Xn = (X - med) / iqr
        return Xn[:, keep].astype(np.float32)

    return xform, keep, dict(med=med.tolist(), q1=q1.tolist(), q3=q3.tolist(), iqr=iqr.tolist())

def select_tau(wtil: np.ndarray, tau_mode: str, tau_fixed: float, low_frac_target: float, tau_min: float, tau_max: float) -> float:
    if tau_mode == "fixed":
        tau = float(tau_fixed)
    elif tau_mode == "quantile":
        tau = float(np.quantile(wtil, float(low_frac_target)))
    else:
        raise ValueError(f"Unknown tau_mode: {tau_mode}")
    # IMPORTANT: 不要粗暴夹到 0.05 这种，会把大量序列全部判 low（你之前的问题之一）
    tau = float(np.clip(tau, float(tau_min), float(tau_max)))
    return tau

def run_stage3(
    train_dir: Path,
    val_dir: Path,
    out_dir: Path,
    sp_conf: dict,
    aob_conf: dict,
    metrics_conf: dict,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    train_items = load_split(train_dir)
    val_items   = load_split(val_dir)
    print("[Stage3] train seq:", len(train_items), "val seq:", len(val_items))

    X_tr, y_tr = stack_frames(train_items)
    X_va, y_va = stack_frames(val_items)
    print("[Data] X_tr", X_tr.shape, "pos_rate", float(y_tr.mean()))
    print("[Data] X_va", X_va.shape, "pos_rate", float(y_va.mean()))

    if len(np.unique(y_tr)) < 2:
        raise RuntimeError(f"[Stage3] y_tr has only one class: unique={np.unique(y_tr)}. "
                           "Try lowering IOU_THR or inspect Stage2 outputs.")

    xform, keep, feat_stats = robust_fit_transform(X_tr)
    print("[Feat] keep dims:", keep, "num_keep:", int(np.sum(keep)))

    clf = LogisticRegression(solver="lbfgs", max_iter=400, class_weight="balanced")
    clf.fit(xform(X_tr), y_tr)

    p_va = clf.predict_proba(xform(X_va))[:, 1].astype(np.float32)
    auc = roc_auc_score(y_va, p_va) if len(np.unique(y_va)) > 1 else float("nan")
    ece = expected_calibration_error(y_va, p_va, n_bins=int(sp_conf.get("ece_bins", 15)))
    cov, risk = coverage_risk(y_va, p_va)
    print(f"[Emission] AUROC={auc:.4f}  ECE={ece:.4f}  risk@50%={float(risk[int(0.5*len(risk))-1]):.4f}")

    per_seq = []
    audit_all = {}

    for it in tqdm(val_items, desc="Stage3(val)"):
        E = it["E"]
        z_gt = it["z_gt"]
        z_obs = it["z_obs"]
        chi = it["chi"]
        seq = it["seq"]

        # emission
        e = clf.predict_proba(xform(E))[:, 1].astype(np.float32)
        e = np.clip(e, 1e-6, 1 - 1e-6)

        # enforce t=0 reliable (DAVIS semi-supervised first frame annotated)
        e[0] = max(float(e[0]), 1.0 - 1e-6)

        w = hmm_smooth(
            e,
            p01=float(sp_conf["trans_p01"]),
            p10=float(sp_conf["trans_p10"]),
            pi1=float(sp_conf.get("pi1", 0.99))
        )
        wtil = np.clip(w, 1e-6, 1 - 1e-6).astype(np.float32)

        tau = select_tau(
            wtil=wtil,
            tau_mode=str(aob_conf.get("tau_mode", "fixed")),
            tau_fixed=float(aob_conf.get("tau_fixed", 0.45)),
            low_frac_target=float(aob_conf.get("low_frac_target", 0.25)),
            tau_min=float(aob_conf.get("tau_min", 1e-6)),
            tau_max=float(aob_conf.get("tau_max", 0.999999)),
        )

        z_final, audit = apply_aob(
            z_bar=z_obs, wtil=wtil,
            tau=tau,
            eta_l=float(aob_conf["eta_l"]),
            eta_u=float(aob_conf["eta_u"]),
            eps_gate=float(aob_conf["eps_gate"]),
            abstain_mode=str(aob_conf.get("abstain_mode", "hold")),
        )

        m_obs = traj_metrics(z_obs, z_gt, float(metrics_conf["fail_thresh"]))
        m_fin = traj_metrics(z_final, z_gt, float(metrics_conf["fail_thresh"]))

        per_seq.append(dict(
            seq=seq, T=int(len(chi)),
            pos_rate=float(chi.mean()),
            tau=float(tau),
            low_frac=float(audit["low_frac"]),
            obs_p95=m_obs["p95"], fin_p95=m_fin["p95"],
            obs_fail=m_obs["fail_rate"], fin_fail=m_fin["fail_rate"],
            bridged_frames=audit["bridged_frames"],
            abstained_frames=audit["abstained_frames"],
            invalid_segments=audit["invalid_segments"],
            num_segments=audit["num_segments"],
            wtil_mean=audit["wtil_mean"]
        ))

        audit_all[seq] = dict(audit=audit, metrics_obs=m_obs, metrics_final=m_fin)
        np.savez_compressed(
            out_dir / f"{seq}.npz",
            seq=seq, z_gt=z_gt, z_obs=z_obs, z_final=z_final,
            chi=chi, E=E, e=e, w=w, wtil=wtil
        )

    def agg(key):
        vals = [r[key] for r in per_seq if np.isfinite(r[key])]
        return float(np.mean(vals)) if len(vals) else float("nan")

    print("\n[Val aggregate]")
    print("mean obs_p95", agg("obs_p95"), "mean fin_p95", agg("fin_p95"))
    print("mean obs_fail", agg("obs_fail"), "mean fin_fail", agg("fin_fail"))
    print("mean low_frac", agg("low_frac"), "mean bridged_frames", agg("bridged_frames"))

    report = dict(
        stage="stage3_sp_hmm_aob",
        sp=dict(auroc=float(auc), ece=float(ece), feat_keep=keep.tolist(), feat_stats=feat_stats),
        aob=aob_conf,
        metrics=metrics_conf,
        per_seq=per_seq,
    )
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    (out_dir / "audit_all.json").write_text(json.dumps(audit_all, indent=2))
    print(f"\n[OK] wrote stage3 to: {out_dir}")
