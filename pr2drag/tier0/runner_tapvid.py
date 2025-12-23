# pr2drag/tier0/runner_tapvid.py
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any, List
import json
import numpy as np
import pandas as pd

from pr2drag.datasets.tapvid import build_tapvid_dataset, TapVidSeq
from pr2drag.trackers.base import load_pred_npz
from pr2drag.tier0.metrics_tapvid import compute_tapvid_metrics, DEFAULT_THRESHOLDS_PX
from pr2drag.tier0.audit_schema import AuditMeta, write_audit_json


def _ensure_dir(p: str | Path) -> Path:
    pp = Path(p)
    pp.mkdir(parents=True, exist_ok=True)
    return pp


def _strict_check_pred(seq: TapVidSeq, pred_tracks: np.ndarray, pred_vis: np.ndarray) -> None:
    T, Q = seq.gt_vis.shape
    if pred_tracks.shape != (T, Q, 2):
        raise ValueError(f"[Runner] {seq.name}: pred tracks {pred_tracks.shape} != {(T,Q,2)}")
    if pred_vis.shape != (T, Q):
        raise ValueError(f"[Runner] {seq.name}: pred vis {pred_vis.shape} != {(T,Q)}")


def run_tapvid_eval(
    davis_root: str,
    res: str,
    split: str,
    pkl_path: str,
    query_mode: str,
    stride: int,
    pred_dir: str,
    out_dir: str,
    resize_to_256: bool,
    config_path: str | None = None,
    config_sha1: str | None = None,
) -> Dict[str, Any]:
    outp = _ensure_dir(out_dir)
    errlog = outp / "errors.log"

    # 1) dataset
    seqs = build_tapvid_dataset(
        davis_root=davis_root,
        pkl_path=pkl_path,
        split=split,
        res=res,
        query_mode=query_mode,
        stride=stride,
    )

    rows: List[Dict[str, Any]] = []
    errors: List[str] = []

    # 2) per-seq eval
    for seq in seqs:
        try:
            npz_path = Path(pred_dir) / f"{seq.name}.npz"
            pred = load_pred_npz(str(npz_path))

            _strict_check_pred(seq, pred.tracks_xy, pred.vis)

            m = compute_tapvid_metrics(
                gt_tracks_xy=seq.gt_tracks_xy,
                gt_vis=seq.gt_vis,
                pred_tracks_xy=pred.tracks_xy,
                pred_vis=pred.vis,
                queries_txy=seq.queries_txy,
                thresholds_px=DEFAULT_THRESHOLDS_PX,
                resize_to_256=resize_to_256,
                video_hw=seq.video_hw if resize_to_256 else None,
            )

            row = {
                "seq": seq.name,
                "T": m.t,
                "Q": m.q,
                "AJ": m.aj,
                "OA": m.oa,
                "delta_x": m.delta_x,
            }
            for thr, v in m.jaccard_by_thr.items():
                row[f"J@{thr}"] = v
            for thr, v in m.pck_by_thr.items():
                row[f"PCK@{thr}"] = v
            rows.append(row)

        except Exception as e:
            msg = f"[{seq.name}] {type(e).__name__}: {e}"
            errors.append(msg)

    if errors:
        errlog.write_text("\n".join(errors) + "\n", encoding="utf-8")
        # 非 MVP 路线：默认严格失败
        raise RuntimeError(
            f"[TapVidEval] {len(errors)} sequences failed. See {errlog}.\n"
            f"First error: {errors[0]}"
        )

    df = pd.DataFrame(rows).sort_values("seq").reset_index(drop=True)
    df_path = outp / "metrics_per_seq.csv"
    df.to_csv(df_path, index=False)

    # 3) summary
    summary = {
        "num_seqs": int(df.shape[0]),
        "mean": {k: float(df[k].mean()) for k in ["AJ", "OA", "delta_x"] if k in df.columns},
        "std": {k: float(df[k].std(ddof=0)) for k in ["AJ", "OA", "delta_x"] if k in df.columns},
        "weighted_by_Q": {
            k: float(np.average(df[k].to_numpy(), weights=df["Q"].to_numpy()))
            for k in ["AJ", "OA", "delta_x"]
            if k in df.columns
        },
    }
    (outp / "metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # 4) audit
    audit = AuditMeta.make(
        config_path=config_path,
        config_sha1=config_sha1,
        inputs={
            "davis_root": davis_root,
            "res": res,
            "split": split,
            "pkl_path": pkl_path,
            "query_mode": query_mode,
            "stride": stride,
            "pred_dir": pred_dir,
            "resize_to_256": resize_to_256,
        },
        outputs={
            "out_dir": str(outp.resolve()),
            "metrics_per_seq_csv": str(df_path.resolve()),
            "metrics_summary_json": str((outp / "metrics_summary.json").resolve()),
        },
    )
    write_audit_json(outp / "audit.json", audit)

    return {"df_path": str(df_path), "summary": summary, "out_dir": str(outp)}