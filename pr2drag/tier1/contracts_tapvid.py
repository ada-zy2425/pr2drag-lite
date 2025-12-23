# pr2drag/tier1/contracts_tapvid.py
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np


class TapVidContractError(RuntimeError):
    """Raised when TAP-Vid input/output contract is violated."""


def _as_path(p: Union[str, Path]) -> Path:
    return p if isinstance(p, Path) else Path(p)


def _is_np_array(x: Any) -> bool:
    return isinstance(x, np.ndarray)


def _shape_str(a: np.ndarray) -> str:
    return f"shape={tuple(a.shape)} dtype={a.dtype}"


def _require_key(d: Mapping[str, Any], key: str, *, ctx: str) -> Any:
    if key not in d:
        raise TapVidContractError(f"[TapVidContract] missing key='{key}' ({ctx}).")
    return d[key]


def _require_ndarray(d: Mapping[str, Any], key: str, *, ctx: str) -> np.ndarray:
    v = _require_key(d, key, ctx=ctx)
    if not _is_np_array(v):
        raise TapVidContractError(
            f"[TapVidContract] key='{key}' must be np.ndarray, got {type(v)} ({ctx})."
        )
    return v


def _require_rank(a: np.ndarray, rank: int, *, key: str, ctx: str) -> None:
    if a.ndim != rank:
        raise TapVidContractError(
            f"[TapVidContract] key='{key}' must have ndim={rank}, got ndim={a.ndim} "
            f"({_shape_str(a)}) ({ctx})."
        )


def _require_last_dim(a: np.ndarray, last_dim: int, *, key: str, ctx: str) -> None:
    if a.shape[-1] != last_dim:
        raise TapVidContractError(
            f"[TapVidContract] key='{key}' last-dim must be {last_dim}, got {a.shape[-1]} "
            f"({_shape_str(a)}) ({ctx})."
        )


def _ensure_finite(a: np.ndarray, *, key: str, ctx: str) -> None:
    if not np.isfinite(a).all():
        bad = np.argwhere(~np.isfinite(a))
        # show at most first few
        head = bad[:5].tolist()
        raise TapVidContractError(
            f"[TapVidContract] key='{key}' contains NaN/Inf; first bad indices={head} "
            f"({_shape_str(a)}) ({ctx})."
        )


def _coerce_float32(a: np.ndarray, *, key: str, ctx: str, strict: bool) -> np.ndarray:
    if a.dtype == np.float32:
        return a
    if strict:
        raise TapVidContractError(
            f"[TapVidContract] key='{key}' must be float32, got {a.dtype} "
            f"({_shape_str(a)}) ({ctx})."
        )
    # permissive: cast numeric -> float32
    if not np.issubdtype(a.dtype, np.number):
        raise TapVidContractError(
            f"[TapVidContract] key='{key}' must be numeric to cast, got {a.dtype} "
            f"({_shape_str(a)}) ({ctx})."
        )
    return a.astype(np.float32, copy=False)


def _coerce_int32(a: np.ndarray, *, key: str, ctx: str, strict: bool) -> np.ndarray:
    if a.dtype == np.int32:
        return a
    if strict:
        raise TapVidContractError(
            f"[TapVidContract] key='{key}' must be int32, got {a.dtype} "
            f"({_shape_str(a)}) ({ctx})."
        )
    if not np.issubdtype(a.dtype, np.integer):
        raise TapVidContractError(
            f"[TapVidContract] key='{key}' must be integer to cast, got {a.dtype} "
            f"({_shape_str(a)}) ({ctx})."
        )
    return a.astype(np.int32, copy=False)


def _coerce_bool(a: np.ndarray, *, key: str, ctx: str, strict: bool) -> np.ndarray:
    if a.dtype == np.bool_:
        return a
    if a.dtype in (np.uint8, np.int8, np.int32, np.int64):
        # accept {0,1} mask; strict checks value set
        if strict:
            uniq = np.unique(a)
            if uniq.size > 2 or not np.isin(uniq, [0, 1]).all():
                raise TapVidContractError(
                    f"[TapVidContract] key='{key}' integer mask must be in {{0,1}}, got uniq={uniq[:10]} "
                    f"({_shape_str(a)}) ({ctx})."
                )
        return a.astype(np.bool_, copy=False)
    if strict:
        raise TapVidContractError(
            f"[TapVidContract] key='{key}' must be bool or uint8/int mask, got {a.dtype} "
            f"({_shape_str(a)}) ({ctx})."
        )
    # permissive: try bool cast
    return a.astype(np.bool_, copy=False)


def _parse_meta_json(meta: Any, *, ctx: str, strict: bool) -> Dict[str, Any]:
    if meta is None:
        return {}
    if isinstance(meta, (bytes, bytearray)):
        meta = meta.decode("utf-8", errors="replace")
    if isinstance(meta, np.ndarray):
        # npz may store 0-d array
        if meta.ndim == 0:
            meta = meta.item()
        else:
            raise TapVidContractError(f"[TapVidContract] meta_json must be scalar, got {_shape_str(meta)} ({ctx}).")
    if not isinstance(meta, str):
        raise TapVidContractError(f"[TapVidContract] meta_json must be str, got {type(meta)} ({ctx}).")

    meta = meta.strip()
    if meta == "":
        return {}

    try:
        obj = json.loads(meta)
    except Exception as e:
        raise TapVidContractError(f"[TapVidContract] meta_json JSON parse failed: {e} ({ctx}).") from e

    if not isinstance(obj, dict):
        raise TapVidContractError(f"[TapVidContract] meta_json must decode to dict, got {type(obj)} ({ctx}).")

    if strict:
        # minimal required fields (you can extend later, but don't weaken)
        req = ["tracker", "query_mode", "resize_to_256", "keep_aspect"]
        missing = [k for k in req if k not in obj]
        if missing:
            raise TapVidContractError(f"[TapVidContract] meta_json missing fields={missing} ({ctx}).")
    return obj


@dataclass(frozen=True)
class PredBundle:
    """Normalized prediction bundle after contract validation."""
    tracks_xy: np.ndarray     # float32 [T,N,2] in (x,y) original coords
    vis: np.ndarray           # bool    [T,N]
    queries_txy: np.ndarray   # float32 [N,3] (t0,x0,y0)
    video_hw: Tuple[int, int] # (H,W)
    meta: Dict[str, Any]


def validate_paths_exist(frame_paths: Sequence[Union[str, Path]], *, ctx: str = "") -> None:
    """Ensure all frame paths exist on disk (hard error)."""
    missing: list[str] = []
    for p in frame_paths:
        pp = _as_path(p)
        if not pp.exists():
            missing.append(str(pp))
            if len(missing) >= 10:
                break
    if missing:
        hint = f" first_missing={missing[:3]}" if missing else ""
        raise TapVidContractError(
            f"[TapVidContract] missing {len(missing)} frame files.{hint} ({ctx})."
        )


def validate_pred_npz(
    npz: Mapping[str, Any],
    *,
    strict: bool = True,
    ctx: str = "",
) -> PredBundle:
    """
    Validate and normalize a loaded TAP-Vid prediction npz dict.

    Required keys:
      - tracks_xy: float32 [T,N,2]
      - vis: bool/uint8/int mask [T,N]
      - queries_txy: float32/int [N,3]
      - video_hw: int32 [2] => [H,W]
      - meta_json: (optional but recommended) JSON string

    Returns:
      PredBundle with canonical dtypes.
    """
    tracks_xy = _require_ndarray(npz, "tracks_xy", ctx=ctx)
    vis = _require_ndarray(npz, "vis", ctx=ctx)
    queries_txy = _require_ndarray(npz, "queries_txy", ctx=ctx)
    video_hw = _require_ndarray(npz, "video_hw", ctx=ctx)

    _require_rank(tracks_xy, 3, key="tracks_xy", ctx=ctx)
    _require_last_dim(tracks_xy, 2, key="tracks_xy", ctx=ctx)
    _require_rank(vis, 2, key="vis", ctx=ctx)
    _require_rank(queries_txy, 2, key="queries_txy", ctx=ctx)
    _require_rank(video_hw, 1, key="video_hw", ctx=ctx)

    if queries_txy.shape[1] != 3:
        raise TapVidContractError(
            f"[TapVidContract] key='queries_txy' must have shape [N,3], got {_shape_str(queries_txy)} ({ctx})."
        )
    if video_hw.shape[0] != 2:
        raise TapVidContractError(
            f"[TapVidContract] key='video_hw' must have shape [2] as [H,W], got {_shape_str(video_hw)} ({ctx})."
        )

    # dtype normalization
    tracks_xy = _coerce_float32(tracks_xy, key="tracks_xy", ctx=ctx, strict=strict)
    vis = _coerce_bool(vis, key="vis", ctx=ctx, strict=strict)

    # queries: allow int -> float32 (t0 is integer but we store float32 for simplicity)
    if np.issubdtype(queries_txy.dtype, np.integer):
        if strict and queries_txy.dtype not in (np.int32, np.int64, np.int16, np.int8):
            raise TapVidContractError(
                f"[TapVidContract] key='queries_txy' integer dtype unexpected {queries_txy.dtype} ({ctx})."
            )
        queries_txy = queries_txy.astype(np.float32, copy=False)
    else:
        queries_txy = _coerce_float32(queries_txy, key="queries_txy", ctx=ctx, strict=strict)

    video_hw = _coerce_int32(video_hw, key="video_hw", ctx=ctx, strict=False)  # safe to cast
    H = int(video_hw[0])
    W = int(video_hw[1])
    if H <= 0 or W <= 0:
        raise TapVidContractError(f"[TapVidContract] invalid video_hw=[{H},{W}] ({ctx}).")

    T, N, _ = tracks_xy.shape
    if vis.shape != (T, N):
        raise TapVidContractError(
            f"[TapVidContract] key='vis' shape mismatch: expected {(T,N)}, got {tuple(vis.shape)} ({ctx})."
        )
    if queries_txy.shape[0] != N:
        raise TapVidContractError(
            f"[TapVidContract] key='queries_txy' N mismatch: tracks N={N} vs queries N={queries_txy.shape[0]} ({ctx})."
        )

    # numeric sanity
    _ensure_finite(tracks_xy, key="tracks_xy", ctx=ctx)
    _ensure_finite(queries_txy, key="queries_txy", ctx=ctx)

    # optional bounds check (strict => hard error)
    # allow slightly out-of-bound due to rounding? no: for paper-grade, keep strict.
    xy_min = tracks_xy.reshape(-1, 2).min(axis=0)
    xy_max = tracks_xy.reshape(-1, 2).max(axis=0)
    if strict:
        if xy_min[0] < -1e-3 or xy_min[1] < -1e-3 or xy_max[0] > (W - 1 + 1e-3) or xy_max[1] > (H - 1 + 1e-3):
            raise TapVidContractError(
                f"[TapVidContract] tracks_xy out of bounds: min={xy_min.tolist()} max={xy_max.tolist()} "
                f"for video_hw=({H},{W}) ({ctx})."
            )

    # t0 range check
    t0 = queries_txy[:, 0]
    # allow float but must be integer-valued
    if strict:
        if not np.all(np.abs(t0 - np.round(t0)) < 1e-6):
            bad = np.where(np.abs(t0 - np.round(t0)) >= 1e-6)[0][:10].tolist()
            raise TapVidContractError(
                f"[TapVidContract] queries_txy[:,0] must be integer-valued frame indices; bad idx={bad} ({ctx})."
            )
    t0i = np.round(t0).astype(np.int64)
    if strict:
        if (t0i < 0).any() or (t0i >= T).any():
            bad = np.where((t0i < 0) | (t0i >= T))[0][:10].tolist()
            raise TapVidContractError(
                f"[TapVidContract] queries_txy t0 out of range [0,{T-1}]; bad idx={bad} ({ctx})."
            )

    # meta_json
    meta = {}
    if "meta_json" in npz:
        meta = _parse_meta_json(npz["meta_json"], ctx=ctx, strict=False)
    elif strict:
        # 不强制，但推荐：若你想强制就把 strict 时 raise
        meta = {}

    return PredBundle(
        tracks_xy=tracks_xy,
        vis=vis,
        queries_txy=queries_txy,
        video_hw=(H, W),
        meta=meta,
    )


def load_and_validate_pred_npz(
    npz_path: Union[str, Path],
    *,
    strict: bool = True,
    allow_pickle: bool = False,
) -> PredBundle:
    """Load npz from disk, then validate to PredBundle."""
    p = _as_path(npz_path)
    if not p.exists():
        raise TapVidContractError(f"[TapVidContract] pred file not found: {p}")

    try:
        with np.load(str(p), allow_pickle=allow_pickle) as z:
            # materialize to plain dict so the file can close
            d = {k: z[k] for k in z.files}
    except Exception as e:
        raise TapVidContractError(f"[TapVidContract] failed to load npz: {p} err={e}") from e

    return validate_pred_npz(d, strict=strict, ctx=f"pred={p.name}")


def validate_seq_consistency(
    *,
    seq_name: str,
    frame_paths: Sequence[Union[str, Path]],
    video_hw: Tuple[int, int],
    gt_tracks_xy: np.ndarray,
    gt_vis: np.ndarray,
    pred: PredBundle,
    strict: bool = True,
) -> None:
    """
    Validate consistency between GT (dataset) and PredBundle.
    This is called per-sequence in runner.

    Requirements:
      - T matches frames and GT and pred
      - N matches GT and pred
      - video_hw matches pred.video_hw
    """
    ctx = f"seq={seq_name}"

    if not isinstance(video_hw, tuple) or len(video_hw) != 2:
        raise TapVidContractError(f"[TapVidContract] invalid dataset video_hw={video_hw} ({ctx}).")
    H, W = int(video_hw[0]), int(video_hw[1])
    if H <= 0 or W <= 0:
        raise TapVidContractError(f"[TapVidContract] invalid dataset video_hw=({H},{W}) ({ctx}).")

    # frames existence check (hard)
    validate_paths_exist(frame_paths, ctx=ctx)

    if gt_tracks_xy.ndim != 3 or gt_tracks_xy.shape[-1] != 2:
        raise TapVidContractError(f"[TapVidContract] gt_tracks_xy must be [T,N,2], got {_shape_str(gt_tracks_xy)} ({ctx}).")
    if gt_vis.ndim != 2:
        raise TapVidContractError(f"[TapVidContract] gt_vis must be [T,N], got {_shape_str(gt_vis)} ({ctx}).")

    T = len(frame_paths)
    if gt_tracks_xy.shape[0] != T or gt_vis.shape[0] != T:
        raise TapVidContractError(
            f"[TapVidContract] GT T mismatch: frames T={T}, gt_tracks T={gt_tracks_xy.shape[0]}, gt_vis T={gt_vis.shape[0]} ({ctx})."
        )

    if pred.tracks_xy.shape[0] != T:
        raise TapVidContractError(
            f"[TapVidContract] pred T mismatch: frames T={T}, pred_tracks T={pred.tracks_xy.shape[0]} ({ctx})."
        )

    N = gt_tracks_xy.shape[1]
    if gt_vis.shape[1] != N:
        raise TapVidContractError(
            f"[TapVidContract] GT N mismatch: gt_tracks N={N}, gt_vis N={gt_vis.shape[1]} ({ctx})."
        )
    if pred.tracks_xy.shape[1] != N:
        raise TapVidContractError(
            f"[TapVidContract] pred N mismatch: gt N={N}, pred N={pred.tracks_xy.shape[1]} ({ctx})."
        )

    if pred.video_hw != (H, W):
        # 若你允许 dataset video_hw 由第一帧读出来而略有差异，可在这里做更复杂逻辑；
        # 但 paper-grade 推荐严格一致。
        raise TapVidContractError(
            f"[TapVidContract] video_hw mismatch: dataset=({H},{W}) pred={pred.video_hw} ({ctx})."
        )

    if strict:
        # quick bounds sanity on GT too (detect corrupted pkl/resize bug)
        _ensure_finite(gt_tracks_xy.astype(np.float32, copy=False), key="gt_tracks_xy", ctx=ctx)
        gmin = gt_tracks_xy.reshape(-1, 2).min(axis=0)
        gmax = gt_tracks_xy.reshape(-1, 2).max(axis=0)
        if gmin[0] < -1e-3 or gmin[1] < -1e-3 or gmax[0] > (W - 1 + 1e-3) or gmax[1] > (H - 1 + 1e-3):
            raise TapVidContractError(
                f"[TapVidContract] gt_tracks_xy out of bounds: min={gmin.tolist()} max={gmax.tolist()} "
                f"for video_hw=({H},{W}) ({ctx})."
            )