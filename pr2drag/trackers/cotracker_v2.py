# pr2drag/trackers/cotracker_v2.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple, Dict
import importlib
import inspect

import numpy as np


@dataclass(frozen=True)
class CoTrackerV2Output:
    # We expose a stable contract to runner_tapvid.py
    # tracks_xy: [T,Q,2] float32 in (x,y)
    # occluded : [T,Q] bool, True=occluded
    tracks_xy: np.ndarray
    occluded: np.ndarray


def _require(cond: bool, msg: str, exc: type[Exception] = ValueError) -> None:
    if not cond:
        raise exc(msg)


def _as_bool(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.dtype == np.bool_:
        return a
    return (a.astype(np.int32) != 0)


def _resolve_build_cotracker() -> Any:
    """
    Resolve a callable that builds a CoTracker model.
    Support common layouts across:
      - official facebookresearch/co-tracker repo (editable install)
      - pip package variants (if any)
    """
    tried: list[str] = []

    # (A) official repo layout: function inside module
    # cotracker/models/build_cotracker.py defines build_cotracker(...)
    for spec in [
        ("cotracker.models.build_cotracker", "build_cotracker"),
        ("cotracker.models.core", "build_cotracker"),
        ("cotracker", "build_cotracker"),
    ]:
        mod_name, attr = spec
        try:
            mod = importlib.import_module(mod_name)
            tried.append(f"{mod_name}:{attr}")
            fn = getattr(mod, attr, None)
            if callable(fn):
                return fn
        except Exception as e:
            tried.append(f"{mod_name}:{attr} -> {repr(e)}")

    # (B) Sometimes people do `from cotracker.models import build_cotracker`
    # which gives you a module; then the function is module.build_cotracker
    for mod_name in ["cotracker.models"]:
        try:
            mod = importlib.import_module(mod_name)
            tried.append(f"{mod_name}:build_cotracker(module?)")
            maybe_mod = getattr(mod, "build_cotracker", None)
            if maybe_mod is None:
                continue
            # if it's a module, try attribute inside it
            if inspect.ismodule(maybe_mod):
                fn = getattr(maybe_mod, "build_cotracker", None)
                if callable(fn):
                    return fn
            if callable(maybe_mod):
                return maybe_mod
        except Exception as e:
            tried.append(f"{mod_name}:build_cotracker -> {repr(e)}")

    raise ImportError(
        "[cotracker_v2] Could not import/build CoTracker.\n"
        "Tried:\n  - " + "\n  - ".join(tried) + "\n\n"
        "Fix:\n"
        "  - Ensure you installed the official CoTracker repo with: pip install -e /content/co-tracker\n"
    )


def _resolve_predictor_class() -> Optional[Any]:
    """
    Prefer the official predictor API if present:
      from cotracker.predictor import CoTrackerPredictor
    """
    try:
        m = importlib.import_module("cotracker.predictor")
        cls = getattr(m, "CoTrackerPredictor", None)
        if cls is not None:
            return cls
    except Exception:
        pass
    return None


def _to_torch_video(video_uint8: np.ndarray) -> "Any":
    """
    Convert [T,H,W,3] uint8 RGB -> torch tensor [1,T,3,H,W] float32 in [0,1].
    """
    try:
        import torch
    except Exception as e:
        raise RuntimeError("PyTorch is required for CoTracker.") from e

    v = np.asarray(video_uint8)
    _require(v.ndim == 4 and v.shape[-1] == 3, f"[cotracker_v2] video must be [T,H,W,3], got {v.shape}")
    if v.dtype != np.uint8:
        v = np.clip(v, 0, 255).astype(np.uint8)

    t = torch.from_numpy(v).to(torch.float32) / 255.0   # [T,H,W,3]
    t = t.permute(0, 3, 1, 2).unsqueeze(0)              # [1,T,3,H,W]
    return t


def _to_torch_queries(queries: np.ndarray) -> "Any":
    """
    queries: [Q,3] (t,y,x) float32
    CoTracker commonly uses [1,Q,3].
    """
    try:
        import torch
    except Exception as e:
        raise RuntimeError("PyTorch is required for CoTracker.") from e

    q = np.asarray(queries, dtype=np.float32)
    _require(q.ndim == 2 and q.shape[1] == 3, f"[cotracker_v2] queries must be [Q,3], got {q.shape}")
    return torch.from_numpy(q).to(torch.float32).unsqueeze(0)  # [1,Q,3]


def _pick_device(device: Optional[str]) -> "Any":
    try:
        import torch
    except Exception as e:
        raise RuntimeError("PyTorch is required for CoTracker.") from e

    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _normalize_outputs(out: Any, T: int, Q: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize to:
      tracks_xy_TQ2: [T,Q,2] float32
      occluded_TQ  : [T,Q] bool (True=occluded)
    Accept common shapes:
      - tracks: [1,T,Q,2] or [T,Q,2] or [1,Q,T,2]
      - vis/occluded: [1,T,Q] / [T,Q] / [1,Q,T]
    """
    def _np(a: Any) -> np.ndarray:
        if hasattr(a, "detach"):
            a = a.detach().cpu().numpy()
        return np.asarray(a)

    tracks = None
    vis = None
    occ = None

    # dict
    if isinstance(out, dict):
        tracks = out.get("tracks", out.get("tracks_xy", None))
        vis = out.get("vis", out.get("visibility", None))
        occ = out.get("occluded", out.get("occ", None))

    # object
    if tracks is None:
        tracks = getattr(out, "tracks", None)
        if tracks is None:
            tracks = getattr(out, "tracks_xy", None)
    if vis is None:
        vis = getattr(out, "vis", None)
        if vis is None:
            vis = getattr(out, "visibility", None)
    if occ is None:
        occ = getattr(out, "occluded", None)
        if occ is None:
            occ = getattr(out, "occ", None)

    _require(tracks is not None, "[cotracker_v2] output missing tracks/tracks_xy")
    tracks = _np(tracks).astype(np.float32)

    # decide occlusion/visibility
    if occ is not None:
        occ = _as_bool(_np(occ))
    elif vis is not None:
        vis = _as_bool(_np(vis))
        occ = (~vis).astype(bool)
    else:
        raise KeyError("[cotracker_v2] output missing occluded/vis")

    # squeeze batch if present
    if tracks.ndim == 4 and tracks.shape[0] == 1:
        tracks = tracks[0]
    if occ.ndim == 3 and occ.shape[0] == 1:
        occ = occ[0]

    # now tracks is either [T,Q,2] or [Q,T,2]
    _require(tracks.ndim == 3 and tracks.shape[-1] == 2, f"[cotracker_v2] bad tracks shape {tracks.shape}")
    _require(occ.ndim == 2, f"[cotracker_v2] bad occ shape {occ.shape}")

    if tracks.shape[0] == T and tracks.shape[1] == Q:
        _require(occ.shape == (T, Q), f"[cotracker_v2] occ shape {occ.shape} != {(T,Q)}")
        return tracks, occ

    if tracks.shape[0] == Q and tracks.shape[1] == T:
        _require(occ.shape == (Q, T), f"[cotracker_v2] occ shape {occ.shape} != {(Q,T)}")
        return np.transpose(tracks, (1, 0, 2)).astype(np.float32), np.transpose(occ, (1, 0)).astype(bool)

    raise ValueError(
        f"[cotracker_v2] cannot align shapes: tracks={tracks.shape}, occ={occ.shape}, "
        f"expected (T,Q,2)=({T},{Q},2) or (Q,T,2)=({Q},{T},2)"
    )


def run_cotracker_v2(
    *,
    video_uint8: np.ndarray,          # [T,H,W,3] RGB uint8
    queries_tyx: np.ndarray,          # [Q,3] (t,y,x)
    checkpoint: Optional[str] = None,
    device: Optional[str] = None,
    **_: Any,
) -> CoTrackerV2Output:
    """
    Stable wrapper used by tier0/runner_tapvid.py.

    Notes:
      - We treat queries as (t,y,x) which matches TAP-Vid output format.
      - This wrapper supports both official Predictor API and raw build_cotracker API.
    """
    T = int(np.asarray(video_uint8).shape[0])
    Q = int(np.asarray(queries_tyx).shape[0])

    dev = _pick_device(device)

    # Prefer Predictor if available
    Predictor = _resolve_predictor_class()
    if Predictor is not None:
        # CoTrackerPredictor usually accepts checkpoint=... or model=...
        try:
            kwargs: Dict[str, Any] = {}
            sig = inspect.signature(Predictor)
            if "checkpoint" in sig.parameters:
                kwargs["checkpoint"] = checkpoint
            predictor = Predictor(**kwargs)
        except Exception as e:
            raise ImportError(f"[cotracker_v2] Failed to init CoTrackerPredictor: {repr(e)}") from e

        try:
            import torch  # noqa
            predictor = predictor.to(dev) if hasattr(predictor, "to") else predictor
            predictor.eval() if hasattr(predictor, "eval") else None

            video_t = _to_torch_video(video_uint8).to(dev)       # [1,T,3,H,W]
            queries_t = _to_torch_queries(queries_tyx).to(dev)   # [1,Q,3]

            # call predictor in a version-robust way
            with torch.no_grad():
                if callable(predictor):
                    out = predictor(video_t, queries_t)
                elif hasattr(predictor, "forward"):
                    out = predictor.forward(video_t, queries_t)
                elif hasattr(predictor, "predict"):
                    out = predictor.predict(video_t, queries_t)
                else:
                    raise TypeError("[cotracker_v2] CoTrackerPredictor has no callable/forward/predict method")

            tracks_TQ2, occ_TQ = _normalize_outputs(out, T=T, Q=Q)
            return CoTrackerV2Output(tracks_xy=tracks_TQ2.astype(np.float32), occluded=occ_TQ.astype(bool))

        except Exception as e:
            raise RuntimeError(f"[cotracker_v2] Predictor inference failed: {repr(e)}") from e

    # Fallback: build model directly
    build_fn = _resolve_build_cotracker()
    try:
        sig = inspect.signature(build_fn)
        kwargs: Dict[str, Any] = {}
        if "checkpoint" in sig.parameters:
            kwargs["checkpoint"] = checkpoint
        model = build_fn(**kwargs)
    except Exception as e:
        raise ImportError(f"[cotracker_v2] build_cotracker() failed: {repr(e)}") from e

    # Try to locate a runner interface
    try:
        import torch

        model = model.to(dev) if hasattr(model, "to") else model
        model.eval() if hasattr(model, "eval") else None

        video_t = _to_torch_video(video_uint8).to(dev)
        queries_t = _to_torch_queries(queries_tyx).to(dev)

        with torch.no_grad():
            if callable(model):
                out = model(video_t, queries_t)
            elif hasattr(model, "forward"):
                out = model.forward(video_t, queries_t)
            else:
                raise TypeError("[cotracker_v2] built model is not callable and has no forward()")

        tracks_TQ2, occ_TQ = _normalize_outputs(out, T=T, Q=Q)
        return CoTrackerV2Output(tracks_xy=tracks_TQ2.astype(np.float32), occluded=occ_TQ.astype(bool))

    except Exception as e:
        raise RuntimeError(f"[cotracker_v2] Model inference failed: {repr(e)}") from e