# pr2drag/trackers/cotracker_v2.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import importlib
import inspect

import numpy as np


@dataclass(frozen=True)
class CoTrackerV2Output:
    """
    Stable contract for PR2-Drag Lite.

    tracks_xy: float32 [T,Q,2] in (x,y)
    occluded : bool    [T,Q]   True=occluded
    """
    tracks_xy: np.ndarray
    occluded: np.ndarray


# ---------------------------
# small utils
# ---------------------------
def _require(cond: bool, msg: str, exc: type[Exception] = ValueError) -> None:
    if not cond:
        raise exc(msg)


def _as_bool(a: Any) -> np.ndarray:
    a = np.asarray(a)
    if a.dtype == np.bool_:
        return a
    return (a.astype(np.int32) != 0)


def _np(a: Any) -> np.ndarray:
    if hasattr(a, "detach"):
        a = a.detach().cpu().numpy()
    return np.asarray(a)


def _to_torch_video(video_uint8: np.ndarray) -> Any:
    """
    [T,H,W,3] uint8 RGB -> torch [1,T,3,H,W] float32 in [0,1]
    """
    try:
        import torch
    except Exception as e:
        raise RuntimeError("[cotracker_v2] PyTorch is required.") from e

    v = np.asarray(video_uint8)
    _require(v.ndim == 4 and v.shape[-1] == 3, f"[cotracker_v2] video must be [T,H,W,3], got {v.shape}")
    if v.dtype != np.uint8:
        v = np.clip(v, 0, 255).astype(np.uint8)

    t = torch.from_numpy(v).to(torch.float32) / 255.0          # [T,H,W,3]
    t = t.permute(0, 3, 1, 2).unsqueeze(0).contiguous()        # [1,T,3,H,W]
    return t


def _to_torch_queries(q: np.ndarray) -> Any:
    """
    [Q,3] -> torch [1,Q,3] float32
    """
    try:
        import torch
    except Exception as e:
        raise RuntimeError("[cotracker_v2] PyTorch is required.") from e

    q = np.asarray(q, dtype=np.float32)
    _require(q.ndim == 2 and q.shape[1] == 3, f"[cotracker_v2] queries must be [Q,3], got {q.shape}")
    return torch.from_numpy(q).to(torch.float32).unsqueeze(0).contiguous()


def _pick_device(device: Optional[str]) -> Any:
    try:
        import torch
    except Exception as e:
        raise RuntimeError("[cotracker_v2] PyTorch is required.") from e

    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _tyx_to_txy(q_tyx: np.ndarray) -> np.ndarray:
    q = np.asarray(q_tyx, dtype=np.float32)
    out = q.copy()
    out[:, 1] = q[:, 2]  # x
    out[:, 2] = q[:, 1]  # y
    return out


# ---------------------------
# resolve builders / predictor
# ---------------------------
def _resolve_predictor_class() -> Optional[type]:
    """
    Official repo often provides:
      from cotracker.predictor import CoTrackerPredictor
    """
    try:
        m = importlib.import_module("cotracker.predictor")
        cls = getattr(m, "CoTrackerPredictor", None)
        if isinstance(cls, type):
            return cls
    except Exception:
        pass
    return None


def _resolve_build_cotracker_fn() -> Optional[Any]:
    """
    Try to find a callable build_cotracker() in common places.
    IMPORTANT: must return a *callable function*, not a module.
    """
    candidates = [
        ("cotracker.models.build_cotracker", "build_cotracker"),
        ("cotracker.models.core", "build_cotracker"),
        ("cotracker", "build_cotracker"),
        ("cotracker.models", "build_cotracker"),  # sometimes re-exported
    ]
    for mod_name, attr in candidates:
        try:
            mod = importlib.import_module(mod_name)
            fn = getattr(mod, attr, None)
            # some repos expose build_cotracker as a *module*
            if inspect.ismodule(fn):
                fn2 = getattr(fn, "build_cotracker", None)
                if callable(fn2):
                    return fn2
            if callable(fn):
                return fn
        except Exception:
            continue
    return None


def _resolve_torchhub_model(checkpoint: Optional[str]) -> Optional[Any]:
    """
    Official hubconf commonly supports:
      torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline", pretrained=True)
    In offline/editable installs, the repo may ship hubconf.py locally.
    """
    try:
        import torch
    except Exception:
        return None

    # First try local hubconf (if installed from repo)
    for repo in ["cotracker", "facebookresearch/co-tracker"]:
        for name in ["cotracker3_offline", "cotracker3_online", "cotracker2", "cotracker"]:
            try:
                # NOTE: if local 'cotracker' package has hubconf.py, torch.hub.load('cotracker', ...) can work.
                model = torch.hub.load(repo, name, pretrained=True)
                return model
            except Exception:
                continue

    # If user passes a checkpoint, torchhub path is ambiguous; we won't guess.
    _ = checkpoint
    return None


# ---------------------------
# calling + output normalization
# ---------------------------
def _normalize_outputs(out: Any, T: int, Q: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      tracks_xy_TQ2: [T,Q,2] float32
      occluded_TQ  : [T,Q] bool (True=occluded)
    """
    tracks = None
    vis = None
    occ = None

    if isinstance(out, dict):
        tracks = out.get("tracks", out.get("tracks_xy", out.get("pred_tracks", None)))
        vis = out.get("vis", out.get("visibility", out.get("pred_visibility", None)))
        occ = out.get("occluded", out.get("occ", None))
    else:
        tracks = getattr(out, "tracks", None)
        if tracks is None:
            tracks = getattr(out, "tracks_xy", None)
        if tracks is None:
            tracks = getattr(out, "pred_tracks", None)

        vis = getattr(out, "vis", None)
        if vis is None:
            vis = getattr(out, "visibility", None)
        if vis is None:
            vis = getattr(out, "pred_visibility", None)

        occ = getattr(out, "occluded", None)
        if occ is None:
            occ = getattr(out, "occ", None)

    _require(tracks is not None, "[cotracker_v2] output missing tracks/tracks_xy/pred_tracks")
    tracks = _np(tracks).astype(np.float32)

    if occ is not None:
        occ = _as_bool(_np(occ))
    elif vis is not None:
        vis = _as_bool(_np(vis))
        occ = (~vis).astype(bool)
    else:
        raise KeyError("[cotracker_v2] output missing occluded/vis/visibility")

    # squeeze batch if present
    if tracks.ndim == 4 and tracks.shape[0] == 1:
        tracks = tracks[0]
    if occ.ndim == 3 and occ.shape[0] == 1:
        occ = occ[0]

    _require(tracks.ndim == 3 and tracks.shape[-1] == 2, f"[cotracker_v2] bad tracks shape {tracks.shape}")
    _require(occ.ndim == 2, f"[cotracker_v2] bad occ shape {occ.shape}")

    # accept [T,Q,2] or [Q,T,2]
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


def _call_model_version_robust(model: Any, video_t: Any, queries_t: Any) -> Any:
    """
    Try multiple calling conventions, because CoTracker versions differ.
    """
    # 1) callable(model)(video, queries)
    try:
        if callable(model):
            return model(video_t, queries_t)
    except Exception:
        pass

    # 2) model.forward(video, queries)
    try:
        if hasattr(model, "forward"):
            return model.forward(video_t, queries_t)
    except Exception:
        pass

    # 3) model.predict(video, queries)
    try:
        if hasattr(model, "predict"):
            return model.predict(video_t, queries_t)
    except Exception:
        pass

    # 4) keyword variants
    for kw in ["queries", "query_points", "queries_txy", "queries_tyx"]:
        try:
            if callable(model):
                return model(video_t, **{kw: queries_t})
        except Exception:
            continue

    raise TypeError("[cotracker_v2] Cannot call model with provided (video, queries) under any known API variant.")


# ---------------------------
# public entry
# ---------------------------
def run_cotracker_v2(
    *,
    video_uint8: np.ndarray,          # [T,H,W,3] RGB uint8
    queries_tyx: np.ndarray,          # [Q,3] (t,y,x)
    checkpoint: Optional[str] = None,
    device: Optional[str] = None,
    **_: Any,
) -> CoTrackerV2Output:
    """
    Stable wrapper for TAP-Vid runner.

    We accept queries in TAP-Vid format (t,y,x).
    Internally we will try both (t,y,x) and (t,x,y) when calling CoTracker,
    because different releases use different conventions.
    """
    v = np.asarray(video_uint8)
    q_tyx = np.asarray(queries_tyx, dtype=np.float32)
    _require(v.ndim == 4 and v.shape[-1] == 3, f"[cotracker_v2] video must be [T,H,W,3], got {v.shape}")
    _require(q_tyx.ndim == 2 and q_tyx.shape[1] == 3, f"[cotracker_v2] queries_tyx must be [Q,3], got {q_tyx.shape}")

    T = int(v.shape[0])
    Q = int(q_tyx.shape[0])

    dev = _pick_device(device)

    # Build / resolve model in 3 tiers:
    # A) Predictor class
    Predictor = _resolve_predictor_class()
    model = None

    if Predictor is not None:
        try:
            # Some Predictor __init__ accepts checkpoint, some doesn't.
            try:
                sig = inspect.signature(Predictor)
                if "checkpoint" in sig.parameters:
                    model = Predictor(checkpoint=checkpoint)
                else:
                    model = Predictor()
            except Exception:
                model = Predictor()
        except Exception as e:
            raise ImportError(f"[cotracker_v2] Failed to init CoTrackerPredictor: {repr(e)}") from e

    # B) build_cotracker()
    if model is None:
        build_fn = _resolve_build_cotracker_fn()
        if build_fn is not None:
            try:
                kwargs: Dict[str, Any] = {}
                try:
                    sig = inspect.signature(build_fn)
                    if "checkpoint" in sig.parameters:
                        kwargs["checkpoint"] = checkpoint
                except Exception:
                    pass
                model = build_fn(**kwargs)
            except Exception as e:
                raise ImportError(f"[cotracker_v2] build_cotracker() failed: {repr(e)}") from e

    # C) torch.hub
    if model is None:
        model = _resolve_torchhub_model(checkpoint=checkpoint)

    if model is None:
        raise ImportError(
            "[cotracker_v2] Could not construct a CoTracker model.\n"
            "Fix (Colab recommended):\n"
            "  1) git clone https://github.com/facebookresearch/co-tracker /content/co-tracker\n"
            "  2) pip install -e /content/co-tracker\n"
            "  3) restart runtime, then rerun.\n"
        )

    # Move to device if possible
    try:
        model = model.to(dev) if hasattr(model, "to") else model
        model.eval() if hasattr(model, "eval") else None
    except Exception:
        pass

    # Prepare torch inputs
    try:
        import torch
        video_t = _to_torch_video(v).to(dev)
        q1 = _to_torch_queries(q_tyx).to(dev)           # (t,y,x)
        q2 = _to_torch_queries(_tyx_to_txy(q_tyx)).to(dev)  # (t,x,y)

        with torch.no_grad():
            # try tyx first
            try:
                out = _call_model_version_robust(model, video_t, q1)
            except Exception:
                out = _call_model_version_robust(model, video_t, q2)

        tracks_TQ2, occ_TQ = _normalize_outputs(out, T=T, Q=Q)
        return CoTrackerV2Output(tracks_xy=tracks_TQ2.astype(np.float32), occluded=occ_TQ.astype(bool))

    except Exception as e:
        raise RuntimeError(f"[cotracker_v2] Inference failed: {repr(e)}") from e