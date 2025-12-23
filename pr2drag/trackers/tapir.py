# pr2drag/trackers/tapir.py
from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from pr2drag.datasets.tapvid import TapVidSeq
from pr2drag.trackers.base import TapVidPred


class TapirDependencyError(RuntimeError):
    """Raised when TAPIR backend dependencies are missing or not usable."""


class TapirInputError(ValueError):
    """Raised when inputs (video / queries) violate the tracker contract."""


@dataclass(frozen=True)
class TapirTrackerConfig:
    """
    TAPIR wrapper config.

    Contract (this wrapper):
      - Input video: [T,H,W,3] uint8 (or float -> auto converted)
      - Input queries: [Q,3] float32 in (t, y, x)
        * If resize_to_256=True: queries are assumed already in 256 raster coords
          (recommended: let datasets/tapvid.py do it).
        * If resize_to_256=False: queries are in original pixel coords.

      - Backend expects:
        video_256_uint8: [T,256,256,3] uint8
        queries_tyx_norm: [Q,3] float32 in [t, y_norm, x_norm], y/x in [0,1]
      - Backend returns:
        tracks_yx_norm: [T,Q,2] float32 in (y_norm, x_norm), normalized [0,1]
        occluded: [T,Q] bool, True=occluded

      - Wrapper outputs:
        tracks_xy: [T,Q,2] float32 in (x,y) in 256 raster pixel coords (0..255)
        vis: [T,Q] bool True=visible (i.e., ~occluded)
    """

    ckpt_path: str = ""  # optional; backend may ignore

    resize_to_256: bool = True
    keep_aspect: bool = True
    interp: str = "bilinear"  # "bilinear" | "nearest"

    backend: str = "auto"  # "auto" | "external_fn" | "jax_tapir"
    external_infer: str = ""  # "module:callable" override; else env PR2DRAG_TAPIR_INFER

    clamp_coords: bool = True
    clamp_eps: float = 1e-3


def _require(cond: bool, msg: str, exc_type=TapirInputError) -> None:
    if not cond:
        raise exc_type(msg)


def _as_uint8_video(video: np.ndarray) -> np.ndarray:
    """
    Accept:
      - uint8 [T,H,W,3]
      - float [T,H,W,3] in [0,1] or [0,255]
      - int   [T,H,W,3] in [0,255] (clipped)
    Return uint8 [T,H,W,3].
    """
    _require(isinstance(video, np.ndarray), f"video must be np.ndarray, got {type(video)}")
    _require(video.ndim == 4 and video.shape[-1] == 3, f"video must be [T,H,W,3], got {video.shape}")

    if video.dtype == np.uint8:
        return video

    if np.issubdtype(video.dtype, np.floating):
        v = video.astype(np.float32)
        vmax = float(np.nanmax(v)) if v.size else 0.0
        if vmax <= 1.5:
            v = v * 255.0
        v = np.clip(v, 0.0, 255.0)
        return v.astype(np.uint8)

    if np.issubdtype(video.dtype, np.integer):
        v = np.clip(video, 0, 255).astype(np.uint8)
        return v

    raise TapirInputError(f"Unsupported video dtype: {video.dtype}")


def _resize_pad_to_256(
    frames_uint8: np.ndarray,
    keep_aspect: bool,
    interp: str,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Resize frames to 256x256, optionally keep aspect with padding.

    Returns:
      frames_256: uint8 [T,256,256,3]
      meta:
        - orig_hw: (H,W)
        - scale: (sy,sx) mapping orig->resized (before pad)
        - pad: (py,px) padding added (float)
        - resized_hw: (Hr,Wr)
        - pad_int: (y0,x0) integer insertion
    """
    _require(interp in ("bilinear", "nearest"), f"interp must be bilinear/nearest, got {interp!r}")

    try:
        import cv2  # type: ignore
    except Exception as e:
        raise TapirDependencyError(
            "OpenCV (cv2) is required for TAPIR preprocessing. "
            "Install with: pip install opencv-python"
        ) from e

    T, H, W, _ = frames_uint8.shape

    inter = cv2.INTER_LINEAR if interp == "bilinear" else cv2.INTER_NEAREST

    if not keep_aspect:
        Hr = Wr = 256
        fx = 256.0 / float(W)
        fy = 256.0 / float(H)
        out = np.stack([cv2.resize(frames_uint8[t], (Wr, Hr), interpolation=inter) for t in range(T)], axis=0)
        meta = {
            "orig_hw": (H, W),
            "resized_hw": (Hr, Wr),
            "scale": (fy, fx),
            "pad": (0.0, 0.0),
            "pad_int": (0, 0),
        }
        return out.astype(np.uint8), meta

    s = min(256.0 / float(H), 256.0 / float(W))
    Hr = int(round(H * s))
    Wr = int(round(W * s))
    Hr = max(1, min(256, Hr))
    Wr = max(1, min(256, Wr))

    py = (256 - Hr) / 2.0
    px = (256 - Wr) / 2.0
    y0 = int(np.floor(py))
    x0 = int(np.floor(px))

    resized = np.stack([cv2.resize(frames_uint8[t], (Wr, Hr), interpolation=inter) for t in range(T)], axis=0)

    out = np.zeros((T, 256, 256, 3), dtype=np.uint8)
    out[:, y0 : y0 + Hr, x0 : x0 + Wr, :] = resized

    meta = {
        "orig_hw": (H, W),
        "resized_hw": (Hr, Wr),
        "scale": (s, s),  # isotropic
        "pad": (py, px),
        "pad_int": (y0, x0),
    }
    return out, meta


def _xy_orig_to_256xy(xy: np.ndarray, meta: Dict[str, Any]) -> np.ndarray:
    """
    Map original pixel (x,y) -> 256 raster (x,y) after resize+pad.
    """
    s_y, s_x = meta["scale"]
    py, px = meta["pad"]
    x = xy[..., 0] * s_x + px
    y = xy[..., 1] * s_y + py
    return np.stack([x, y], axis=-1)


def _xy256_to_norm_yx(xy256: np.ndarray, clamp: bool, eps: float) -> np.ndarray:
    """
    256 pixel (x,y) -> normalized (y,x) in [0,1]
    """
    x = xy256[..., 0] / 255.0
    y = xy256[..., 1] / 255.0
    if clamp:
        x = np.clip(x, 0.0 + eps, 1.0 - eps)
        y = np.clip(y, 0.0 + eps, 1.0 - eps)
    return np.stack([y, x], axis=-1)


def _queries_tyx_to_tyx_norm(
    queries_tyx: np.ndarray,
    meta: Dict[str, Any],
    cfg: TapirTrackerConfig,
) -> np.ndarray:
    """
    Convert queries [Q,3] (t,y,x) into backend queries [Q,3] (t,y_norm,x_norm).
    Assumptions:
      - if cfg.resize_to_256=True: queries are already in 256 raster coords (recommended).
      - else: queries are in original pixel coords and must be mapped to 256 with meta.
    """
    _require(isinstance(queries_tyx, np.ndarray), f"queries_tyx must be np.ndarray, got {type(queries_tyx)}")
    _require(queries_tyx.ndim == 2 and queries_tyx.shape[1] == 3, f"queries_tyx must be [Q,3], got {queries_tyx.shape}")
    q = queries_tyx.astype(np.float32).copy()

    # sanitize t (will be clipped later)
    t = q[:, 0]

    yx = q[:, 1:3]  # (y,x)
    xy = np.stack([yx[:, 1], yx[:, 0]], axis=-1)  # -> (x,y)

    if cfg.resize_to_256:
        xy256 = xy
    else:
        xy256 = _xy_orig_to_256xy(xy, meta)

    yx_norm = _xy256_to_norm_yx(xy256, clamp=cfg.clamp_coords, eps=cfg.clamp_eps)  # (y,x) norm

    out = np.zeros((q.shape[0], 3), dtype=np.float32)
    out[:, 0] = t
    out[:, 1] = yx_norm[:, 0]
    out[:, 2] = yx_norm[:, 1]
    return out


def _load_external_infer(spec: str):
    """
    spec format: "module.submodule:callable"
    """
    _require(":" in spec, f"external_infer must be 'module:callable', got {spec!r}", TapirDependencyError)
    mod_name, fn_name = spec.split(":", 1)
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, fn_name, None)
    _require(callable(fn), f"external infer '{spec}' not callable", TapirDependencyError)
    return fn


def _infer_backend(cfg: TapirTrackerConfig) -> str:
    if cfg.backend != "auto":
        return cfg.backend

    if cfg.external_infer or os.environ.get("PR2DRAG_TAPIR_INFER", ""):
        return "external_fn"

    # If no external infer, we still choose jax_tapir but will error with a clear message.
    return "jax_tapir"


def _tapir_infer_external(
    video_256_uint8: np.ndarray,
    queries_tyx_norm: np.ndarray,
    cfg: TapirTrackerConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Call user-provided inference fn:
      infer(video_256_uint8, queries_tyx_norm) -> (tracks_yx_norm, occluded)
    """
    spec = cfg.external_infer or os.environ.get("PR2DRAG_TAPIR_INFER", "")
    _require(bool(spec), "external infer not specified", TapirDependencyError)
    fn = _load_external_infer(spec)

    out = fn(video_256_uint8, queries_tyx_norm)
    _require(
        isinstance(out, (tuple, list)) and len(out) == 2,
        "external infer must return (tracks_yx_norm, occluded)",
        TapirDependencyError,
    )
    tracks_yx_norm, occluded = out
    tracks_yx_norm = np.asarray(tracks_yx_norm, dtype=np.float32)
    occluded = np.asarray(occluded).astype(bool)

    _require(tracks_yx_norm.ndim == 3 and tracks_yx_norm.shape[-1] == 2, f"tracks must be [T,Q,2], got {tracks_yx_norm.shape}")
    _require(occluded.shape == tracks_yx_norm.shape[:2], f"occluded must be [T,Q], got {occluded.shape}")
    return tracks_yx_norm, occluded


def _tapir_infer_jax_stub(
    video_256_uint8: np.ndarray,
    queries_tyx_norm: np.ndarray,
    cfg: TapirTrackerConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    raise TapirDependencyError(
        "TAPIR JAX backend is not wired in pr2drag-lite.\n"
        "Solid + low-coupling solution:\n"
        "  1) Add a small wrapper around official TAPIR/TAPNet code:\n"
        "       def infer(video_256_uint8, queries_tyx_norm) -> (tracks_yx_norm, occluded)\n"
        "  2) Export it via env:\n"
        "       export PR2DRAG_TAPIR_INFER='your.module:infer'\n"
        "  3) Set tapvid.tracker='tapir' and re-run tapvid_pred.\n"
    )


class TapirTracker:
    """
    TAPIR tracker wrapper aligned with pr2drag contracts.

    predict(seq) returns:
      TapVidPred:
        tracks_xy: float32 [T,Q,2] (x,y) in 256 raster pixel coords
        vis: bool [T,Q]
    """

    def __init__(self, cfg: Optional[TapirTrackerConfig] = None):
        self.cfg = cfg or TapirTrackerConfig()
        self.backend = _infer_backend(self.cfg)

    @property
    def name(self) -> str:
        return "tapir"

    def predict(self, seq: TapVidSeq) -> TapVidPred:
        _require(hasattr(seq, "video"), "TapVidSeq must have 'video' for tapir tracker.")
        _require(seq.video is not None, "TapVidSeq.video is None; provide video in pkl or implement frame loading.")

        video = _as_uint8_video(seq.video)
        T, H, W, _ = video.shape

        queries = np.asarray(seq.query_points_tyx, dtype=np.float32)
        _require(queries.ndim == 2 and queries.shape[1] == 3, f"seq.query_points_tyx must be [Q,3], got {queries.shape}")
        Q = int(queries.shape[0])
        _require(Q > 0, "No queries (Q==0). Nothing to track.")

        # Validate t0 range now (clip is dangerous)
        t0 = np.floor(queries[:, 0]).astype(np.int32)
        _require(np.all(t0 >= 0) and np.all(t0 < T), f"query t out of range [0,{T-1}] in {seq.name}")

        # Resize/pad video to 256
        if self.cfg.resize_to_256:
            v256, meta = _resize_pad_to_256(video, keep_aspect=self.cfg.keep_aspect, interp=self.cfg.interp)
        else:
            _require(H == 256 and W == 256, "resize_to_256=False requires input video already 256x256")
            v256, meta = video, {"scale": (1.0, 1.0), "pad": (0.0, 0.0), "orig_hw": (H, W), "resized_hw": (H, W), "pad_int": (0, 0)}

        # Convert queries to backend normalized coordinates
        queries_tyx_norm = _queries_tyx_to_tyx_norm(queries, meta=meta, cfg=self.cfg)

        # Dispatch backend
        if self.backend == "external_fn":
            tracks_yx_norm, occ = _tapir_infer_external(v256, queries_tyx_norm, self.cfg)
        elif self.backend == "jax_tapir":
            tracks_yx_norm, occ = _tapir_infer_jax_stub(v256, queries_tyx_norm, self.cfg)
        else:
            raise TapirDependencyError(f"Unknown TAPIR backend: {self.backend}")

        _require(tracks_yx_norm.shape[0] == v256.shape[0], f"tracks T mismatch: got {tracks_yx_norm.shape[0]} vs {v256.shape[0]}")
        _require(tracks_yx_norm.shape[1] == Q, f"tracks Q mismatch: got {tracks_yx_norm.shape[1]} vs {Q}")

        # Convert (y,x) norm -> (x,y) in 256 pixel coords
        y = tracks_yx_norm[..., 0] * 255.0
        x = tracks_yx_norm[..., 1] * 255.0
        tracks_xy = np.stack([x, y], axis=-1).astype(np.float32)

        if not np.isfinite(tracks_xy).all():
            raise TapirInputError(f"[tapir] backend returned NaN/Inf tracks for seq={seq.name}")

        if self.cfg.clamp_coords:
            tracks_xy[..., 0] = np.clip(tracks_xy[..., 0], 0.0, 255.0)
            tracks_xy[..., 1] = np.clip(tracks_xy[..., 1], 0.0, 255.0)

        vis = (~occ).astype(bool)

        return TapVidPred(
            tracks_xy=tracks_xy,
            vis=vis,
            queries_tyx=queries.astype(np.float32),
            video_hw=(int(H), int(W)),
        )