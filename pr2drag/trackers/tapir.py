# pr2drag/trackers/tapir.py
from __future__ import annotations

import os
import importlib
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np


class TapirDependencyError(RuntimeError):
    """Raised when TAPIR backend dependencies are missing or not usable."""


class TapirInputError(ValueError):
    """Raised when inputs (video / queries) violate the tracker contract."""


@dataclass(frozen=True)
class TapirTrackerConfig:
    """
    Contract for TAPIR tracker wrapper.

    Output coordinate convention (this wrapper):
      - pred_tracks_xy: float32 [T, N, 2] in (x, y) coordinates
      - pred_occluded : bool    [T, N]
      - By default, coords are in 256x256 raster pixel space (0..255-ish),
        after resize+pad preprocessing (keep_aspect optional).
    """
    ckpt_path: str = ""                 # optional; backend may have its own default
    resize_to_256: bool = True
    keep_aspect: bool = True
    interp: str = "bilinear"            # "bilinear" or "nearest" (points -> bilinear)
    # Backend selection:
    backend: str = "auto"               # "auto" | "jax_tapir" | "external_fn"
    # If you want to use your own inference callable, you can register it via env:
    #   PR2DRAG_TAPIR_INFER="your.module:infer_fn"
    # infer_fn signature requirement: infer_fn(video_256_uint8, queries_tyx_norm) -> (tracks_yx_norm, occluded)
    # where:
    #   video_256_uint8: [T,256,256,3] uint8
    #   queries_tyx_norm: [N,3] float32 in [t, y, x] with y/x normalized to [0,1] in 256 raster
    # returns:
    #   tracks_yx_norm: [T,N,2] float32 in (y,x) normalized [0,1]
    #   occluded: [T,N] bool
    external_infer: str = ""            # optional override; same as env var format

    # Optional numeric guardrails
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


def _infer_coord_space_xy(points_xy: np.ndarray) -> str:
    """
    Heuristic: decide if xy is normalized [0,1]/[0,2] or pixel coords.
    """
    if points_xy.size == 0:
        return "pixel"
    m = float(np.nanmax(points_xy))
    if m <= 2.0:
        return "normalized"
    return "pixel"


def _resize_pad_to_256(
    frames_uint8: np.ndarray,
    keep_aspect: bool,
    interp: str,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Resize frames to 256x256, optionally keeping aspect with padding.

    Returns:
      frames_256: uint8 [T,256,256,3]
      meta: dict with mapping info to transform coords.

    meta fields:
      - orig_hw: (H,W)
      - scale: (sy,sx) scaling from orig -> resized (before pad)
      - pad: (py,px) padding added on top/left (pixels) to reach 256
      - resized_hw: (Hr,Wr)
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
    if not keep_aspect:
        Hr = Wr = 256
        fx = 256.0 / float(W)
        fy = 256.0 / float(H)
        inter = cv2.INTER_LINEAR if interp == "bilinear" else cv2.INTER_NEAREST
        out = np.stack([cv2.resize(frames_uint8[t], (Wr, Hr), interpolation=inter) for t in range(T)], axis=0)
        meta = {
            "orig_hw": (H, W),
            "resized_hw": (Hr, Wr),
            "scale": (fy, fx),
            "pad": (0.0, 0.0),
        }
        return out.astype(np.uint8), meta

    # keep_aspect: fit within 256, pad the rest
    s = min(256.0 / float(H), 256.0 / float(W))
    Hr = int(round(H * s))
    Wr = int(round(W * s))
    Hr = max(1, min(256, Hr))
    Wr = max(1, min(256, Wr))

    py = (256 - Hr) / 2.0
    px = (256 - Wr) / 2.0

    inter = cv2.INTER_LINEAR if interp == "bilinear" else cv2.INTER_NEAREST
    resized = np.stack([cv2.resize(frames_uint8[t], (Wr, Hr), interpolation=inter) for t in range(T)], axis=0)

    out = np.zeros((T, 256, 256, 3), dtype=np.uint8)
    y0 = int(np.floor(py))
    x0 = int(np.floor(px))
    out[:, y0:y0 + Hr, x0:x0 + Wr, :] = resized

    meta = {
        "orig_hw": (H, W),
        "resized_hw": (Hr, Wr),
        "scale": (s, s),         # isotropic
        "pad": (py, px),
        "pad_int": (y0, x0),
    }
    return out, meta


def _xy_orig_to_256xy(
    xy: np.ndarray, orig_hw: Tuple[int, int], meta: Dict[str, Any]
) -> np.ndarray:
    """
    Map (x,y) in original pixel coords -> 256 raster pixel coords after resize+pad.
    """
    H, W = orig_hw
    s_y, s_x = meta["scale"]  # usually equal if keep_aspect
    py, px = meta["pad"]
    # xy: [...,2] (x,y)
    x = xy[..., 0] * s_x + px
    y = xy[..., 1] * s_y + py
    return np.stack([x, y], axis=-1)


def _xy256_to_norm_yx(
    xy256: np.ndarray,
    clamp: bool,
    eps: float,
) -> np.ndarray:
    """
    Convert 256-raster (x,y) pixel coords -> normalized (y,x) in [0,1] for TAPIR-like backends.
    """
    x = xy256[..., 0] / 255.0
    y = xy256[..., 1] / 255.0
    if clamp:
        x = np.clip(x, 0.0 + eps, 1.0 - eps)
        y = np.clip(y, 0.0 + eps, 1.0 - eps)
    return np.stack([y, x], axis=-1)  # (y,x)


def _queries_tapvid_to_queries_tyx_norm(
    points_xy: np.ndarray,  # [N,T,2] or [K,T,2] in (x,y) original or normalized
    occluded: np.ndarray,   # [N,T] bool
    video_hw: Tuple[int, int],
    meta: Dict[str, Any],
    cfg: TapirTrackerConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build query_points for TAPIR backend: [N,3] float32 [t, y_norm, x_norm].

    TAP-Vid "points" stores per-track trajectory points; query frame is typically t=0 for each track
    in DAVIS split, but do NOT assume — we select the first visible frame per track.

    Returns:
      queries_tyx_norm: [N,3] float32
      query_track_ids:  [N] int64 (0..N-1), identity here
    """
    _require(points_xy.ndim == 3 and points_xy.shape[-1] == 2, f"points must be [N,T,2], got {points_xy.shape}")
    _require(occluded.shape == points_xy.shape[:2], "occluded must be [N,T] aligned with points")
    N, T, _ = points_xy.shape

    # Determine whether points are normalized or pixel coords
    space = _infer_coord_space_xy(points_xy)
    H, W = video_hw
    pts = points_xy.astype(np.float32)

    if space == "normalized":
        # normalized in original raster => convert to original pixel coords first
        # NOTE: ambiguity (0..1) could mean relative to W/H; assume that.
        pts_px = pts.copy()
        pts_px[..., 0] = pts_px[..., 0] * float(W - 1)
        pts_px[..., 1] = pts_px[..., 1] * float(H - 1)
    else:
        pts_px = pts

    # pick query frame per track = first non-occluded frame
    vis = ~occluded.astype(bool)
    queries = np.zeros((N, 3), dtype=np.float32)
    for i in range(N):
        idx = np.where(vis[i])[0]
        if idx.size == 0:
            # fully occluded track: fall back to t=0
            t0 = 0
        else:
            t0 = int(idx[0])
        xy0_px = pts_px[i, t0]  # (x,y) in orig pixels
        xy0_256 = _xy_orig_to_256xy(xy0_px, (H, W), meta)  # (x,y) in 256 raster pixels
        yx_norm = _xy256_to_norm_yx(xy0_256, clamp=cfg.clamp_coords, eps=cfg.clamp_eps)  # (y,x) norm
        queries[i, 0] = float(t0)
        queries[i, 1] = float(yx_norm[0])
        queries[i, 2] = float(yx_norm[1])

    track_ids = np.arange(N, dtype=np.int64)
    return queries, track_ids


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

    # Prefer explicit external infer if provided
    if cfg.external_infer or os.environ.get("PR2DRAG_TAPIR_INFER", ""):
        return "external_fn"

    # If jax seems installed, try jax backend
    try:
        import jax  # noqa: F401
        import haiku  # noqa: F401
        return "jax_tapir"
    except Exception:
        return "jax_tapir"  # still return jax_tapir; we’ll error with clear msg


def _tapir_infer_external(
    video_256_uint8: np.ndarray,
    queries_tyx_norm: np.ndarray,
    cfg: TapirTrackerConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    spec = cfg.external_infer or os.environ.get("PR2DRAG_TAPIR_INFER", "")
    _require(bool(spec), "external infer not specified", TapirDependencyError)
    fn = _load_external_infer(spec)

    out = fn(video_256_uint8, queries_tyx_norm)
    _require(isinstance(out, (tuple, list)) and len(out) == 2, "external infer must return (tracks, occluded)", TapirDependencyError)
    tracks_yx_norm, occluded = out
    tracks_yx_norm = np.asarray(tracks_yx_norm, dtype=np.float32)
    occluded = np.asarray(occluded).astype(bool)

    _require(tracks_yx_norm.ndim == 3 and tracks_yx_norm.shape[-1] == 2, f"tracks must be [T,N,2], got {tracks_yx_norm.shape}")
    _require(occluded.shape == tracks_yx_norm.shape[:2], f"occluded must be [T,N], got {occluded.shape}")
    return tracks_yx_norm, occluded


def _tapir_infer_jax_stub(
    video_256_uint8: np.ndarray,
    queries_tyx_norm: np.ndarray,
    cfg: TapirTrackerConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    JAX TAPIR backend placeholder.

    Why placeholder?
      - Official TAPIR uses JAX/Haiku and model code from a separate repo.
      - Your pr2drag-lite currently does NOT vendor that repo.

    What you should do (solid route):
      A) Vendor official TAPIR/TAPNet code into your project (e.g., third_party/tapnet) and expose
         a function with signature:
             infer(video_256_uint8, queries_tyx_norm) -> (tracks_yx_norm, occluded)
      B) Set env:
             PR2DRAG_TAPIR_INFER="third_party.tapnet_wrapper:infer"
         so this wrapper uses the external_fn backend.

    This keeps pr2drag clean and avoids pinning jax/haiku internals into your repo.
    """
    raise TapirDependencyError(
        "TAPIR JAX backend is not wired yet.\n"
        "Recommended (solid + low-coupling) solution:\n"
        "  1) Add a small wrapper function around official TAPIR/TAPNet code:\n"
        "       def infer(video_256_uint8, queries_tyx_norm) -> (tracks_yx_norm, occluded)\n"
        "  2) Export it via env:\n"
        "       export PR2DRAG_TAPIR_INFER='your.module:infer'\n"
        "  3) Set tapvid.tracker='tapir' and re-run tapvid_pred.\n"
        "This file already supports that via backend='external_fn'."
    )


class TapirTracker:
    """
    High-level tracker wrapper.

    Main API:
      predict(video_uint8, points_xy, occluded, video_hw) -> (pred_tracks_xy_256, pred_occluded)
    """
    def __init__(self, cfg: Optional[TapirTrackerConfig] = None):
        self.cfg = cfg or TapirTrackerConfig()
        self.backend = _infer_backend(self.cfg)

    @property
    def name(self) -> str:
        return "tapir"

    def predict_from_tapvid_item(
        self,
        video: np.ndarray,        # [T,H,W,3] uint8 or float
        points_xy: np.ndarray,    # [N,T,2] (x,y) original or normalized
        occluded: np.ndarray,     # [N,T] bool
        video_hw: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
          pred_tracks_xy_256: float32 [T,N,2] in (x,y) 256 raster pixel coords
          pred_occluded     : bool    [T,N]
        """
        v = _as_uint8_video(video)
        T, H, W, _ = v.shape
        _require((H, W) == tuple(video_hw), f"video_hw {video_hw} mismatch actual {(H,W)}")

        # resize/pad video to 256
        if self.cfg.resize_to_256:
            v256, meta = _resize_pad_to_256(v, keep_aspect=self.cfg.keep_aspect, interp=self.cfg.interp)
        else:
            # still enforce 256 contract for TAPIR: require already 256
            _require(H == 256 and W == 256, "resize_to_256=False requires input video already 256x256")
            v256, meta = v, {"orig_hw": (H, W), "scale": (1.0, 1.0), "pad": (0.0, 0.0)}

        # build query points in [t, y_norm, x_norm]
        queries_tyx_norm, _ = _queries_tapvid_to_queries_tyx_norm(
            points_xy=points_xy,
            occluded=occluded,
            video_hw=(H, W),
            meta=meta,
            cfg=self.cfg,
        )

        # run backend => tracks in (y,x) normalized
        if self.backend == "external_fn":
            tracks_yx_norm, pred_occ = _tapir_infer_external(v256, queries_tyx_norm, self.cfg)
        elif self.backend == "jax_tapir":
            tracks_yx_norm, pred_occ = _tapir_infer_jax_stub(v256, queries_tyx_norm, self.cfg)
        else:
            raise TapirDependencyError(f"Unknown TAPIR backend: {self.backend}")

        # convert tracks from (y,x) norm -> (x,y) 256 pixel coords
        _require(tracks_yx_norm.shape[0] == v256.shape[0], "tracks T mismatch")
        y = tracks_yx_norm[..., 0] * 255.0
        x = tracks_yx_norm[..., 1] * 255.0
        pred_tracks_xy_256 = np.stack([x, y], axis=-1).astype(np.float32)

        if self.cfg.clamp_coords:
            pred_tracks_xy_256[..., 0] = np.clip(pred_tracks_xy_256[..., 0], 0.0, 255.0)
            pred_tracks_xy_256[..., 1] = np.clip(pred_tracks_xy_256[..., 1], 0.0, 255.0)

        return pred_tracks_xy_256, pred_occ.astype(bool)