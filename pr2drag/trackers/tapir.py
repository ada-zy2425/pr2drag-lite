# pr2drag/trackers/tapir.py
from __future__ import annotations

import os
import importlib
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from pr2drag.trackers.base import TapVidPred
from pr2drag.datasets.tapvid import TapVidSeq


class TapirDependencyError(RuntimeError):
    """Raised when TAPIR backend dependencies are missing or not usable."""


class TapirInputError(ValueError):
    """Raised when inputs (video / queries) violate the tracker contract."""


@dataclass(frozen=True)
class TapirTrackerConfig:
    """
    This wrapper is backend-agnostic. It only handles:
      - video -> 256x256 preprocessing (optionally keep aspect with padding)
      - queries_tyx -> normalized queries for backend
      - backend output -> map back to ORIGINAL pixel coords

    Backend contract (external):
      infer(video_256_uint8, queries_tyx_norm) -> (tracks_yx_norm, occluded)
        - tracks_yx_norm: float32 [T,Q,2] in (y,x) normalized [0,1]
        - occluded: bool [T,Q]
    """

    # preprocessing
    resize_to_256: bool = True
    keep_aspect: bool = True
    interp: str = "bilinear"  # points => bilinear

    # external backend hook: "module.submodule:infer_fn"
    external_infer: str = ""  # if empty, uses env PR2DRAG_TAPIR_INFER

    # guardrails
    clamp_norm: bool = True
    clamp_eps: float = 1e-4


def _require(cond: bool, msg: str, exc_type=TapirInputError) -> None:
    if not cond:
        raise exc_type(msg)


def _as_uint8_video(video: np.ndarray) -> np.ndarray:
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
        return np.clip(video, 0, 255).astype(np.uint8)

    raise TapirInputError(f"Unsupported video dtype: {video.dtype}")


def _load_external_infer(spec: str):
    _require(":" in spec, f"external infer must be 'module:callable', got {spec!r}", TapirDependencyError)
    mod_name, fn_name = spec.split(":", 1)
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, fn_name, None)
    _require(callable(fn), f"external infer '{spec}' not callable", TapirDependencyError)
    return fn


def _resize_pad_to_256(frames_uint8: np.ndarray, keep_aspect: bool, interp: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    _require(interp in ("bilinear", "nearest"), f"interp must be bilinear/nearest, got {interp!r}")

    try:
        import cv2  # type: ignore
    except Exception as e:
        raise TapirDependencyError("OpenCV (cv2) required: pip install opencv-python") from e

    T, H, W, _ = frames_uint8.shape
    inter = cv2.INTER_LINEAR if interp == "bilinear" else cv2.INTER_NEAREST

    if not keep_aspect:
        fx = 256.0 / float(W)
        fy = 256.0 / float(H)
        out = np.stack([cv2.resize(frames_uint8[t], (256, 256), interpolation=inter) for t in range(T)], axis=0)
        meta = {
            "orig_hw": (H, W),
            "scale": (fy, fx),     # (sy, sx)
            "pad": (0.0, 0.0),     # (py, px)
        }
        return out.astype(np.uint8), meta

    s = min(256.0 / float(H), 256.0 / float(W))
    Hr = int(round(H * s))
    Wr = int(round(W * s))
    Hr = max(1, min(256, Hr))
    Wr = max(1, min(256, Wr))

    py = (256.0 - float(Hr)) * 0.5
    px = (256.0 - float(Wr)) * 0.5
    y0 = int(np.floor(py))
    x0 = int(np.floor(px))

    resized = np.stack([cv2.resize(frames_uint8[t], (Wr, Hr), interpolation=inter) for t in range(T)], axis=0)
    out = np.zeros((T, 256, 256, 3), dtype=np.uint8)
    out[:, y0 : y0 + Hr, x0 : x0 + Wr, :] = resized

    meta = {
        "orig_hw": (H, W),
        "scale": (s, s),          # isotropic
        "pad": (py, px),          # float pad
        "pad_int": (y0, x0),
        "resized_hw": (Hr, Wr),
    }
    return out, meta


def _xy_orig_to_256(xy: np.ndarray, meta: Dict[str, Any]) -> np.ndarray:
    sy, sx = meta["scale"]
    py, px = meta["pad"]
    x = xy[..., 0] * sx + px
    y = xy[..., 1] * sy + py
    return np.stack([x, y], axis=-1)


def _xy256_to_orig(xy256: np.ndarray, meta: Dict[str, Any]) -> np.ndarray:
    sy, sx = meta["scale"]
    py, px = meta["pad"]
    x = (xy256[..., 0] - px) / max(1e-12, sx)
    y = (xy256[..., 1] - py) / max(1e-12, sy)
    return np.stack([x, y], axis=-1)


def _queries_tyx_to_norm(queries_tyx: np.ndarray, meta: Dict[str, Any], cfg: TapirTrackerConfig) -> np.ndarray:
    """
    queries_tyx: [Q,3] (t, y, x) in ORIGINAL pixel coords.
    Return: [Q,3] (t, y_norm, x_norm) in 256 raster normalized space.
    """
    _require(isinstance(queries_tyx, np.ndarray), "queries must be np.ndarray")
    _require(queries_tyx.ndim == 2 and queries_tyx.shape[1] == 3, f"queries must be [Q,3], got {queries_tyx.shape}")

    q = queries_tyx.astype(np.float32).copy()
    # (y,x) -> (x,y) for mapping
    xy = np.stack([q[:, 2], q[:, 1]], axis=-1)  # x,y
    xy256 = _xy_orig_to_256(xy, meta)
    x_norm = xy256[:, 0] / 255.0
    y_norm = xy256[:, 1] / 255.0

    if cfg.clamp_norm:
        eps = float(cfg.clamp_eps)
        x_norm = np.clip(x_norm, 0.0 + eps, 1.0 - eps)
        y_norm = np.clip(y_norm, 0.0 + eps, 1.0 - eps)

    out = np.zeros_like(q, dtype=np.float32)
    out[:, 0] = q[:, 0]
    out[:, 1] = y_norm
    out[:, 2] = x_norm
    return out


class TapirTracker:
    """
    BaseTapVidTracker-compatible wrapper.
    Produces predictions in ORIGINAL pixel coords: TapVidPred(tracks_xy [T,Q,2], vis [T,Q]).
    """

    def __init__(self, cfg: Optional[TapirTrackerConfig] = None):
        self.cfg = cfg or TapirTrackerConfig()

    @property
    def name(self) -> str:
        return "tapir"

    def predict(self, seq: TapVidSeq) -> TapVidPred:
        _require(hasattr(seq, "video") and isinstance(seq.video, np.ndarray),  # type: ignore[attr-defined]
                 "TapirTracker requires seq.video to be present as np.ndarray [T,H,W,3].",
                 TapirInputError)

        video = _as_uint8_video(seq.video)  # type: ignore[attr-defined]
        T, H, W, _ = video.shape
        _require((H, W) == tuple(seq.video_hw), f"seq.video_hw={seq.video_hw} mismatch actual {(H,W)}")

        # preprocess to 256
        if self.cfg.resize_to_256:
            v256, meta = _resize_pad_to_256(video, keep_aspect=self.cfg.keep_aspect, interp=self.cfg.interp)
        else:
            _require(H == 256 and W == 256, "resize_to_256=False requires input already 256x256")
            v256, meta = video, {"orig_hw": (H, W), "scale": (1.0, 1.0), "pad": (0.0, 0.0)}

        # build backend queries (normalized)
        queries_tyx = np.asarray(seq.query_points_tyx, dtype=np.float32)
        queries_tyx_norm = _queries_tyx_to_norm(queries_tyx, meta, self.cfg)

        # external infer
        spec = self.cfg.external_infer or os.environ.get("PR2DRAG_TAPIR_INFER", "")
        if not spec:
            raise TapirDependencyError(
                "TAPIR backend not provided.\n"
                "Set env: PR2DRAG_TAPIR_INFER='your.module:infer'\n"
                "infer(video_256_uint8, queries_tyx_norm)->(tracks_yx_norm, occluded)"
            )
        infer_fn = _load_external_infer(spec)

        out = infer_fn(v256, queries_tyx_norm)
        _require(isinstance(out, (tuple, list)) and len(out) == 2,
                 "external infer must return (tracks_yx_norm, occluded)",
                 TapirDependencyError)
        tracks_yx_norm, occluded = out

        tracks_yx_norm = np.asarray(tracks_yx_norm, dtype=np.float32)
        occluded = np.asarray(occluded).astype(bool)

        _require(tracks_yx_norm.ndim == 3 and tracks_yx_norm.shape[-1] == 2,
                 f"tracks must be [T,Q,2], got {tracks_yx_norm.shape}",
                 TapirDependencyError)
        _require(occluded.shape == tracks_yx_norm.shape[:2],
                 f"occluded must be [T,Q], got {occluded.shape} vs {tracks_yx_norm.shape[:2]}",
                 TapirDependencyError)
        _require(tracks_yx_norm.shape[0] == T, "tracks T mismatch with video", TapirDependencyError)

        # yx_norm -> xy256 pixels
        y256 = tracks_yx_norm[..., 0] * 255.0
        x256 = tracks_yx_norm[..., 1] * 255.0
        xy256 = np.stack([x256, y256], axis=-1).astype(np.float32)

        # map back to ORIGINAL coords
        xy_orig = _xy256_to_orig(xy256, meta).astype(np.float32)

        vis = (~occluded).astype(bool)
        return TapVidPred(tracks_xy=xy_orig, vis=vis, queries_txy=queries_tyx)