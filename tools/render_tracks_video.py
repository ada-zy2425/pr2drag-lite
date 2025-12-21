# tools/render_tracks_video.py
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np

def _safe_np_load(npz_path: Path) -> Dict[str, np.ndarray]:
    try:
        with np.load(npz_path, allow_pickle=True) as z:
            return {k: z[k] for k in z.files}
    except Exception as e:
        raise RuntimeError(f"Failed to load npz: {npz_path} ({e})")

def _find_seq_npz(out_dir: Path, seq: str) -> Optional[Path]:
    # Heuristics: search common patterns
    cands: List[Path] = []
    for p in out_dir.rglob("*.npz"):
        name = p.stem.lower()
        if seq.lower() in name:
            cands.append(p)
    if not cands:
        return None
    # prefer ones that contain stage3-ish keys
    def score(p: Path) -> int:
        try:
            d = _safe_np_load(p)
            keys = set(d.keys())
            s = 0
            for k in ["z_fin", "z_obs", "wtil", "w", "p", "err_obs"]:
                if k in keys:
                    s += 10
            return s
        except Exception:
            return -1
    cands.sort(key=score, reverse=True)
    return cands[0]

def _read_frames(davis_root: Path, res: str, seq: str) -> Tuple[List[np.ndarray], int, int]:
    # DAVIS: JPEGImages/<res>/<seq>/*.jpg
    img_dir = davis_root / "JPEGImages" / res / seq
    if not img_dir.exists():
        raise FileNotFoundError(f"Missing DAVIS frames dir: {img_dir}")
    paths = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in [".jpg", ".png"]])
    if not paths:
        raise FileNotFoundError(f"No frames found in: {img_dir}")

    # try cv2, fallback to imageio
    frames = []
    try:
        import cv2  # type: ignore
        for p in paths:
            im = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if im is None:
                raise RuntimeError(f"cv2.imread failed: {p}")
            frames.append(im)
        h, w = frames[0].shape[:2]
        return frames, w, h
    except Exception:
        import imageio.v2 as imageio  # type: ignore
        for p in paths:
            im = imageio.imread(p)
            if im.ndim == 2:
                im = np.stack([im, im, im], axis=-1)
            if im.shape[-1] == 4:
                im = im[..., :3]
            frames.append(im[..., ::-1].copy())  # to BGR-like for consistent drawing later
        h, w = frames[0].shape[:2]
        return frames, w, h

def _maybe_scale_xy(z: np.ndarray, w: int, h: int) -> np.ndarray:
    z = np.asarray(z).astype(np.float32)
    if z.ndim != 2 or z.shape[1] != 2:
        raise ValueError(f"Expected (T,2) coords, got {z.shape}")
    m = float(np.nanmax(np.abs(z)))
    # heuristic: normalized coords
    if m <= 2.0:
        zz = z.copy()
        zz[:, 0] *= float(w)
        zz[:, 1] *= float(h)
        return zz
    return z

def _draw(frames: List[np.ndarray], z_obs: np.ndarray, z_fin: np.ndarray,
          low_mask: Optional[np.ndarray], radius: int = 4) -> List[np.ndarray]:
    try:
        import cv2  # type: ignore
    except Exception as e:
        raise RuntimeError("Please `pip install opencv-python` (or use an environment with cv2).") from e

    out = []
    T = min(len(frames), z_obs.shape[0], z_fin.shape[0])
    for t in range(T):
        im = frames[t].copy()
        x1, y1 = z_obs[t]
        x2, y2 = z_fin[t]
        # low-confidence overlay
        if low_mask is not None and t < low_mask.shape[0] and bool(low_mask[t]):
            overlay = im.copy()
            overlay[:] = (overlay * 0.85).astype(np.uint8)
            im = cv2.addWeighted(overlay, 0.7, im, 0.3, 0)

        # obs (red) and fin (green)
        cv2.circle(im, (int(round(x1)), int(round(y1))), radius, (0, 0, 255), -1)
        cv2.circle(im, (int(round(x2)), int(round(y2))), radius, (0, 255, 0), -1)
        # small line between them
        cv2.line(im, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), (255, 255, 255), 1)
        out.append(im)
    return out

def _write_mp4(frames: List[np.ndarray], out_path: Path, fps: int = 20) -> None:
    import cv2  # type: ignore
    out_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(out_path), fourcc, float(fps), (w, h))
    if not vw.isOpened():
        raise RuntimeError(f"VideoWriter failed for: {out_path}")
    for im in frames:
        vw.write(im)
    vw.release()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--davis_root", type=str, required=True)
    ap.add_argument("--res", type=str, default="480p")
    ap.add_argument("--seq", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True, help="stage3 output dir, e.g. /.../davis2016_E0b4_safe_bridge_speed")
    ap.add_argument("--tau", type=float, default=None, help="optional tau for low-mask (wtil < tau). If None, use median(wtil) as weak fallback.")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--radius", type=int, default=4)
    args = ap.parse_args()

    davis_root = Path(args.davis_root)
    out_dir = Path(args.out_dir)
    seq = args.seq

    frames, W, H = _read_frames(davis_root, args.res, seq)

    npz = _find_seq_npz(out_dir, seq)
    if npz is None:
        raise FileNotFoundError(f"Could not find a per-seq npz under: {out_dir} (seq={seq}). "
                                f"If your pipeline doesn't save npz, tell me what file it saves; we'll adapt.")
    d = _safe_np_load(npz)

    # expected keys
    if "z_obs" not in d or "z_fin" not in d:
        raise KeyError(f"npz missing z_obs/z_fin keys. keys={sorted(d.keys())} path={npz}")

    z_obs = _maybe_scale_xy(d["z_obs"], W, H)
    z_fin = _maybe_scale_xy(d["z_fin"], W, H)

    low_mask = None
    if "wtil" in d:
        wtil = np.asarray(d["wtil"]).reshape(-1).astype(np.float32)
        tau = float(args.tau) if args.tau is not None else float(np.nanmedian(wtil))
        low_mask = (wtil < tau)
    elif "w" in d:
        wtil = np.asarray(d["w"]).reshape(-1).astype(np.float32)
        tau = float(args.tau) if args.tau is not None else float(np.nanmedian(wtil))
        low_mask = (wtil < tau)

    drawn = _draw(frames, z_obs, z_fin, low_mask, radius=args.radius)

    out_path = out_dir / "_analysis" / f"{seq}_tracks.mp4"
    _write_mp4(drawn, out_path, fps=args.fps)
    print(f"[OK] wrote: {out_path}")

if __name__ == "__main__":
    main()