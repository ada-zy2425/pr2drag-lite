# tools/render_npz_to_mp4.py
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

def _sorted_frames(img_dir: Path):
    exts = (".jpg", ".jpeg", ".png")
    fs = [p for p in img_dir.iterdir() if p.suffix.lower() in exts]
    fs.sort()
    return fs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--davis_root", type=str, required=True)
    ap.add_argument("--res", type=str, default="480p")
    ap.add_argument("--run_dir", type=str, required=True, help=".../pr2drag_data/<out_dir>")
    ap.add_argument("--seq", type=str, required=True)
    ap.add_argument("--npz_rel", type=str, default="_analysis/npz")
    ap.add_argument("--out_mp4", type=str, default=None)
    ap.add_argument("--fps", type=int, default=25)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    npz_path = run_dir / args.npz_rel / f"{args.seq}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"npz not found: {npz_path}")

    # lazy import cv2 so environment without it fails nicely
    import cv2

    data = np.load(npz_path, allow_pickle=False)
    if "z_obs" not in data or "z_fin" not in data:
        raise KeyError(f"npz must contain z_obs and z_fin, got keys={list(data.keys())}")

    z_obs = data["z_obs"].astype(np.float32)
    z_fin = data["z_fin"].astype(np.float32)
    T = int(z_fin.shape[0])

    davis_root = Path(args.davis_root)
    img_dir = davis_root / "JPEGImages" / args.res / args.seq
    if not img_dir.exists():
        raise FileNotFoundError(f"DAVIS frames dir not found: {img_dir}")

    frames = _sorted_frames(img_dir)
    if len(frames) == 0:
        raise RuntimeError(f"no frames under: {img_dir}")

    # align length
    N = min(len(frames), T)
    frames = frames[:N]
    z_obs = z_obs[:N]
    z_fin = z_fin[:N]

    first = cv2.imread(str(frames[0]))
    if first is None:
        raise RuntimeError(f"failed to read frame: {frames[0]}")
    H, W = first.shape[:2]

    out_mp4 = args.out_mp4
    if out_mp4 is None:
        out_mp4 = str(run_dir / f"{args.seq}_overlay.mp4")
    out_path = Path(out_mp4)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(out_path), fourcc, args.fps, (W, H))
    if not vw.isOpened():
        raise RuntimeError("cv2.VideoWriter failed to open (codec issue). Try different codec/container.")

    # optional masks
    low_mask = data["low_mask"].astype(np.int32)[:N] if "low_mask" in data else None
    bridge_mask = data["bridge_mask"].astype(np.int32)[:N] if "bridge_mask" in data else None
    abstain_mask = data["abstain_mask"].astype(np.int32)[:N] if "abstain_mask" in data else None

    def draw_point(img, xy, r=4, thickness=-1):
        x, y = float(xy[0]), float(xy[1])
        if np.isnan(x) or np.isnan(y):
            return
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < W and 0 <= yi < H:
            # BGR
            cv2.circle(img, (xi, yi), r, (0, 255, 0), thickness)

    for t, fp in enumerate(frames):
        img = cv2.imread(str(fp))
        if img is None:
            raise RuntimeError(f"failed to read frame: {fp}")

        # annotate text
        tag = []
        if low_mask is not None and int(low_mask[t]) == 1:
            tag.append("LOW")
        if bridge_mask is not None and int(bridge_mask[t]) == 1:
            tag.append("BRIDGE")
        if abstain_mask is not None and int(abstain_mask[t]) == 1:
            tag.append("ABSTAIN")
        text = f"{args.seq} t={t}"
        if tag:
            text += " [" + ",".join(tag) + "]"
        cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        # draw obs and fin
        # obs: blue, fin: green
        xo, yo = z_obs[t]
        xf, yf = z_fin[t]
        if not (np.isnan(xo) or np.isnan(yo)):
            cv2.circle(img, (int(round(xo)), int(round(yo))), 4, (255, 0, 0), -1)
        if not (np.isnan(xf) or np.isnan(yf)):
            cv2.circle(img, (int(round(xf)), int(round(yf))), 4, (0, 255, 0), -1)

        vw.write(img)

    vw.release()
    print(f"[OK] wrote: {out_path}")

if __name__ == "__main__":
    main()