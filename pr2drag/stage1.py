from __future__ import annotations
from pathlib import Path
import json
import numpy as np
from PIL import Image
from tqdm import tqdm

from .core import load_mask_uint8, choose_target_id, mask_to_binary, centroid_from_binary, bbox_from_binary

def read_txt_lines(p: Path):
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")
    return [l.strip() for l in p.read_text().splitlines() if l.strip()]

def parse_sequences_from_imageset_lines(lines):
    seqs = set()
    for l in lines:
        parts = l.split()
        if len(parts) == 1 and ("/" not in parts[0]) and ("\\" not in parts[0]):
            seqs.add(parts[0]); continue
        img_rel = parts[0].lstrip("/")
        pp = Path(img_rel)
        if len(pp.parts) >= 2:
            seqs.add(pp.parts[-2])
    return sorted(seqs)

def stage1_precompute(
    davis_root: Path,
    res: str,
    split_txt: Path,
    out_dir: Path,
    split_name: str,
    overwrite: bool = False,
    skip_if_exists: bool = True,
    verbose_skip: bool = True,
):
    lines = read_txt_lines(split_txt)
    seqs = parse_sequences_from_imageset_lines(lines)
    print(f"[Stage1:{split_name}] split={split_txt} num_seqs(parsed)={len(seqs)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    meta = dict(
        davis_root=str(davis_root),
        res=res,
        split=str(split_txt),
        split_name=split_name,
        num_sequences=0,
        sequences=[]
    )

    for seq in tqdm(seqs, desc=f"Stage1({split_name})"):
        out_path = out_dir / f"{seq}.npz"
        if skip_if_exists and out_path.exists() and (not overwrite):
            if verbose_skip:
                print(f"[skip] Stage1 exists: {out_path.name}")
            meta["sequences"].append({"seq": seq, "npz": str(out_path)})
            continue

        img_dir = davis_root / "JPEGImages" / res / seq
        ann_dir = davis_root / "Annotations" / res / seq
        if (not img_dir.exists()) or (not ann_dir.exists()):
            print(f"[warn] missing dirs for seq={seq}, skip.")
            continue

        frames = sorted(img_dir.glob("*.jpg"))
        masks  = sorted(ann_dir.glob("*.png"))
        if len(frames) == 0 or len(masks) == 0:
            print(f"[warn] empty seq={seq}, skip.")
            continue

        frame_map = {f.stem: f for f in frames}
        mask_map  = {m.stem: m for m in masks}
        common = sorted(list(set(frame_map.keys()) & set(mask_map.keys())))
        if len(common) == 0:
            print(f"[warn] no aligned frames in seq={seq}, skip.")
            continue

        im0 = np.array(Image.open(frame_map[common[0]]).convert("RGB"))
        H, W = im0.shape[0], im0.shape[1]

        m0 = load_mask_uint8(mask_map[common[0]])
        tid = choose_target_id(m0)

        z_gt, area, bbox, touch_border = [], [], [], []
        frame_paths, mask_paths = [], []

        for stem in common:
            fp = frame_map[stem]
            mp = mask_map[stem]
            m = load_mask_uint8(mp)
            fg = mask_to_binary(m, tid)
            c = centroid_from_binary(fg)
            a = float(fg.sum())
            b = bbox_from_binary(fg)
            tb = bool(np.any(np.isnan(b)) or (b[0] <= 0) or (b[1] <= 0) or (b[2] >= W-1) or (b[3] >= H-1))

            z_gt.append(c); area.append(a); bbox.append(b); touch_border.append(tb)
            frame_paths.append(str(fp)); mask_paths.append(str(mp))

        z_gt = np.stack(z_gt, axis=0).astype(np.float32)
        area = np.array(area, dtype=np.float32)
        bbox = np.stack(bbox, axis=0).astype(np.float32)
        touch_border = np.array(touch_border, dtype=np.uint8)

        np.savez_compressed(
            out_path,
            seq=seq,
            target_id=(-1 if tid is None else int(tid)),
            H=np.int32(H), W=np.int32(W),
            frames=np.array(frame_paths, dtype=object),
            masks=np.array(mask_paths, dtype=object),
            z_gt=z_gt,
            area=area,
            bbox=bbox,
            touch_border=touch_border
        )
        meta["sequences"].append({"seq": seq, "T": int(z_gt.shape[0]), "npz": str(out_path)})

    meta["num_sequences"] = len(meta["sequences"])
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[OK] Stage1({split_name}) meta: {out_dir/'meta.json'}")
    print(f"[OK] Stage1({split_name}) npz_dir: {out_dir}  num_npz={len(list(out_dir.glob('*.npz')))}")
