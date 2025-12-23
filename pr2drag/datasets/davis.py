# pr2drag/datasets/davis.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional
import glob


@dataclass(frozen=True)
class DavisSeq:
    name: str
    frame_paths: List[str]
    video_hw: tuple[int, int]  # (H, W)


def _guess_jpeg_dir(davis_root: str, res: str) -> Path:
    root = Path(davis_root)
    cands = [
        root / "JPEGImages" / res,
        root / "JPEGImages" / "480p",
        root / "JPEGImages" / "Full-Resolution",
        root / "JPEGImages",
    ]
    for c in cands:
        if c.exists() and c.is_dir():
            return c
    raise FileNotFoundError(
        f"[DAVIS] cannot find JPEGImages directory under {root}. Tried: {', '.join(str(x) for x in cands)}"
    )


def resolve_davis_frames(davis_root: str, seq: str, res: str = "480p") -> List[str]:
    jpeg_dir = _guess_jpeg_dir(davis_root, res)
    seq_dir = jpeg_dir / seq
    if not seq_dir.exists():
        # fallback: try recursive search
        hits = glob.glob(str(jpeg_dir / "**" / seq), recursive=True)
        if hits:
            seq_dir = Path(hits[0])
        else:
            raise FileNotFoundError(f"[DAVIS] sequence folder not found: {seq_dir}")

    frames = sorted(seq_dir.glob("*.jpg"))
    if not frames:
        frames = sorted(seq_dir.glob("*.png"))
    if not frames:
        raise FileNotFoundError(f"[DAVIS] no frames found under {seq_dir} (*.jpg/*.png)")
    return [str(p) for p in frames]


def load_davis_split_list(davis_root: str, res: str, split_txt: str) -> List[str]:
    p = Path(split_txt)
    if not p.exists():
        raise FileNotFoundError(f"[DAVIS] split file not found: {p}")
    seqs: List[str] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        # DAVIS ImageSets/480p/*.txt usually has "seq_name/00000" per line, but your log shows dedup=30/20
        # We robustly take the first token before '/'
        name = s.split()[0].split("/")[0]
        seqs.append(name)
    # dedup keep order
    seen = set()
    out = []
    for x in seqs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out