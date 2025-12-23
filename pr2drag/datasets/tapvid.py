# pr2drag/datasets/tapvid.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pickle


@dataclass(frozen=True)
class TapVidSeq:
    name: str
    frame_paths: List[str]                 # len=T; 允许为空（pkl 内带 video 时可为空）
    video_hw: Tuple[int, int]              # (H, W)
    gt_tracks_xy: np.ndarray               # float32 [Q, T, 2] in (x,y)
    gt_vis: np.ndarray                     # bool   [Q, T]
    queries_txy: np.ndarray                # float32/int [Q, 3] (t0, x0, y0)


def load_tapvid_pkl(pkl_path: str) -> Dict[str, Any]:
    p = Path(pkl_path)
    if not p.exists():
        raise FileNotFoundError(f"[tapvid] pkl not found: {p}")
    with p.open("rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict) or not obj:
        raise TypeError(f"[tapvid] pkl root must be non-empty dict, got {type(obj)}")
    return obj


def _infer_hw_from_video(video: np.ndarray) -> Tuple[int, int]:
    if not isinstance(video, np.ndarray) or video.ndim != 4 or video.shape[-1] != 3:
        raise ValueError(f"[tapvid] 'video' must be [T,H,W,3], got {getattr(video, 'shape', None)}")
    return int(video.shape[1]), int(video.shape[2])


def _ensure_bool(a: np.ndarray) -> np.ndarray:
    if a.dtype == np.bool_:
        return a
    return (a.astype(np.int32) != 0)


def _infer_xy_order(points: np.ndarray, hw: Tuple[int, int]) -> str:
    """
    points[...,2] 可能是 (y,x) 或 (x,y)。用 (H,W) 做一次鲁棒推断。
    """
    H, W = hw
    p0 = float(np.nanmax(points[..., 0]))
    p1 = float(np.nanmax(points[..., 1]))

    # 典型：W > H。若维0更像 W，说明维0是 x；反之维1是 x。
    score0_x = abs(p0 - W) < abs(p0 - H)
    score1_x = abs(p1 - W) < abs(p1 - H)

    if score0_x and (not score1_x):
        return "xy"
    if score1_x and (not score0_x):
        return "yx"

    # fallback：谁更大谁更像 x（因为 W 通常更大）
    return "xy" if p0 >= p1 else "yx"


def _to_xy(points: np.ndarray, order: str) -> np.ndarray:
    if order == "xy":
        return points
    if order == "yx":
        return points[..., ::-1]
    raise ValueError(f"Unknown order={order}")


def _first_visible_t(vis_row: np.ndarray) -> int:
    idx = np.flatnonzero(vis_row)
    return int(idx[0]) if idx.size > 0 else 0


def _make_queries_txy_from_tracks(
    tracks_xy: np.ndarray,   # [Q,T,2]
    vis: np.ndarray,         # [Q,T]
    mode: str,
    stride: int,
    max_queries: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    返回 (queries_txy, tracks_out, vis_out)
      - queries_txy: [Q',3] (t,x,y)
      - tracks_out : [Q',T,2]
      - vis_out    : [Q',T]
    """
    if mode not in ("tapvid", "strided"):
        raise ValueError(f"[tapvid] query_mode must be one of ['tapvid','strided'], got {mode!r}")

    Q, T, _ = tracks_xy.shape
    if Q == 0 or T == 0:
        raise ValueError(f"[tapvid] empty tracks shape={tracks_xy.shape}")

    if mode == "tapvid":
        q_list: List[List[float]] = []
        for i in range(Q):
            t0 = _first_visible_t(vis[i])
            x0, y0 = tracks_xy[i, t0].tolist()
            q_list.append([float(t0), float(x0), float(y0)])
        queries = np.asarray(q_list, dtype=np.float32)
        return queries, tracks_xy.astype(np.float32), vis.astype(np.bool_)

    # mode == "strided"
    if stride <= 0:
        raise ValueError(f"[tapvid] stride must be >0 for strided mode, got {stride}")

    q_list = []
    tracks_list = []
    vis_list = []

    for i in range(Q):
        for t in range(0, T, stride):
            if not bool(vis[i, t]):
                continue
            x, y = tracks_xy[i, t].tolist()
            q_list.append([float(t), float(x), float(y)])
            tracks_list.append(tracks_xy[i])
            vis_list.append(vis[i])

    if not q_list:
        # 保底：退化成 tapvid，避免“无 query”导致后续崩
        return _make_queries_txy_from_tracks(tracks_xy, vis, mode="tapvid", stride=stride, max_queries=0, seed=seed)

    queries = np.asarray(q_list, dtype=np.float32)
    tracks_out = np.stack(tracks_list, axis=0).astype(np.float32)  # [Q',T,2]
    vis_out = np.stack(vis_list, axis=0).astype(np.bool_)

    # 可复现下采样
    if max_queries and queries.shape[0] > int(max_queries):
        rng = np.random.default_rng(int(seed))
        idx = rng.choice(queries.shape[0], size=int(max_queries), replace=False)
        idx = np.sort(idx)
        queries = queries[idx]
        tracks_out = tracks_out[idx]
        vis_out = vis_out[idx]

    return queries, tracks_out, vis_out


def build_tapvid_dataset(
    davis_root: str,
    pkl_path: str,
    split: str = "davis",
    query_mode: str = "tapvid",
    stride: int = 5,
    max_queries: int = 0,
    seed: int = 0,
) -> List[TapVidSeq]:
    """
    Solid contract:
      - 输入 pkl: dict[name] = {'points':[Q,T,2], 'occluded':[Q,T], 'video':[T,H,W,3] (可选)}
      - 输出 tracks: gt_tracks_xy [Q',T,2] / gt_vis [Q',T] / queries_txy [Q',3] (t,x,y)
    """
    _ = davis_root  # 当前 split=davis 且 pkl 内带 video 时可不需要；先保留接口
    data = load_tapvid_pkl(pkl_path)

    seqs: List[TapVidSeq] = []
    for name, ex in data.items():
        if not isinstance(ex, dict):
            raise TypeError(f"[tapvid:{name}] each entry must be dict, got {type(ex)}")
        if "points" not in ex or "occluded" not in ex:
            raise KeyError(f"[tapvid:{name}] missing keys, need 'points' and 'occluded'")

        points = np.asarray(ex["points"])
        occ = _ensure_bool(np.asarray(ex["occluded"]))

        if points.ndim != 3 or points.shape[-1] != 2:
            raise ValueError(f"[tapvid:{name}] points must be [Q,T,2], got {points.shape}")
        if occ.shape != points.shape[:2]:
            raise ValueError(f"[tapvid:{name}] occluded shape {occ.shape} != (Q,T) {points.shape[:2]}")

        # 解析 H,W：优先用 pkl 内 video
        frame_paths: List[str] = []
        if "video" in ex and isinstance(ex["video"], np.ndarray):
            H, W = _infer_hw_from_video(ex["video"])
        else:
            # 没 video 的情况：你后面如果要从 davis_root 取帧，再在这里补全。
            # 先给一个明确错误，避免 silent wrong。
            raise KeyError(
                f"[tapvid:{name}] missing 'video' array in pkl. "
                "Either include video in pkl, or implement frame path resolving from davis_root."
            )

        # 坐标系转成 (x,y)
        order = _infer_xy_order(points, (H, W))
        tracks_xy = _to_xy(points.astype(np.float32), order=order)  # [Q,T,2] xy
        vis = (~occ).astype(np.bool_)                                # [Q,T]

        # 生成 queries（并在 strided 情况复制样本）
        queries_txy, tracks_out, vis_out = _make_queries_txy_from_tracks(
            tracks_xy=tracks_xy,
            vis=vis,
            mode=query_mode,
            stride=int(stride),
            max_queries=int(max_queries),
            seed=int(seed),
        )

        if queries_txy.shape[0] == 0:
            raise ValueError(f"[tapvid:{name}] no queries produced (mode={query_mode})")

        seqs.append(
            TapVidSeq(
                name=str(name),
                frame_paths=frame_paths,     # 暂时为空：oracle pred/eval 不需要；tracker 接入时再补
                video_hw=(H, W),
                gt_tracks_xy=tracks_out,
                gt_vis=vis_out,
                queries_txy=queries_txy,
            )
        )

    if not seqs:
        raise ValueError(f"[tapvid] empty dataset from pkl={pkl_path}")
    return seqs