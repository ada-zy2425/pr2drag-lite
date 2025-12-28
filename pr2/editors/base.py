from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Protocol


@dataclass
class Task:
    task_id: str
    video_path: str
    T: int
    guidance: Dict[str, Any]
    meta: Dict[str, Any]


@dataclass
class EditorOutput:
    """
    你接入真实 editor 后建议至少包含：
    - result: 任务级指标（成功、灾难、flicker、coverage）
    - frame_logs: per-frame logs（也可以只写 jsonl 路径；这里先用内存列表以便 demo）
    - artifacts_dir: 可选（视频、图片、曲线）
    """
    result: Dict[str, Any]
    frame_logs: Optional[list[Dict[str, Any]]] = None
    artifacts_dir: Optional[str] = None


class EditorBase(Protocol):
    name: str

    def run_task(
        self,
        task: Task,
        controller: Any,
        seed: int,
        out_dir: Path,
    ) -> EditorOutput:
        ...
