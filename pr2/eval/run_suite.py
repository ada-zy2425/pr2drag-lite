from __future__ import annotations

import argparse
import platform
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from pr2.editors.base import Task
from pr2.eval.config import load_yaml
from pr2.eval.run_one import run_one
from pr2.utils.io import iter_jsonl, ensure_dir, atomic_write_json, atomic_write_text
from pr2.utils.hashing import hash_dict
from pr2.utils.git import get_git_commit
from pr2.utils.trace import format_exception


def task_exists(task_dir: Path) -> bool:
    return (task_dir / "result.json").exists()


def write_run_meta(out_dir: Path, cfg: dict, config_path: str, tasks_path: str) -> None:
    meta = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "config_path": config_path,
        "tasks_path": tasks_path,
        "config_hash": hash_dict(cfg),
        "git_commit": get_git_commit(Path(__file__).resolve().parents[2]),
    }
    atomic_write_json(out_dir / "run_meta.json", meta, indent=2)
    # 保存 config snapshot（yaml 原样保存更利于审计；这里用 json 也行）
    atomic_write_json(out_dir / "config_snapshot.json", cfg, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="path to YAML config")
    ap.add_argument("--tasks", required=True, help="jsonl tasks file")
    ap.add_argument("--out", required=True, help="output run directory")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    out_dir = ensure_dir(args.out)
    resume = bool(cfg.get("runtime", {}).get("resume", True))

    write_run_meta(out_dir, cfg, args.config, args.tasks)

    tasks = []
    for row in iter_jsonl(args.tasks):
        tasks.append(Task(
            task_id=row["task_id"],
            video_path=row.get("video_path", ""),
            T=int(row.get("T", 0)),
            guidance=row.get("guidance", {}),
            meta=row.get("meta", {}),
        ))

    for task in tqdm(tasks, desc="Running tasks"):
        task_dir = out_dir / task.task_id
        if resume and task_exists(task_dir):
            continue
        try:
            run_one(task, cfg, out_dir)
        except Exception as e:
            err_dir = ensure_dir(out_dir / task.task_id)
            atomic_write_text(err_dir / "error.txt", str(e))
            atomic_write_text(err_dir / "traceback.txt", format_exception(e))
            continue


if __name__ == "__main__":
    main()
