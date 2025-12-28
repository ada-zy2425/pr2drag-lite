from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from pr2.editors.base import Task
from pr2.editors.factory import make_editor
from pr2.policy.schedules import RadiusSchedule, GMaxSchedule, LambdaSafeSchedule
from pr2.policy.controller import PR2Controller
from pr2.utils.io import atomic_write_json, ensure_dir, write_jsonl
from pr2.utils.seeding import set_global_seed


def build_controller(cfg: Dict[str, Any]) -> PR2Controller:
    pol = cfg["policy"]
    thr = pol["thresholding"]
    sched = pol["schedules"]
    comp = pol["completion"]

    controller = PR2Controller(
        tau=float(thr.get("tau", 0.5)),
        radius_schedule=RadiusSchedule(**sched["radius"]),
        gmax_schedule=GMaxSchedule(**sched["gmax"]),
        lambda_schedule=LambdaSafeSchedule(**sched["lambda_safe"]),
        completion_enabled=bool(comp.get("enabled", True)),
        completion_solver=str(comp.get("solver", "minaccel")),
        completion_fallback=str(comp.get("fallback", "abstain")),
        max_seg_len=int(comp.get("max_seg_len", 12)),
        v_max=float(comp.get("v_max", 40.0)),
    )
    return controller


def run_one(task: Task, cfg: Dict[str, Any], out_dir: Path) -> None:
    seed = int(cfg.get("seed", 0))
    set_global_seed(seed)

    out_dir = ensure_dir(out_dir)
    task_dir = ensure_dir(out_dir / task.task_id)

    # audit: save task payload
    atomic_write_json(task_dir / "task.json", {
        "task_id": task.task_id,
        "video_path": task.video_path,
        "T": task.T,
        "guidance": task.guidance,
        "meta": task.meta,
    }, indent=2)

    editor = make_editor(cfg["editor"])
    controller = build_controller(cfg)

    out = editor.run_task(task=task, controller=controller, seed=seed, out_dir=task_dir)

    atomic_write_json(task_dir / "result.json", out.result)
    if out.frame_logs is not None:
        write_jsonl(task_dir / "frame_log.jsonl", out.frame_logs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="path to YAML config")
    ap.add_argument("--task_json", required=True, help="single task JSON (inline) or path to .json")
    ap.add_argument("--out", required=True, help="output directory")
    args = ap.parse_args()

    from pr2.eval.config import load_yaml
    import json

    cfg = load_yaml(args.config)

    p = Path(args.task_json)
    if p.exists():
        task_payload = json.loads(p.read_text(encoding="utf-8"))
    else:
        task_payload = json.loads(args.task_json)

    task = Task(
        task_id=task_payload["task_id"],
        video_path=task_payload.get("video_path", ""),
        T=int(task_payload.get("T", 0)),
        guidance=task_payload.get("guidance", {}),
        meta=task_payload.get("meta", {}),
    )

    run_one(task, cfg, Path(args.out))


if __name__ == "__main__":
    main()
