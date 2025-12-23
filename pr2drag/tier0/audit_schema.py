# pr2drag/tier0/audit_schema.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional
import json
import os
import platform
import subprocess
from datetime import datetime, timezone


def _try_git_hash(repo_root: Optional[str] = None) -> str:
    try:
        cwd = repo_root or os.getcwd()
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=cwd).decode().strip()
        return out
    except Exception:
        return "unknown"


@dataclass(frozen=True)
class AuditMeta:
    timestamp_utc: str
    git_hash: str
    python: str
    platform: str
    config_path: Optional[str]
    config_sha1: Optional[str]
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]

    @staticmethod
    def make(
        config_path: Optional[str],
        config_sha1: Optional[str],
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        repo_root: Optional[str] = None,
    ) -> "AuditMeta":
        ts = datetime.now(timezone.utc).isoformat()
        return AuditMeta(
            timestamp_utc=ts,
            git_hash=_try_git_hash(repo_root),
            python=platform.python_version(),
            platform=f"{platform.system()}-{platform.release()} ({platform.machine()})",
            config_path=config_path,
            config_sha1=config_sha1,
            inputs=inputs,
            outputs=outputs,
        )


def write_audit_json(path: str | Path, meta: AuditMeta) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(asdict(meta), indent=2, ensure_ascii=False), encoding="utf-8")