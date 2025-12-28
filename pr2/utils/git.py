from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional


def get_git_commit(repo_dir: str | Path) -> Optional[str]:
    """
    返回当前 repo 的 git commit hash（若不可用则返回 None）。
    """
    repo_dir = str(repo_dir)
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_dir, stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return None
