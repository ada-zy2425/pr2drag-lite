from __future__ import annotations

import contextlib
import datetime as _dt
import io
import json
import os
import platform
import shutil
import subprocess
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


def _now_utc_iso() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _safe_mkdir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _readable_bool(x: Any) -> bool:
    return bool(x) if x is not None else False


def _run_shell(cmd: str) -> str:
    try:
        out = subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT)
        return out.strip()
    except Exception:
        return "NA"


class _Tee(io.TextIOBase):
    """Write-through stream that duplicates to multiple streams."""

    def __init__(self, *streams: io.TextIOBase):
        self._streams = [s for s in streams if s is not None]

    def write(self, s: str) -> int:
        n = 0
        for st in self._streams:
            try:
                n = st.write(s)
            except Exception:
                # don't crash logging on a broken stream
                pass
        self.flush()
        return n

    def flush(self) -> None:
        for st in self._streams:
            try:
                st.flush()
            except Exception:
                pass


@contextlib.contextmanager
def tee_stdout_stderr(log_path: Path):
    """
    Tee both stdout and stderr into a file, while still showing them in notebook.
    Robust to failures; always restores original streams.
    """
    _safe_mkdir(log_path.parent)
    f = None
    old_out, old_err = sys.stdout, sys.stderr
    try:
        f = open(log_path, "w", encoding="utf-8", buffering=1)
        sys.stdout = _Tee(old_out, f)
        sys.stderr = _Tee(old_err, f)
        yield
    finally:
        sys.stdout = old_out
        sys.stderr = old_err
        if f is not None:
            try:
                f.flush()
                f.close()
            except Exception:
                pass


@dataclass
class ArchiveOptions:
    enable: bool = False
    pack_root: str = "_paperpack"   # relative to base_out by default
    zip: bool = True
    copy_out_dir: bool = False      # if True, copy entire out_dir (may be larger)


def parse_archive_options(cfg: Dict[str, Any]) -> ArchiveOptions:
    s3 = dict(cfg.get("stage3", {}))
    ana = dict(s3.get("analysis", {}))
    arch = dict(ana.get("archive", {}))

    opt = ArchiveOptions(
        enable=_readable_bool(arch.get("enable", False)),
        pack_root=str(arch.get("pack_root", "_paperpack")),
        zip=_readable_bool(arch.get("zip", True)),
        copy_out_dir=_readable_bool(arch.get("copy_out_dir", False)),
    )
    return opt


def _resolve_pack_root(base_out: Path, pack_root: str) -> Path:
    pr = Path(pack_root)
    if pr.is_absolute():
        return pr
    return base_out / pr


def _copytree_clean(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(src, dst)


def write_manifest(
    manifest_path: Path,
    *,
    status: str,
    cmd: str,
    tag: str,
    out_dir: Path,
    cfg_path: Optional[Path],
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    m: Dict[str, Any] = {
        "timestamp_utc": _now_utc_iso(),
        "status": status,
        "cmd": cmd,
        "tag": tag,
        "out_dir": str(out_dir),
        "config_path": str(cfg_path) if cfg_path is not None else "NA",
        "git_commit": _run_shell("git rev-parse HEAD"),
        "git_dirty_files": _run_shell("git status --porcelain | wc -l"),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "cwd": os.getcwd(),
    }
    if extra:
        m.update(extra)

    _safe_mkdir(manifest_path.parent)
    manifest_path.write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")


def archive_run(
    *,
    base_out: Path,
    out_dir: Path,
    tag: str,
    cmd: str,
    cfg_path: Optional[Path],
    cfg: Dict[str, Any],
    opts: ArchiveOptions,
) -> Optional[Path]:
    """
    Create a paper-ready pack:
      base_out/<pack_root>/<tag>/{analysis/,configs/,manifest.json,(optional)out_dir_copy/}
    and optionally zip it.

    Returns zip path if created, else pack dir path, else None if disabled.
    """
    if not opts.enable:
        return None

    pack_root = _resolve_pack_root(base_out, opts.pack_root)
    pack_dir = pack_root / tag
    analysis_src = out_dir / "_analysis"

    _safe_mkdir(pack_dir)
    _safe_mkdir(pack_dir / "analysis")
    _safe_mkdir(pack_dir / "configs")

    # 1) copy analysis
    if analysis_src.exists():
        _copytree_clean(analysis_src, pack_dir / "analysis")
    else:
        # still create empty folder; manifest will indicate missing
        _safe_mkdir(pack_dir / "analysis")

    # 2) copy config
    if cfg_path is not None and Path(cfg_path).exists():
        try:
            shutil.copy2(str(cfg_path), str(pack_dir / "configs" / Path(cfg_path).name))
        except Exception:
            # do not crash packaging
            pass

    # 3) optional: copy whole out_dir (may be bigger)
    if opts.copy_out_dir and out_dir.exists():
        try:
            _copytree_clean(out_dir, pack_dir / "out_dir_copy")
        except Exception:
            pass

    # 4) manifest
    write_manifest(
        pack_dir / "manifest.json",
        status="ok",
        cmd=cmd,
        tag=tag,
        out_dir=out_dir,
        cfg_path=cfg_path,
        extra={
            "seed": cfg.get("seed", "NA"),
            "davis_root": cfg.get("davis_root", "NA"),
            "res": cfg.get("res", "NA"),
        },
    )

    # 5) zip
    if opts.zip:
        _safe_mkdir(pack_root)
        zip_base = str(pack_root / tag)
        # shutil.make_archive wants base_name without extension
        zip_path = shutil.make_archive(base_name=zip_base, format="zip", root_dir=str(pack_root), base_dir=str(tag))
        return Path(zip_path)

    return pack_dir


def archive_on_exception(
    *,
    base_out: Path,
    out_dir: Path,
    tag: str,
    cmd: str,
    cfg_path: Optional[Path],
    cfg: Dict[str, Any],
    opts: ArchiveOptions,
    exc: BaseException,
) -> None:
    """
    If stage3 crashes, still attempt to write a manifest + traceback into out_dir/_analysis and paperpack.
    """
    try:
        _safe_mkdir(out_dir / "_analysis")
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        (out_dir / "_analysis" / "traceback.txt").write_text(tb, encoding="utf-8")
    except Exception:
        pass

    try:
        pack_root = _resolve_pack_root(base_out, opts.pack_root)
        pack_dir = pack_root / tag
        _safe_mkdir(pack_dir)
        (pack_dir / "traceback.txt").write_text(
            "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            encoding="utf-8",
        )
        write_manifest(
            pack_dir / "manifest.json",
            status="error",
            cmd=cmd,
            tag=tag,
            out_dir=out_dir,
            cfg_path=cfg_path,
            extra={
                "seed": cfg.get("seed", "NA"),
                "davis_root": cfg.get("davis_root", "NA"),
                "res": cfg.get("res", "NA"),
            },
        )
    except Exception:
        pass
