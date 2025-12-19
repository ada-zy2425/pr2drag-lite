from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import yaml

@dataclass
class Config:
    raw: Dict[str, Any]

    @property
    def seed(self) -> int:
        return int(self.raw.get("seed", 0))

    @property
    def davis_root(self) -> Path:
        return Path(self.raw["paths"]["davis_root"])

    @property
    def res(self) -> str:
        return str(self.raw["paths"].get("res", "480p"))

    @property
    def base_out(self) -> Path:
        return Path(self.raw["paths"]["base_out"])

    @property
    def overwrite(self) -> bool:
        return bool(self.raw.get("io", {}).get("overwrite", False))

    @property
    def skip_if_exists(self) -> bool:
        return bool(self.raw.get("io", {}).get("skip_if_exists", True))

    @property
    def verbose_skip(self) -> bool:
        return bool(self.raw.get("io", {}).get("verbose_skip", True))

    def get(self, key: str, default=None):
        return self.raw.get(key, default)

def load_config(path: Path) -> Config:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("YAML config must be a dict at top-level.")
    return Config(raw)
