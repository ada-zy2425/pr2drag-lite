from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np

from pr2.posterior.model import PosteriorModel
from pr2.posterior.calibrate import Calibrator


@dataclass
class PosteriorBundle:
    """
    一个可保存/加载的 posterior bundle：
    - model: f_phi(E) -> p_raw
    - calibrator: Calib(p_raw) -> w_tilde
    """
    model: Any
    calibrator: Any

    def predict_wtilde(self, X: np.ndarray) -> np.ndarray:
        p = self.model.predict_proba(X)
        return self.calibrator.predict(p)

    def save(self, out_dir: str | Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "bundle.pkl", "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(out_dir: str | Path) -> "PosteriorBundle":
        out_dir = Path(out_dir)
        with open(out_dir / "bundle.pkl", "rb") as f:
            return pickle.load(f)
