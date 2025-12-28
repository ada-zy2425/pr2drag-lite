from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np

from pr2.posterior.model import SklearnLogRegPosterior, TorchMLPPosterior, PosteriorModel
from pr2.posterior.calibrate import (
    IdentityCalibrator,
    TemperatureScalingCalibrator,
    IsotonicCalibrator,
    Calibrator,
)
from pr2.posterior.metrics import ece as ece_metric, brier as brier_metric, risk_coverage_curve
from pr2.utils.io import atomic_write_json


@dataclass
class TrainConfig:
    model: str = "logreg"  # ["logreg","mlp"]
    calibrator: str = "temperature"  # ["identity","temperature","isotonic"]
    out_dir: str = "runs/posterior"
    seed: int = 0


def train_posterior(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_calib: np.ndarray,
    y_calib: np.ndarray,
    cfg: TrainConfig,
) -> Dict[str, Any]:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if cfg.model == "logreg":
        model: PosteriorModel = SklearnLogRegPosterior()
    elif cfg.model == "mlp":
        model = TorchMLPPosterior(seed=cfg.seed)
    else:
        raise ValueError(f"Unknown model: {cfg.model}")

    model.fit(X_train, y_train)
    p_calib_raw = model.predict_proba(X_calib)

    if cfg.calibrator == "identity":
        cal: Calibrator = IdentityCalibrator()
    elif cfg.calibrator == "temperature":
        cal = TemperatureScalingCalibrator()
    elif cfg.calibrator == "isotonic":
        cal = IsotonicCalibrator()
    else:
        raise ValueError(f"Unknown calibrator: {cfg.calibrator}")

    cal.fit(p_calib_raw, y_calib)
    p_calib = cal.predict(p_calib_raw)

    report = {
        "ece": ece_metric(p_calib, y_calib),
        "brier": brier_metric(p_calib, y_calib),
        "risk_coverage": {k: v.tolist() for k, v in risk_coverage_curve(p_calib, y_calib).items()},
        "cfg": cfg.__dict__,
    }

    atomic_write_json(out_dir / "tier0_calibration_report.json", report)
    return report
