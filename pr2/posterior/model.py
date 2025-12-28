from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol

import numpy as np


class PosteriorModel(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray) -> "PosteriorModel":
        ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        ...


@dataclass
class SklearnLogRegPosterior:
    """
    默认优先用 sklearn 的逻辑回归（稳定、可解释、训练快，适合 Tier-0）。
    如果你后续想要更强的非线性，可以换 torch MLP（见 TorchMLPPosterior）。
    """
    C: float = 1.0
    max_iter: int = 200
    _model: Any = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SklearnLogRegPosterior":
        try:
            from sklearn.linear_model import LogisticRegression
        except Exception as e:
            raise ImportError("需要 scikit-learn 才能使用 SklearnLogRegPosterior。请 pip install scikit-learn") from e

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        self._model = LogisticRegression(C=self.C, max_iter=self.max_iter, solver="lbfgs")
        self._model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Posterior model not fitted yet.")
        X = np.asarray(X, dtype=np.float32)
        p = self._model.predict_proba(X)[:, 1]
        return p.astype(np.float32)


@dataclass
class TorchMLPPosterior:
    """
    可选：torch MLP。你可以在 evidence 更复杂时使用。
    """
    hidden: int = 128
    lr: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 20
    batch_size: int = 256
    seed: int = 0
    _state: Optional[Dict[str, Any]] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TorchMLPPosterior":
        try:
            import torch
            import torch.nn as nn
        except Exception as e:
            raise ImportError("需要 torch 才能使用 TorchMLPPosterior。请 pip install torch") from e

        rng = np.random.default_rng(self.seed)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = nn.Sequential(
            nn.Linear(X.shape[1], self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, 1),
        ).to(device)

        opt = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.BCEWithLogitsLoss()

        idx = np.arange(len(X))
        for _ep in range(self.epochs):
            rng.shuffle(idx)
            for s in range(0, len(X), self.batch_size):
                j = idx[s:s+self.batch_size]
                xb = torch.from_numpy(X[j]).to(device)
                yb = torch.from_numpy(y[j].astype(np.float32)).to(device).view(-1, 1)
                logits = model(xb)
                loss = loss_fn(logits, yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

        self._state = {
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "in_dim": int(X.shape[1]),
            "hidden": int(self.hidden),
        }
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._state is None:
            raise RuntimeError("Posterior model not fitted yet.")
        try:
            import torch
            import torch.nn as nn
        except Exception as e:
            raise ImportError("需要 torch 才能使用 TorchMLPPosterior。请 pip install torch") from e

        X = np.asarray(X, dtype=np.float32)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        in_dim = int(self._state["in_dim"])
        hidden = int(self._state["hidden"])

        model = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        ).to(device)
        model.load_state_dict(self._state["state_dict"])
        model.eval()

        with torch.no_grad():
            xb = torch.from_numpy(X).to(device)
            logits = model(xb).view(-1)
            p = torch.sigmoid(logits).detach().cpu().numpy()
        return p.astype(np.float32)
