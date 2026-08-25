"""
continuous.py — Continuous EMG decoding
==========================================

Goes beyond discrete gesture classification to continuous outputs:
- Joint angle regression
- Force level prediction
- Continuous kinematics

Two model architectures:
- LSTMRegressor        — 2-layer LSTM + linear head
- TransformerRegressor — Transformer encoder + linear head

Both inherit from ``ContinuousDecoder``, the canonical wrapper class.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from myoadapt.models.base import BaseModel, register_model

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch not installed — ContinuousDecoder disabled")


if HAS_TORCH:

    class _LSTMNet(nn.Module):
        """2-layer LSTM + linear regression head."""

        def __init__(self, n_channels: int = 12, hidden_dim: int = 64,
                     n_outputs: int = 1, n_layers: int = 2,
                     dropout: float = 0.3):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=n_channels, hidden_size=hidden_dim,
                num_layers=n_layers, batch_first=True, dropout=dropout,
            )
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, 32), nn.ReLU(),
                nn.Linear(32, n_outputs),
            )

        def forward(self, x):
            """x: (batch, seq_len, n_channels)"""
            lstm_out, _ = self.lstm(x)
            # Take last timestep
            last = lstm_out[:, -1, :]
            return self.head(last)

        def count_parameters(self) -> int:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)


    class _TransformerNet(nn.Module):
        """Transformer encoder + linear regression head."""

        def __init__(self, n_channels: int = 12, d_model: int = 64,
                     n_heads: int = 4, n_layers: int = 2,
                     n_outputs: int = 1, dropout: float = 0.1):
            super().__init__()
            self.input_proj = nn.Linear(n_channels, d_model)
            self.pos_encoder = nn.Parameter(torch.randn(1, 1024, d_model) * 0.02)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
                dropout=dropout, batch_first=True, activation="gelu",
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.head = nn.Sequential(
                nn.Linear(d_model, 32), nn.ReLU(),
                nn.Linear(32, n_outputs),
            )

        def forward(self, x):
            """x: (batch, seq_len, n_channels)"""
            seq_len = x.shape[1]
            x = self.input_proj(x) + self.pos_encoder[:, :seq_len]
            x = self.transformer(x)
            # Mean-pool over sequence
            x = x.mean(dim=1)
            return self.head(x)

        def count_parameters(self) -> int:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)


    class ContinuousDecoder(BaseModel):
        """
        Continuous EMG decoder for joint angle / force regression.

        Parameters
        ----------
        n_channels : int (default 12)
        n_outputs : int — dimensionality of regression target (e.g., 1 for force, 5 for 5-DOF hand)
        architecture : 'lstm' or 'transformer'
        hidden_dim / d_model : hidden state size
        n_epochs : int
        lr : float
        """

        def __init__(self,
                     n_channels: int = 12,
                     n_outputs: int = 1,
                     architecture: str = "lstm",
                     hidden_dim: int = 64,
                     n_heads: int = 4,
                     n_layers: int = 2,
                     n_epochs: int = 50,
                     lr: float = 1e-3,
                     batch_size: int = 64,
                     device: str = "auto",
                     random_state: int = 42):
            self.n_channels = n_channels
            self.n_outputs = n_outputs
            self.architecture = architecture
            self.hidden_dim = hidden_dim
            self.n_heads = n_heads
            self.n_layers = n_layers
            self.n_epochs = n_epochs
            self.lr = lr
            self.batch_size = batch_size
            self.device = ("cuda" if device == "auto" and torch.cuda.is_available()
                          else "cpu" if device == "auto" else device)
            self.random_state = random_state
            self.net: Optional[nn.Module] = None
            torch.manual_seed(random_state)

        def _build_net(self):
            if self.architecture == "lstm":
                return _LSTMNet(
                    n_channels=self.n_channels,
                    hidden_dim=self.hidden_dim,
                    n_outputs=self.n_outputs,
                    n_layers=self.n_layers,
                )
            elif self.architecture == "transformer":
                return _TransformerNet(
                    n_channels=self.n_channels,
                    d_model=self.hidden_dim,
                    n_heads=self.n_heads,
                    n_layers=self.n_layers,
                    n_outputs=self.n_outputs,
                )
            else:
                raise ValueError(f"Unknown architecture: {self.architecture}")

        def fit(self, X: np.ndarray, y: np.ndarray,
                X_val: Optional[np.ndarray] = None,
                y_val: Optional[np.ndarray] = None) -> ContinuousDecoder:
            """
            X : (n_samples, seq_len, n_channels)
            y : (n_samples, n_outputs) regression targets
            """
            X = np.asarray(X, dtype=np.float32)
            y = np.asarray(y, dtype=np.float32)
            self.net = self._build_net().to(self.device)
            optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
            criterion = nn.MSELoss()

            X_t = torch.FloatTensor(X).to(self.device)
            y_t = torch.FloatTensor(y).to(self.device)
            ds = TensorDataset(X_t, y_t)
            loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

            self.net.train()
            for epoch in range(self.n_epochs):
                total_loss = 0
                for xb, yb in loader:
                    optimizer.zero_grad()
                    out = self.net(xb)
                    loss = criterion(out, yb)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                if (epoch + 1) % 10 == 0:
                    logger.info(f"  [ContinuousDecoder] epoch {epoch+1}/{self.n_epochs} "
                                f"mse={total_loss/len(loader):.6f}")
            return self

        def predict(self, X: np.ndarray) -> np.ndarray:
            if self.net is None:
                raise RuntimeError("Not trained")
            X = np.asarray(X, dtype=np.float32)
            self.net.eval()
            with torch.no_grad():
                X_t = torch.FloatTensor(X).to(self.device)
                return self.net(X_t).cpu().numpy()

        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            # For regression, predict == predict_proba
            return self.predict(X)

        def save(self, path: str) -> None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            # Exclude the live nn.Module from the saved config — it is
            # restored separately via state_dict on load.
            cfg = {k: v for k, v in self.__dict__.items()
                   if k != "net" and not k.startswith("_")}
            torch.save({
                "state_dict": self.net.state_dict() if self.net else None,
                "config": cfg,
            }, path)

        @classmethod
        def load(cls, path: str) -> ContinuousDecoder:
            state = torch.load(path, map_location="cpu", weights_only=True)
            cfg = state["config"]
            obj = cls(**cfg)
            obj.net = obj._build_net()
            if state["state_dict"] is not None:
                obj.net.load_state_dict(state["state_dict"])
            return obj

        def count_parameters(self) -> int:
            return self.net.count_parameters() if self.net else 0

        def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
            """Compute regression metrics."""
            from sklearn.metrics import mean_squared_error, r2_score
            y_pred = self.predict(X)
            mse = float(mean_squared_error(y, y_pred))
            rmse = float(np.sqrt(mse))
            r2 = float(r2_score(y, y_pred))
            mae = float(np.mean(np.abs(y - y_pred)))
            return {"mse": mse, "rmse": rmse, "r2": r2, "mae": mae}


    # Aliases / register
    @register_model("lstm_regressor")
    class LSTMRegressor(ContinuousDecoder):
        def __init__(self, **kwargs):
            kwargs["architecture"] = "lstm"
            super().__init__(**kwargs)


    @register_model("transformer_regressor")
    class TransformerRegressor(ContinuousDecoder):
        def __init__(self, **kwargs):
            kwargs["architecture"] = "transformer"
            super().__init__(**kwargs)

else:
    class ContinuousDecoder:
        def __init__(self, *args, **kwargs):
            raise ImportError("ContinuousDecoder requires PyTorch. Install with: pip install torch")


    class LSTMRegressor:
        """Stub registered when PyTorch is unavailable — raises on use."""

        def __init__(self, *args, **kwargs):
            raise ImportError("LSTMRegressor requires PyTorch. Install with: pip install torch")


    class TransformerRegressor:
        """Stub registered when PyTorch is unavailable — raises on use."""

        def __init__(self, *args, **kwargs):
            raise ImportError("TransformerRegressor requires PyTorch. Install with: pip install torch")
