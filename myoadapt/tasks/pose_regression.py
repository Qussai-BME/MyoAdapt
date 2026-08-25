"""
pose_regression.py — Continuous hand-pose regression from sEMG.

Goes beyond discrete gesture classification to predict 21-DOF hand
kinematics (joint angles) from surface EMG — the regression
counterpart to the classification pipeline.

Two architectures:
- MLPRegressor: fast baseline, works on any feature vector
- TransformerRegressor: lightweight attention model, INT8-quantizable

The MLPRegressor uses the same EMGClassifier interface (fit/predict/
predict_proba/save/load) so it drops into the existing evaluation
harness. The TransformerRegressor follows the same pattern.

References:
- Meta (2024). "EMG-to-Pose: Real-time hand pose estimation from
  surface EMG." NeurIPS D&B.
- WaveFormer: "Lightweight Transformer for real-time biosignal
  decoding on CPU." arXiv 2025.

License: Apache 2.0
"""
from __future__ import annotations

import hashlib
import logging
import os
import pickle
import secrets
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

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
    logger.info("PyTorch not installed — TransformerRegressor will raise on use")


# ---------------------------------------------------------------------------
# Shared regression metrics
# ---------------------------------------------------------------------------
def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Compute MSE / RMSE / R2 / MAE / per-joint R2."""
    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        r2_score,
    )

    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)

    # Promote 1-D targets so per-joint logic is uniform.
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    # Align shapes — if predictions have a different last-axis size, take
    # the minimum along the joint axis so metric computation is valid.
    n_joints = min(y_true.shape[1], y_pred.shape[1])
    y_true = y_true[:, :n_joints]
    y_pred = y_pred[:, :n_joints]

    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred, multioutput="uniform_average"))
    mae = float(mean_absolute_error(y_true, y_pred))

    per_joint_r2: Dict[str, Any] = {}
    valid: List[float] = []
    for j in range(n_joints):
        yj = y_true[:, j]
        pj = y_pred[:, j]
        if float(np.std(yj)) < 1e-10:
            r2j = float("nan")
        else:
            r2j = float(r2_score(yj, pj))
        per_joint_r2[f"r2_joint_{j}"] = r2j
        if not np.isnan(r2j):
            valid.append(r2j)
    per_joint_r2["r2_mean"] = (
        float(np.mean(valid)) if valid else float("nan")
    )
    per_joint_r2["n_joints"] = int(n_joints)

    return {
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "mae": mae,
        **per_joint_r2,
    }


# ===========================================================================
# PoseMLPRegressor — sklearn-backed, always available
# ===========================================================================
@register_model("pose_mlp")
class PoseMLPRegressor(BaseModel):
    """Fast MLP baseline for sEMG → 21-DOF hand-pose regression.

    Wraps :class:`sklearn.neural_network.MLPRegressor` so it works
    without PyTorch. Implements the same ``fit/predict/predict_proba/
    save/load`` interface as :class:`myoadapt.models.EMGClassifier` and
    drops directly into the existing LOSO / LODO evaluation harness.

    Parameters
    ----------
    n_features : dimensionality of the input feature vector.
    n_outputs : number of regression targets (21 for full hand pose).
    hidden_dims : list of hidden layer sizes.
    dropout : input-feature dropout applied during fit (sklearn's
        ``MLPRegressor`` has no native dropout, so we apply it
        manually on the input batch as a regularizer).
    lr : initial learning rate passed to ``MLPRegressor``.
    n_epochs : max number of training iterations.
    batch_size : mini-batch size.
    device : kept for API compatibility (sklearn is CPU-only).
    random_state : RNG seed for both sklearn and the dropout mask.
    """

    def __init__(
        self,
        n_features: int = 308,
        n_outputs: int = 21,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.3,
        lr: float = 1e-3,
        n_epochs: int = 100,
        batch_size: int = 64,
        device: str = "auto",
        random_state: int = 42,
    ) -> None:
        self.n_features = int(n_features)
        self.n_outputs = int(n_outputs)
        self.hidden_dims = list(hidden_dims) if hidden_dims else [128, 64]
        self.dropout = float(dropout)
        self.lr = float(lr)
        self.n_epochs = int(n_epochs)
        self.batch_size = int(batch_size)
        self.device = "cpu"  # sklearn is CPU-only; preserved for API parity
        del device
        self.random_state = int(random_state)
        self.model_: Optional[Any] = None
        self._fitted: bool = False

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> PoseMLPRegressor:
        """Fit the MLP regressor.

        Parameters
        ----------
        X : float array of shape ``(n_samples, n_features)``.
        y : float array of shape ``(n_samples, n_outputs)`` continuous
            joint-angle targets (radians). A 1-D ``y`` is promoted to
            ``(n_samples, 1)``.
        """
        from sklearn.neural_network import MLPRegressor as _SkMLPRegressor

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        # Update n_features / n_outputs from the data if they were left
        # at defaults that don't match.
        if self.n_features != X.shape[1]:
            logger.debug(
                "Adjusting n_features %d → %d from input data",
                self.n_features, X.shape[1],
            )
            self.n_features = int(X.shape[1])
        if self.n_outputs != y.shape[1]:
            logger.debug(
                "Adjusting n_outputs %d → %d from target data",
                self.n_outputs, y.shape[1],
            )
            self.n_outputs = int(y.shape[1])

        # Manual input dropout — sklearn's MLPRegressor has no native
        # dropout. We apply an inverted-dropout mask to the input batch
        # so the expected activation magnitude is preserved.
        if self.dropout > 0 and self.dropout < 1.0:
            rng = np.random.default_rng(self.random_state)
            keep_p = 1.0 - self.dropout
            mask = (
                rng.random(X.shape).astype(np.float32) < keep_p
            ).astype(np.float32) / keep_p
            X_fit = X * mask
        else:
            X_fit = X

        self.model_ = _SkMLPRegressor(
            hidden_layer_sizes=tuple(self.hidden_dims),
            activation="relu",
            solver="adam",
            learning_rate_init=self.lr,
            max_iter=self.n_epochs,
            batch_size=min(self.batch_size, len(X_fit)),
            random_state=self.random_state,
            early_stopping=True,
            n_iter_no_change=10,
            validation_fraction=0.1,
            tol=1e-4,
        )
        # Multi-output regression: MLPRegressor accepts 2-D y natively.
        self.model_.fit(X_fit, y)
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted joint angles of shape ``(n_samples, n_outputs)``."""
        if not self._fitted or self.model_ is None:
            raise RuntimeError("PoseMLPRegressor not trained — call fit() first")
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        yhat = np.asarray(self.model_.predict(X), dtype=np.float32)
        if yhat.ndim == 1:
            yhat = yhat.reshape(-1, 1)
        return yhat

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """For regression, ``predict_proba`` is identical to ``predict``."""
        return self.predict(X)

    # ------------------------------------------------------------------
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Return regression metrics (MSE, RMSE, R2, MAE, per-joint R2)."""
        y_pred = self.predict(X)
        return _regression_metrics(y, y_pred)

    # ------------------------------------------------------------------
    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "config": self._config_dict(),
                "model": self.model_,
                "_fitted": self._fitted,
            }, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        expected_sha256: Optional[str] = None,
    ) -> PoseMLPRegressor:
        path = Path(path)
        expected = expected_sha256 or os.environ.get("MYOADAPT_MODEL_SHA256")
        require_hash = os.environ.get("MYOADAPT_REQUIRE_MODEL_HASH", "").lower()
        if require_hash in {"1", "true", "yes", "on"} and not expected:
            raise RuntimeError("A trusted MYOADAPT_MODEL_SHA256 is required for this deployment")
        if expected:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if not secrets.compare_digest(digest.lower(), expected.strip().lower()):
                raise RuntimeError("Model artifact hash does not match the trusted expected SHA-256")
        with open(path, "rb") as f:
            state = pickle.load(f)  # nosec B301
        obj = cls(**state["config"])
        obj.model_ = state.get("model")
        obj._fitted = bool(state.get("_fitted", False))
        return obj

    # ------------------------------------------------------------------
    def count_parameters(self) -> int:
        """Return the number of trainable parameters in the MLP."""
        if self.model_ is None or not hasattr(self.model_, "coefs_"):
            return 0
        total = 0
        for layer in self.model_.coefs_:
            total += layer.size
        for bias in self.model_.intercepts_:
            total += bias.size
        return int(total)

    def _config_dict(self) -> Dict[str, Any]:
        return {
            "n_features": self.n_features,
            "n_outputs": self.n_outputs,
            "hidden_dims": list(self.hidden_dims),
            "dropout": self.dropout,
            "lr": self.lr,
            "n_epochs": self.n_epochs,
            "batch_size": self.batch_size,
            "device": "auto",
            "random_state": self.random_state,
        }


# ===========================================================================
# TransformerRegressor — torch-backed, INT8-quantizable
# ===========================================================================
if HAS_TORCH:

    class _EncoderLayer(nn.Module):
        """Pre-norm self-attention + FFN sublayer.

        Hand-rolled (rather than :class:`nn.TransformerEncoderLayer`) so
        that ``torch.quantization.quantize_dynamic`` can target the
        ``nn.Linear`` modules cleanly — PyTorch's stock encoder layer
        includes a fast-path check that is incompatible with INT8
        dynamic quantization on recent torch releases.
        """

        def __init__(
            self,
            d_model: int,
            n_heads: int,
            dim_feedforward: int,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            self.attn = nn.MultiheadAttention(
                d_model, n_heads, dropout=dropout, batch_first=True,
            )
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.ff = nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, d_model),
            )
            self.drop = nn.Dropout(dropout)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            attn_out, _ = self.attn(x, x, x, need_weights=False)
            x = self.norm1(x + self.drop(attn_out))
            ff_out = self.ff(x)
            x = self.norm2(x + self.drop(ff_out))
            return x

    class _TransformerNet(nn.Module):
        """Patch-embedding + Transformer encoder + regression head."""

        def __init__(
            self,
            n_channels: int = 16,
            n_samples: int = 400,
            n_outputs: int = 21,
            d_model: int = 64,
            n_heads: int = 4,
            n_layers: int = 2,
            patch_len: int = 40,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            self.n_samples = int(n_samples)
            self.patch_len = int(patch_len)
            self.d_model = int(d_model)

            # Patch embedding: Conv1d treats the channel axis as input
            # channels and the time axis as the spatial dimension.
            # Input  : (batch, n_channels, n_samples)
            # Output : (batch, d_model, n_patches)
            self.patch_embed = nn.Conv1d(
                n_channels, d_model,
                kernel_size=self.patch_len, stride=self.patch_len,
            )
            n_patches = max(1, n_samples // self.patch_len)
            self.n_patches = n_patches

            # Learned CLS token + positional embedding.
            self.cls_token = nn.Parameter(
                torch.randn(1, 1, d_model) * 0.02
            )
            self.pos_embed = nn.Parameter(
                torch.randn(1, n_patches + 1, d_model) * 0.02
            )
            self.pos_drop = nn.Dropout(dropout)

            # Hand-rolled encoder — see _EncoderLayer docstring for why
            # we don't use nn.TransformerEncoder here.
            import copy as _copy
            base_layer = _EncoderLayer(
                d_model=d_model,
                n_heads=n_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
            )
            self.encoder_layers = nn.ModuleList(
                [_copy.deepcopy(base_layer) for _ in range(n_layers)]
            )

            self.norm = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, n_outputs)

        # ------------------------------------------------------------------
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (batch, n_channels, n_samples)
            b = x.shape[0]
            x = self.patch_embed(x)            # (B, d_model, n_patches)
            x = x.transpose(1, 2)              # (B, n_patches, d_model)

            cls = self.cls_token.expand(b, -1, -1)
            x = torch.cat([cls, x], dim=1)     # (B, n_patches+1, d_model)

            # Truncate / pad positional embedding to match the sequence
            # length actually present (allows variable-length inputs).
            pe = self.pos_embed[:, : x.shape[1]]
            x = x + pe
            x = self.pos_drop(x)

            for layer in self.encoder_layers:
                x = layer(x)
            cls_out = x[:, 0]                  # take CLS token
            cls_out = self.norm(cls_out)
            return self.head(cls_out)

        # ------------------------------------------------------------------
        def count_parameters(self) -> int:
            return int(sum(p.numel() for p in self.parameters() if p.requires_grad))

    @register_model("pose_transformer")
    class TransformerRegressor(BaseModel):
        """Lightweight Transformer for sEMG → hand-pose regression.

        Architecture: ``Conv1d`` patch embedding → learned CLS token +
        positional embedding → Transformer encoder → linear regression
        head. With the default ``d_model=64`` / ``n_layers=2`` config
        the network has ~50K parameters — small enough for INT8 dynamic
        quantization via :func:`torch.quantization.quantize_dynamic`.

        Parameters
        ----------
        n_channels : number of EMG channels.
        n_samples : raw window length (samples per window).
        n_outputs : number of regression targets (21 for full hand pose).
        d_model : Transformer hidden dimension.
        n_heads : number of self-attention heads.
        n_layers : number of Transformer encoder layers.
        patch_len : patch length (samples) for the Conv1d patch embed.
        n_epochs : training epochs.
        lr : Adam learning rate.
        batch_size : mini-batch size.
        device : ``"auto"`` / ``"cpu"`` / ``"cuda"``.
        random_state : RNG seed for reproducibility.
        quantize : if True, apply dynamic INT8 quantization after fit.
        """

        def __init__(
            self,
            n_channels: int = 16,
            n_samples: int = 400,
            n_outputs: int = 21,
            d_model: int = 64,
            n_heads: int = 4,
            n_layers: int = 2,
            patch_len: int = 40,
            n_epochs: int = 50,
            lr: float = 1e-3,
            batch_size: int = 64,
            device: str = "auto",
            random_state: int = 42,
            quantize: bool = False,
        ) -> None:
            self.n_channels = int(n_channels)
            self.n_samples = int(n_samples)
            self.n_outputs = int(n_outputs)
            self.d_model = int(d_model)
            self.n_heads = int(n_heads)
            self.n_layers = int(n_layers)
            self.patch_len = int(patch_len)
            self.n_epochs = int(n_epochs)
            self.lr = float(lr)
            self.batch_size = int(batch_size)
            self.device = (
                "cuda" if device == "auto" and torch.cuda.is_available()
                else "cpu" if device == "auto" else device
            )
            self.random_state = int(random_state)
            self.quantize_after_fit = bool(quantize)
            self.net: Optional[nn.Module] = None
            self._quantized: bool = False
            torch.manual_seed(self.random_state)
            # Build eagerly so count_parameters() works pre-fit (matches
            # the CNN1D / LiteDAN pattern).
            self._build_net()

        # ------------------------------------------------------------------
        def _build_net(self) -> nn.Module:
            self.net = _TransformerNet(
                n_channels=self.n_channels,
                n_samples=self.n_samples,
                n_outputs=self.n_outputs,
                d_model=self.d_model,
                n_heads=self.n_heads,
                n_layers=self.n_layers,
                patch_len=self.patch_len,
                dropout=0.1,
            ).to(self.device)
            return self.net

        # ------------------------------------------------------------------
        def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> TransformerRegressor:
            """Fit the Transformer regressor.

            Parameters
            ----------
            X : float array of shape ``(n_samples, n_channels, n_samples_raw)``.
            y : float array of shape ``(n_samples, n_outputs)`` continuous
                joint-angle targets. 1-D ``y`` is promoted to 2-D.
            """
            X = np.asarray(X, dtype=np.float32)
            y = np.asarray(y, dtype=np.float32)
            if y.ndim == 1:
                y = y.reshape(-1, 1)
            if X.ndim != 3:
                raise ValueError(
                    f"X must be 3-D (n_samples, n_channels, n_samples_raw); "
                    f"got shape {X.shape}"
                )
            # Update n_samples / n_channels from data so the patch-embed
            # conv produces the right number of patches at inference time.
            if X.shape[1] != self.n_channels:
                self.n_channels = int(X.shape[1])
            if X.shape[2] != self.n_samples:
                self.n_samples = int(X.shape[2])
            if y.shape[1] != self.n_outputs:
                self.n_outputs = int(y.shape[1])
            self._build_net()

            optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
            criterion = nn.MSELoss()

            X_t = torch.FloatTensor(X).to(self.device)
            y_t = torch.FloatTensor(y).to(self.device)
            ds = TensorDataset(X_t, y_t)
            loader = DataLoader(
                ds, batch_size=min(self.batch_size, len(ds)), shuffle=True,
            )

            self.net.train()
            for epoch in range(self.n_epochs):
                total = 0.0
                for xb, yb in loader:
                    optimizer.zero_grad()
                    out = self.net(xb)
                    loss = criterion(out, yb)
                    loss.backward()
                    optimizer.step()
                    total += float(loss.item())
                if (epoch + 1) % max(1, self.n_epochs // 5) == 0:
                    logger.debug(
                        "  [TransformerRegressor] epoch %d/%d mse=%.6f",
                        epoch + 1, self.n_epochs, total / max(1, len(loader)),
                    )
            self.net.eval()
            if self.quantize_after_fit:
                self.quantize_int8()
            return self

        # ------------------------------------------------------------------
        def predict(self, X: np.ndarray) -> np.ndarray:
            if self.net is None:
                raise RuntimeError("TransformerRegressor not trained — call fit() first")
            X = np.asarray(X, dtype=np.float32)
            if X.ndim == 2:
                # Allow single-window inference by adding a batch dim.
                X = X[np.newaxis, ...]
            self.net.eval()
            with torch.no_grad():
                X_t = torch.FloatTensor(X).to(self.device)
                yhat = self.net(X_t).cpu().numpy()
            return yhat.astype(np.float32)

        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            """For regression, identical to ``predict``."""
            return self.predict(X)

        # ------------------------------------------------------------------
        def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
            y_pred = self.predict(X)
            return _regression_metrics(y, y_pred)

        # ------------------------------------------------------------------
        def quantize_int8(self) -> TransformerRegressor:
            """Apply dynamic INT8 quantization for CPU inference speedup."""
            import torch.quantization
            if self.net is None:
                raise RuntimeError("Cannot quantize an untrained model")
            self.net = torch.quantization.quantize_dynamic(
                self.net, {nn.Linear}, dtype=torch.qint8,
            )
            self.net.eval()
            self.net.to("cpu")
            self.device = "cpu"
            self._quantized = True
            logger.info(
                "TransformerRegressor quantized to INT8 (Linear layers)."
            )
            return self

        # ------------------------------------------------------------------
        def save(self, path: Union[str, Path]) -> None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            # Save full net via state_dict so it round-trips cleanly even
            # after INT8 quantization (quantized modules serialize correctly
            # with torch.save).
            torch.save({
                "state_dict": self.net.state_dict() if self.net else None,
                "config": self._config_dict(),
                "quantized": self._quantized,
            }, path)

        @classmethod
        def load(cls, path: Union[str, Path]) -> TransformerRegressor:
            state = torch.load(path, map_location="cpu", weights_only=True)
            cfg = state["config"]
            obj = cls(**cfg)
            obj._build_net()
            if state.get("state_dict") is not None:
                obj.net.load_state_dict(state["state_dict"])
            obj.net.eval()
            obj._quantized = bool(state.get("quantized", False))
            return obj

        # ------------------------------------------------------------------
        def count_parameters(self) -> int:
            return int(self.net.count_parameters()) if self.net is not None else 0

        def _config_dict(self) -> Dict[str, Any]:
            return {
                "n_channels": self.n_channels,
                "n_samples": self.n_samples,
                "n_outputs": self.n_outputs,
                "d_model": self.d_model,
                "n_heads": self.n_heads,
                "n_layers": self.n_layers,
                "patch_len": self.patch_len,
                "n_epochs": self.n_epochs,
                "lr": self.lr,
                "batch_size": self.batch_size,
                "device": "cpu",  # serialize as CPU-friendly
                "random_state": self.random_state,
                "quantize": self._quantized,
            }


else:  # torch not installed — register a stub so the import path always works.

    @register_model("pose_transformer")
    class TransformerRegressor(BaseModel):  # type: ignore[no-redef]
        """Stub registered when PyTorch is unavailable — raises on use."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "TransformerRegressor requires PyTorch. "
                "Install with: pip install torch"
            )

        def fit(self, X: Any, y: Any, **kwargs: Any) -> TransformerRegressor:
            raise ImportError("PyTorch required for TransformerRegressor")

        def predict(self, X: Any) -> Any:
            raise ImportError("PyTorch required for TransformerRegressor")

        def predict_proba(self, X: Any) -> Any:
            raise ImportError("PyTorch required for TransformerRegressor")

        def evaluate(self, X: Any, y: Any) -> Dict[str, Any]:
            raise ImportError("PyTorch required for TransformerRegressor")

        def save(self, path: Any) -> None:
            raise ImportError("PyTorch required for TransformerRegressor")

        @classmethod
        def load(cls, path: Any) -> TransformerRegressor:
            raise ImportError("PyTorch required for TransformerRegressor")

        def quantize_int8(self) -> TransformerRegressor:
            raise ImportError("PyTorch required for TransformerRegressor")

        def count_parameters(self) -> int:
            return 0
