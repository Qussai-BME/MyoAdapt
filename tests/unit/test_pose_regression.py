"""Unit tests for sEMG-to-hand-pose regression (myoadapt.tasks.pose_regression)."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from myoadapt.tasks.pose_regression import PoseMLPRegressor, TransformerRegressor as PoseTransformerRegressor
from myoadapt.models.base import MODEL_REGISTRY, get_model


# ---------------------------------------------------------------------------
# PoseMLPRegressor (sklearn-backed, no torch needed)
# ---------------------------------------------------------------------------
def _linear_pose_data(n=50, n_feat=20, n_out=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_feat)).astype(np.float32)
    y = (X[:, :n_out] * 2 + 0.05 * rng.standard_normal((n, n_out))).astype(np.float32)
    return X, y


def test_pose_mlp_fit_predict_learns_signal():
    X, y = _linear_pose_data(n=150, n_feat=20, n_out=5)
    m = PoseMLPRegressor(n_features=X.shape[1], n_outputs=y.shape[1],
                          hidden_dims=[32, 16], n_epochs=300, random_state=0,
                          dropout=0.0)  # isolate fitting correctness from regularization strength
    m.fit(X, y)
    preds = m.predict(X)
    assert preds.shape == y.shape
    from sklearn.metrics import r2_score
    assert r2_score(y, preds) > 0.3  # not a strict bound, just "actually learning"


def test_pose_mlp_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        PoseMLPRegressor().predict(np.zeros((2, 10)))


def test_pose_mlp_save_load_roundtrip():
    X, y = _linear_pose_data()
    m = PoseMLPRegressor(n_features=X.shape[1], n_outputs=y.shape[1], n_epochs=50).fit(X, y)
    preds_before = m.predict(X)
    with tempfile.TemporaryDirectory() as td:
        path = str(Path(td) / "pose_mlp.pkl")
        m.save(path)
        m2 = PoseMLPRegressor.load(path)
        preds_after = m2.predict(X)
    assert np.allclose(preds_before, preds_after)


def test_pose_mlp_adjusts_dims_from_data():
    """n_features/n_outputs given at construction should be corrected to
    match the data actually passed to fit(), not silently mismatch."""
    X, y = _linear_pose_data(n_feat=20, n_out=5)
    m = PoseMLPRegressor(n_features=999, n_outputs=999).fit(X, y)
    assert m.n_features == 20
    assert m.n_outputs == 5


# ---------------------------------------------------------------------------
# TransformerRegressor (torch-backed, raw (n_channels, n_samples) windows)
# ---------------------------------------------------------------------------
torch = pytest.importorskip("torch")


def test_pose_transformer_fit_predict_shapes():
    rng = np.random.default_rng(0)
    n, ch, ws, n_out = 16, 6, 160, 4
    X = rng.standard_normal((n, ch, ws)).astype(np.float32)
    y = rng.standard_normal((n, n_out)).astype(np.float32)
    m = PoseTransformerRegressor(n_channels=ch, n_samples=ws, n_outputs=n_out,
                                  d_model=16, n_heads=2, n_layers=1, patch_len=20,
                                  n_epochs=2, batch_size=8, device="cpu")
    m.fit(X, y)
    preds = m.predict(X)
    assert preds.shape == (n, n_out)
    assert m.count_parameters() > 0


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
def test_registered_in_model_registry():
    assert "pose_mlp" in MODEL_REGISTRY
    assert "pose_transformer" in MODEL_REGISTRY
    m = get_model("pose_mlp", n_features=10, n_outputs=3)
    assert isinstance(m, PoseMLPRegressor)
