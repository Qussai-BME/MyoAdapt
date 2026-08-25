"""Unit tests for continuous EMG decoding (myoadapt.tasks.continuous).

Previously had zero test coverage, and the underlying models never
registered into MODEL_REGISTRY at all (models/__init__.py never
imported myoadapt.tasks — see CHANGELOG [Unreleased]).
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from myoadapt.tasks.continuous import ContinuousDecoder, LSTMRegressor, TransformerRegressor
from myoadapt.models.base import MODEL_REGISTRY, get_model


def _synthetic_continuous(n_samples=24, seq_len=16, n_channels=6, n_outputs=2, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, seq_len, n_channels)).astype(np.float32)
    y = np.stack([X.mean(axis=(1, 2)), X.std(axis=(1, 2))], axis=1).astype(np.float32)
    return X, y


@pytest.mark.parametrize("cls", [LSTMRegressor, TransformerRegressor])
def test_fit_predict_shapes(cls):
    X, y = _synthetic_continuous(n_outputs=2)
    m = cls(n_channels=X.shape[2], n_outputs=2, hidden_dim=8, n_epochs=2,
            batch_size=8, device="cpu")
    m.fit(X, y)
    preds = m.predict(X)
    assert preds.shape == y.shape
    assert m.count_parameters() > 0


@pytest.mark.parametrize("cls", [LSTMRegressor, TransformerRegressor])
def test_evaluate_returns_expected_keys(cls):
    X, y = _synthetic_continuous()
    m = cls(n_channels=X.shape[2], n_outputs=2, hidden_dim=8, n_epochs=2,
            batch_size=8, device="cpu")
    m.fit(X, y)
    metrics = m.evaluate(X, y)
    assert set(metrics) == {"mse", "rmse", "r2", "mae"}
    assert metrics["mse"] >= 0 and metrics["rmse"] >= 0


@pytest.mark.parametrize("cls", [LSTMRegressor, TransformerRegressor])
def test_save_load_roundtrip_is_exact(cls):
    X, y = _synthetic_continuous()
    m = cls(n_channels=X.shape[2], n_outputs=2, hidden_dim=8, n_epochs=2,
            batch_size=8, device="cpu")
    m.fit(X, y)
    preds_before = m.predict(X)
    with tempfile.TemporaryDirectory() as td:
        path = str(Path(td) / "model.pt")
        m.save(path)
        m2 = cls.load(path)
        preds_after = m2.predict(X)
    assert np.allclose(preds_before, preds_after)


def test_predict_before_fit_raises():
    m = LSTMRegressor(n_channels=4, n_outputs=1, device="cpu")
    with pytest.raises(RuntimeError):
        m.predict(np.zeros((2, 10, 4), dtype=np.float32))


def test_continuous_decoder_unknown_architecture_raises():
    m = ContinuousDecoder(n_channels=4, n_outputs=1, architecture="not_a_real_arch", device="cpu")
    X, y = _synthetic_continuous(n_channels=4, n_outputs=1)
    with pytest.raises(ValueError):
        m.fit(X, y)


# ---------------------------------------------------------------------------
# Registration — regression test for the models/tasks circular-import fix
# ---------------------------------------------------------------------------
def test_registered_in_model_registry():
    """LSTMRegressor/TransformerRegressor must be reachable via the shared
    MODEL_REGISTRY regardless of which module is imported first — this
    used to silently fail because models/__init__.py never imported
    myoadapt.tasks at all."""
    assert "lstm_regressor" in MODEL_REGISTRY
    assert "transformer_regressor" in MODEL_REGISTRY
    m = get_model("lstm_regressor", n_channels=4, n_outputs=1, device="cpu")
    assert isinstance(m, LSTMRegressor)
