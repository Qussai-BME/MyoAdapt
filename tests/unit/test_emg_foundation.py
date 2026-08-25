"""
Unit tests for EMG Foundation Model.
"""
import numpy as np
import pytest

from myoadapt.models.base import MODEL_REGISTRY


def test_emg_foundation_registered():
    assert "emg_foundation" in MODEL_REGISTRY


def test_emg_foundation_creation():
    from myoadapt.models.emg_foundation import EMGFoundation
    fm = EMGFoundation(n_channels=12, n_samples=400, n_classes=3,
                       ssl_pretrain_epochs=2, finetune_epochs=2)
    n = fm.count_parameters()
    assert n > 0, "EMGFoundation must have >0 params"


def test_emg_foundation_ssl_pretrain_smoke():
    """SSL pretraining should reduce loss (or at least not crash)."""
    from myoadapt.models.emg_foundation import EMGFoundation
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 12, 400)).astype(np.float32)
    fm = EMGFoundation(n_channels=12, n_samples=400, n_classes=None,
                       ssl_pretrain_epochs=3, batch_size=8)
    fm.pretrain(X)
    hist = fm.training_history()
    assert len(hist["ssl_loss"]) == 3
    assert len(hist["finetune_loss"]) == 0  # not fine-tuned yet


def test_emg_foundation_finetune_predict():
    """End-to-end: pretrain -> fine-tune -> predict."""
    from myoadapt.models.emg_foundation import EMGFoundation
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 12, 400)).astype(np.float32)
    y = rng.integers(0, 3, size=40)
    # Inject class signal
    for i, c in enumerate(y):
        X[i] += c * 0.3
    fm = EMGFoundation(n_channels=12, n_samples=400, n_classes=3,
                       ssl_pretrain_epochs=2, finetune_epochs=3, batch_size=8)
    fm.pretrain(X)
    fm.fit(X, y)
    preds = fm.predict(X)
    assert preds.shape == (40,)
    proba = fm.predict_proba(X)
    assert proba.shape == (40, 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    # Features extracted
    feats = fm.extract_features(X)
    assert feats.shape == (40, fm.d_model)


def test_emg_foundation_save_load(tmp_path):
    from myoadapt.models.emg_foundation import EMGFoundation
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 12, 400)).astype(np.float32)
    y = rng.integers(0, 3, size=20)
    fm = EMGFoundation(n_channels=12, n_samples=400, n_classes=3,
                       ssl_pretrain_epochs=1, finetune_epochs=1, batch_size=8)
    fm.pretrain(X)
    fm.fit(X, y)
    path = str(tmp_path / "emg_fm.pt")
    fm.save(path)
    loaded = EMGFoundation.load(path)
    preds_before = fm.predict(X)
    preds_after = loaded.predict(X)
    np.testing.assert_array_equal(preds_before, preds_after)
