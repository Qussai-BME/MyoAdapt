"""Unit tests for models."""
import hashlib

import numpy as np
import pytest

from myoadapt.models.classical import EMGClassifier
from myoadapt.models.base import MODEL_REGISTRY, get_model


def test_classical_xgboost_unavailable_falls_back():
    """If xgboost is not installed, falls back to RandomForest."""
    try:
        import xgboost
        pytest.skip("xgboost is installed — fallback not tested")
    except ImportError:
        clf = EMGClassifier(model_type="xgboost")
        assert "RandomForest" in type(clf.model).__name__ or "XGB" in type(clf.model).__name__


def test_classical_random_forest_train_predict():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 20)).astype(np.float32)
    y = rng.integers(0, 3, size=100)
    clf = EMGClassifier(model_type="random_forest", n_estimators=10)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert len(preds) == 100
    proba = clf.predict_proba(X)
    assert proba.shape == (100, 3)


def test_classical_lda():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 20)).astype(np.float32)
    y = rng.integers(0, 3, size=100)
    clf = EMGClassifier(model_type="lda")
    clf.fit(X, y)
    preds = clf.predict(X)
    assert len(preds) == 100


def test_classical_save_load(tmp_path):
    rng = np.random.default_rng(42)
    X = rng.standard_normal((50, 10)).astype(np.float32)
    y = rng.integers(0, 2, size=50)
    clf = EMGClassifier(model_type="random_forest", n_estimators=5)
    clf.fit(X, y)
    path = str(tmp_path / "model.pkl")
    clf.save(path)
    loaded = EMGClassifier.load(path)
    preds = loaded.predict(X)
    assert len(preds) == 50


def test_classical_load_enforces_expected_hash(tmp_path):
    rng = np.random.default_rng(7)
    X = rng.standard_normal((30, 6)).astype(np.float32)
    y = rng.integers(0, 2, size=30)
    clf = EMGClassifier(model_type="random_forest", n_estimators=5)
    clf.fit(X, y)
    path = tmp_path / "trusted-model.pkl"
    clf.save(path)
    expected = hashlib.sha256(path.read_bytes()).hexdigest()

    loaded = EMGClassifier.load(path, expected_sha256=expected)
    assert len(loaded.predict(X)) == len(X)
    with pytest.raises(RuntimeError, match="hash does not match"):
        EMGClassifier.load(path, expected_sha256="0" * 64)


def test_model_registry():
    """Registry must contain all four canonical models."""
    assert "classical" in MODEL_REGISTRY
    # cnn1d and lite_dan should be registered either via eager import (torch
    # available) or via stub (torch unavailable).
    assert "cnn1d" in MODEL_REGISTRY, f"registry has: {list(MODEL_REGISTRY.keys())}"
    assert "lite_dan" in MODEL_REGISTRY, f"registry has: {list(MODEL_REGISTRY.keys())}"


def test_get_model_dispatch():
    clf = get_model("classical", model_type="lda")
    assert clf is not None


def test_cnn1d_creation():
    from myoadapt.models.cnn1d import CNN1D
    model = CNN1D(n_channels=12, n_classes=3, n_samples=400, n_epochs=2)
    n = model.count_parameters()
    assert n > 0, "CNN1D must have >0 params after eager init"
    # Realistic count for the strawman architecture (~15K for default config).
    assert 10_000 < n < 25_000, f"expected ~15K params, got {n}"


def test_cnn1d_train_predict_smoke():
    """End-to-end smoke: train on small synthetic data, predict, check shapes."""
    from myoadapt.models.cnn1d import CNN1D
    rng = np.random.default_rng(0)
    # 40 samples, 12 channels, 400 samples each
    X = rng.standard_normal((40, 12, 400)).astype(np.float32)
    # Class-correlated: add small offset per class
    y = rng.integers(0, 3, size=40)
    for i, c in enumerate(y):
        X[i] += c * 0.5
    model = CNN1D(n_channels=12, n_classes=3, n_samples=400, n_epochs=2, batch_size=8)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (40,)
    proba = model.predict_proba(X)
    assert proba.shape == (40, 3)
    # Probabilities must sum to 1
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_cnn1d_save_load(tmp_path):
    """Verify save/load round-trip preserves prediction behavior."""
    from myoadapt.models.cnn1d import CNN1D
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 12, 400)).astype(np.float32)
    y = rng.integers(0, 3, size=20)
    model = CNN1D(n_channels=12, n_classes=3, n_samples=400, n_epochs=1, batch_size=8)
    model.fit(X, y)
    path = str(tmp_path / "cnn1d.pt")
    model.save(path)
    loaded = CNN1D.load(path)
    preds_before = model.predict(X)
    preds_after = loaded.predict(X)
    np.testing.assert_array_equal(preds_before, preds_after)


def test_lite_dan_creation():
    from myoadapt.models.lite_dan import LiteDAN
    model = LiteDAN(n_features=100, n_classes=3, n_domains=5, n_epochs=2)
    n = model.count_parameters()
    assert n > 0, "LiteDAN must have >0 params after eager init"


def test_lite_dan_train_predict_smoke():
    """End-to-end smoke: train on small synthetic data with subject groups."""
    from myoadapt.models.lite_dan import LiteDAN
    rng = np.random.default_rng(0)
    n_samples = 60
    n_features = 30
    n_subjects = 5
    n_classes = 3
    X = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    y = rng.integers(0, n_classes, size=n_samples)
    groups = rng.integers(0, n_subjects, size=n_samples)
    # Inject class signal so model can learn something
    for i, c in enumerate(y):
        X[i, :10] += c * 0.8
    model = LiteDAN(
        n_features=n_features, n_classes=n_classes, n_domains=n_subjects,
        n_epochs=3, batch_size=16, lambda_schedule='gradual', use_grl=True,
    )
    model.fit(X, y, groups=groups)
    preds = model.predict(X)
    assert preds.shape == (n_samples,)
    proba = model.predict_proba(X)
    assert proba.shape == (n_samples, n_classes)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    # Lambda schedule history should be present
    hist = model.training_history()
    assert 'lambda' in hist
    assert len(hist['lambda']) == 3  # 3 epochs


def test_lite_dan_ablation_no_grl():
    """Ablation hook: use_grl=False must still train (no GRL)."""
    from myoadapt.models.lite_dan import LiteDAN
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 20)).astype(np.float32)
    y = rng.integers(0, 3, size=40)
    groups = rng.integers(0, 4, size=40)
    model = LiteDAN(
        n_features=20, n_classes=3, n_domains=4,
        n_epochs=2, batch_size=16, use_grl=False, lambda_schedule='none',
    )
    model.fit(X, y, groups=groups)
    preds = model.predict(X)
    assert preds.shape == (40,)


def test_lite_dan_ablation_fixed_lambda():
    """Ablation hook: lambda_schedule='fixed' must train with constant lambda."""
    from myoadapt.models.lite_dan import LiteDAN
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 20)).astype(np.float32)
    y = rng.integers(0, 3, size=40)
    groups = rng.integers(0, 4, size=40)
    model = LiteDAN(
        n_features=20, n_classes=3, n_domains=4,
        n_epochs=3, batch_size=16, lambda_schedule='fixed', lambda_max=0.5,
    )
    model.fit(X, y, groups=groups)
    hist = model.training_history()
    # Fixed schedule: lambda should be constant at lambda_max
    assert all(abs(l - 0.5) < 1e-6 for l in hist['lambda']), \
        f"fixed schedule must give lambda=0.5, got {hist['lambda']}"


def test_dan_trainer_explicit_train_and_epoch_override():
    """DANTrainer wraps LiteDAN with an explicit train() call and logging;
    had no test of its own even though LiteDAN.fit() itself is tested."""
    pytest.importorskip("torch")
    import numpy as np
    from myoadapt.models.lite_dan import LiteDAN, DANTrainer

    rng = np.random.default_rng(0)
    n, n_feat, n_cls, n_dom = 60, 16, 3, 3
    X = rng.standard_normal((n, n_feat)).astype(np.float32)
    y = rng.integers(0, n_cls, n)
    groups = np.repeat(np.arange(n_dom), n // n_dom)

    model = LiteDAN(n_features=n_feat, n_classes=n_cls, n_domains=n_dom,
                     n_epochs=20, device="cpu")
    trainer = DANTrainer(model, lr=0.001)
    history = trainer.train(X, y, groups, n_epochs=3, batch_size=16)
    assert set(history.keys()) >= {"loss", "lambda"}
    assert len(history["loss"]) == 3  # n_epochs override respected
    assert model.n_epochs == 20  # restored after train(), not left mutated
    preds = model.predict(X)
    assert preds.shape == (n,)
