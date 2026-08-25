"""Unit tests for myoadapt.evaluation.electrode_shift.ElectrodeShiftRobustness."""
from __future__ import annotations

import numpy as np
import pytest

from myoadapt.evaluation.electrode_shift import ElectrodeShiftRobustness
from myoadapt.models.classical import EMGClassifier


def _fitted_model_and_data(n=150, n_ch=8, n_feat_per_ch=4, n_cls=3, seed=0):
    rng = np.random.default_rng(seed)
    X3d = rng.standard_normal((n, n_ch, n_feat_per_ch))
    y = rng.integers(0, n_cls, n)
    clf = EMGClassifier(model_type="random_forest", k_features=None)
    clf.fit(X3d.reshape(n, -1), y)

    class _FlatWrap:
        """Adapts a flat-feature classifier to the 3-D (n, ch, feat)
        convention ElectrodeShiftRobustness's perturbations operate on."""
        def __init__(self, inner):
            self.inner = inner

        def predict(self, X):
            return self.inner.predict(np.asarray(X).reshape(len(X), -1))

    return _FlatWrap(clf), X3d, y, n_ch


def test_requires_predict_method():
    with pytest.raises(TypeError):
        ElectrodeShiftRobustness(model=object(), n_channels=8)


def test_requires_at_least_2_channels():
    class Dummy:
        def predict(self, X):
            return np.zeros(len(X))
    with pytest.raises(ValueError):
        ElectrodeShiftRobustness(model=Dummy(), n_channels=1)


def test_circular_shift_is_a_rotation():
    model, X, y, n_ch = _fitted_model_and_data()
    ev = ElectrodeShiftRobustness(model=model, n_channels=n_ch)
    shifted = ev.circular_shift(X, k=1)
    assert np.allclose(shifted[:, 1, :], X[:, 0, :])
    # full rotation returns to the original
    full = ev.circular_shift(X, k=n_ch)
    assert np.allclose(full, X)


def test_channel_dropout_zeroes_chosen_channels():
    model, X, y, n_ch = _fitted_model_and_data()
    ev = ElectrodeShiftRobustness(model=model, n_channels=n_ch, random_state=1)
    dropped = ev.channel_dropout(X, k=2, prob=1.0)
    n_fully_zero_channels = (dropped == 0).all(axis=(0, 2)).sum()
    assert n_fully_zero_channels == 2


def test_channel_dropout_prob_zero_is_noop():
    model, X, y, n_ch = _fitted_model_and_data()
    ev = ElectrodeShiftRobustness(model=model, n_channels=n_ch)
    out = ev.channel_dropout(X, k=3, prob=0.0)
    assert np.array_equal(out, X)


def test_noise_injection_scales_with_sigma():
    model, X, y, n_ch = _fitted_model_and_data()
    ev = ElectrodeShiftRobustness(model=model, n_channels=n_ch, random_state=0)
    small = ev.noise_injection(X, sigma=0.01)
    large = ev.noise_injection(X, sigma=1.0)
    assert np.abs(large - X).mean() > np.abs(small - X).mean()


def test_evaluate_shift_baseline_matches_unperturbed_accuracy():
    model, X, y, n_ch = _fitted_model_and_data()
    ev = ElectrodeShiftRobustness(model=model, n_channels=n_ch, random_state=0)
    result = ev.evaluate_shift(X, y, shift_type="circular", max_shift=4, step=1)
    from sklearn.metrics import accuracy_score
    expected_baseline = accuracy_score(y, model.predict(X))
    assert abs(result["baseline_accuracy"] - expected_baseline) < 1e-9
    assert len(result["per_shift"]) == 5  # 0..4 inclusive, step 1


def test_evaluate_shift_invalid_type_raises():
    model, X, y, n_ch = _fitted_model_and_data()
    ev = ElectrodeShiftRobustness(model=model, n_channels=n_ch)
    with pytest.raises(ValueError):
        ev.evaluate_shift(X, y, shift_type="not_a_real_shift")


def test_evaluate_all_shifts_covers_all_four_types_and_picks_worst():
    model, X, y, n_ch = _fitted_model_and_data()
    ev = ElectrodeShiftRobustness(model=model, n_channels=n_ch, random_state=0)
    full = ev.evaluate_all_shifts(X, y, max_shift=3, step=1)
    assert set(full.keys()) == {"circular", "dropout", "swap", "noise", "summary"}
    assert full["summary"]["worst_shift_type"] in ElectrodeShiftRobustness.SHIFT_TYPES
    assert ev.results_ is full  # cached for plot_degradation_curve
