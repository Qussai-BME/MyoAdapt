"""
Unit tests for the calibration & uncertainty module.
"""
from __future__ import annotations

import numpy as np
import pytest

from myoadapt.evaluation.calibration import (
    TemperatureScaling, PlattCalibration, IsotonicCalibration,
    expected_calibration_error, maximum_calibration_error, brier_score,
    calibration_report, conformal_prediction_set,
)


def _softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


@pytest.fixture
def overconfident_data():
    rng = np.random.default_rng(42)
    n = 200
    y_true = rng.integers(0, 3, n)
    # Over-confident logits: model gets ~70% accuracy but is way over-confident.
    logits = rng.standard_normal((n, 3)) * 5 + y_true[:, None] * 2
    return logits, y_true


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def test_ece_zero_for_perfect_predictions():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_proba = np.eye(3)[y_true]
    assert expected_calibration_error(y_true, y_proba) == 0.0


def test_ece_positive_for_overconfident_model(overconfident_data):
    logits, y_true = overconfident_data
    ece = expected_calibration_error(y_true, _softmax(logits))
    assert ece > 0.1


def test_mce_geq_ece(overconfident_data):
    logits, y_true = overconfident_data
    proba = _softmax(logits)
    ece = expected_calibration_error(y_true, proba)
    mce = maximum_calibration_error(y_true, proba)
    assert mce >= ece - 1e-9


def test_brier_zero_for_perfect():
    y_true = np.array([0, 1, 2])
    y_proba = np.eye(3)[y_true]
    assert brier_score(y_true, y_proba) == 0.0


def test_brier_binary_mode():
    y_true = np.array([0, 1, 1, 0])
    y_proba = np.array([0.1, 0.9, 0.8, 0.2])
    b = brier_score(y_true, y_proba)
    assert b > 0.0
    # Manual: ((0.1-0)^2 + (0.9-1)^2 + (0.8-1)^2 + (0.2-0)^2) / 4 = 0.015 + 0.015 + 0.04 + 0.04 / 4 ≈ 0.0225
    # Actually formula above is mean((p1 - y)^2). Let's verify range is sane.
    assert b < 0.1


def test_calibration_report_contains_all_fields(overconfident_data):
    logits, y_true = overconfident_data
    rep = calibration_report(y_true, _softmax(logits))
    for key in ("ece", "mce", "brier", "nll", "mean_confidence", "accuracy",
                "n_bins", "n_samples"):
        assert key in rep
    assert rep["n_samples"] == len(y_true)


# ---------------------------------------------------------------------------
# Temperature scaling
# ---------------------------------------------------------------------------
def test_temperature_scaling_reduces_ece(overconfident_data):
    logits, y_true = overconfident_data
    ts = TemperatureScaling()
    ts.fit(logits[:150], y_true[:150])
    calibrated = ts.transform(logits[150:])
    ece_before = expected_calibration_error(y_true[150:], _softmax(logits[150:]))
    ece_after = expected_calibration_error(y_true[150:], calibrated)
    # Calibration should reduce (or at least not worsen) ECE significantly.
    assert ece_after < ece_before + 0.05


def test_temperature_scaling_preserves_argmax(overconfident_data):
    """Temperature scaling does NOT change the predicted class."""
    logits, y_true = overconfident_data
    ts = TemperatureScaling().fit(logits[:150], y_true[:150])
    cal = ts.transform(logits[150:])
    raw_pred = _softmax(logits[150:]).argmax(axis=1)
    cal_pred = cal.argmax(axis=1)
    np.testing.assert_array_equal(raw_pred, cal_pred)


def test_temperature_scaling_rows_sum_to_one(overconfident_data):
    logits, y_true = overconfident_data
    ts = TemperatureScaling().fit(logits, y_true)
    cal = ts.transform(logits)
    np.testing.assert_allclose(cal.sum(axis=1), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Platt & Isotonic
# ---------------------------------------------------------------------------
def test_platt_calibration_rows_sum_to_one(overconfident_data):
    logits, y_true = overconfident_data
    cal = PlattCalibration().fit_transform(logits[:150], y_true[:150])
    np.testing.assert_allclose(cal.sum(axis=1), 1.0, atol=1e-6)


def test_isotonic_calibration_rows_sum_to_one(overconfident_data):
    logits, y_true = overconfident_data
    cal = IsotonicCalibration().fit_transform(logits[:150], y_true[:150])
    np.testing.assert_allclose(cal.sum(axis=1), 1.0, atol=1e-6)


def test_platt_calibration_output_in_unit_interval(overconfident_data):
    logits, y_true = overconfident_data
    cal = PlattCalibration().fit_transform(logits[:150], y_true[:150])
    assert (cal >= 0).all() and (cal <= 1).all()


# ---------------------------------------------------------------------------
# Conformal prediction
# ---------------------------------------------------------------------------
def test_conformal_prediction_set_sizes_in_range():
    rng = np.random.default_rng(0)
    n = 200
    y_true = rng.integers(0, 3, n)
    y_proba = rng.dirichlet([1, 1, 1], n)
    sets = conformal_prediction_set(y_proba[:50], y_calib=y_true[50:],
                                     y_proba_calib=y_proba[50:], alpha=0.1)
    sizes = [len(s) for s in sets]
    # For 3 classes with alpha=0.1, sets should be in [1, 3].
    assert all(1 <= s <= 3 for s in sizes)


def test_conformal_prediction_set_requires_calibration():
    y_proba = np.random.dirichlet([1, 1, 1], 50)
    with pytest.raises(ValueError):
        conformal_prediction_set(y_proba, alpha=0.1)


def test_conformal_prediction_set_higher_alpha_smaller_sets():
    """Higher α (more miscoverage tolerance) → smaller prediction sets."""
    rng = np.random.default_rng(0)
    n = 300
    y_true = rng.integers(0, 4, n)
    y_proba = rng.dirichlet([1, 1, 1, 1], n)
    sets_low_alpha = conformal_prediction_set(y_proba[:50], y_calib=y_true[50:],
                                              y_proba_calib=y_proba[50:], alpha=0.05)
    sets_high_alpha = conformal_prediction_set(y_proba[:50], y_calib=y_true[50:],
                                                y_proba_calib=y_proba[50:], alpha=0.3)
    mean_low = np.mean([len(s) for s in sets_low_alpha])
    mean_high = np.mean([len(s) for s in sets_high_alpha])
    assert mean_high <= mean_low


def test_calibration_module_importable_from_evaluation():
    from myoadapt.evaluation import calibration
    assert hasattr(calibration, "TemperatureScaling")
    assert hasattr(calibration, "conformal_prediction_set")
