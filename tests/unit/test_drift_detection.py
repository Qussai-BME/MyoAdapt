"""Unit tests for myoadapt.evaluation.drift_detection.DriftDetector."""
from __future__ import annotations

import numpy as np
import pytest

from myoadapt.evaluation.drift_detection import DriftDetector


def _reference_and_shifted(n_ref=200, n_cur=100, n_feat=5, shift=1.2, seed=0):
    rng = np.random.default_rng(seed)
    X_ref = rng.standard_normal((n_ref, n_feat))
    X_shifted = rng.standard_normal((n_cur, n_feat)) + shift
    return X_ref, X_shifted


def test_invalid_significance_level_raises():
    with pytest.raises(ValueError):
        DriftDetector(significance_level=0.0)
    with pytest.raises(ValueError):
        DriftDetector(significance_level=1.0)


def test_fit_requires_2d_input():
    with pytest.raises(ValueError):
        DriftDetector().fit(np.zeros(10))


def test_detect_before_fit_raises():
    with pytest.raises(RuntimeError):
        DriftDetector().detect_feature_drift(np.zeros((5, 3)))


def test_feature_drift_detected_on_real_shift():
    X_ref, X_shifted = _reference_and_shifted()
    det = DriftDetector(significance_level=0.05).fit(X_ref)
    result = det.detect_feature_drift(X_shifted)
    assert result["any_drift"] is True
    assert result["n_drifted"] == 5  # all 5 features shifted by +1.2
    assert result["mmd"] > 0


def test_feature_drift_not_detected_on_same_distribution():
    X_ref, _ = _reference_and_shifted()
    rng = np.random.default_rng(99)
    X_same = rng.standard_normal((100, 5))  # same distribution, no shift
    det = DriftDetector(significance_level=0.01).fit(X_ref)  # strict alpha
    result = det.detect_feature_drift(X_same)
    assert result["drift_ratio"] < 0.5  # most features should NOT flag drift


def test_prediction_drift_detects_distribution_change():
    rng = np.random.default_rng(0)
    y_ref = rng.integers(0, 3, 300)  # balanced 3-class
    y_cur = np.concatenate([np.zeros(150, dtype=int), rng.integers(1, 3, 30)])  # class-0 heavy
    det = DriftDetector().fit(rng.standard_normal((300, 4)))
    result = det.detect_prediction_drift(y_ref, y_cur)
    assert result["drifted"] is True


def test_performance_drift_detects_degradation():
    rng = np.random.default_rng(0)
    perf_ref = rng.uniform(0.80, 0.88, 20)
    perf_cur = rng.uniform(0.55, 0.65, 20)
    det = DriftDetector().fit(rng.standard_normal((50, 4)))
    result = det.detect_performance_drift(perf_ref, perf_cur)
    assert result["detected"] is True


def test_detect_all_combines_all_three_and_narrative_is_nonempty():
    X_ref, X_cur = _reference_and_shifted()
    rng = np.random.default_rng(0)
    y_ref = rng.integers(0, 3, 200)
    y_cur = rng.integers(0, 3, 100)
    scores_ref = rng.uniform(0.75, 0.85, 20)
    scores_cur = rng.uniform(0.55, 0.65, 20)
    det = DriftDetector().fit(X_ref)
    result = det.detect_all(X_current=X_cur, y_ref=y_ref, y_current=y_cur,
                             scores_ref=scores_ref, scores_current=scores_cur)
    assert set(result.keys()) == {"feature", "prediction", "performance", "summary"}
    assert result["summary"]["any_drift"] is True
    text = det.drift_narrative(result)
    assert isinstance(text, str) and len(text) > 0


def test_detect_all_without_optional_args_only_runs_feature_drift():
    X_ref, X_cur = _reference_and_shifted()
    det = DriftDetector().fit(X_ref)
    result = det.detect_all(X_current=X_cur)
    assert "feature" in result and result["feature"] is not None
    assert result["prediction"] is None
    assert result["performance"] is None


def test_single_feature_1d_input_handled_correctly():
    """Regression test: _mmd_squared used to misinterpret a flat 1-D
    single-feature array as (1, n) — one sample across n features —
    instead of (n, 1). Fixed alongside the same bug in FatigueTracker."""
    rng = np.random.default_rng(0)
    ref_1d = rng.standard_normal(200)
    cur_1d = rng.standard_normal(100) + 2.0
    det = DriftDetector().fit(ref_1d.reshape(-1, 1))
    result = det.detect_feature_drift(cur_1d.reshape(-1, 1))
    assert result["any_drift"] is True
