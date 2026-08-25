"""Unit tests for evaluation metrics & statistics."""
import numpy as np
import pytest

from myoadapt.evaluation.metrics import compute_metrics, per_class_report
from myoadapt.evaluation.statistics import (
    friedman_test, wilcoxon_pairwise, cohen_d,
    holm_sidak_correction, bootstrap_ci,
)


def test_compute_metrics_basic():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 2, 2])
    classes = [0, 1, 2]
    m = compute_metrics(y_true, y_pred, classes=classes, rest_class=0)
    assert 0 <= m["accuracy"] <= 1
    assert 0 <= m["macro_f1"] <= 1
    assert "inflation" in m
    assert "per_class" in m


def test_inflation_zero_when_no_rest():
    """If all predictions are correct and rest is absent, inflation should be ~0."""
    y_true = np.array([1, 2, 1, 2])
    y_pred = np.array([1, 2, 1, 2])
    classes = [1, 2]
    m = compute_metrics(y_true, y_pred, classes=classes, rest_class="rest")
    assert abs(m["inflation"]) < 1e-6


def test_inflation_positive_with_rest_dominance():
    """Rest-class dominance should produce positive inflation."""
    # 80% rest, 20% active
    y_true = np.array([0] * 80 + [1] * 20)
    # Predict rest for everything — high accuracy, terrible active recall
    y_pred = np.array([0] * 100)
    classes = [0, 1]
    m = compute_metrics(y_true, y_pred, classes=classes, rest_class=0)
    assert m["accuracy"] == 0.8
    assert m["active_only_accuracy"] == 0.0
    assert m["inflation"] == pytest.approx(0.8)


def test_friedman_test():
    scores = {
        "model_a": [0.8, 0.85, 0.9, 0.82, 0.87],
        "model_b": [0.7, 0.75, 0.8, 0.72, 0.77],
        "model_c": [0.6, 0.65, 0.7, 0.62, 0.67],
    }
    res = friedman_test(scores)
    assert res["n_models"] == 3
    assert res["n_folds"] == 5
    assert res["chi2"] > 0
    assert "mean_rank_per_model" in res


def test_wilcoxon_pairwise():
    scores = {
        "model_a": [0.8, 0.85, 0.9, 0.82, 0.87, 0.83, 0.88, 0.91, 0.84, 0.86],
        "model_b": [0.7, 0.75, 0.8, 0.72, 0.77, 0.73, 0.78, 0.81, 0.74, 0.76],
    }
    res = wilcoxon_pairwise(scores, reference="model_a")
    assert res["n_tests"] == 1
    assert "tests" in res


def test_cohen_d():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
    d = cohen_d(a, b)
    assert d < 0  # a < b on average


def test_holm_sidak():
    pvals = [0.001, 0.04, 0.03, 0.20, 0.01]
    out = holm_sidak_correction(pvals, alpha=0.05)
    assert len(out) == 5
    # The smallest p-value should be significant
    assert out[0]["significant"] is True


def test_bootstrap_ci():
    rng = np.random.default_rng(42)
    scores = rng.standard_normal(100).tolist()
    lo, hi = bootstrap_ci(scores, n_bootstrap=100)
    assert lo < hi
