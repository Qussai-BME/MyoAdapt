"""
Unit tests for the statistical-rigor suite additions.

Covers:
- Hedges' g vs Cohen's d (small-N correction)
- interpret_effect_size thresholds
- BCa bootstrap CI (returns sensible bounds, includes z0/a)
- paired_wilcoxon_test (single pair, with effect size r)
- power_analysis_paired_ttest (monotone in N and effect size)
- minimum_sample_size_paired (returns N at target power)
- Critical Difference diagram (figure renders, CD value sane)
- per_fold_boxplot, per_class_heatmap, reliability_diagram
"""
from __future__ import annotations

import numpy as np
import pytest

from myoadapt.evaluation.statistics import (
    hedges_g, interpret_effect_size, bootstrap_ci_bca,
    paired_wilcoxon_test, power_analysis_paired_ttest,
    minimum_sample_size_paired,
)
from myoadapt.evaluation.diagrams import (
    critical_difference_diagram, per_fold_boxplot, per_class_heatmap,
    reliability_diagram,
)


# ---------------------------------------------------------------------------
# Hedges' g
# ---------------------------------------------------------------------------
def test_hedges_g_zero_for_identical_samples():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert hedges_g(a, a) == 0.0


def test_hedges_g_smaller_in_magnitude_than_cohen_d():
    """Hedges' g applies J < 1 correction, so |g| <= |d|."""
    from myoadapt.evaluation.statistics import cohen_d
    rng = np.random.default_rng(42)
    a = rng.normal(0.5, 1.0, 8)
    b = rng.normal(0.0, 1.0, 8)
    assert abs(hedges_g(a, b)) <= abs(cohen_d(a, b)) + 1e-9


def test_hedges_g_too_few_samples_returns_zero():
    assert hedges_g(np.array([1.0]), np.array([2.0])) == 0.0


def test_interpret_effect_size_thresholds():
    assert interpret_effect_size(0.0) == "negligible"
    assert interpret_effect_size(0.19) == "negligible"
    assert interpret_effect_size(0.2) == "small"
    assert interpret_effect_size(0.49) == "small"
    assert interpret_effect_size(0.5) == "medium"
    assert interpret_effect_size(0.79) == "medium"
    assert interpret_effect_size(0.8) == "large"
    assert interpret_effect_size(2.0) == "large"
    # Sign-insensitive.
    assert interpret_effect_size(-0.6) == "medium"


# ---------------------------------------------------------------------------
# BCa bootstrap
# ---------------------------------------------------------------------------
def test_bca_ci_contains_mean():
    rng = np.random.default_rng(0)
    data = rng.normal(0.5, 0.1, 100).tolist()
    ci = bootstrap_ci_bca(data, n_bootstrap=500, random_state=0)
    assert ci["lower"] < 0.5 < ci["upper"]
    assert "z0_bias" in ci
    assert "acceleration" in ci
    assert ci["n_bootstrap"] == 500


def test_bca_ci_single_sample_returns_nan_bounds():
    ci = bootstrap_ci_bca([0.5])
    assert ci["point"] == 0.5
    assert np.isnan(ci["lower"])
    assert np.isnan(ci["upper"])


def test_bca_ci_shrinks_with_more_data():
    """With 4x more data, CI should be roughly half as wide."""
    rng = np.random.default_rng(0)
    small = rng.normal(0.5, 1.0, 25).tolist()
    large = rng.normal(0.5, 1.0, 400).tolist()
    ci_small = bootstrap_ci_bca(small, n_bootstrap=300, random_state=0)
    ci_large = bootstrap_ci_bca(large, n_bootstrap=300, random_state=0)
    width_small = ci_small["upper"] - ci_small["lower"]
    width_large = ci_large["upper"] - ci_large["lower"]
    assert width_large < width_small


# ---------------------------------------------------------------------------
# Paired Wilcoxon
# ---------------------------------------------------------------------------
def test_paired_wilcoxon_too_few_returns_p1():
    a = np.array([1.0, 2.0])
    b = np.array([2.0, 1.0])
    r = paired_wilcoxon_test(a, b)
    assert r["p_value"] == 1.0
    assert r["effect_size_r"] == 0.0


def test_paired_wilcoxon_returns_effect_size_in_range():
    rng = np.random.default_rng(0)
    a = rng.normal(0.7, 0.05, 30)
    b = rng.normal(0.65, 0.05, 30)
    r = paired_wilcoxon_test(a, b)
    assert -1.0 <= r["effect_size_r"] <= 1.0
    assert r["n"] == 30


# ---------------------------------------------------------------------------
# Power analysis
# ---------------------------------------------------------------------------
def test_power_monotone_in_n():
    """For fixed effect size, power increases with n."""
    p_low = power_analysis_paired_ttest(0.5, 10)
    p_high = power_analysis_paired_ttest(0.5, 50)
    assert p_high > p_low


def test_power_monotone_in_effect_size():
    """For fixed n, power increases with effect size."""
    p_small = power_analysis_paired_ttest(0.2, 30)
    p_large = power_analysis_paired_ttest(1.0, 30)
    assert p_large > p_small


def test_minimum_sample_size_decreases_with_larger_effect():
    """Larger effect sizes require fewer samples for the same power."""
    n_small_effect = minimum_sample_size_paired(0.2, target_power=0.8)
    n_large_effect = minimum_sample_size_paired(1.0, target_power=0.8)
    assert n_large_effect < n_small_effect


def test_minimum_sample_size_returns_at_least_3():
    """Statistical power is undefined for n < 3."""
    n = minimum_sample_size_paired(2.0, target_power=0.8)
    assert n >= 3


# ---------------------------------------------------------------------------
# Diagrams
# ---------------------------------------------------------------------------
def test_critical_difference_diagram_returns_figure():
    import matplotlib
    matplotlib.use("Agg")
    rng = np.random.default_rng(0)
    folds = {
        "RF": rng.normal(0.7, 0.05, 12).tolist(),
        "LDA": rng.normal(0.65, 0.06, 12).tolist(),
        "XGB": rng.normal(0.71, 0.04, 12).tolist(),
    }
    fig = critical_difference_diagram(folds)
    assert fig is not None
    # Figure should have at least one axes with content.
    assert len(fig.axes) >= 1


def test_critical_difference_diagram_rejects_one_model():
    with pytest.raises(ValueError):
        critical_difference_diagram({"RF": [0.7, 0.6]})


def test_per_fold_boxplot_returns_figure():
    import matplotlib
    matplotlib.use("Agg")
    rng = np.random.default_rng(0)
    folds = {
        "RF": rng.normal(0.7, 0.05, 10).tolist(),
        "LDA": rng.normal(0.65, 0.06, 10).tolist(),
    }
    fig = per_fold_boxplot(folds)
    assert fig is not None


def test_per_class_heatmap_returns_figure():
    import matplotlib
    matplotlib.use("Agg")
    data = {
        "RF": {"0": 0.8, "1": 0.7, "2": 0.6},
        "LDA": {"0": 0.75, "1": 0.65, "2": 0.55},
    }
    fig = per_class_heatmap(data)
    assert fig is not None


def test_reliability_diagram_returns_figure():
    import matplotlib
    matplotlib.use("Agg")
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 3, 100)
    y_proba = rng.dirichlet([1, 1, 1], 100)
    fig = reliability_diagram(y_true, y_proba)
    assert fig is not None
