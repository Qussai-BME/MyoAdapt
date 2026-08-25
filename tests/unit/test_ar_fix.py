"""
Unit tests for the Yule-Walker AR coefficient estimator.

Validates the implementation against a synthetic AR(2) process with
known coefficients [-0.6, 0.3].
"""
import numpy as np
from myoadapt.features.time_domain import (
    ar_coefficients, extract_per_channel, AR_COEFFICIENTS_YULE_WALKER,
)


def test_ar_coefficients_yule_walker_flag():
    """Confirm the Yule-Walker implementation flag is set."""
    assert AR_COEFFICIENTS_YULE_WALKER is True


def test_ar_synthetic_recovery():
    """Recover AR(2) coefficients from a synthetic process."""
    rng = np.random.default_rng(42)
    n = 5000
    x = np.zeros(n)
    for i in range(2, n):
        x[i] = 0.6 * x[i - 1] - 0.3 * x[i - 2] + rng.standard_normal() * 0.1
    coeffs = ar_coefficients(x, order=2)
    # Should recover approximately [0.6, -0.3] (note sign convention)
    # The Yule-Walker solve gives coefficients of the AR model
    # x[n] = -a[0]*x[n-1] - a[1]*x[n-2] + noise
    # So for our generating process (0.6, -0.3), expected a = [-0.6, 0.3]
    assert len(coeffs) == 2
    assert abs(coeffs[0] + 0.6) < 0.2, f"Expected ~-0.6, got {coeffs[0]}"
    assert abs(coeffs[1] - 0.3) < 0.2, f"Expected ~0.3, got {coeffs[1]}"


def test_ar_zero_input():
    """Zero input → zero coefficients (no crash)."""
    assert ar_coefficients(np.zeros(100), order=4) == [0.0] * 4


def test_ar_short_input():
    """Short input → zero coefficients (no crash)."""
    assert ar_coefficients(np.array([1.0]), order=4) == [0.0] * 4


def test_ar_constant_input():
    """Constant input → zero coefficients."""
    assert ar_coefficients(np.ones(100), order=4) == [0.0] * 4


def test_ar_order():
    """Verify the order parameter is respected."""
    for order in [2, 4, 6, 8]:
        x = np.random.randn(500)
        coeffs = ar_coefficients(x, order=order)
        assert len(coeffs) == order


def test_extract_per_channel_has_ar():
    """Verify AR features appear in the full per-channel feature dict."""
    seg = np.random.randn(400)
    feats = extract_per_channel(seg, ar_order=4)
    assert "AR_0" in feats
    assert "AR_1" in feats
    assert "AR_2" in feats
    assert "AR_3" in feats
