"""Unit tests for MiniROCKET verifier."""
import numpy as np
import pytest

from myoadapt.features.minirocket import (
    MiniRocketFeatures, MiniRocketVerifier, _generate_kernels, _apply_kernel,
)


def test_generate_kernels_deterministic():
    k1 = _generate_kernels(100)
    k2 = _generate_kernels(100)
    assert len(k1["weights"]) == 100
    np.testing.assert_array_equal(k1["biases"], k2["biases"])


def test_apply_kernel_basic():
    series = np.random.randn(400)
    w = np.random.randn(7)
    ppv = _apply_kernel(series, w, bias=0.0, dilation=1, padding=0)
    assert isinstance(ppv, float)
    assert 0.0 <= ppv <= 1.0


def test_minirocket_features_small():
    """Use small n_kernels for test speed."""
    windows = np.random.randn(5, 12, 400).astype(np.float32)
    mr = MiniRocketFeatures(n_kernels=50)
    mr.fit(windows)
    out = mr.transform(windows)
    assert out.shape == (5, 50 * 12)


def test_verifier_confirms_near_singularity():
    """The verifier should confirm near-singular covariance on PPV features."""
    rng = np.random.default_rng(42)
    # Generate PPV-like features: binary values, highly correlated (low rank)
    n_subj = 5
    n_windows = 50
    n_features = 100
    features_per_subject = []
    for _ in range(n_subj):
        # Generate features that are correlated (low rank = near-singular)
        base = rng.standard_normal((n_windows, 3))  # rank 3 << n_features
        proj = rng.standard_normal((3, n_features))
        # Quantize to binary PPV-like values
        feats = (base @ proj > 0).astype(np.float32)
        features_per_subject.append(feats)

    verifier = MiniRocketVerifier(n_kernels=100, epsilon=1e-10)
    diag = verifier.diagnose(features_per_subject)
    # Should detect high condition number (binary rank-deficient features)
    assert diag["condition_number_global"] > 1e2  # at least 100
    assert diag["n_subjects"] == n_subj


def test_verdict_text():
    rng = np.random.default_rng(42)
    features = [rng.standard_normal((50, 100)) for _ in range(3)]
    verifier = MiniRocketVerifier(n_kernels=50)
    diag = verifier.diagnose(features)
    assert diag["verdict"] in [
        "CONFIRMED — near-singular covariance",
        "REFUTED — covariance is well-conditioned",
    ]
