"""
Unit tests for the correct Per-Subject Euclidean Alignment (Hahne 2014).

These tests verify that:
  1. After per-subject EA, every subject's per-sample Frobenius norm equals
     sqrt(n_channels) — i.e. covariances are whitened to identity.
  2. Cross-subject scale differences (2x, 4x, 10x) are eliminated.
  3. Unseen test subjects can be aligned unsupervised (no labels needed).
  4. The legacy ``EuclideanAlignment`` (average matrix) is kept for backward
     compatibility but does NOT equalize scales — that's why PerSubjectEA
     is the recommended class for the EA ablation.
"""
import numpy as np
import pytest

from myoadapt.adaptation.ea import PerSubjectEA, EuclideanAlignment
from myoadapt.data.preprocessing import compute_subject_alignment, apply_euclidean_alignment


def _make_signal(n_samples, n_channels, scale, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_samples, n_channels)) * scale


def test_per_subject_ea_whitens_each_subject():
    """After EA, every subject's per-sample norm ≈ sqrt(n_channels)."""
    n_channels = 8
    ea = PerSubjectEA()
    scales = [0.5, 1.0, 2.0, 5.0]  # 10x spread
    for i, s in enumerate(scales):
        sig = _make_signal(2000, n_channels, s, seed=i)
        ea.fit_subject(f"s{i}", sig)

    expected_norm = np.sqrt(n_channels)
    for i, s in enumerate(scales):
        sig = _make_signal(2000, n_channels, s, seed=i)  # regenerate same signal
        aligned = ea.transform_subject(f"s{i}", sig)
        per_sample_norm = np.linalg.norm(aligned) / np.sqrt(aligned.shape[0])
        assert abs(per_sample_norm - expected_norm) < 0.05, (
            f"subject s{i} (scale={s}): per-sample norm {per_sample_norm:.3f} "
            f"!= expected {expected_norm:.3f}"
        )


def test_per_subject_ea_unseen_subject_unsupervised():
    """Unseen test subject can be aligned without labels."""
    n_channels = 6
    ea = PerSubjectEA()
    # Train on subjects 0, 1
    for i in range(2):
        sig = _make_signal(2000, n_channels, scale=2.0, seed=i)
        ea.fit_subject(i, sig)
    # Test on unseen subject
    test_sig = _make_signal(2000, n_channels, scale=0.3, seed=99)
    aligned = ea.transform_subject("test_unseen", test_sig)
    per_sample_norm = np.linalg.norm(aligned) / np.sqrt(aligned.shape[0])
    expected = np.sqrt(n_channels)
    assert abs(per_sample_norm - expected) < 0.05, (
        f"unseen subject norm {per_sample_norm:.3f} != {expected:.3f}"
    )


def test_per_subject_ea_mixed_scales_ratio_close_to_one():
    """Subjects with very different scales should converge to same norm."""
    n_channels = 10
    ea = PerSubjectEA()
    sig_big = _make_signal(3000, n_channels, scale=5.0, seed=1)
    sig_small = _make_signal(3000, n_channels, scale=0.2, seed=2)  # 25x ratio
    ea.fit_subject("big", sig_big)
    ea.fit_subject("small", sig_small)
    aligned_big = ea.transform_subject("big", sig_big)
    aligned_small = ea.transform_subject("small", sig_small)
    n_big = np.linalg.norm(aligned_big) / np.sqrt(aligned_big.shape[0])
    n_small = np.linalg.norm(aligned_small) / np.sqrt(aligned_small.shape[0])
    ratio = n_big / n_small
    assert abs(ratio - 1.0) < 0.05, f"EA should equalize, ratio={ratio}"


def test_per_subject_ea_transform_signal_no_subject_id():
    """transform_signal() with no subject_id computes alignment on-the-fly."""
    n_channels = 6
    ea = PerSubjectEA()
    sig = _make_signal(2000, n_channels, scale=3.0, seed=0)
    aligned = ea.transform_signal(sig)
    assert aligned.shape == sig.shape
    per_sample_norm = np.linalg.norm(aligned) / np.sqrt(aligned.shape[0])
    expected = np.sqrt(n_channels)
    assert abs(per_sample_norm - expected) < 0.05


def test_compute_subject_alignment_returns_correct_shape():
    """Alignment matrix must be (n_channels, n_channels)."""
    for n_ch in [4, 8, 12, 16]:
        sig = _make_signal(2000, n_ch, scale=1.5)
        R = compute_subject_alignment(sig)
        assert R.shape == (n_ch, n_ch)
        assert np.all(np.isfinite(R))


def test_compute_subject_alignment_singular_input():
    """Singular covariance (constant channel) must not crash."""
    n_channels = 4
    sig = np.ones((1000, n_channels))  # rank 1 → singular cov
    R = compute_subject_alignment(sig, eps=1e-3)
    assert R.shape == (n_channels, n_channels)
    assert np.all(np.isfinite(R))


def test_legacy_ea_kept_for_backward_compat():
    """Legacy EuclideanAlignment should still instantiate and run."""
    ea = EuclideanAlignment()
    sig1 = _make_signal(1000, 8, scale=2.0, seed=1)
    sig2 = _make_signal(1000, 8, scale=0.5, seed=2)
    ea.fit_from_signals([sig1, sig2])
    assert ea.R_ is not None and ea.R_.shape == (8, 8)
    aligned1 = ea.transform_signal(sig1)
    assert aligned1.shape == sig1.shape
    # Legacy EA does NOT equalize scales — that's why PerSubjectEA exists.
    # We only verify it runs without crashing.
