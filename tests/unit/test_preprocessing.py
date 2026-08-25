"""Unit tests for preprocessing (filtering, windowing, EA)."""
import numpy as np
import pytest

from myoadapt.data.preprocessing import (
    filter_signal, notch_filter, preprocess_signal,
    segment_windows, compute_alignment_matrix, apply_euclidean_alignment,
)


def test_filter_signal_shape():
    sig = np.random.randn(2000, 12).astype(np.float32)
    out = filter_signal(sig, fs=2000)
    assert out.shape == sig.shape


def test_filter_signal_univariate():
    sig = np.random.randn(2000).astype(np.float32)
    out = filter_signal(sig, fs=2000)
    assert out.shape == sig.shape


def test_notch_filter_shape():
    sig = np.random.randn(2000, 12).astype(np.float32)
    out = notch_filter(sig, fs=2000, freq=50.0)
    assert out.shape == sig.shape


def test_preprocess_signal_pipeline():
    sig = np.random.randn(2000, 12).astype(np.float32)
    out = preprocess_signal(sig, fs=2000, notch_freq=50.0)
    assert out.shape == sig.shape


def test_segment_windows_basic():
    sig = np.random.randn(1000, 12).astype(np.float32)
    windows, labels = segment_windows(sig, fs=2000, window_ms=200, increment_ms=50)
    assert windows.shape[1] == 12
    assert windows.shape[2] == 400  # 200ms * 2kHz
    # 1000 samples, 400 window, 100 step → (1000-400)/100 + 1 = 7 windows
    assert windows.shape[0] == 7
    assert labels is None


def test_segment_windows_with_labels():
    sig = np.random.randn(1000, 12).astype(np.float32)
    labels = np.zeros(1000, dtype=np.int32)
    labels[200:800] = 3
    windows, wlabels = segment_windows(sig, fs=2000, window_ms=200, increment_ms=50,
                                        labels=labels)
    assert wlabels is not None
    assert len(wlabels) == windows.shape[0]


def test_segment_windows_short_signal():
    sig = np.random.randn(100, 12).astype(np.float32)
    windows, _ = segment_windows(sig, fs=2000, window_ms=200, increment_ms=50)
    # Should pad
    assert windows.shape[0] == 1


def test_compute_alignment_matrix_shape():
    sigs = [np.random.randn(500, 12) for _ in range(3)]
    R = compute_alignment_matrix(sigs)
    assert R.shape == (12, 12)


def test_apply_alignment():
    sig = np.random.randn(100, 12)
    R = np.eye(12, dtype=np.float32)
    out = apply_euclidean_alignment(sig, R)
    assert out.shape == sig.shape
    # Identity alignment → no change
    np.testing.assert_allclose(out, sig, atol=1e-5)
