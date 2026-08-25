"""
preprocessing.py — Signal filtering, windowing, and Euclidean Alignment
========================================================================

Implements the full sEMG preprocessing chain:
  1. Butterworth bandpass (20-450 Hz, order 4) - IEEE/ISEK standard
  2. Notch filter at 50 Hz (EU) / 60 Hz (US) - power-line interference
  3. Optional Euclidean Alignment (Hahne et al. 2014) - per-subject whitening
  4. Sliding-window segmentation (default 200 ms / 50 ms increment)

All functions accept either 1-D (single channel) or 2-D (n_samples x n_channels)
input. Output preserves the input shape.

License: Apache 2.0
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt

logger = logging.getLogger(__name__)


def filter_signal(
    signal: np.ndarray,
    fs: int = 2000,
    low: float = 20.0,
    high: float = 450.0,
    order: int = 4,
    btype: str = "bandpass",
) -> np.ndarray:
    """Butterworth bandpass filter (zero-phase via filtfilt)."""
    nyq = 0.5 * fs
    if btype == "bandpass":
        low_n = max(low / nyq, 1e-6)
        high_n = min(high / nyq, 0.999999)
        sos = butter(order, [low_n, high_n], btype="bandpass", output="sos")
    elif btype == "lowpass":
        wn = min(high / nyq, 0.999999)
        sos = butter(order, wn, btype="lowpass", output="sos")
    elif btype == "highpass":
        wn = max(low / nyq, 1e-6)
        sos = butter(order, wn, btype="highpass", output="sos")
    else:
        raise ValueError(f"Unknown btype: {btype}")

    if signal.ndim == 1:
        return sosfiltfilt(sos, signal).astype(signal.dtype, copy=False)
    elif signal.ndim == 2:
        out = np.empty_like(signal, dtype=np.float64)
        for ch in range(signal.shape[1]):
            out[:, ch] = sosfiltfilt(sos, signal[:, ch])
        return out.astype(signal.dtype, copy=False)
    else:
        raise ValueError(f"signal must be 1-D or 2-D, got {signal.ndim}-D")


def notch_filter(
    signal: np.ndarray,
    fs: int = 2000,
    freq: float = 50.0,
    quality: float = 30.0,
) -> np.ndarray:
    """Second-order IIR notch filter at `freq` Hz (power-line interference)."""
    w0 = freq / (0.5 * fs)
    b, a = iirnotch(w0, quality)

    if signal.ndim == 1:
        return filtfilt(b, a, signal).astype(signal.dtype, copy=False)
    elif signal.ndim == 2:
        out = np.empty_like(signal, dtype=np.float64)
        for ch in range(signal.shape[1]):
            out[:, ch] = filtfilt(b, a, signal[:, ch])
        return out.astype(signal.dtype, copy=False)
    else:
        raise ValueError(f"signal must be 1-D or 2-D, got {signal.ndim}-D")


def preprocess_signal(
    signal: np.ndarray,
    fs: int = 2000,
    low: float = 20.0,
    high: float = 450.0,
    notch_freq: Optional[float] = 50.0,
    order: int = 4,
) -> np.ndarray:
    """Full preprocessing chain: bandpass -> notch (optional)."""
    out = filter_signal(signal, fs=fs, low=low, high=high, order=order)
    if notch_freq is not None and notch_freq > 0:
        out = notch_filter(out, fs=fs, freq=notch_freq)
    return out


def segment_windows(
    signal: np.ndarray,
    fs: int = 2000,
    window_ms: int = 200,
    increment_ms: int = 50,
    labels: Optional[np.ndarray] = None,
    label_strategy: str = "majority",
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Sliding-window segmentation.

    Returns
    -------
    windows : (n_windows, n_channels, window_samples)
    window_labels : (n_windows,) or None
    """
    win_samples = int(fs * window_ms / 1000)
    step_samples = int(fs * increment_ms / 1000)

    if signal.ndim == 1:
        signal = signal[:, np.newaxis]
    n_samples, n_channels = signal.shape

    if n_samples < win_samples:
        pad = np.zeros((win_samples - n_samples, n_channels), dtype=signal.dtype)
        signal = np.concatenate([signal, pad], axis=0)
        n_samples = signal.shape[0]
        if labels is not None:
            labels = np.concatenate([labels, np.zeros(pad.shape[0], dtype=labels.dtype)])

    starts = np.arange(0, n_samples - win_samples + 1, step_samples)
    n_windows = len(starts)

    windows = np.empty((n_windows, n_channels, win_samples), dtype=np.float32)
    for i, s in enumerate(starts):
        windows[i] = signal[s : s + win_samples].T

    window_labels = None
    if labels is not None:
        window_labels = np.empty(n_windows, dtype=labels.dtype)
        for i, s in enumerate(starts):
            chunk = labels[s : s + win_samples]
            if label_strategy == "majority":
                values, counts = np.unique(chunk, return_counts=True)
                window_labels[i] = values[np.argmax(counts)]
            elif label_strategy == "first":
                window_labels[i] = chunk[0]
            else:
                raise ValueError(f"Unknown label_strategy: {label_strategy}")

    return windows, window_labels


def compute_alignment_matrix(
    signals: List[np.ndarray],
    method: str = "hahne",
) -> np.ndarray:
    """
    Compute the Euclidean Alignment matrix R (Hahne et al. 2014).

    For each subject's covariance C_i, the whitening matrix is
    R = sqrt(C)^-1. After applying x' = R @ x, the subject's
    covariance becomes identity, removing subject-specific scale/rotation.

    Parameters
    ----------
    signals : list of (n_samples_i, n_channels) arrays - one per subject
    method  : 'hahne' (per-subject averaged, default) or 'legacy' (pooled)
    """
    if isinstance(signals, np.ndarray):
        signals = [signals]
    if len(signals) == 0:
        raise ValueError("signals must contain at least one array")

    n_channels = signals[0].shape[1]
    if method == "hahne":
        cov_sum = np.zeros((n_channels, n_channels), dtype=np.float64)
        for sig in signals:
            sig = np.asarray(sig, dtype=np.float64)
            C = (sig.T @ sig) / max(sig.shape[0] - 1, 1)
            cov_sum += C
        C_avg = cov_sum / len(signals)
    elif method == "legacy":
        pooled = np.concatenate(signals, axis=0)
        C_avg = (pooled.T @ pooled) / max(pooled.shape[0] - 1, 1)
    else:
        raise ValueError(f"Unknown method: {method}")

    eigvals, eigvecs = np.linalg.eigh(C_avg)
    eigvals = np.clip(eigvals, 1e-10, None)
    R = (eigvecs * (1.0 / np.sqrt(eigvals))) @ eigvecs.T
    return R.astype(np.float32)


def compute_subject_alignment(
    signal: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Compute the per-subject Euclidean Alignment matrix (Hahne et al. 2014).

    For a single subject's signal, compute C = (signal^T @ signal) / (n-1),
    then return R = sqrt(C)^-1 so that applying R whitens this subject's
    covariance to identity.

    Parameters
    ----------
    signal : (n_samples, n_channels) — single subject's raw signal
    eps    : small constant for numerical stability

    Returns
    -------
    R : (n_channels, n_channels) whitening matrix
    """
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim == 1:
        signal = signal[:, np.newaxis]
    n_samples, n_channels = signal.shape
    C = (signal.T @ signal) / max(n_samples - 1, 1)
    C = C + eps * np.eye(n_channels)
    eigvals, eigvecs = np.linalg.eigh(C)
    eigvals = np.clip(eigvals, eps, None)
    R = (eigvecs * (1.0 / np.sqrt(eigvals))) @ eigvecs.T
    return R.astype(np.float32)


def apply_euclidean_alignment(
    signal: np.ndarray,
    R: np.ndarray,
) -> np.ndarray:
    """Apply alignment matrix R to a signal (n_samples, n_channels)."""
    signal = np.asarray(signal, dtype=np.float32)
    R = np.asarray(R, dtype=np.float32)
    if signal.ndim == 1:
        signal = signal[:, np.newaxis]
        squeeze = True
    else:
        squeeze = False
    if R.shape != (signal.shape[1], signal.shape[1]):
        raise ValueError(
            f"R shape {R.shape} does not match n_channels={signal.shape[1]}"
        )
    out = signal @ R.T
    return out.squeeze() if squeeze else out
