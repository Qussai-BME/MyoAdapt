"""
correlation.py — Inter-channel correlation features
===================================================

For N channels, computes pairwise Pearson correlation: C(N,2) features.
For 12 channels: 66 features.

"""
from __future__ import annotations

from typing import List, Optional

import numpy as np


def inter_channel_correlation(window: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    window : (n_channels, n_samples) array

    Returns
    -------
    corr : (n_channels*(n_channels-1)//2,) array of upper-triangular
           pairwise Pearson correlations.
    """
    n_channels = window.shape[0]
    if n_channels < 2:
        return np.zeros(0, dtype=np.float32)
    # Normalize each channel
    centered = window - window.mean(axis=1, keepdims=True)
    stds = window.std(axis=1, keepdims=True)
    stds[stds < 1e-12] = 1.0
    normalized = centered / stds
    # Correlation matrix
    corr_matrix = (normalized @ normalized.T) / window.shape[1]
    # Upper-triangular (excluding diagonal)
    iu = np.triu_indices(n_channels, k=1)
    return corr_matrix[iu].astype(np.float32)


class CorrelationFeatures:
    """Stateful inter-channel correlation extractor."""

    name = "correlation"

    def __init__(self):
        self._n_channels_seen: Optional[int] = None

    def fit(self, windows: np.ndarray, y: Optional[np.ndarray] = None):
        if windows.ndim == 3:
            self._n_channels_seen = int(windows.shape[1])
        return self

    def transform(self, windows: np.ndarray) -> np.ndarray:
        n_windows, n_channels, _ = windows.shape
        self._n_channels_seen = int(n_channels)
        n_pairs = n_channels * (n_channels - 1) // 2
        out = np.empty((n_windows, n_pairs), dtype=np.float32)
        for w in range(n_windows):
            out[w] = inter_channel_correlation(windows[w])
        return out

    @property
    def feature_names(self) -> List[str]:
        # Prefer the channel count observed during fit / transform so the
        # names line up with the actual feature dimensionality. Fall back
        # to the canonical 12-channel (66-feature) layout only when no
        # data has been seen yet.
        n = self._n_channels_seen if self._n_channels_seen is not None else 12
        n_pairs = n * (n - 1) // 2
        return [f"ICC_{i}" for i in range(n_pairs)]

    def n_features(self, n_channels: int) -> int:
        return n_channels * (n_channels - 1) // 2
