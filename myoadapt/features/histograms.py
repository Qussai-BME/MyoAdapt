"""
histograms.py — Amplitude-distribution histogram features
=========================================================

Per channel: 10 normalized bin counts of the absolute amplitude.

"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


def histogram_features(seg: np.ndarray, n_bins: int = 10) -> Dict[str, float]:
    """Normalized amplitude histogram (n_bins counts)."""
    abs_seg = np.abs(seg).ravel()
    if abs_seg.max() < 1e-12:
        return {f"Hist_{i}": 0.0 for i in range(n_bins)}
    counts, _ = np.histogram(abs_seg, bins=n_bins,
                              range=(0, abs_seg.max() + 1e-9))
    total = counts.sum()
    if total == 0:
        return {f"Hist_{i}": 0.0 for i in range(n_bins)}
    return {f"Hist_{i}": float(counts[i] / total) for i in range(n_bins)}


class HistogramFeatures:
    """Stateful histogram extractor."""

    name = "histogram"

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins

    def fit(self, windows: np.ndarray, y: Optional[np.ndarray] = None):
        return self

    def transform(self, windows: np.ndarray) -> np.ndarray:
        n_windows, n_channels, _ = windows.shape
        per_ch = self.n_bins
        out = np.empty((n_windows, n_channels * per_ch), dtype=np.float32)
        for w in range(n_windows):
            for ch in range(n_channels):
                feats = histogram_features(windows[w, ch], self.n_bins)
                out[w, ch * per_ch : (ch + 1) * per_ch] = list(feats.values())
        return out

    @property
    def feature_names(self) -> List[str]:
        return [f"Hist_{i}" for i in range(self.n_bins)]

    def n_features_per_channel(self) -> int:
        return self.n_bins
