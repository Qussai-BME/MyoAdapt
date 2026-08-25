"""
time_frequency.py — Time-frequency features (Wavelets + STFT)
=============================================================

Per channel:
- Wavelet packet energy at 5 decomposition levels (db4)
- STFT mean energy in 3 frequency bands

Total: 8 per channel.

"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import pywt
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False
    logger.warning("PyWavelets not installed — TimeFrequencyFeatures disabled")


def wavelet_packet_energy(seg: np.ndarray, wavelet: str = "db4",
                           level: int = 5) -> Dict[str, float]:
    """Energy at each wavelet packet node (level 5, db4 → 32 nodes)."""
    if not HAS_PYWT:
        return {f"WPE_{i}": 0.0 for i in range(2 ** level)}
    seg = np.asarray(seg, dtype=np.float64).ravel()
    try:
        wp = pywt.WaveletPacket(data=seg, wavelet=wavelet, mode="symmetric")
        leaves = [node.data for node in wp.get_level(level, "freq")]
        # truncate/pad to 2^level
        n = 2 ** level
        leaves = leaves[:n]
        while len(leaves) < n:
            leaves.append(np.array([0.0]))
        return {f"WPE_{i}": float(np.sum(leaves[i] ** 2)) for i in range(n)}
    except Exception as e:
        logger.debug(f"Wavelet packet failed: {e}")
        return {f"WPE_{i}": 0.0 for i in range(2 ** level)}


def stft_band_energies(seg: np.ndarray, fs: int,
                        bands: tuple = ((20, 80), (80, 250), (250, 450)),
                        nperseg: int = 256) -> Dict[str, float]:
    """Mean STFT energy in N frequency bands."""
    from scipy import signal as scipy_signal
    seg = np.asarray(seg, dtype=np.float64).ravel()
    nperseg = min(nperseg, len(seg))
    if nperseg < 4:
        return {f"STFT_B{i}": 0.0 for i in range(len(bands))}
    freqs, _, Zxx = scipy_signal.stft(seg, fs=fs, nperseg=nperseg)
    mag = np.abs(Zxx) ** 2
    out = {}
    for i, (lo, hi) in enumerate(bands):
        mask = (freqs >= lo) & (freqs <= hi)
        out[f"STFT_B{i}"] = float(np.mean(mag[mask]) if mask.any() else 0.0)
    return out


class TimeFrequencyFeatures:
    """Stateful time-frequency extractor."""

    name = "time_frequency"

    def __init__(self, fs: int = 2000, wavelet: str = "db4",
                 level: int = 5, nperseg: int = 256):
        self.fs = fs
        self.wavelet = wavelet
        self.level = level
        self.nperseg = nperseg

    def fit(self, windows: np.ndarray, y: Optional[np.ndarray] = None):
        return self

    def transform(self, windows: np.ndarray) -> np.ndarray:
        n_windows, n_channels, _ = windows.shape
        sample = {**wavelet_packet_energy(np.zeros(64), self.wavelet, self.level),
                  **stft_band_energies(np.zeros(64), self.fs, nperseg=self.nperseg)}
        per_ch = len(sample)
        out = np.empty((n_windows, n_channels * per_ch), dtype=np.float32)
        for w in range(n_windows):
            for ch in range(n_channels):
                feats = {
                    **wavelet_packet_energy(windows[w, ch], self.wavelet, self.level),
                    **stft_band_energies(windows[w, ch], self.fs, nperseg=self.nperseg),
                }
                out[w, ch * per_ch : (ch + 1) * per_ch] = list(feats.values())
        return out

    @property
    def feature_names(self) -> List[str]:
        return list({**wavelet_packet_energy(np.zeros(64), self.wavelet, self.level),
                     **stft_band_energies(np.zeros(64), self.fs, nperseg=self.nperseg)}.keys())

    def n_features_per_channel(self) -> int:
        return len(self.feature_names)
