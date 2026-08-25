"""
frequency_domain.py — Frequency-domain sEMG features
====================================================

Per channel:
- MNF  : Mean Frequency
- MDF  : Median Frequency
- PKF  : Peak Frequency
- SM1, SM2, SM3 : Spectral Moments 1-3
- PSR  : Power Spectrum Ratio
- SNR  : estimated signal-to-noise ratio (dB)

Total: 8 per channel.

"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
from scipy import signal as scipy_signal

logger = logging.getLogger(__name__)


def _psd(seg: np.ndarray, fs: int, nperseg: int = 256):
    """Compute one-sided PSD via Welch."""
    seg = np.asarray(seg, dtype=np.float64).ravel()
    nperseg = min(nperseg, len(seg))
    if nperseg < 4:
        return np.array([0.0]), np.array([0.0])
    freqs, psd = scipy_signal.welch(seg, fs=fs, nperseg=nperseg)
    return freqs, psd


def mean_frequency(seg: np.ndarray, fs: int, nperseg: int = 256) -> float:
    """MNF: sum(f·PSD) / sum(PSD)."""
    freqs, psd = _psd(seg, fs, nperseg)
    if psd.sum() < 1e-12:
        return 0.0
    return float(np.sum(freqs * psd) / np.sum(psd))


def median_frequency(seg: np.ndarray, fs: int, nperseg: int = 256) -> float:
    """MDF: frequency that splits PSD area into two equal halves."""
    freqs, psd = _psd(seg, fs, nperseg)
    if psd.sum() < 1e-12:
        return 0.0
    cum = np.cumsum(psd) / np.sum(psd)
    idx = np.searchsorted(cum, 0.5)
    return float(freqs[min(idx, len(freqs) - 1)])


def peak_frequency(seg: np.ndarray, fs: int, nperseg: int = 256) -> float:
    """PKF: frequency with maximum PSD."""
    freqs, psd = _psd(seg, fs, nperseg)
    if len(psd) == 0:
        return 0.0
    return float(freqs[np.argmax(psd)])


def spectral_moments(seg: np.ndarray, fs: int, nperseg: int = 256) -> Dict[str, float]:
    """SM1, SM2, SM3."""
    freqs, psd = _psd(seg, fs, nperseg)
    if psd.sum() < 1e-12:
        return {"SM1": 0.0, "SM2": 0.0, "SM3": 0.0}
    sm1 = float(np.sum(freqs * psd) / np.sum(psd))
    sm2 = float(np.sum((freqs ** 2) * psd) / np.sum(psd))
    sm3 = float(np.sum((freqs ** 3) * psd) / np.sum(psd))
    return {"SM1": sm1, "SM2": sm2, "SM3": sm3}


def power_spectrum_ratio(seg: np.ndarray, fs: int,
                          low: float = 20.0, high: float = 250.0,
                          nperseg: int = 256) -> float:
    """PSR: power in [low, high] / total power."""
    freqs, psd = _psd(seg, fs, nperseg)
    if psd.sum() < 1e-12:
        return 0.0
    in_band = (freqs >= low) & (freqs <= high)
    return float(np.sum(psd[in_band]) / np.sum(psd))


def snr_db(seg: np.ndarray, fs: int,
           signal_band: tuple = (20.0, 450.0),
           noise_band: tuple = (550.0, 950.0),
           nperseg: int = 256) -> float:
    """Estimate SNR (dB) as signal-band power / noise-band power."""
    freqs, psd = _psd(seg, fs, nperseg)
    if psd.sum() < 1e-12:
        return 0.0
    s_mask = (freqs >= signal_band[0]) & (freqs <= signal_band[1])
    n_mask = (freqs >= noise_band[0]) & (freqs <= noise_band[1])
    p_sig = np.sum(psd[s_mask])
    p_noise = np.sum(psd[n_mask])
    if p_noise < 1e-15:
        return 60.0
    return float(10 * np.log10(p_sig / p_noise))


def extract_per_channel_freq(seg: np.ndarray, fs: int,
                              nperseg: int = 256) -> Dict[str, float]:
    """All 8 frequency-domain features for one channel."""
    feats = {
        "MNF": mean_frequency(seg, fs, nperseg),
        "MDF": median_frequency(seg, fs, nperseg),
        "PKF": peak_frequency(seg, fs, nperseg),
        "PSR": power_spectrum_ratio(seg, fs, nperseg=nperseg),
        "SNR": snr_db(seg, fs, nperseg=nperseg),
    }
    feats.update(spectral_moments(seg, fs, nperseg))
    return feats


class FrequencyFeatures:
    """Stateful frequency-domain extractor."""

    name = "frequency_domain"

    def __init__(self, fs: int = 2000, nperseg: int = 256):
        self.fs = fs
        self.nperseg = nperseg

    def fit(self, windows: np.ndarray, y: Optional[np.ndarray] = None):
        return self

    def transform(self, windows: np.ndarray) -> np.ndarray:
        n_windows, n_channels, _ = windows.shape
        sample = extract_per_channel_freq(windows[0, 0], self.fs, self.nperseg)
        per_ch = len(sample)
        out = np.empty((n_windows, n_channels * per_ch), dtype=np.float32)
        for w in range(n_windows):
            for ch in range(n_channels):
                feats = extract_per_channel_freq(windows[w, ch], self.fs, self.nperseg)
                out[w, ch * per_ch : (ch + 1) * per_ch] = list(feats.values())
        return out

    @property
    def feature_names(self) -> List[str]:
        return list(extract_per_channel_freq(np.zeros(100), self.fs, self.nperseg).keys())

    def n_features_per_channel(self) -> int:
        return len(self.feature_names)
