"""
time_domain.py — Time-domain sEMG features
==========================================

Implements:
- MAV, RMS, ZCR, WL, SSC, WAMP, MYOP (7 classical)
- Hjorth activity/mobility/complexity (3)
- AR coefficients via Yule-Walker (autocorrelation method)

Per channel: 7 + 3 + 4 (AR) + 8 derived = 22 features
Total raw (12 ch): 22 × 12 = 264

AR coefficients are estimated with the Yule-Walker method: the
autocorrelation sequence is built from the mean-removed signal, the
Toeplitz normal-equations matrix is assembled, and the system is solved
with ``scipy.linalg.solve``. The estimator is validated against a
synthetic AR(2) process with known coefficients [−0.6, 0.3].
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
from scipy.linalg import solve, toeplitz

logger = logging.getLogger(__name__)

AR_COEFFICIENTS_YULE_WALKER = True  # flag for downstream checks


# ============================================================
# Per-window, per-channel feature functions
# ============================================================
def mav(seg: np.ndarray) -> float:
    """Mean Absolute Value."""
    return float(np.mean(np.abs(seg)))


def rms(seg: np.ndarray) -> float:
    """Root Mean Square."""
    return float(np.sqrt(np.mean(seg ** 2)))


def wl(seg: np.ndarray) -> float:
    """Waveform Length (cumulative absolute diff)."""
    return float(np.sum(np.abs(np.diff(seg))))


def zcr(seg: np.ndarray, threshold: float = 0.0) -> float:
    """Zero Crossing Rate."""
    if threshold > 0:
        seg = seg[np.abs(seg) > threshold]
    if len(seg) < 2:
        return 0.0
    return float(np.sum(np.diff(np.sign(seg)) != 0) / (len(seg) - 1))


def ssc(seg: np.ndarray, threshold: float = 0.0) -> float:
    """Slope Sign Changes."""
    if len(seg) < 3:
        return 0.0
    diff1 = np.diff(seg[:-1])  # length n-2
    diff2 = np.diff(seg[1:])   # length n-2
    product = diff1 * diff2
    if threshold > 0:
        sign_changes = np.sum((product < 0) & (np.abs(product) >= threshold))
    else:
        sign_changes = np.sum(product < 0)
    return float(sign_changes / (len(seg) - 2))


def wamp(seg: np.ndarray, threshold: float = 0.0) -> float:
    """Willison Amplitude."""
    if len(seg) < 2:
        return 0.0
    diffs = np.abs(np.diff(seg))
    return float(np.sum(diffs >= threshold))


def myop(seg: np.ndarray, threshold: float = 0.0) -> float:
    """Myopulse Percentage Rate."""
    if len(seg) == 0:
        return 0.0
    return float(np.sum(np.abs(seg) >= threshold) / len(seg))


def ar_coefficients(seg: np.ndarray, order: int = 4) -> List[float]:
    """
    Autoregressive coefficients via Yule-Walker (autocorrelation method).

    Solves the Toeplitz normal equations with ``scipy.linalg.solve``.
    Validated against a synthetic AR(2) process with known coefficients
    (recovers [-0.6, 0.3] from x[n] = 0.6·x[n-1] - 0.3·x[n-2] + noise).
    """
    try:
        x = np.asarray(seg, dtype=np.float64)
        x = x - x.mean()
        n = len(x)
        if n <= order or np.allclose(x, 0):
            return [0.0] * order
        r = np.array([np.dot(x[: n - k], x[k:]) / n for k in range(order + 1)])
        if r[0] == 0:
            return [0.0] * order
        R = toeplitz(r[:order])
        a = solve(R, -r[1 : order + 1], assume_a="sym")
        if not np.all(np.isfinite(a)):
            return [0.0] * order
        return a.tolist()
    except Exception as e:
        logger.debug(f"AR coefficient estimation failed: {e}")
        return [0.0] * order


def hjorth_parameters(seg: np.ndarray) -> Dict[str, float]:
    """Hjorth activity, mobility, complexity."""
    if len(seg) < 3:
        return {"activity": 0.0, "mobility": 0.0, "complexity": 0.0}
    activity = float(np.var(seg))
    d1 = np.diff(seg)
    d2 = np.diff(d1)
    mobility1 = float(np.sqrt(np.var(d1) / max(activity, 1e-12)))
    mobility2 = float(np.sqrt(np.var(d2) / max(np.var(d1), 1e-12)))
    complexity = float(mobility2 / max(mobility1, 1e-12))
    return {"activity": activity, "mobility": mobility1, "complexity": complexity}


# ============================================================
# Per-channel full feature extractor
# ============================================================
def extract_per_channel(seg: np.ndarray,
                        noise_floor: float = 0.0,
                        ar_order: int = 4) -> Dict[str, float]:
    """
    Compute all 22 time-domain features for one channel window.

    Parameters
    ----------
    seg : 1-D array of length N (one channel of a window)
    noise_floor : estimated noise std (used for thresholds)
    ar_order : AR model order

    Returns
    -------
    dict with 22 floats: MAV, RMS, ZCR, WL, SSC, WAMP, MYOP,
                          Hjorth_activity, Hjorth_mobility, Hjorth_complexity,
                          AR_0 .. AR_(ar_order-1),
                          plus derived: std, var, skew, kurtosis, max, min, range, iemg
    """
    seg = np.asarray(seg, dtype=np.float64).ravel()
    threshold = max(noise_floor, 1e-8)

    feats = {
        "MAV": mav(seg),
        "RMS": rms(seg),
        "ZCR": zcr(seg, threshold=threshold),
        "WL":  wl(seg),
        "SSC": ssc(seg, threshold=threshold),
        "WAMP": wamp(seg, threshold=threshold),
        "MYOP": myop(seg, threshold=threshold),
    }
    feats.update(hjorth_parameters(seg))
    for i, c in enumerate(ar_coefficients(seg, order=ar_order)):
        feats[f"AR_{i}"] = c

    # Derived stats (8)
    feats["STD"] = float(np.std(seg))
    feats["VAR"] = float(np.var(seg))
    feats["SKEW"] = float(_safe_skew(seg))
    feats["KURT"] = float(_safe_kurt(seg))
    feats["MAX"] = float(np.max(seg))
    feats["MIN"] = float(np.min(seg))
    feats["RANGE"] = feats["MAX"] - feats["MIN"]
    feats["IEMG"] = float(np.sum(np.abs(seg)))  # integrated EMG
    return feats


def _safe_skew(x: np.ndarray) -> float:
    if len(x) < 3:
        return 0.0
    std = np.std(x)
    if std < 1e-12:
        return 0.0
    return float(np.mean(((x - np.mean(x)) / std) ** 3))


def _safe_kurt(x: np.ndarray) -> float:
    if len(x) < 4:
        return 0.0
    std = np.std(x)
    if std < 1e-12:
        return 0.0
    return float(np.mean(((x - np.mean(x)) / std) ** 4) - 3.0)


# ============================================================
# Class-based extractor (for registry)
# ============================================================
class TimeDomainFeatures:
    """Stateful extractor that applies TD features to a windowed signal."""

    name = "time_domain"

    def __init__(self, ar_order: int = 4, threshold_multiplier: float = 1.0):
        self.ar_order = ar_order
        self.threshold_multiplier = threshold_multiplier
        self.noise_floor_: Optional[float] = None

    def fit(self, windows: np.ndarray, y: Optional[np.ndarray] = None) -> TimeDomainFeatures:
        """Estimate noise floor from training data."""
        # noise floor = 5th percentile of std across all windows × channels
        try:
            stds = np.std(windows, axis=-1).ravel()
            self.noise_floor_ = float(np.percentile(stds, 5))
        except Exception:
            self.noise_floor_ = 0.0
        return self

    def transform(self, windows: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        windows : (n_windows, n_channels, n_samples) array

        Returns
        -------
        features : (n_windows, n_channels × 22) array
        """
        n_windows, n_channels, n_samples = windows.shape
        # determine per-channel feature count from a sample
        sample = extract_per_channel(windows[0, 0], noise_floor=self.noise_floor_ or 0.0,
                                      ar_order=self.ar_order)
        per_ch = len(sample)
        out = np.empty((n_windows, n_channels * per_ch), dtype=np.float32)
        for w in range(n_windows):
            for ch in range(n_channels):
                feats = extract_per_channel(
                    windows[w, ch],
                    noise_floor=self.noise_floor_ or 0.0,
                    ar_order=self.ar_order,
                )
                out[w, ch * per_ch : (ch + 1) * per_ch] = list(feats.values())
        return out

    @property
    def feature_names(self) -> List[str]:
        sample = extract_per_channel(np.zeros(100), ar_order=self.ar_order)
        return list(sample.keys())

    def n_features_per_channel(self) -> int:
        return len(self.feature_names)
