"""
intent_detection.py — Intent detection (rest vs active movement).

Distinguishes "rest" (no movement intent) from "active" (movement
intended) in real-time EMG streams. This is the gatekeeper for
prosthetic control: the device must know WHEN to act, not just
WHAT gesture.

Two approaches:
- EnergyThresholdDetector: simple, real-time, no training needed.
  Uses a rolling energy threshold (RMS or MAV) — if energy exceeds
  threshold, intent is detected; otherwise rest.
- LearnedIntentDetector: trains a binary classifier (rest vs active)
  on labeled data. More accurate but requires labeled training data.

References
----------
- ReactEMG (2025). "Real-time intent detection for zero-latency
  myoelectric control."
- MyoGestic (2025). "No-movement rejection in prosthetic control."
  Science Advances.

License: Apache 2.0
"""
from __future__ import annotations

import hashlib
import logging
import os
import pickle
import secrets
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from myoadapt.models.base import BaseModel, register_model

logger = logging.getLogger(__name__)

# scipy is a hard dependency of the package, but import lazily so
# the module still loads cleanly if the import path changes.
try:
    from scipy import signal as scipy_signal  # noqa: F401
    _HAS_SCIPY = True
except ImportError:  # pragma: no cover — scipy is a hard dep, but be safe
    _HAS_SCIPY = False

# sklearn is a hard dependency — fall back to a trivial classifier if
# something is unavailable, but never silently misreport.
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.preprocessing import StandardScaler
    _HAS_SKLEARN = True
except ImportError:  # pragma: no cover
    _HAS_SKLEARN = False


# ---------------------------------------------------------------------------
# Energy helpers
# ---------------------------------------------------------------------------
def _rms(window: np.ndarray) -> float:
    """Root-mean-square of a single EMG window (any shape)."""
    arr = np.asarray(window, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(arr ** 2)))


def _mav(window: np.ndarray) -> float:
    """Mean-absolute-value of a single EMG window (any shape)."""
    arr = np.asarray(window, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(np.mean(np.abs(arr)))


def _rolling_mean(values: Sequence[float], window: int) -> np.ndarray:
    """Causal rolling mean over a 1-D sequence.

    Output[i] = mean(values[max(0, i-window+1) : i+1]). Used by
    EnergyThresholdDetector to smooth the per-window energy time
    series so a single noisy window does not spuriously trigger
    intent detection.
    """
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    window = max(1, int(window))
    out = np.zeros(n, dtype=np.float64)
    running_sum = 0.0
    for i in range(n):
        running_sum += values[i]
        if i >= window:
            running_sum -= values[i - window]
        denom = min(i + 1, window)
        out[i] = running_sum / denom if denom > 0 else 0.0
    return out


# ---------------------------------------------------------------------------
# EnergyThresholdDetector
# ---------------------------------------------------------------------------
class EnergyThresholdDetector:
    """Real-time intent detector based on EMG energy thresholding.

    The simplest viable gatekeeper: compute the RMS (or MAV) of each
    incoming EMG window, compare it to a calibrated threshold, and
    emit ``True`` (active intent) when the *smoothed* energy exceeds
    the threshold for the current window.

    Calibration
    -----------
    Call :meth:`fit` with a sample of *rest* EMG windows (any shape
    per window — they are flattened internally). The threshold is set
    to ``mean_rms + n_std * std_rms`` of the rest distribution
    (default ``n_std=3``). Equivalently: a 3-sigma upper bound on
    the rest-energy distribution.

    Real-time use
    -------------
    Feed each new window to :meth:`detect` (single window) or batch
    them through :meth:`detect_stream`. The optional ``smoothing``
    parameter applies a causal rolling mean to the per-window energy
    series, eliminating flicker on the intent/rest boundary.

    Parameters
    ----------
    threshold : float (default 0.05)
        Energy threshold (in the same units as the EMG signal). Used
        directly if ``fit`` is never called.
    window_ms : int (default 200)
        Window length in milliseconds — informational only; the
        detector itself operates on already-windowed inputs.
    fs : int (default 2000)
        Sampling rate in Hz — informational, used for logging and
        for the rolling window length.
    smoothing : int (default 5)
        Length (in windows) of the causal rolling-mean smoother applied
        to the per-window energy time series.
    metric : 'rms' or 'mav' (default 'rms')
        Energy metric to use.
    n_std : float (default 3.0)
        Number of standard deviations above the rest mean used by
        :meth:`fit` to set the threshold.

    References
    ----------
    - ReactEMG (2025). "Real-time intent detection for zero-latency
      myoelectric control."
    """

    def __init__(self,
                 threshold: float = 0.05,
                 window_ms: int = 200,
                 fs: int = 2000,
                 smoothing: int = 5,
                 metric: str = "rms",
                 n_std: float = 3.0):
        if metric not in ("rms", "mav"):
            raise ValueError(f"metric must be 'rms' or 'mav' (got {metric!r})")
        self.threshold = float(threshold)
        self.window_ms = int(window_ms)
        self.fs = int(fs)
        self.smoothing = max(1, int(smoothing))
        self.metric = metric
        self.n_std = float(n_std)
        # Calibration state — populated by fit().
        self.rest_mean_: Optional[float] = None
        self.rest_std_: Optional[float] = None
        self.calibrated_: bool = False
        # Rolling-energy buffer used by detect_stream() and by repeated
        # detect() calls so smoothing is applied consistently.
        self._energy_buffer: List[float] = []

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------
    def fit(self, X_rest: np.ndarray) -> EnergyThresholdDetector:
        """Calibrate the threshold from rest data.

        Parameters
        ----------
        X_rest : array-like
            Either ``(n_windows, n_samples, n_channels)`` raw EMG
            windows, or ``(n_windows, n_features)`` already-extracted
            features. Each row/window is treated as one observation.

        Sets
        ----
        - ``rest_mean_``, ``rest_std_`` — mean / std of per-window energy.
        - ``threshold`` — set to ``rest_mean_ + n_std * rest_std_``.
        """
        X_rest = np.asarray(X_rest, dtype=np.float64)
        if X_rest.size == 0:
            raise ValueError("X_rest is empty — cannot calibrate threshold")
        if X_rest.ndim < 2:
            X_rest = X_rest.reshape(1, -1)
        # Per-window energy (RMS or MAV of the whole window).
        energies = np.array([self._energy(w) for w in X_rest], dtype=np.float64)
        self.rest_mean_ = float(np.mean(energies))
        self.rest_std_ = float(np.std(energies, ddof=1)) if len(energies) > 1 else 0.0
        # Guard against a degenerate (zero-variance) rest distribution.
        if not np.isfinite(self.rest_std_) or self.rest_std_ < 1e-12:
            self.rest_std_ = max(self.rest_mean_ * 0.1, 1e-6)
        self.threshold = float(self.rest_mean_ + self.n_std * self.rest_std_)
        self.calibrated_ = True
        logger.info(
            f"EnergyThresholdDetector calibrated: rest_mean={self.rest_mean_:.6f} "
            f"rest_std={self.rest_std_:.6f} threshold={self.threshold:.6f} "
            f"(n={len(energies)} rest windows, metric={self.metric})"
        )
        return self

    def set_threshold(self, threshold: float) -> EnergyThresholdDetector:
        """Manual threshold override (bypasses fit())."""
        if threshold < 0:
            raise ValueError(f"threshold must be >= 0 (got {threshold})")
        self.threshold = float(threshold)
        logger.info(f"EnergyThresholdDetector threshold manually set to {self.threshold:.6f}")
        return self

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------
    def _energy(self, window: np.ndarray) -> float:
        if self.metric == "rms":
            return _rms(window)
        return _mav(window)

    def detect(self, window: np.ndarray) -> bool:
        """Decide intent for a single EMG window.

        Returns ``True`` if the (optionally smoothed) energy of
        ``window`` exceeds ``self.threshold`` — i.e. movement intent is
        present — and ``False`` otherwise (rest).

        Parameters
        ----------
        window : array-like
            Single EMG window of any shape (samples × channels, or a
            pre-extracted feature vector). Flattened internally.

        Notes
        -----
        When ``smoothing > 1`` and a sequence of ``detect`` calls is
        made, each call appends its raw energy to an internal rolling
        buffer; the detector compares the *smoothed* energy (causal
        rolling mean) to the threshold. Call :meth:`reset` between
        unrelated streams to clear the buffer.
        """
        e = self._energy(window)
        self._energy_buffer.append(e)
        if len(self._energy_buffer) > self.smoothing:
            self._energy_buffer = self._energy_buffer[-self.smoothing:]
        smoothed = float(np.mean(self._energy_buffer))
        return bool(smoothed > self.threshold)

    def detect_stream(self, windows: Iterable[np.ndarray]) -> np.ndarray:
        """Vectorised intent detection over a stream of windows.

        Equivalent to calling :meth:`detect` on each window in
        sequence (so the causal smoothing is applied across the
        stream), but returns a single NumPy boolean array.
        """
        windows_list = list(windows)
        out = np.zeros(len(windows_list), dtype=bool)
        for i, w in enumerate(windows_list):
            out[i] = self.detect(w)
        return out

    def reset(self) -> EnergyThresholdDetector:
        """Clear the rolling-energy buffer between unrelated streams."""
        self._energy_buffer = []
        return self

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
        """Compute detection metrics on labelled (X, y) data.

        Parameters
        ----------
        X : array-like
            ``(n_windows, ...)`` EMG windows.
        y_true : array-like
            Binary labels — 0/False = rest, 1/True = active.

        Returns
        -------
        dict with ``tpr`` (recall on active), ``fpr`` (false-positive
        rate on rest), ``f1``, ``accuracy``, ``precision``, plus
        ``n_samples`` and ``threshold``.
        """
        X = np.asarray(X, dtype=np.float64)
        y_true = np.asarray(y_true).astype(int).ravel()
        if X.ndim < 2:
            X = X.reshape(1, -1) if X.size else X.reshape(0, 0)
        if len(y_true) != len(X):
            raise ValueError(
                f"len(X)={len(X)} != len(y_true)={len(y_true)}"
            )
        # Reset the smoothing buffer so detect_stream starts fresh.
        self.reset()
        y_pred = self.detect_stream(list(X)).astype(int)

        # Confusion matrix → tn, fp, fn, tp (binary 0=rest, 1=active).
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        tpr = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        f1 = float(2 * prec * tpr / (prec + tpr)) if (prec + tpr) > 0 else 0.0
        acc = float((tp + tn) / max(1, tp + tn + fp + fn))
        out: Dict[str, float] = {
            "tpr": tpr,
            "fpr": fpr,
            "f1": f1,
            "accuracy": acc,
            "precision": prec,
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "n_samples": int(len(y_true)),
            "threshold": float(self.threshold),
        }
        if len(np.unique(y_true)) == 2:
            # AUC over the (smoothed) energy series, not the binary
            # decision — gives a threshold-free quality estimate.
            energies = np.array([self._energy(w) for w in X], dtype=np.float64)
            try:
                out["auc"] = float(roc_auc_score(y_true, energies))
            except Exception:  # pragma: no cover — AUC undefined for single-class
                out["auc"] = float("nan")
        logger.info(
            f"EnergyThresholdDetector evaluate: acc={acc:.4f} "
            f"tpr={tpr:.4f} fpr={fpr:.4f} f1={f1:.4f} "
            f"(threshold={self.threshold:.4f}, n={len(y_true)})"
        )
        return out


# ---------------------------------------------------------------------------
# Feature extractor for LearnedIntentDetector
# ---------------------------------------------------------------------------
def _energy_features(X: np.ndarray, fs: int) -> np.ndarray:
    """Per-window EMG energy features for the learned detector.

    Extracts, for each window:

    - per-channel RMS (n_channels features)
    - per-channel MAV (n_channels features)
    - per-channel WL (waveform length, n_channels features)
    - per-channel ZC (zero crossings, n_channels features)
    - global mean / std / max RMS across channels (3 features)

    Input ``X`` may be:

    - 2-D ``(n_windows, n_features)`` — assumed already-feature; passed
      through unchanged. The caller is responsible for shape.
    - 3-D ``(n_windows, n_samples, n_channels)`` raw EMG windows.

    Output is always 2-D ``(n_windows, n_features)``.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 2:
        # Already-feature path: pass through.
        return X
    if X.ndim != 3:
        raise ValueError(
            f"_energy_features expects 2-D or 3-D input (got {X.ndim}-D)"
        )
    n_windows, n_samples, n_channels = X.shape
    feats: List[np.ndarray] = []
    # Per-channel features.
    rms_per = np.sqrt(np.mean(X ** 2, axis=1))                 # (n_windows, n_channels)
    mav_per = np.mean(np.abs(X), axis=1)                       # (n_windows, n_channels)
    wl_per = np.sum(np.abs(np.diff(X, axis=1)), axis=1)        # (n_windows, n_channels)
    # Zero crossings with a small dead-zone to reject noise around 0.
    dead_zone = 1e-3 * max(1.0, float(np.std(X)))
    sign = np.sign(X)
    sign[np.abs(X) < dead_zone] = 0
    zc_per = np.sum(np.abs(np.diff(sign, axis=1)) > 0, axis=1)  # (n_windows, n_channels)
    feats.extend([rms_per, mav_per, wl_per, zc_per.astype(np.float64)])
    # Global aggregate features.
    g_mean = rms_per.mean(axis=1, keepdims=True)
    g_std = rms_per.std(axis=1, keepdims=True)
    g_max = rms_per.max(axis=1, keepdims=True)
    feats.extend([g_mean, g_std, g_max])
    return np.concatenate(feats, axis=1)


# ---------------------------------------------------------------------------
# LearnedIntentDetector
# ---------------------------------------------------------------------------
@register_model("intent_detector")
class LearnedIntentDetector(BaseModel):
    """Binary classifier distinguishing rest (0) from active (1).

    Wraps an sklearn classifier on top of EMG energy features extracted
    from raw windows. More accurate than the energy threshold but
    requires labelled training data (both rest and active).

    Parameters
    ----------
    n_channels : int (default 12)
        Number of EMG channels — informational, used for logging and
        to sanity-check input shape.
    n_samples : int (default 400)
        Number of samples per window — informational.
    method : 'random_forest', 'logistic' (default 'random_forest')
        Underlying sklearn classifier family.
    fs : int (default 2000)
        Sampling rate in Hz.
    n_estimators : int (default 100)
        Number of trees when ``method='random_forest'``.
    max_depth : Optional[int] (default None)
        Max tree depth (random forest only).
    C : float (default 1.0)
        Inverse regularisation strength (logistic only).
    random_state : int (default 42)
        Reproducibility seed.
    standardize : bool (default True)
        Whether to standardize features (z-score) before fitting.

    Notes
    -----
    When sklearn is not installed, the constructor raises
    ``ImportError`` at fit time (not init time) so the class can still
    be imported and introspected.
    """

    def __init__(self,
                 n_channels: int = 12,
                 n_samples: int = 400,
                 method: str = "random_forest",
                 fs: int = 2000,
                 n_estimators: int = 100,
                 max_depth: Optional[int] = None,
                 C: float = 1.0,
                 random_state: int = 42,
                 standardize: bool = True,
                 **kwargs: Any):
        self.n_channels = int(n_channels)
        self.n_samples = int(n_samples)
        self.method = method
        self.fs = int(fs)
        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.C = float(C)
        self.random_state = int(random_state)
        self.standardize = bool(standardize)
        self.extra_kwargs = kwargs
        self.model: Optional[Any] = None
        self.scaler: Optional[StandardScaler] = None
        self.classes_: Optional[np.ndarray] = None
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------
    def _build_model(self) -> Any:
        if not _HAS_SKLEARN:
            raise ImportError(
                "LearnedIntentDetector requires scikit-learn. "
                "Install with: pip install scikit-learn"
            )
        if self.method == "random_forest":
            return RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=4,
                class_weight="balanced",
            )
        if self.method == "logistic":
            return LogisticRegression(
                C=self.C, max_iter=1000,
                random_state=self.random_state,
                class_weight="balanced",
            )
        raise ValueError(f"Unknown method: {self.method!r}")

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------
    def _extract_features(self, X: np.ndarray) -> np.ndarray:
        return _energy_features(X, fs=self.fs)

    # ------------------------------------------------------------------
    # Fit / predict
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> LearnedIntentDetector:
        """Train the rest-vs-active classifier.

        Parameters
        ----------
        X : array-like
            ``(n_windows, n_samples, n_channels)`` raw EMG windows,
            or ``(n_windows, n_features)`` pre-extracted features.
        y : array-like
            Binary labels — 0 = rest, 1 = active.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).astype(int).ravel()
        if len(X) != len(y):
            raise ValueError(
                f"len(X)={len(X)} != len(y)={len(y)}"
            )
        if len(np.unique(y)) < 2:
            raise ValueError(
                "LearnedIntentDetector.fit needs both classes (0=rest, 1=active) "
                f"but got labels {np.unique(y).tolist()}"
            )
        features = self._extract_features(X)
        self.model = self._build_model()
        if self.standardize:
            self.scaler = StandardScaler()
            features = self.scaler.fit_transform(features)
        else:
            self.scaler = None
        self.model.fit(features, y)
        self.classes_ = np.unique(y)
        self._fitted = True
        logger.info(
            f"LearnedIntentDetector fitted: method={self.method}, "
            f"n_samples={len(y)}, n_features={features.shape[1]}, "
            f"classes={self.classes_.tolist()}"
        )
        return self

    def _apply_pre(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted or self.model is None:
            raise RuntimeError("LearnedIntentDetector not fitted — call fit() first")
        features = self._extract_features(X)
        if self.scaler is not None:
            features = self.scaler.transform(features)
        return features

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict rest (0) vs active (1) for each window."""
        feats = self._apply_pre(X)
        return self.model.predict(feats).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(active=1) for each window as a 1-D array."""
        feats = self._apply_pre(X)
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(feats)
            # Column index of class 1 in self.classes_.
            classes = list(self.classes_)
            if 1 in classes:
                idx = classes.index(1)
                return proba[:, idx].astype(np.float64)
            return proba[:, -1].astype(np.float64)
        # Fall back: use decision_function or hard predictions.
        if hasattr(self.model, "decision_function"):
            scores = self.model.decision_function(feats)
            # Sigmoid → probability.
            scores = np.asarray(scores, dtype=np.float64).ravel()
            return 1.0 / (1.0 + np.exp(-scores))
        preds = self.model.predict(feats).astype(int)
        return preds.astype(np.float64)

    # ------------------------------------------------------------------
    # Persistence (pickle)
    # ------------------------------------------------------------------
    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "scaler": self.scaler,
                "classes_": self.classes_,
                "config": {
                    "n_channels": self.n_channels,
                    "n_samples": self.n_samples,
                    "method": self.method,
                    "fs": self.fs,
                    "n_estimators": self.n_estimators,
                    "max_depth": self.max_depth,
                    "C": self.C,
                    "random_state": self.random_state,
                    "standardize": self.standardize,
                    "extra_kwargs": self.extra_kwargs,
                },
            }, f)
        logger.info(f"LearnedIntentDetector saved to {path}")

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        expected_sha256: Optional[str] = None,
    ) -> LearnedIntentDetector:
        path = Path(path)
        expected = expected_sha256 or os.environ.get("MYOADAPT_MODEL_SHA256")
        require_hash = os.environ.get("MYOADAPT_REQUIRE_MODEL_HASH", "").lower()
        if require_hash in {"1", "true", "yes", "on"} and not expected:
            raise RuntimeError("A trusted MYOADAPT_MODEL_SHA256 is required for this deployment")
        if expected:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if not secrets.compare_digest(digest.lower(), expected.strip().lower()):
                raise RuntimeError("Model artifact hash does not match the trusted expected SHA-256")
        with open(path, "rb") as f:
            state = pickle.load(f)  # nosec B301
        obj = cls(**state["config"])
        obj.model = state["model"]
        obj.scaler = state["scaler"]
        obj.classes_ = state["classes_"]
        obj._fitted = obj.model is not None
        logger.info(f"LearnedIntentDetector loaded from {path}")
        return obj

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------
    def count_parameters(self) -> int:
        if self.model is None:
            return 0
        if hasattr(self.model, "estimators_"):
            total = 0
            for est in getattr(self.model, "estimators_", []):
                if hasattr(est, "tree_"):
                    total += est.tree_.node_count
            return total
        if hasattr(self.model, "coef_"):
            return int(np.asarray(self.model.coef_).size)
        return 0

    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
        """Binary detection metrics on labelled (X, y) data."""
        y_true = np.asarray(y_true).astype(int).ravel()
        y_pred = self.predict(X)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        tpr = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        f1 = float(2 * prec * tpr / (prec + tpr)) if (prec + tpr) > 0 else 0.0
        acc = float((tp + tn) / max(1, tp + tn + fp + fn))
        out: Dict[str, float] = {
            "tpr": tpr, "fpr": fpr, "f1": f1, "accuracy": acc,
            "precision": prec,
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "n_samples": int(len(y_true)),
        }
        try:
            proba = self.predict_proba(X)
            out["auc"] = float(roc_auc_score(y_true, proba))
        except Exception:  # pragma: no cover — single-class test set
            out["auc"] = float("nan")
        return out


__all__ = [
    "EnergyThresholdDetector",
    "LearnedIntentDetector",
]
