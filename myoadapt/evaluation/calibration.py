"""
calibration.py — Probability calibration & uncertainty quantification
=====================================================================

Provides the calibration / uncertainty tooling that safety-critical
EMG applications (prosthetics, surgical-robotics control) and EU AI
Act Article 13 transparency obligations require.

Three calibration strategies:
- ``TemperatureScaling``  — single-parameter, preserves the ranking of
  logits (Guo et al. 2017). Recommended default for neural networks.
- ``PlattCalibration``    — sigmoid fit on logits (Platt 1999).
  Equivalent to logistic regression on the decision score.
- ``IsotonicCalibration`` — non-parametric monotonic fit (Zadrozny &
  Elkan 2002). Most flexible but requires more data.

Plus:
- ``expected_calibration_error`` — ECE, the canonical scalar metric.
- ``maximum_calibration_error``   — MCE, worst-bin gap.
- ``brier_score``                — scalar accuracy+calibration combo.
- ``conformal_prediction_set``   — distribution-free prediction set
  with finite-sample coverage guarantee (Vovk 2005; Angelopoulos 2023).

The reliability-diagram figure companion lives in
:mod:`myoadapt.evaluation.diagrams`.

References
----------
- Guo, C. et al. (2017). *On Calibration of Modern Neural Networks.*
  ICML.
- Platt, J. (1999). *Probabilistic Outputs for Support Vector Machines.*
- Zadrozny, B. & Elkan, C. (2002). *Transforming Classifier Scores
  into Accurate Multiclass Probability Estimates.* KDD.
- Angelopoulos, A. N. & Bates, S. (2023). *Conformal Prediction: A
  Gentle Introduction.* Foundations and Trends in ML.

License: Apache 2.0
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------
def expected_calibration_error(y_true: np.ndarray,
                                y_proba: np.ndarray,
                                n_bins: int = 15) -> float:
    """Expected Calibration Error (ECE).

    Bins predictions by confidence and computes the weighted average
    absolute gap between confidence and accuracy in each bin.

    Parameters
    ----------
    y_true : (n,) integer class labels
    y_proba : (n, n_classes) or (n,) predicted probabilities. When 1-D,
        interpreted as P(class=1) for a binary problem.
    n_bins : number of equal-width bins on the confidence axis [0, 1].
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    if y_proba.ndim == 1:
        confidences = y_proba
        correct = (y_true == 1).astype(float)
    else:
        confidences = y_proba.max(axis=1)
        preds = y_proba.argmax(axis=1)
        correct = (preds == y_true).astype(float)
    n = len(confidences)
    if n == 0:
        return 0.0
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
        if i == n_bins - 1:
            mask = mask | (confidences == bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = correct[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def maximum_calibration_error(y_true: np.ndarray,
                               y_proba: np.ndarray,
                               n_bins: int = 15) -> float:
    """Maximum Calibration Error (MCE) — worst-bin gap."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    if y_proba.ndim == 1:
        confidences = y_proba
        correct = (y_true == 1).astype(float)
    else:
        confidences = y_proba.max(axis=1)
        preds = y_proba.argmax(axis=1)
        correct = (preds == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    mce = 0.0
    for i in range(n_bins):
        mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
        if i == n_bins - 1:
            mask = mask | (confidences == bins[i + 1])
        if mask.sum() == 0:
            continue
        gap = abs(correct[mask].mean() - confidences[mask].mean())
        mce = max(mce, gap)
    return float(mce)


def brier_score(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Multi-class Brier score (lower is better; 0 = perfect).

    For multi-class problems, the Brier score is the mean squared error
    between the one-hot label encoding and the predicted probability
    vector, averaged over classes.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=np.float64)
    if y_proba.ndim == 1:
        # Binary case.
        p1 = y_proba
        one_hot = (y_true == 1).astype(float)
        return float(np.mean((p1 - one_hot) ** 2))
    n_classes = y_proba.shape[1]
    one_hot = np.zeros_like(y_proba)
    one_hot[np.arange(len(y_true)), y_true.astype(int)] = 1.0
    return float(np.mean(np.sum((y_proba - one_hot) ** 2, axis=1) / n_classes))


# ---------------------------------------------------------------------------
# Temperature scaling (Guo et al. 2017)
# ---------------------------------------------------------------------------
class TemperatureScaling:
    """Single-parameter temperature scaling for neural-network logits.

    Fits a scalar ``T > 0`` on a held-out validation set so that
    ``softmax(logits / T)`` is calibrated. Preserves the ranking of
    logits (and hence the argmax / accuracy).

    Parameters
    ----------
    max_iter : int (default 100)
        Maximum number of L-BFGS-B iterations.
    init_temperature : float (default 1.5)
        Starting value for ``T``. Guo et al. recommend > 1 because
        modern NNs are typically over-confident.
    """

    def __init__(self, max_iter: int = 100, init_temperature: float = 1.5):
        self.max_iter = max_iter
        self.init_temperature = init_temperature
        self.temperature_: float = 1.0
        self.fitted_: bool = False

    def _nll(self, T: float, logits: np.ndarray, y: np.ndarray) -> float:
        from scipy.special import logsumexp
        scaled = logits / max(T, 1e-6)
        log_z = logsumexp(scaled, axis=1, keepdims=True)
        log_probs = scaled - log_z
        return float(-np.mean(log_probs[np.arange(len(y)), y]))

    def fit(self, logits: np.ndarray, y: np.ndarray) -> TemperatureScaling:
        """Fit ``T`` on validation logits and labels.

        Parameters
        ----------
        logits : (n, n_classes) pre-softmax logits.
        y : (n,) integer class labels in ``[0, n_classes)``.
        """
        from scipy.optimize import minimize
        logits = np.asarray(logits, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        # L-BFGS-B with positivity constraint on T.
        result = minimize(
            fun=lambda T: self._nll(float(T[0]), logits, y),
            x0=np.array([self.init_temperature], dtype=np.float64),
            method="L-BFGS-B",
            bounds=[(1e-3, 100.0)],
            options={"maxiter": self.max_iter, "ftol": 1e-7},
        )
        self.temperature_ = float(result.x[0])
        self.fitted_ = True
        logger.info(f"TemperatureScaling fitted: T={self.temperature_:.4f}")
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        """Apply temperature scaling to logits, returning probabilities."""
        if not self.fitted_:
            raise RuntimeError("TemperatureScaling not fitted")
        from scipy.special import softmax
        return softmax(np.asarray(logits, dtype=np.float64) / self.temperature_, axis=1)

    def fit_transform(self, logits: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.fit(logits, y).transform(logits)


# ---------------------------------------------------------------------------
# Platt sigmoid calibration (binary, then one-vs-rest for multiclass)
# ---------------------------------------------------------------------------
class PlattCalibration:
    """Platt (1999) sigmoid calibration via logistic regression on logits.

    Multiclass is handled one-vs-rest: for each class ``c`` a binary
    logistic regression is fit on ``logit_c`` vs the rest. The
    resulting per-class probabilities are renormalised to sum to 1.
    """

    def __init__(self, C: float = 1.0, max_iter: int = 1000):
        self.C = C
        self.max_iter = max_iter
        self.models_: List[LogisticRegression] = []
        self.fitted_: bool = False

    def fit(self, logits: np.ndarray, y: np.ndarray) -> PlattCalibration:
        logits = np.asarray(logits, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        n_classes = logits.shape[1]
        self.models_ = []
        for c in range(n_classes):
            lr = LogisticRegression(C=self.C, max_iter=self.max_iter)
            y_bin = (y == c).astype(int)
            # Fit on the single-column logit for class c.
            lr.fit(logits[:, c:c + 1], y_bin)
            self.models_.append(lr)
        self.fitted_ = True
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("PlattCalibration not fitted")
        logits = np.asarray(logits, dtype=np.float64)
        n_samples, n_classes = logits.shape
        probs = np.zeros((n_samples, n_classes), dtype=np.float64)
        for c, lr in enumerate(self.models_):
            probs[:, c] = lr.predict_proba(logits[:, c:c + 1])[:, 1]
        # Renormalise so rows sum to 1.
        row_sums = probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return probs / row_sums

    def fit_transform(self, logits: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.fit(logits, y).transform(logits)


# ---------------------------------------------------------------------------
# Isotonic regression calibration (non-parametric)
# ---------------------------------------------------------------------------
class IsotonicCalibration:
    """Isotonic-regression calibration (Zadrozny & Elkan 2002).

    Multiclass is handled one-vs-rest, same as :class:`PlattCalibration`.
    Isotonic is the most flexible calibrator but needs more data —
    a rule of thumb is at least 1000 calibration samples per class.
    """

    def __init__(self, y_min: float = 0.0, y_max: float = 1.0):
        self.y_min = y_min
        self.y_max = y_max
        self.models_: List[IsotonicRegression] = []
        self.fitted_: bool = False

    def fit(self, logits: np.ndarray, y: np.ndarray) -> IsotonicCalibration:
        logits = np.asarray(logits, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        n_classes = logits.shape[1]
        self.models_ = []
        for c in range(n_classes):
            ir = IsotonicRegression(y_min=self.y_min, y_max=self.y_max,
                                     out_of_bounds="clip")
            y_bin = (y == c).astype(float)
            ir.fit(logits[:, c], y_bin)
            self.models_.append(ir)
        self.fitted_ = True
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("IsotonicCalibration not fitted")
        logits = np.asarray(logits, dtype=np.float64)
        n_samples, n_classes = logits.shape
        probs = np.zeros((n_samples, n_classes), dtype=np.float64)
        for c, ir in enumerate(self.models_):
            probs[:, c] = ir.predict(logits[:, c])
        # Renormalise.
        row_sums = probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return probs / row_sums

    def fit_transform(self, logits: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.fit(logits, y).transform(logits)


# ---------------------------------------------------------------------------
# Conformal prediction (distribution-free finite-sample coverage)
# ---------------------------------------------------------------------------
def conformal_prediction_set(
    y_proba: np.ndarray,
    calibration_scores: Optional[np.ndarray] = None,
    y_calib: Optional[np.ndarray] = None,
    y_proba_calib: Optional[np.ndarray] = None,
    alpha: float = 0.1,
) -> List[np.ndarray]:
    """Distribution-free prediction sets with finite-sample coverage.

    Returns a list of arrays — one per row of ``y_proba`` — listing
    the classes included in the prediction set. Coverage guarantee:
    with probability ≥ 1 - α, the true class is in the returned set.

    Two call modes:

    1. ``calibration_scores`` is provided directly (pre-computed).
    2. ``y_calib`` + ``y_proba_calib`` are provided — the function
       computes the 1-vs-rest softmax score ``1 - p_true`` on the
       calibration set, then takes the ⌈(n+1)(1-α)/n⌉-th quantile.

    Parameters
    ----------
    y_proba : (n, n_classes) test-set probabilities.
    calibration_scores : (n_calib,) pre-computed non-conformity scores.
    y_calib, y_proba_calib : calibration set, used to compute scores
        when ``calibration_scores`` is None.
    alpha : float in (0, 1). Miscoverage rate (1 - target coverage).

    Returns
    -------
    list of arrays. ``result[i]`` is the array of class indices in the
    prediction set for test row ``i``.

    References
    ----------
    - Vovk, V. et al. (2005). *Algorithmic Learning in a Random World.*
    - Angelopoulos, A. N. & Bates, S. (2023). *Conformal Prediction: A
      Gentle Introduction.* Foundations and Trends in ML 16(4).
    """
    y_proba = np.asarray(y_proba, dtype=np.float64)

    if calibration_scores is None:
        if y_calib is None or y_proba_calib is None:
            raise ValueError(
                "Either calibration_scores or (y_calib, y_proba_calib) must be provided"
            )
        y_calib = np.asarray(y_calib, dtype=np.int64)
        y_proba_calib = np.asarray(y_proba_calib, dtype=np.float64)
        # Non-conformity score: 1 - p(y_true).
        true_probs = y_proba_calib[np.arange(len(y_calib)), y_calib]
        calibration_scores = 1.0 - true_probs
    else:
        calibration_scores = np.asarray(calibration_scores, dtype=np.float64)

    n_calib = len(calibration_scores)
    # Quantile with the (n+1)(1-alpha)/n finite-sample correction.
    quantile_level = np.ceil((n_calib + 1) * (1 - alpha)) / n_calib
    quantile_level = min(quantile_level, 1.0)
    threshold = float(np.quantile(calibration_scores, quantile_level,
                                   method="higher"))
    # For each test row: include every class c such that 1 - p(c) <= threshold,
    # i.e. p(c) >= 1 - threshold.
    cutoff = 1.0 - threshold
    return [np.where(row >= cutoff)[0] for row in y_proba]


# ---------------------------------------------------------------------------
# Convenience: full calibration report
# ---------------------------------------------------------------------------
def calibration_report(y_true: np.ndarray,
                        y_proba: np.ndarray,
                        n_bins: int = 15) -> Dict[str, float]:
    """Scalar calibration summary: ECE, MCE, Brier, and NLL.

    Useful as a one-call summary for tables / dashboards.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=np.float64)
    if y_proba.ndim == 1:
        # Binary: extend to (n, 2) for NLL computation.
        p1 = y_proba
        p0 = 1.0 - p1
        y_proba_2d = np.column_stack([p0, p1])
    else:
        y_proba_2d = y_proba
    eps = 1e-12
    clipped = np.clip(y_proba_2d, eps, 1 - eps)
    # Negative log-likelihood.
    nll = -float(np.mean(np.log(clipped[np.arange(len(y_true)), y_true.astype(int)])))
    return {
        "ece": expected_calibration_error(y_true, y_proba, n_bins=n_bins),
        "mce": maximum_calibration_error(y_true, y_proba, n_bins=n_bins),
        "brier": brier_score(y_true, y_proba),
        "nll": nll,
        "mean_confidence": float(np.max(y_proba_2d, axis=1).mean()),
        "accuracy": float((y_proba_2d.argmax(axis=1) == y_true).mean()),
        "n_bins": n_bins,
        "n_samples": int(len(y_true)),
    }
