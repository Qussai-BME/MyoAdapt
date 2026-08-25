"""
trust_scores.py — Calibration-aware trust scores
================================================

A trust score is a 0-1 value that combines:
- Model confidence (softmax)
- Calibration (is the model right when it's confident?)
- Feature distribution (is the input similar to training data?)
- Temporal consistency (does this prediction agree with neighbors?)

For high-risk clinical applications, predictions below a trust threshold
should be rejected (not deployed).

"""
from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class TrustScorer:
    """
    Multi-factor trust scoring.

    Parameters
    ----------
    confidence_weight : float
    calibration_weight : float
    distribution_weight : float
    temporal_weight : float
    threshold : float — reject predictions below this
    """

    def __init__(self,
                 confidence_weight: float = 0.4,
                 calibration_weight: float = 0.3,
                 distribution_weight: float = 0.2,
                 temporal_weight: float = 0.1,
                 threshold: float = 0.5):
        self.w = {
            "confidence": confidence_weight,
            "calibration": calibration_weight,
            "distribution": distribution_weight,
            "temporal": temporal_weight,
        }
        self.threshold = threshold
        self.training_mean_: Optional[np.ndarray] = None
        self.training_std_: Optional[np.ndarray] = None
        self.recent_predictions_: List[int] = []

    def fit(self, X_train: np.ndarray) -> TrustScorer:
        """Fit the distribution model from training features."""
        self.training_mean_ = X_train.mean(axis=0)
        self.training_std_ = X_train.std(axis=0) + 1e-8
        return self

    def score(self, X: np.ndarray, proba: np.ndarray,
              pred_idx: int) -> float:
        """
        Compute a trust score for a single prediction.

        Parameters
        ----------
        X : (1, n_features) input features
        proba : (1, n_classes) softmax probabilities
        pred_idx : int — predicted class index
        """
        # 1. Confidence
        confidence = float(proba[0, pred_idx])

        # 2. Calibration (margin between top-2)
        sorted_p = np.sort(proba[0])[::-1]
        margin = float(sorted_p[0] - sorted_p[1]) if len(sorted_p) > 1 else 1.0
        calibration = margin

        # 3. Distribution similarity (z-score distance)
        if self.training_mean_ is not None:
            z = np.abs((X[0] - self.training_mean_) / self.training_std_)
            # Fraction of features within 3 sigma
            distribution = float(np.mean(z < 3.0))
        else:
            distribution = 0.5

        # 4. Temporal consistency
        if self.recent_predictions_:
            agree = sum(1 for p in self.recent_predictions_[-5:] if p == pred_idx)
            temporal = agree / min(5, len(self.recent_predictions_))
        else:
            temporal = 0.5
        self.recent_predictions_.append(pred_idx)

        trust = (
            self.w["confidence"] * confidence +
            self.w["calibration"] * calibration +
            self.w["distribution"] * distribution +
            self.w["temporal"] * temporal
        )
        return float(trust)

    def should_reject(self, trust_score: float) -> bool:
        return trust_score < self.threshold

    def reset(self):
        self.recent_predictions_.clear()
