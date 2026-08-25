"""
ea.py — Euclidean Alignment (Hahne et al. 2014)
================================================

Provides TWO complementary EA strategies:

1. ``EuclideanAlignment`` (legacy) — single average alignment matrix.
   Kept for backward compatibility; does NOT equalize per-subject scales.

2. ``PerSubjectEA`` (correct Hahne 2014) — per-subject whitening.
   Each subject gets R_i = sqrt(C_i)^-1 applied to their OWN data only.
   After alignment, every subject's covariance is the identity matrix.

For ablation, use ``PerSubjectEA``.

"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from myoadapt.data.preprocessing import (
    apply_euclidean_alignment,
    compute_alignment_matrix,
    compute_subject_alignment,
)

logger = logging.getLogger(__name__)


class EuclideanAlignment:
    """
    Legacy average-alignment EA adapter.

    .. warning::
        This computes a SINGLE alignment matrix from the average of subject
        covariances. A single linear transform CANNOT whiten multiple
        distributions, so this does NOT equalize per-subject scales.

        For correct Hahne 2014 EA, use :class:`PerSubjectEA`.
    """

    name = "ea"

    def __init__(self):
        self.R_: Optional[np.ndarray] = None

    def fit_from_signals(self, training_signals: List[np.ndarray]) -> EuclideanAlignment:
        """Fit from raw (n_samples, n_channels) signals."""
        self.R_ = compute_alignment_matrix(training_signals)
        return self

    def fit(self, X_source: np.ndarray, X_target: Optional[np.ndarray] = None,
            n_channels: int = 12) -> EuclideanAlignment:
        """
        Fit from feature matrix (heuristic).

        NOTE: This is an approximation. For the true EA effect, use
        .fit_from_signals() on raw signals BEFORE feature extraction.
        """
        if X_target is not None:
            X_combined = np.vstack([X_source, X_target])
        else:
            X_combined = X_source
        cov = np.cov(X_combined, rowvar=False)
        cov = cov + 1e-6 * np.eye(cov.shape[0])
        try:
            eigvals, eigvecs = np.linalg.eigh(cov)
            eigvals = np.clip(eigvals, 1e-10, None)
            sqrt_cov = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            self.R_ = np.linalg.inv(sqrt_cov).astype(np.float32)
        except np.linalg.LinAlgError:
            logger.warning("EA matrix inversion failed — using identity")
            self.R_ = np.eye(cov.shape[0], dtype=np.float32)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.R_ is None:
            raise RuntimeError("EA not fitted")
        return (X.astype(np.float32) @ self.R_).astype(np.float32)

    def transform_signal(self, signal: np.ndarray) -> np.ndarray:
        """Apply alignment to a raw (n_samples, n_channels) signal."""
        if self.R_ is None:
            raise RuntimeError("EA not fitted")
        return apply_euclidean_alignment(signal, self.R_)

    def fit_transform(self, X_source: np.ndarray,
                       X_target: Optional[np.ndarray] = None) -> np.ndarray:
        self.fit(X_source, X_target)
        return self.transform(X_source)


class PerSubjectEA:
    """
    Per-subject Euclidean Alignment (correct Hahne 2014 implementation).

    Each subject's raw signal is whitened by its own covariance matrix.
    After alignment, every subject's covariance is identity, eliminating
    cross-subject scale and orientation differences.

    Usage in a LOSO fold:
        1. For each training subject i, compute R_i = compute_subject_alignment(signal_i)
        2. Apply R_i to signal_i (each subject uses their own matrix)
        3. For the test subject, compute R_test from their test signal (unsupervised)
        4. Apply R_test to the test signal

    This is unsupervised on the test subject — no labels are needed — and is
    the standard EA used in the cross-subject EMG literature.
    """

    name = "per_subject_ea"

    def __init__(self, eps: float = 1e-6):
        self.eps = eps
        # Map: subject_id -> R matrix
        self.R_per_subject_: Dict[Any, np.ndarray] = {}

    def fit_subject(self, subject_id, signal: np.ndarray) -> np.ndarray:
        """Compute and store the alignment matrix for one subject."""
        R = compute_subject_alignment(signal, eps=self.eps)
        self.R_per_subject_[subject_id] = R
        return R

    def fit_subjects(self, signals_by_subject: Dict[Any, np.ndarray]) -> PerSubjectEA:
        """Fit alignment matrices for multiple subjects at once."""
        for sid, sig in signals_by_subject.items():
            self.fit_subject(sid, sig)
        return self

    def transform_subject(self, subject_id, signal: np.ndarray) -> np.ndarray:
        """Apply the alignment matrix for ``subject_id`` to ``signal``.

        If the subject was never fit, computes the matrix on-the-fly (this is
        the unsupervised test-subject case — no labels needed).
        """
        if subject_id not in self.R_per_subject_:
            logger.info(f"EA: computing on-the-fly alignment for unseen subject '{subject_id}'")
            self.fit_subject(subject_id, signal)
        R = self.R_per_subject_[subject_id]
        return apply_euclidean_alignment(signal, R)

    def transform_signal(self, signal: np.ndarray, subject_id=None) -> np.ndarray:
        """Apply alignment; if subject_id is unknown, computes on-the-fly."""
        if subject_id is None:
            # No subject ID given: compute fresh alignment for this signal
            R = compute_subject_alignment(signal, eps=self.eps)
        else:
            R = self.R_per_subject_.get(subject_id)
            if R is None:
                R = compute_subject_alignment(signal, eps=self.eps)
                self.R_per_subject_[subject_id] = R
        return apply_euclidean_alignment(signal, R)
