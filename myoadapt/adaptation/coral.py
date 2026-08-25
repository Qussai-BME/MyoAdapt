"""
coral.py — Correlation Alignment (CORAL)
========================================

Aligns the covariance of source features to target features.

Note: On MiniROCKET PPV features, CORAL INCREASES domain shift
by ~9.1× because the covariance is near-singular (condition number ~1e11).
Use only on well-conditioned feature spaces (e.g., hand-crafted TD features).

"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class CORAL:
    """
    Correlation Alignment.

    Given source X_s and target X_t (both centered), compute:
        A = U_s * (S_s^(-1/2)) * U_s^T * U_t * (S_t^(1/2)) * U_t^T

    Then X_s_aligned = X_s @ A.
    """

    name = "coral"

    def __init__(self, n_components: Optional[int] = None,
                 eps: float = 1e-6):
        self.n_components = n_components
        self.eps = eps
        self.A_: Optional[np.ndarray] = None
        self.condition_number_: Optional[float] = None

    def fit(self, X_source: np.ndarray, X_target: np.ndarray) -> CORAL:
        X_s = X_source.astype(np.float64)
        X_t = X_target.astype(np.float64)
        # Center
        X_s = X_s - X_s.mean(axis=0, keepdims=True)
        X_t = X_t - X_t.mean(axis=0, keepdims=True)

        # Source covariance
        cov_s = np.cov(X_s, rowvar=False) + self.eps * np.eye(X_s.shape[1])
        cov_t = np.cov(X_t, rowvar=False) + self.eps * np.eye(X_t.shape[1])

        # Track condition number
        eigvals_s = np.linalg.eigvalsh(cov_s)
        eigvals_s = np.clip(eigvals_s, 0, None)
        self.condition_number_ = float(eigvals_s.max() / max(eigvals_s.min(), 1e-30))

        # Whitening source
        U_s, S_s, _ = np.linalg.svd(cov_s)
        S_s_inv_sqrt = np.diag(1.0 / np.sqrt(np.clip(S_s, self.eps, None)))
        # Coloring target
        U_t, S_t, _ = np.linalg.svd(cov_t)
        S_t_sqrt = np.diag(np.sqrt(np.clip(S_t, self.eps, None)))

        self.A_ = U_s @ S_s_inv_sqrt @ U_s.T @ U_t @ S_t_sqrt @ U_t.T
        if self.n_components is not None:
            # Truncate to top-k components
            self.A_ = self.A_[:, : self.n_components]

        if self.condition_number_ > 1e8:
            logger.warning(
                f"CORAL: source covariance is near-singular "
                f"(condition number={self.condition_number_:.2e}). "
                f"This typically INCREASES domain shift on MiniROCKET PPV "
                f"features. Consider Euclidean Alignment "
                f"or use a different feature space."
            )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.A_ is None:
            raise RuntimeError("CORAL not fitted — call .fit(X_s, X_t) first")
        return (X.astype(np.float64) @ self.A_).astype(np.float32)

    def fit_transform(self, X_source: np.ndarray,
                       X_target: np.ndarray) -> np.ndarray:
        return self.fit(X_source, X_target).transform(X_source)
