"""
sa.py — Subspace Alignment (Fernando et al. 2013)
=================================================

Aligns the PCA subspace of source to target.

"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class SubspaceAlignment:
    """
    Subspace Alignment.

    1. Compute PCA of source (top-d components): U_s
    2. Compute PCA of target (top-d components): U_t
    3. Transformation M = U_s^T U_t
    4. Aligned source = X_s @ U_s @ M = X_s @ U_s @ U_s^T @ U_t
    """

    name = "sa"

    def __init__(self, n_components: int = 30):
        self.n_components = n_components
        self.U_s_: Optional[np.ndarray] = None
        self.M_: Optional[np.ndarray] = None

    def fit(self, X_source: np.ndarray, X_target: np.ndarray) -> SubspaceAlignment:
        from sklearn.decomposition import PCA
        d = min(self.n_components, X_source.shape[1], X_target.shape[1])
        pca_s = PCA(n_components=d)
        pca_s.fit(X_source)
        pca_t = PCA(n_components=d)
        pca_t.fit(X_target)
        self.U_s_ = pca_s.components_.T  # (n_features, d)
        U_t = pca_t.components_.T
        self.M_ = self.U_s_.T @ U_t  # (d, d)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.U_s_ is None or self.M_ is None:
            raise RuntimeError("SA not fitted")
        return (X.astype(np.float64) @ self.U_s_ @ self.M_).astype(np.float32)

    def fit_transform(self, X_source: np.ndarray,
                       X_target: np.ndarray) -> np.ndarray:
        return self.fit(X_source, X_target).transform(X_source)
