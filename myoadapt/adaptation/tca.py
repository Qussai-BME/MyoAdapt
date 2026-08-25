"""
tca.py — Transfer Component Analysis
====================================

Finds a shared subspace in which source and target distributions are close
(measured by Maximum Mean Discrepancy). Uses eigendecomposition of a joint
kernel matrix.

Note: On MiniROCKET PPV features, TCA's eigenvalues span ~14 orders
of magnitude, making scale selection arbitrary and unstable.

"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class TCA:
    """
    Transfer Component Analysis (Pan et al. 2010).

    Parameters
    ----------
    n_components : int — dimensionality of the transfer subspace
    kernel : str — 'rbf' (default), 'linear'
    gamma : float — RBF bandwidth (auto = 1/n_features if None)
    mu : float — trade-off between MMD minimization and variance preservation
    """

    name = "tca"

    def __init__(self, n_components: int = 30,
                 kernel: str = "rbf",
                 gamma: Optional[float] = None,
                 mu: float = 0.5,
                 eps: float = 1e-6):
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.mu = mu
        self.eps = eps
        self.W_: Optional[np.ndarray] = None
        self.eigvals_: Optional[np.ndarray] = None
        self.scale_gap_: Optional[float] = None
        self.X_train_: Optional[np.ndarray] = None

    def _kernel(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        if self.kernel == "linear":
            if Y is None:
                return X @ X.T
            return X @ Y.T
        # RBF
        gamma = self.gamma or (1.0 / X.shape[1])
        if Y is None:
            sq = np.sum(X ** 2, axis=1)
            D = sq[:, None] + sq[None, :] - 2 * X @ X.T
        else:
            sq_x = np.sum(X ** 2, axis=1)
            sq_y = np.sum(Y ** 2, axis=1)
            D = sq_x[:, None] + sq_y[None, :] - 2 * X @ Y.T
        return np.exp(-gamma * np.clip(D, 0, None))

    def fit(self, X_source: np.ndarray, X_target: np.ndarray) -> TCA:
        X_s = X_source.astype(np.float64)
        X_t = X_target.astype(np.float64)
        n_s, n_t = X_s.shape[0], X_t.shape[0]
        n_total = n_s + n_t
        X = np.vstack([X_s, X_t])

        K = self._kernel(X)
        # L matrix (MMD)
        e_s = np.ones((n_s, 1)) / n_s
        e_t = np.ones((n_t, 1)) / n_t
        L_top = np.hstack([e_s @ e_s.T, -e_s @ e_t.T])
        L_bot = np.hstack([-e_t @ e_s.T, e_t @ e_t.T])
        L = np.vstack([L_top, L_bot])

        # Centering matrix H = I - 11^T / n
        H = np.eye(n_total) - np.ones((n_total, n_total)) / n_total

        # Solve generalized eigenproblem: (K L K + μI) w = λ K H K w
        A = K @ L @ K + self.mu * np.eye(n_total)
        B = K @ H @ K + self.eps * np.eye(n_total)

        try:
            from scipy.linalg import eigh
            eigvals, eigvecs = eigh(A, B)
        except Exception as e:
            logger.error(f"TCA eigenproblem failed: {e}")
            self.W_ = np.eye(n_total, self.n_components)
            return self

        # Sort ascending (smallest MMD first)
        order = np.argsort(eigvals)
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        # Scale diagnostic
        valid = eigvals[eigvals > self.eps]
        if len(valid) > 1:
            self.scale_gap_ = float(valid.max() / valid.min())
            if self.scale_gap_ > 1e8:
                logger.warning(
                    f"TCA eigenvalue scale gap = {self.scale_gap_:.2e} — "
                    f"near-singular. Scale selection is unstable on this feature space."
                )

        self.eigvals_ = eigvals
        self.W_ = eigvecs[:, : self.n_components]
        self.X_train_ = X  # store training stack for out-of-sample projection
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.W_ is None or self.X_train_ is None:
            raise RuntimeError("TCA not fitted — call .fit(X_s, X_t) first")
        # Out-of-sample extension via the kernel trick:
        # K(X_new, X_train) @ W_  (Pan et al. 2010, Eq. 17)
        X_new = X.astype(np.float64)
        K_new = self._kernel(X_new, self.X_train_)
        return (K_new @ self.W_).astype(np.float32)

    def fit_transform(self, X_source: np.ndarray,
                       X_target: np.ndarray) -> np.ndarray:
        self.fit(X_source, X_target)
        # Project the source rows of the training stack via the kernel embedding
        n_s = X_source.shape[0]
        K_full = self._kernel(self.X_train_)  # (n_s + n_t, n_s + n_t)
        return (K_full @ self.W_)[:n_s].astype(np.float32)
