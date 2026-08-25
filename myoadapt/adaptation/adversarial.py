"""
adversarial.py — Adversarial Domain Adaptation wrapper
======================================================

Thin wrapper that delegates to myoadapt.models.lite_dan.LiteDAN.
Provided here so all DA methods share the same registry interface.

Note: AdversarialDA is jointly trained with the classifier (not a
preprocessing step like CORAL/TCA/SA). The `fit` signature differs
intentionally — it requires labels and subject groups for the source
domain and (optionally) unlabelled target-domain features.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class AdversarialDA:
    """
    Adversarial Domain Adaptation via Gradient Reversal Layer.

    A thin wrapper around ``LiteDAN`` exposing the same predict/predict_proba
    surface but with a fit signature that accepts labels and groups (required
    for adversarial training). Use ``LiteDAN`` directly for full ablation
    control (lambda schedule, with/without GRL).
    """

    name = "adversarial"

    def __init__(self,
                 n_features: int = 308,
                 n_classes: int = 12,
                 n_domains: int = 12,
                 lambda_schedule: str = "gradual",
                 use_grl: bool = True,
                 n_epochs: int = 100,
                 lr: float = 1e-3,
                 device: str = "auto"):
        from myoadapt.models.lite_dan import LiteDAN
        self.n_features = n_features
        self.n_classes = n_classes
        self.n_domains = n_domains
        self.lite_dan = LiteDAN(
            n_features=n_features,
            n_classes=n_classes,
            n_domains=n_domains,
            lambda_schedule=lambda_schedule,
            use_grl=use_grl,
            n_epochs=n_epochs,
            lr=lr,
            device=device,
        )

    def fit(self,
            X_source: np.ndarray,
            y_source: np.ndarray,
            groups_source: np.ndarray,
            X_target: Optional[np.ndarray] = None,
            groups_target: Optional[np.ndarray] = None) -> AdversarialDA:
        """
        Train the adversarial network on labelled source data.

        Parameters
        ----------
        X_source, y_source, groups_source
            Labelled source-domain features, labels, and subject IDs.
        X_target, groups_target
            Optional unlabelled target-domain data and groups. When provided,
            these are concatenated with the source during training so the
            domain discriminator sees both domains. Labels for target rows
            are set to -1 (ignored by ``CrossEntropyLoss`` via masking).
        """
        if X_target is not None and len(X_target) > 0:
            X = np.vstack([X_source, X_target])
            y = np.concatenate([y_source, np.full(len(X_target), -1)])
            groups = np.concatenate([groups_source, groups_target])
            # Mask -1 labels out of the gesture loss
            self.lite_dan.fit(X, y, groups=groups, ignore_label=-1)
        else:
            self.lite_dan.fit(X_source, y_source, groups=groups_source)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return the encoder's 64-D feature embedding."""
        return self.lite_dan.extract_features(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.lite_dan.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.lite_dan.predict_proba(X)
