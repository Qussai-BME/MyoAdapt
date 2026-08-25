"""Unit tests for myoadapt.adaptation.adversarial.AdversarialDA.

AdversarialDA is an intentional thin wrapper around models.lite_dan.LiteDAN
(see its module docstring) so all domain-adaptation methods share one
registry interface. LiteDAN itself already has coverage; this file checks
the wrapper's own fit/predict/transform delegation and its target-domain
path, which had no test of their own.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from myoadapt.adaptation.adversarial import AdversarialDA


def _source_data(n=60, n_feat=16, n_cls=4, n_dom=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_feat)).astype(np.float32)
    y = rng.integers(0, n_cls, n)
    groups = np.repeat(np.arange(n_dom), n // n_dom)
    return X, y, groups


def test_fit_predict_shapes():
    X, y, groups = _source_data()
    ada = AdversarialDA(n_features=16, n_classes=4, n_domains=3, n_epochs=3, device="cpu")
    ada.fit(X, y, groups)
    preds = ada.predict(X)
    proba = ada.predict_proba(X)
    assert preds.shape == (60,)
    assert proba.shape == (60, 4)
    assert set(np.unique(preds).tolist()).issubset(set(range(4)))


def test_transform_returns_embedding():
    X, y, groups = _source_data()
    ada = AdversarialDA(n_features=16, n_classes=4, n_domains=3, n_epochs=3, device="cpu")
    ada.fit(X, y, groups)
    emb = ada.transform(X)
    assert emb.shape == (60, 64)  # LiteDAN's fixed encoder width


def test_fit_with_unlabelled_target_domain():
    """Target-domain rows get label -1 and must be masked out of the
    classification loss rather than crashing or corrupting predictions."""
    X, y, groups = _source_data(n_dom=3)
    X_t = np.random.default_rng(1).standard_normal((15, 16)).astype(np.float32)
    g_t = np.full(15, 3)  # a 4th, unseen domain id
    ada = AdversarialDA(n_features=16, n_classes=4, n_domains=4, n_epochs=3, device="cpu")
    ada.fit(X, y, groups, X_target=X_t, groups_target=g_t)
    preds = ada.predict(X_t)
    assert preds.shape == (15,)
    assert not np.isnan(ada.predict_proba(X_t)).any()
