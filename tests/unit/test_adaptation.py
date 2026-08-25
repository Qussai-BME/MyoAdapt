"""Unit tests for adaptation methods."""
import numpy as np
import pytest

from myoadapt.adaptation.coral import CORAL
from myoadapt.adaptation.tca import TCA
from myoadapt.adaptation.sa import SubspaceAlignment
from myoadapt.adaptation.ea import EuclideanAlignment


def test_coral_basic():
    rng = np.random.default_rng(42)
    X_s = rng.standard_normal((100, 50))
    X_t = rng.standard_normal((100, 50))
    coral = CORAL()
    coral.fit(X_s, X_t)
    X_s_aligned = coral.transform(X_s)
    assert X_s_aligned.shape == X_s.shape


def test_coral_condition_number():
    rng = np.random.default_rng(42)
    X_s = rng.standard_normal((100, 50))
    X_t = rng.standard_normal((100, 50))
    coral = CORAL()
    coral.fit(X_s, X_t)
    assert coral.condition_number_ > 0


def test_tca_basic():
    rng = np.random.default_rng(42)
    X_s = rng.standard_normal((100, 50))
    X_t = rng.standard_normal((100, 50))
    tca = TCA(n_components=10)
    tca.fit(X_s, X_t)
    assert tca.W_ is not None
    assert tca.W_.shape[1] == 10


def test_sa_basic():
    rng = np.random.default_rng(42)
    X_s = rng.standard_normal((100, 50))
    X_t = rng.standard_normal((100, 50))
    sa = SubspaceAlignment(n_components=20)
    sa.fit(X_s, X_t)
    X_s_aligned = sa.transform(X_s)
    assert X_s_aligned.shape[0] == 100
    assert X_s_aligned.shape[1] == 20


def test_ea_basic():
    rng = np.random.default_rng(42)
    X_s = rng.standard_normal((100, 50))
    ea = EuclideanAlignment()
    ea.fit(X_s)
    X_aligned = ea.transform(X_s)
    assert X_aligned.shape == X_s.shape


def test_ea_from_signals():
    rng = np.random.default_rng(42)
    sigs = [rng.standard_normal((500, 12)) for _ in range(3)]
    ea = EuclideanAlignment()
    ea.fit_from_signals(sigs)
    assert ea.R_.shape == (12, 12)
