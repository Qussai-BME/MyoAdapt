"""Unit tests for myoadapt.evaluation.zero_shot.ZeroShotEvaluator."""
from __future__ import annotations

import numpy as np
import pytest

from myoadapt.evaluation.zero_shot import ZeroShotEvaluator
from myoadapt.models.classical import EMGClassifier


def _multi_subject_data(n_subjects=20, n_per_subj=25, n_feat=10, n_cls=4, seed=0):
    rng = np.random.default_rng(seed)
    X, y, groups = [], [], []
    for s in range(n_subjects):
        X.append(rng.standard_normal((n_per_subj, n_feat)))
        y.append(rng.integers(0, n_cls, n_per_subj))
        groups.append(np.full(n_per_subj, s))
    return np.concatenate(X), np.concatenate(y), np.concatenate(groups)


def _factory():
    return EMGClassifier("random_forest", k_features=None)


def test_requires_at_least_3_subjects():
    X, y, groups = _multi_subject_data(n_subjects=2)
    ev = ZeroShotEvaluator(model_factory=_factory, verbose=False)
    with pytest.raises(ValueError):
        ev.run(X, y, groups)


def test_train_test_pools_are_disjoint():
    X, y, groups = _multi_subject_data()
    ev = ZeroShotEvaluator(model_factory=_factory, n_train_subjects=15,
                            n_test_subjects=5, random_state=0, verbose=False,
                            rest_class=None)
    result = ev.run(X, y, groups)
    assert not (set(result["train_subjects"]) & set(result["test_subjects"]))
    assert result["n_train_subjects"] == 15
    assert result["n_test_subjects"] == 5


def test_default_pool_split_is_80_20():
    X, y, groups = _multi_subject_data(n_subjects=20)
    ev = ZeroShotEvaluator(model_factory=_factory, random_state=0, verbose=False,
                            rest_class=None)
    result = ev.run(X, y, groups)
    assert result["n_train_subjects"] == 16
    assert result["n_test_subjects"] == 4


def test_per_subject_results_and_aggregate_consistent():
    X, y, groups = _multi_subject_data()
    ev = ZeroShotEvaluator(model_factory=_factory, n_train_subjects=15,
                            n_test_subjects=5, random_state=0, verbose=False,
                            rest_class=None)
    result = ev.run(X, y, groups)
    accs = [e["accuracy"] for e in result["per_subject"]]
    assert len(result["per_subject"]) == 5
    assert abs(np.mean(accs) - result["aggregate"]["accuracy_mean"]) < 1e-9
    assert result["worst_subject"]["accuracy"] <= result["best_subject"]["accuracy"]


def test_reproducible_with_fixed_random_state():
    X, y, groups = _multi_subject_data()
    ev1 = ZeroShotEvaluator(model_factory=_factory, n_train_subjects=15,
                             n_test_subjects=5, random_state=7, verbose=False,
                             rest_class=None)
    ev2 = ZeroShotEvaluator(model_factory=_factory, n_train_subjects=15,
                             n_test_subjects=5, random_state=7, verbose=False,
                             rest_class=None)
    r1, r2 = ev1.run(X, y, groups), ev2.run(X, y, groups)
    assert r1["train_subjects"] == r2["train_subjects"]
    assert r1["test_subjects"] == r2["test_subjects"]


def test_compare_with_loso_returns_paired_stats_with_enough_overlap():
    X, y, groups = _multi_subject_data(n_subjects=15, n_per_subj=25, n_cls=3)
    ev = ZeroShotEvaluator(model_factory=_factory, n_train_subjects=8,
                            n_test_subjects=7, random_state=0, verbose=False,
                            rest_class=None)
    comp = ev.compare_with_loso(X, y, groups)
    assert set(comp.keys()) == {"zero_shot", "loso", "comparison"}
    assert comp["comparison"]["n_overlapping_subjects"] == 7
    assert comp["comparison"]["cohen_dz"] is not None


def test_compare_with_loso_handles_insufficient_overlap():
    X, y, groups = _multi_subject_data(n_subjects=10, n_cls=3)
    ev = ZeroShotEvaluator(model_factory=_factory, n_train_subjects=7,
                            n_test_subjects=3, random_state=0, verbose=False,
                            rest_class=None)
    comp = ev.compare_with_loso(X, y, groups)
    assert comp["comparison"]["cohen_dz"] is None
    assert "insufficient overlap" in comp["comparison"]["effect_interpretation"]
