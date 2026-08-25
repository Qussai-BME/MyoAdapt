"""Unit tests for myoadapt.evaluation.hyperopt.HyperparameterOptimizer."""
from __future__ import annotations

import numpy as np
import pytest

optuna = pytest.importorskip("optuna")

from myoadapt.evaluation.hyperopt import HyperparameterOptimizer, _HAS_OPTUNA


def _toy_data(n=90, n_feat=10, n_cls=3, n_groups=6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_feat))
    y = rng.integers(0, n_cls, n)
    groups = np.repeat(np.arange(n_groups), n // n_groups)
    return X, y, groups


def test_optuna_actually_available():
    """Regression test: optuna renamed CMAESampler -> CmaEsSampler at
    some point. The old name lived in a single shared try/except with
    the optuna/TPESampler/RandomSampler/pruner imports, so the ImportError
    silently set _HAS_OPTUNA = False and disabled the entire HPO feature
    — even though optuna itself, TPE, Random, and both pruners all
    imported fine. If this assertion ever fails again, check for another
    renamed/moved symbol in that same import block."""
    assert _HAS_OPTUNA is True


def test_invalid_constructor_args_raise():
    with pytest.raises(ValueError):
        HyperparameterOptimizer(protocol="not_a_protocol")
    with pytest.raises(ValueError):
        HyperparameterOptimizer(sampler="not_a_sampler")
    with pytest.raises(ValueError):
        HyperparameterOptimizer(pruner="not_a_pruner")
    with pytest.raises(ValueError):
        HyperparameterOptimizer(scoring="not_a_metric")


def test_default_param_space_known_model():
    opt = HyperparameterOptimizer(model_type="random_forest")
    space = opt.default_param_space()
    assert isinstance(space, dict) and len(space) > 0


def test_default_param_space_unknown_model_raises():
    with pytest.raises(KeyError):
        HyperparameterOptimizer().default_param_space("not_a_real_model")


@pytest.mark.parametrize("sampler", ["tpe", "random", "cmaes"])
def test_optimize_kfold_runs_for_each_sampler(sampler):
    X, y, _ = _toy_data()
    opt = HyperparameterOptimizer(model_type="logistic", protocol="kfold", n_folds=3,
                                   n_trials=4, sampler=sampler, pruner="none", random_state=0)
    result = opt.optimize(X, y)
    assert result["n_trials"] == 4
    assert "best_params" in result and "best_score" in result
    assert 0.0 <= result["best_score"] <= 1.0


def test_optimize_loso_requires_groups():
    X, y, _ = _toy_data()
    opt = HyperparameterOptimizer(protocol="loso", n_trials=2, pruner="none")
    with pytest.raises(ValueError):
        opt.optimize(X, y)  # no groups passed


def test_optimize_loso_runs_with_groups():
    X, y, groups = _toy_data()
    opt = HyperparameterOptimizer(model_type="logistic", protocol="loso",
                                   n_trials=3, pruner="none", random_state=0)
    result = opt.optimize(X, y, groups=groups)
    assert result["n_trials"] == 3
    assert opt.best_params_ == result["best_params"]
    assert opt.best_score_ == result["best_score"]


def test_custom_param_space_is_respected():
    X, y, _ = _toy_data()
    opt = HyperparameterOptimizer(model_type="logistic", protocol="kfold", n_folds=3,
                                   n_trials=3, pruner="none", random_state=0)
    custom_space = {"C": (0.01, 0.1)}  # narrow range, different from default
    result = opt.optimize(X, y, param_space=custom_space)
    assert 0.01 <= result["best_params"]["C"] <= 0.1


def test_custom_model_factory_bypasses_classical_builder():
    """model_factory should be used instead of EMGClassifier when provided."""
    from sklearn.dummy import DummyClassifier
    X, y, _ = _toy_data()
    calls = []

    def factory(params):
        calls.append(params)
        return DummyClassifier(strategy="most_frequent")

    opt = HyperparameterOptimizer(protocol="kfold", n_folds=3, n_trials=2,
                                   pruner="none", model_factory=factory)
    opt.optimize(X, y, param_space={"dummy_param": (0, 1)})
    assert len(calls) > 0
