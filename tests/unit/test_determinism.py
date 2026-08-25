"""Determinism test: two LOSO runs on identical synthetic data with the same
random_state must produce identical accuracy_mean values.

Catches subtle sources of nondeterminism: un-seeded RNGs in the model
factory, parallel-fold ordering, etc. We use RandomForest (which is
deterministic when random_state is set) and force n_jobs=1 so the test
doesn't depend on joblib's thread scheduling.
"""
from __future__ import annotations

import numpy as np
import pytest

from myoadapt.evaluation.loso import LOSOEvaluator
from myoadapt.models.classical import EMGClassifier


def _make_synthetic(
    n_subjects: int = 5,
    n_gestures: int = 4,
    n_windows_per_gesture: int = 12,
    n_features: int = 20,
    seed: int = 42,
):
    """Build a small (X, y, groups) synthetic dataset."""
    rng = np.random.default_rng(seed)
    n_per_subject = n_gestures * n_windows_per_gesture
    n_total = n_subjects * n_per_subject

    X = np.zeros((n_total, n_features), dtype=np.float32)
    y = np.zeros(n_total, dtype=np.int32)
    groups = np.zeros(n_total, dtype=np.int32)

    idx = 0
    for subj in range(1, n_subjects + 1):
        for gesture in range(n_gestures):
            block = rng.standard_normal(
                (n_windows_per_gesture, n_features)
            ).astype(np.float32)
            # Class-correlated signal so the classifier has something to learn.
            block[:, :10] += gesture * 0.8
            # Per-subject offset so domains are distinguishable.
            block += subj * 0.03
            n = n_windows_per_gesture
            X[idx:idx + n] = block
            y[idx:idx + n] = gesture
            groups[idx:idx + n] = subj
            idx += n

    return X, y, groups


def _factory():
    """Fresh, seeded RandomForest for each fold. n_jobs=1 to avoid
    thread-scheduling nondeterminism in the classifier."""
    return EMGClassifier(
        model_type="random_forest",
        n_estimators=20,
        random_state=42,
        n_jobs=1,
    )


def test_loso_deterministic_across_runs():
    """Two LOSOEvaluator.run() calls on the same data with the same
    random_state must produce identical accuracy_mean values."""
    X, y, groups = _make_synthetic()

    evaluator_a = LOSOEvaluator(model_factory=_factory, verbose=False, n_jobs=1)
    evaluator_b = LOSOEvaluator(model_factory=_factory, verbose=False, n_jobs=1)

    results_a = evaluator_a.run(X, y, groups)
    results_b = evaluator_b.run(X, y, groups)

    acc_a = results_a["aggregate"]["accuracy_mean"]
    acc_b = results_b["aggregate"]["accuracy_mean"]
    assert acc_a == pytest.approx(acc_b, abs=1e-12), (
        f"LOSO accuracy_mean not deterministic: run A={acc_a!r} vs run B={acc_b!r}"
    )

    # Stronger check: per-fold accuracies must be bit-identical.
    np.testing.assert_array_equal(
        results_a["aggregate"]["per_fold_accuracies"],
        results_b["aggregate"]["per_fold_accuracies"],
    )

    # Same for macro_f1.
    f1_a = results_a["aggregate"]["macro_f1_mean"]
    f1_b = results_b["aggregate"]["macro_f1_mean"]
    assert f1_a == pytest.approx(f1_b, abs=1e-12)


def test_loso_deterministic_with_ea_disabled():
    """Even with feature selection off and EA off, two runs must match."""
    X, y, groups = _make_synthetic(n_subjects=4, n_gestures=3)

    evaluator = LOSOEvaluator(
        model_factory=_factory,
        use_ea=False,
        use_feature_selection=False,
        verbose=False,
        n_jobs=1,
    )
    results_a = evaluator.run(X, y, groups)
    results_b = evaluator.run(X, y, groups)

    assert results_a["aggregate"]["accuracy_mean"] == pytest.approx(
        results_b["aggregate"]["accuracy_mean"], abs=1e-12
    )
