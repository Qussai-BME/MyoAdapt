"""Unit test for the LODOEvaluator.

Synthetic, minimal — verifies that the evaluator runs end-to-end on a
tiny synthetic dataset and returns the expected result structure.

Configuration:
- 3 databases
- 4 subjects per database (12 subjects total)
- 5 gesture classes
- 10 windows per gesture per subject (5 * 4 * 10 = 200 windows/db,
  600 windows total)
"""
from __future__ import annotations

import numpy as np
import pytest

from myoadapt.evaluation.lodo import LODOEvaluator
from myoadapt.models.classical import EMGClassifier


def _make_synthetic(
    n_databases: int = 3,
    n_subjects_per_db: int = 4,
    n_gestures: int = 5,
    n_windows_per_gesture: int = 10,
    n_features: int = 16,
    seed: int = 42,
):
    """Build a (X, y, db_ids) synthetic dataset for LODO."""
    rng = np.random.default_rng(seed)
    n_per_db = n_subjects_per_db * n_gestures * n_windows_per_gesture
    n_total = n_databases * n_per_db

    X = np.zeros((n_total, n_features), dtype=np.float32)
    y = np.zeros(n_total, dtype=np.int32)
    db_ids = np.zeros(n_total, dtype=np.int32)

    idx = 0
    for db in range(1, n_databases + 1):
        for subj in range(n_subjects_per_db):
            for gesture in range(n_gestures):
                # Class-correlated signal so the classifier has something
                # to learn. Database-specific offset so domains differ
                # slightly (makes LODO harder than LOSO).
                base = rng.standard_normal(
                    (n_windows_per_gesture, n_features)
                ).astype(np.float32)
                base[:, :8] += gesture * 0.7  # class signal
                base += db * 0.05              # domain shift
                n = n_windows_per_gesture
                X[idx:idx + n] = base
                y[idx:idx + n] = gesture
                db_ids[idx:idx + n] = db
                idx += n

    return X, y, db_ids


def _factory():
    """Fresh model for each LODO fold."""
    return EMGClassifier(model_type="random_forest", n_estimators=10, random_state=42)


def test_lodo_runs_and_structure():
    """LODOEvaluator must run on synthetic data and return the documented
    result structure."""
    X, y, db_ids = _make_synthetic(
        n_databases=3,
        n_subjects_per_db=4,
        n_gestures=5,
        n_windows_per_gesture=10,
        n_features=16,
    )
    assert X.shape[0] == 3 * 4 * 5 * 10  # 600
    assert len(np.unique(db_ids)) == 3

    evaluator = LODOEvaluator(model_factory=_factory, verbose=False)
    results = evaluator.run(X, y, database_ids=db_ids)

    # ---- Top-level structure ------------------------------------------------
    assert isinstance(results, dict)
    assert results["protocol"] == "LODO"
    assert results["n_folds"] == 3

    # ---- Per-fold structure -------------------------------------------------
    assert "per_fold" in results
    assert len(results["per_fold"]) == 3
    for fold in results["per_fold"]:
        assert "fold" in fold
        assert "test_database" in fold
        assert "n_train" in fold
        assert "n_test" in fold
        assert "metrics" in fold
        # Each fold trains on 2 dbs and tests on 1 → 2/3 of the data is train.
        assert fold["n_train"] + fold["n_test"] == X.shape[0]
        assert fold["n_test"] == X.shape[0] // 3

        m = fold["metrics"]
        assert "accuracy" in m
        assert "macro_f1" in m
        assert 0.0 <= m["accuracy"] <= 1.0
        assert 0.0 <= m["macro_f1"] <= 1.0

    # ---- Aggregate structure ------------------------------------------------
    agg = results["aggregate"]
    assert "accuracy_mean" in agg
    assert "accuracy_std" in agg
    assert "macro_f1_mean" in agg
    assert "macro_f1_std" in agg
    assert "per_fold_accuracies" in agg
    assert "per_fold_macro_f1s" in agg
    assert len(agg["per_fold_accuracies"]) == 3
    assert len(agg["per_fold_macro_f1s"]) == 3
    # The mean must equal the arithmetic mean of per-fold accuracies.
    np.testing.assert_allclose(
        agg["accuracy_mean"],
        float(np.mean(agg["per_fold_accuracies"])),
        rtol=1e-6,
    )


def test_lodo_each_database_held_out_once():
    """Each database must be the held-out test DB in exactly one fold."""
    X, y, db_ids = _make_synthetic()
    evaluator = LODOEvaluator(model_factory=_factory, verbose=False)
    results = evaluator.run(X, y, database_ids=db_ids)
    held_out = sorted(fold["test_database"] for fold in results["per_fold"])
    assert held_out == [1, 2, 3]


def test_lodo_no_subject_leakage():
    """A sample whose db is the test db must NEVER appear in the train set
    for that fold."""
    X, y, db_ids = _make_synthetic()
    evaluator = LODOEvaluator(model_factory=_factory, verbose=False)
    # We re-implement the split inline to check the invariant directly.
    from myoadapt.data.splits import lodo_splits

    unique_dbs = np.unique(db_ids)
    for (train_idx, test_idx), db in zip(lodo_splits(db_ids), unique_dbs):
        train_dbs = set(db_ids[train_idx].tolist())
        test_dbs = set(db_ids[test_idx].tolist())
        assert db in test_dbs
        assert db not in train_dbs
        # All other dbs should be in train.
        assert all(d in train_dbs for d in unique_dbs if d != db)
