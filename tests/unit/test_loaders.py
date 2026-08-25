"""Unit tests for myoadapt.data.loaders (NinaPro / CapgMyo / UCI).

Previously had zero dedicated test coverage. That's how the bug in
test_ninapro_synthetic_subjects_are_balanced below went undetected: the
array was pre-sized for 10 repetitions per (subject, gesture) but the
generation loop was missing the repetition level, silently leaving 90%
of every default synthetic dataset as zero-signal rows mislabeled under
a phantom "subject 0" — which every other command's synthetic-fallback
testing had been silently running on.
"""
from __future__ import annotations

import numpy as np
import pytest

from myoadapt.data.loaders import load_dataset, NinaProLoader, CapgMyoLoader, UCILoader, DATASET_REGISTRY


@pytest.mark.parametrize("db", ["DB1", "DB2", "DB3", "DB7"])
def test_ninapro_synthetic_subjects_are_balanced(db):
    """Regression test for the missing-repetition-loop bug: every subject
    must get the same number of windows, and none should be all-zero."""
    meta = DATASET_REGISTRY[db]
    loader = NinaProLoader(root="/nonexistent-path", db=db)
    X, y, groups, load_meta = loader.load_split()
    assert load_meta["synthetic"] is True

    uniq, counts = np.unique(groups, return_counts=True)
    assert len(uniq) == meta["n_subjects"]
    assert uniq.min() == 1  # no phantom subject 0
    assert counts.min() == counts.max()  # perfectly balanced

    is_zero_row = (X.reshape(len(X), -1) == 0).all(axis=1)
    assert not is_zero_row.any(), f"{is_zero_row.sum()} all-zero rows found"


def test_ninapro_synthetic_total_matches_subjects_times_gestures_times_reps():
    loader = NinaProLoader(root="/nonexistent-path", db="DB2")
    X, y, groups, meta = loader.load_split()
    n_subjects = DATASET_REGISTRY["DB2"]["n_subjects"]
    n_gestures = DATASET_REGISTRY["DB2"]["n_gestures"]
    assert X.shape[0] == n_subjects * n_gestures * 10
    assert set(np.unique(y).tolist()) == set(range(n_gestures))


def test_ninapro_synthetic_gesture_labels_present_per_subject():
    """Every subject should have all n_gestures represented, each with
    exactly 10 repetitions — not just present once."""
    loader = NinaProLoader(root="/nonexistent-path", db="DB2")
    X, y, groups, meta = loader.load_split()
    n_gestures = DATASET_REGISTRY["DB2"]["n_gestures"]
    for subj in (1, 20, 40):
        sub_labels = y[groups == subj]
        uniq, counts = np.unique(sub_labels, return_counts=True)
        assert len(uniq) == n_gestures
        assert (counts == 10).all()


@pytest.mark.parametrize("cls,db", [
    (CapgMyoLoader, "CapgMyo-DBa"),
    (UCILoader, "UCI"),
])
def test_other_loaders_are_balanced_and_nonzero(cls, db):
    """CapgMyo/UCI use fully vectorized generation (no per-subject Python
    loop), so this checks they don't share the NinaPro bug — they don't,
    but worth a standing guard."""
    loader = cls(root="/nonexistent-path", db=db)
    X, y, groups, meta = loader.load_split()
    uniq, counts = np.unique(groups, return_counts=True)
    assert counts.min() == counts.max()
    is_zero_row = (X.reshape(len(X), -1) == 0).all(axis=1)
    assert not is_zero_row.any()


def test_load_dataset_n_subjects_cap_is_respected():
    X, y, groups, meta = load_dataset(db="DB2", root="/nonexistent-path", n_subjects=5)
    assert set(np.unique(groups).tolist()) == {1, 2, 3, 4, 5}
    assert meta["n_subjects"] == 5


def test_unknown_db_raises():
    with pytest.raises((ValueError, KeyError)):
        load_dataset(db="NotARealDB", root="/nonexistent-path")
