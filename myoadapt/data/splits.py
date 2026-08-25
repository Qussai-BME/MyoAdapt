"""
splits.py — Subject-aware and database-aware cross-validation splits
====================================================================

Implements:
  - loso_splits                  — Leave-One-Subject-Out
  - lodo_splits                  — Leave-One-Database-Out
  - kfold_splits                 — subject-grouped k-fold
  - train_test_split_subject     — hold-out with no subject leakage

All split functions are generators yielding (train_idx, test_idx) pairs.
The key invariant: a subject (or database) in test NEVER appears in train.

License: Apache 2.0
"""
from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import List, Optional, Tuple, Union

import numpy as np
from sklearn.model_selection import GroupKFold

logger = logging.getLogger(__name__)


def loso_splits(groups: np.ndarray) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Leave-One-Subject-Out generator."""
    groups = np.asarray(groups)
    unique = np.unique(groups)
    for subj in unique:
        test_idx = np.where(groups == subj)[0]
        train_idx = np.where(groups != subj)[0]
        yield train_idx, test_idx


def lodo_splits(database_ids: np.ndarray) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Leave-One-Database-Out generator."""
    return loso_splits(database_ids)


def kfold_splits(
    groups: np.ndarray,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = 42,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Subject-grouped k-fold (no subject leakage)."""
    groups = np.asarray(groups)
    gkf = GroupKFold(n_splits=n_splits)
    for train_idx, test_idx in gkf.split(np.zeros_like(groups), groups=groups):
        yield train_idx, test_idx


def train_test_split_subject(
    groups: np.ndarray,
    test_size: Optional[float] = 0.2,
    test_subject: Optional[Union[int, List[int]]] = None,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Subject-level train/test split (no subject in both).

    Parameters
    ----------
    groups        : (n_samples,) subject IDs
    test_size     : fraction of subjects for test (0 < test_size < 1)
    test_subject  : explicit subject(s) for test (overrides test_size)
    random_state  : reproducible subject shuffling
    """
    groups = np.asarray(groups)
    unique_subjects = np.unique(groups)
    rng = np.random.default_rng(random_state)

    if test_subject is not None:
        if np.isscalar(test_subject):
            test_subjects = [test_subject]
        else:
            test_subjects = list(test_subject)
        test_subjects_arr = np.array(test_subjects)
        train_subjects = np.setdiff1d(unique_subjects, test_subjects_arr)
    else:
        if test_size is None or not (0 < test_size < 1):
            raise ValueError("Either test_subject or test_size in (0, 1) must be given")
        n_test = max(1, int(round(len(unique_subjects) * test_size)))
        perm = rng.permutation(len(unique_subjects))
        test_subjects_arr = unique_subjects[perm[:n_test]]
        train_subjects = unique_subjects[perm[n_test:]]

    test_idx = np.where(np.isin(groups, test_subjects_arr))[0]
    train_idx = np.where(np.isin(groups, train_subjects))[0]
    return train_idx, test_idx
