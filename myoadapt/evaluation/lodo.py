"""
lodo.py — Leave-One-Database-Out evaluator
==========================================

For cross-database generalization.

"""
from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional

import numpy as np
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler

from myoadapt.data.splits import lodo_splits
from myoadapt.evaluation.metrics import compute_metrics

logger = logging.getLogger(__name__)


class LODOEvaluator:
    """
    Leave-One-Database-Out evaluator.

    Use case: train on DB1 + DB2 + DB7, test on DB3 (amputees).
    This evaluates cross-database generalization, which is much harder
    than LOSO within a single database.
    """

    def __init__(self,
                 model_factory: Callable,
                 rest_class: str = "rest",
                 n_jobs: int = 1,
                 verbose: bool = True):
        self.model_factory = model_factory
        self.rest_class = rest_class
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.results_: Optional[Dict] = None

    def run(self, X: np.ndarray, y: np.ndarray,
            database_ids: np.ndarray,
            classes: Optional[List[str]] = None) -> Dict:
        if classes is None:
            classes = sorted(np.unique(y).tolist())

        unique_dbs = np.unique(database_ids)
        n_folds = len(unique_dbs)
        if self.verbose:
            logger.info(f"Starting LODO: {n_folds} folds (databases: {unique_dbs.tolist()})")

        splits = list(lodo_splits(database_ids))
        if self.n_jobs == 1:
            per_fold = [self._run_fold(i, X, y, classes, train_idx, test_idx, db, database_ids)
                        for i, ((train_idx, test_idx), db) in enumerate(zip(splits, unique_dbs))]
        else:
            per_fold = Parallel(n_jobs=self.n_jobs)(
                delayed(self._run_fold)(i, X, y, classes, train_idx, test_idx, db, database_ids)
                for i, ((train_idx, test_idx), db) in enumerate(zip(splits, unique_dbs))
            )

        accuracies = [f["metrics"]["accuracy"] for f in per_fold]
        macro_f1s = [f["metrics"]["macro_f1"] for f in per_fold]

        self.results_ = {
            "protocol": "LODO",
            "n_folds": n_folds,
            "per_fold": per_fold,
            "aggregate": {
                "accuracy_mean": float(np.mean(accuracies)),
                "accuracy_std": float(np.std(accuracies, ddof=1)),
                "macro_f1_mean": float(np.mean(macro_f1s)),
                "macro_f1_std": float(np.std(macro_f1s, ddof=1)),
                "per_fold_accuracies": accuracies,
                "per_fold_macro_f1s": macro_f1s,
            },
        }
        return self.results_

    def _run_fold(self, fold_idx: int, X: np.ndarray, y: np.ndarray,
                   classes: List[str], train_idx: np.ndarray,
                   test_idx: np.ndarray, db_id, database_ids=None) -> Dict:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        model = self.model_factory()
        # Some models (LiteDAN) accept a `groups` argument; others (sklearn)
        # do not. Try the rich signature first, fall back to the minimal one.
        try:
            groups_arg = database_ids[train_idx] if database_ids is not None else None
            if groups_arg is not None:
                model.fit(X_train, y_train, groups=groups_arg)
            else:
                model.fit(X_train, y_train)
        except TypeError:
            model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = compute_metrics(y_test, y_pred, classes=classes,
                                   rest_class=self.rest_class)
        if self.verbose:
            logger.info(f"  LODO fold {fold_idx + 1} (DB={db_id}): "
                        f"acc={metrics['accuracy']:.4f} F1={metrics['macro_f1']:.4f}")
        return {
            "fold": fold_idx,
            "test_database": int(db_id),
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "metrics": metrics,
        }
