"""
loso.py — Leave-One-Subject-Out evaluator
==========================================

Strict LOSO with per-fold preprocessing:
1. Split: one subject = test, rest = train
2. Fit Euclidean Alignment on training subjects (optional)
3. Extract features per window
4. Fold-level Z-score normalization (fit on train only)
5. Optional feature selection (SelectKBest)
6. Train classifier
7. Predict on test
8. Record per-fold metrics
9. Aggregate + statistical tests

"""
from __future__ import annotations

import logging
import time
from typing import Callable, Dict, List, Optional

import numpy as np
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler

from myoadapt.data.splits import loso_splits
from myoadapt.evaluation.metrics import compute_metrics
from myoadapt.evaluation.statistics import (
    bootstrap_ci,
    friedman_test,
    wilcoxon_pairwise,
)

logger = logging.getLogger(__name__)


class LOSOEvaluator:
    """
    Leave-One-Subject-Out evaluator with fold-level preprocessing.

    Parameters
    ----------
    model_factory : callable that returns a fresh model each fold
                    (so state doesn't leak across folds).
                    Pass model_factory=lambda: EMGClassifier('xgboost').
                    If None, you must pass `model` per-fold.
    use_ea : bool — apply Euclidean Alignment per fold
    use_feature_selection : bool — apply SelectKBest per fold
    k_features : int — number of features to select
    n_jobs : int — parallel folds (default 1; LOSO folds are independent)
    rest_class : str — name of the rest class for inflation analysis
    """

    def __init__(self,
                 model_factory: Optional[Callable] = None,
                 use_ea: bool = False,
                 use_feature_selection: bool = False,
                 k_features: int = 420,
                 n_jobs: int = 1,
                 rest_class: str = "rest",
                 verbose: bool = True):
        self.model_factory = model_factory
        self.use_ea = use_ea
        self.use_feature_selection = use_feature_selection
        self.k_features = k_features
        self.n_jobs = n_jobs
        self.rest_class = rest_class
        self.verbose = verbose
        self.results_: Optional[Dict] = None

    def run(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray,
            classes: Optional[List[str]] = None) -> Dict:
        """
        Run LOSO evaluation.

        Parameters
        ----------
        X : (n_samples, n_features) feature matrix
        y : (n_samples,) labels
        groups : (n_samples,) subject IDs
        classes : list of class names (default = sorted unique y)

        Returns
        -------
        dict with per_fold, aggregate, statistics
        """
        if classes is None:
            classes = sorted(np.unique(y).tolist())

        unique_subjects = np.unique(groups)
        n_folds = len(unique_subjects)
        if self.verbose:
            logger.info(f"Starting LOSO: {n_folds} folds, "
                        f"{X.shape[0]} samples, {X.shape[1]} features")

        splits = list(loso_splits(groups))

        # Run folds
        if self.n_jobs == 1:
            per_fold = [self._run_fold(i, X, y, groups, classes, train_idx, test_idx)
                        for i, (train_idx, test_idx) in enumerate(splits)]
        else:
            per_fold = Parallel(n_jobs=self.n_jobs)(
                delayed(self._run_fold)(i, X, y, groups, classes, train_idx, test_idx)
                for i, (train_idx, test_idx) in enumerate(splits)
            )

        # Aggregate
        accuracies = [f["metrics"]["accuracy"] for f in per_fold]
        macro_f1s = [f["metrics"]["macro_f1"] for f in per_fold]
        inflations = [f["metrics"]["inflation"] for f in per_fold]

        acc_mean, acc_std = float(np.mean(accuracies)), float(np.std(accuracies, ddof=1))
        f1_mean, f1_std = float(np.mean(macro_f1s)), float(np.std(macro_f1s, ddof=1))
        inf_mean = float(np.mean(inflations))

        acc_ci = bootstrap_ci(accuracies, statistic=np.mean, n_bootstrap=1000)
        f1_ci = bootstrap_ci(macro_f1s, statistic=np.mean, n_bootstrap=1000)

        self.results_ = {
            "protocol": "LOSO",
            "n_folds": n_folds,
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "per_fold": per_fold,
            "aggregate": {
                "accuracy_mean": acc_mean,
                "accuracy_std": acc_std,
                "accuracy_ci_95": acc_ci,
                "macro_f1_mean": f1_mean,
                "macro_f1_std": f1_std,
                "macro_f1_ci_95": f1_ci,
                "inflation_mean_pp": inf_mean,
                "per_fold_accuracies": accuracies,
                "per_fold_macro_f1s": macro_f1s,
                "per_fold_inflations": inflations,
            },
        }
        return self.results_

    def _run_fold(self, fold_idx: int, X: np.ndarray, y: np.ndarray,
                   groups: np.ndarray, classes: List[str],
                   train_idx: np.ndarray, test_idx: np.ndarray) -> Dict:
        t0 = time.time()
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Fold-level scaling
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Optional feature selection
        if self.use_feature_selection and self.k_features < X_train.shape[1]:
            from sklearn.feature_selection import SelectKBest, f_classif
            selector = SelectKBest(f_classif, k=self.k_features)
            X_train = selector.fit_transform(X_train, y_train)
            X_test = selector.transform(X_test)

        # Train
        if self.model_factory is None:
            raise ValueError("model_factory is required")
        model = self.model_factory()
        # Support both classical-style and DAN-style fit
        try:
            model.fit(X_train, y_train, groups=groups[train_idx])
        except TypeError:
            model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)
        metrics = compute_metrics(y_test, y_pred, classes=classes,
                                   rest_class=self.rest_class)
        elapsed = time.time() - t0

        if self.verbose:
            logger.info(
                f"  Fold {fold_idx + 1}/{len(np.unique(groups))} "
                f"acc={metrics['accuracy']:.4f} F1={metrics['macro_f1']:.4f} "
                f"inf={metrics['inflation']*100:.2f}pp ({elapsed:.1f}s)"
            )

        return {
            "fold": fold_idx,
            "test_subject": int(np.unique(groups[test_idx])[0]),
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "metrics": metrics,
            "elapsed_seconds": elapsed,
        }

    def compare(self, other_results: Dict[str, Dict]) -> Dict:
        """
        Statistical comparison against other models' LOSO results.

        Parameters
        ----------
        other_results : dict mapping model_name -> LOSO results dict
                       (must contain 'aggregate.per_fold_accuracies')

        Returns
        -------
        dict with friedman + wilcoxon results
        """
        if self.results_ is None:
            raise RuntimeError("Run .run() first")

        per_fold_scores = {"this_model": self.results_["aggregate"]["per_fold_accuracies"]}
        for name, res in other_results.items():
            per_fold_scores[name] = res["aggregate"]["per_fold_accuracies"]

        friedman = friedman_test(per_fold_scores)
        wilcoxon = wilcoxon_pairwise(per_fold_scores,
                                      reference="this_model")

        return {
            "friedman": friedman,
            "wilcoxon_vs_reference": wilcoxon,
        }
