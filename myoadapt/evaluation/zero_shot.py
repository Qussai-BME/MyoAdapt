"""
zero_shot.py — Zero-Shot Cross-User evaluation protocol.

Trains on a large pool of subjects (e.g., 400+ from EMG-EPN612)
and evaluates zero-shot on completely unseen subjects — no
adaptation, no calibration. This is the strictest generalization
test: can the model generalize to a new user with ZERO
subject-specific data?

This complements LOSO (cross-subject within one database) and
LODO (cross-database). The three protocols form a hierarchy:
  - LOSO: within-database cross-subject
  - LODO: cross-database
  - Zero-Shot: large-scale cross-user (no adaptation)

References:
- EMG-EPN612: "A 612-subject sEMG database for myoelectric
  control." (2024) — largest public sEMG gesture dataset.
- Yang et al. (2025). "Self-supervised pretraining for zero-shot
  EMG gesture recognition."
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from sklearn.preprocessing import StandardScaler

from myoadapt.evaluation.metrics import compute_metrics
from myoadapt.evaluation.statistics import (
    bootstrap_ci,
    cohen_dz,
    interpret_effect_size,
    paired_wilcoxon_test,
)

logger = logging.getLogger(__name__)


class ZeroShotEvaluator:
    """
    Zero-Shot Cross-User evaluator.

    Splits the subject pool into a *training pool* (hundreds of
    subjects) and a disjoint *test pool* of completely unseen
    subjects. A single model is trained on ALL training-pool data
    and then evaluated on EACH test subject individually — with no
    per-subject adaptation, no calibration, no fine-tuning.

    This is the strictest generalization test in the MyoAdapt
    hierarchy and is the protocol of choice for large-scale public
    databases such as EMG-EPN612 (612 subjects).

    Parameters
    ----------
    model_factory : callable
        Returns a fresh model each call (so state does not leak
        between the training pool and any LOSO comparison). Pass
        e.g. ``model_factory=lambda: EMGClassifier('xgboost')``.
    n_train_subjects : int, optional
        Number of subjects to draw for the training pool. If
        ``None`` (default), uses 80% of the available subjects.
    n_test_subjects : int, optional
        Number of held-out subjects to evaluate zero-shot. If
        ``None`` (default), uses the remaining 20% of subjects.
    random_state : int
        Seed for the reproducible subject shuffle.
    rest_class : str
        Label treated as the Rest class for the inflation analysis
        (passed through to :func:`compute_metrics`).
    verbose : bool
        If True, log per-subject progress.

    Attributes
    ----------
    results_ : dict or None
        Populated by :meth:`run`. ``None`` until ``run`` is called.
    """

    def __init__(self,
                 model_factory: Callable,
                 n_train_subjects: Optional[int] = None,
                 n_test_subjects: Optional[int] = None,
                 random_state: int = 42,
                 rest_class: str = "rest",
                 verbose: bool = True):
        self.model_factory = model_factory
        self.n_train_subjects = n_train_subjects
        self.n_test_subjects = n_test_subjects
        self.random_state = random_state
        self.rest_class = rest_class
        self.verbose = verbose
        self.results_: Optional[Dict] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self,
            X: np.ndarray,
            y: np.ndarray,
            groups: np.ndarray,
            classes: Optional[List[str]] = None) -> Dict:
        """Run the Zero-Shot Cross-User evaluation.

        Parameters
        ----------
        X : (n_samples, n_features) feature matrix.
        y : (n_samples,) labels.
        groups : (n_samples,) subject IDs.
        classes : optional list of class names
            (default = sorted unique ``y``).

        Returns
        -------
        dict with:
            ``protocol``             : "ZeroShot"
            ``n_train_subjects``      : int
            ``n_test_subjects``       : int
            ``train_subjects``       : list of subject IDs in train pool
            ``test_subjects``        : list of subject IDs in test pool
            ``per_subject``          : list of per-test-subject dicts
            ``aggregate``            : {accuracy_mean/std,
                                        macro_f1_mean/std,
                                        accuracy_ci_95,
                                        macro_f1_ci_95,
                                        per_subject_accuracies,
                                        per_subject_macro_f1s}
            ``worst_subject``        : {subject, accuracy, macro_f1}
            ``best_subject``         : {subject, accuracy, macro_f1}
            ``n_samples``            : int
            ``n_features``           : int
        """
        X = np.asarray(X)
        y = np.asarray(y)
        groups = np.asarray(groups)
        if classes is None:
            classes = sorted(np.unique(y).tolist())

        unique_subjects = np.unique(groups)
        n_total = len(unique_subjects)
        if n_total < 3:
            raise ValueError(
                f"ZeroShotEvaluator requires at least 3 subjects, got {n_total}."
            )

        # Resolve train / test pool sizes.
        n_train, n_test = self._resolve_pool_sizes(n_total)
        if n_train + n_test > n_total:
            # Clip rather than crash.
            n_test = max(1, n_total - n_train)

        # Reproducible subject shuffle.
        rng = np.random.default_rng(self.random_state)
        perm = rng.permutation(n_total)
        train_subjects = unique_subjects[perm[:n_train]].tolist()
        test_subjects = unique_subjects[perm[n_train:n_train + n_test]].tolist()

        if self.verbose:
            logger.info(
                f"ZeroShot: train_pool={n_train} subjects, "
                f"test_pool={n_test} subjects, "
                f"{X.shape[0]} samples, {X.shape[1]} features"
            )

        train_idx = np.where(np.isin(groups, train_subjects))[0]
        test_idx = np.where(np.isin(groups, test_subjects))[0]

        # ---- Train ONE model on ALL train-pool data -------------------
        t0 = time.time()
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        y_train = y[train_idx]

        model = self.model_factory()
        # Support both classical-style and DAN-style fit.
        try:
            model.fit(X_train, y_train, groups=groups[train_idx])
        except TypeError:
            model.fit(X_train, y_train)

        train_seconds = time.time() - t0
        if self.verbose:
            logger.info(
                f"  Trained on {len(train_idx)} samples "
                f"({n_train} subjects) in {train_seconds:.1f}s"
            )

        # ---- Evaluate zero-shot on EACH test subject ------------------
        per_subject: List[Dict[str, Any]] = []
        for i, subj in enumerate(test_subjects):
            subj_mask = groups == subj
            if subj_mask.sum() == 0:
                continue
            X_subj = scaler.transform(X[subj_mask])
            y_subj = y[subj_mask]
            t_subj = time.time()
            y_pred = model.predict(X_subj)
            m = compute_metrics(y_subj, y_pred, classes=classes,
                                 rest_class=self.rest_class)
            elapsed = time.time() - t_subj
            entry = {
                "subject": _coerce_scalar(subj),
                "n_samples": int(subj_mask.sum()),
                "accuracy": float(m["accuracy"]),
                "macro_f1": float(m["macro_f1"]),
                "weighted_f1": float(m["weighted_f1"]),
                "kappa": float(m["kappa"]),
                "inflation": float(m["inflation"]),
                "metrics": m,
                "elapsed_seconds": elapsed,
            }
            per_subject.append(entry)
            if self.verbose:
                logger.info(
                    f"  Test subject {i + 1}/{n_test} (id={subj}): "
                    f"acc={m['accuracy']:.4f} F1={m['macro_f1']:.4f} "
                    f"({elapsed:.2f}s)"
                )

        # ---- Aggregate ------------------------------------------------
        accuracies = [e["accuracy"] for e in per_subject]
        macro_f1s = [e["macro_f1"] for e in per_subject]
        acc_mean = float(np.mean(accuracies)) if accuracies else 0.0
        acc_std = float(np.std(accuracies, ddof=1)) if len(accuracies) > 1 else 0.0
        f1_mean = float(np.mean(macro_f1s)) if macro_f1s else 0.0
        f1_std = float(np.std(macro_f1s, ddof=1)) if len(macro_f1s) > 1 else 0.0
        acc_ci = bootstrap_ci(accuracies, statistic=np.mean,
                              n_bootstrap=1000,
                              random_state=self.random_state) if len(accuracies) > 1 else (acc_mean, acc_mean)
        f1_ci = bootstrap_ci(macro_f1s, statistic=np.mean,
                             n_bootstrap=1000,
                             random_state=self.random_state) if len(macro_f1s) > 1 else (f1_mean, f1_mean)

        worst = min(per_subject, key=lambda e: e["accuracy"]) if per_subject else None
        best = max(per_subject, key=lambda e: e["accuracy"]) if per_subject else None

        self.results_ = {
            "protocol": "ZeroShot",
            "n_train_subjects": int(n_train),
            "n_test_subjects": int(n_test),
            "train_subjects": [_coerce_scalar(s) for s in train_subjects],
            "test_subjects": [_coerce_scalar(s) for s in test_subjects],
            "per_subject": per_subject,
            "aggregate": {
                "accuracy_mean": acc_mean,
                "accuracy_std": acc_std,
                "accuracy_ci_95": [float(acc_ci[0]), float(acc_ci[1])],
                "macro_f1_mean": f1_mean,
                "macro_f1_std": f1_std,
                "macro_f1_ci_95": [float(f1_ci[0]), float(f1_ci[1])],
                "per_subject_accuracies": accuracies,
                "per_subject_macro_f1s": macro_f1s,
            },
            "worst_subject": _subject_summary(worst),
            "best_subject": _subject_summary(best),
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "train_seconds": float(train_seconds),
        }
        return self.results_

    def compare_with_loso(self,
                          X: np.ndarray,
                          y: np.ndarray,
                          groups: np.ndarray,
                          classes: Optional[List[str]] = None) -> Dict:
        """Run ZeroShot AND LOSO on the same data; return a comparison.

        The two protocols are evaluated on the same (X, y, groups)
        tensors so the comparison is apples-to-apples. The LOSO
        evaluation reuses the same ``model_factory`` and the same
        per-fold preprocessing (StandardScaler) — only the splitting
        strategy differs.

        Returns
        -------
        dict with:
            ``zero_shot``      : ZeroShot ``run`` output
            ``loso``           : LOSO ``run`` output
            ``comparison``     : {
                ``accuracy_gap_pp``, ``macro_f1_gap_pp``,
                ``paired_wilcoxon``, ``cohen_dz``, ``effect_interpretation``
            }
        """
        # Local import to avoid a hard cycle when this module is
        # imported very early in package init.
        from myoadapt.evaluation.loso import LOSOEvaluator

        zs = self.run(X, y, groups, classes=classes)

        loso = LOSOEvaluator(
            model_factory=self.model_factory,
            rest_class=self.rest_class,
            verbose=self.verbose,
        )
        loso_results = loso.run(X, y, groups, classes=classes)

        # Align per-subject accuracies between the two protocols by
        # intersecting their subject IDs (ZeroShot only has the test
        # pool; LOSO has every subject).
        zs_acc_by_subj = {e["subject"]: e["accuracy"] for e in zs["per_subject"]}
        loso_acc_by_subj = {
            int(e["test_subject"]) if isinstance(e["test_subject"], (np.integer, int)) else e["test_subject"]:
                e["metrics"]["accuracy"]
            for e in loso_results["per_fold"]
        }
        common = sorted(set(zs_acc_by_subj.keys()) & set(loso_acc_by_subj.keys()),
                        key=lambda x: str(x))
        zs_vec = np.array([zs_acc_by_subj[s] for s in common], dtype=float)
        loso_vec = np.array([loso_acc_by_subj[s] for s in common], dtype=float)

        comparison: Dict[str, Any] = {
            "n_overlapping_subjects": int(len(common)),
            "zero_shot_accuracy_mean": float(zs["aggregate"]["accuracy_mean"]),
            "loso_accuracy_mean": float(loso_results["aggregate"]["accuracy_mean"]),
            "accuracy_gap_pp": float(
                (zs["aggregate"]["accuracy_mean"]
                 - loso_results["aggregate"]["accuracy_mean"]) * 100.0
            ),
            "macro_f1_gap_pp": float(
                (zs["aggregate"]["macro_f1_mean"]
                 - loso_results["aggregate"]["macro_f1_mean"]) * 100.0
            ),
        }
        if len(common) >= 5:
            wilcoxon = paired_wilcoxon_test(zs_vec, loso_vec)
            dz = cohen_dz(zs_vec - loso_vec)
            comparison["paired_wilcoxon"] = wilcoxon
            comparison["cohen_dz"] = dz
            comparison["effect_interpretation"] = interpret_effect_size(dz)
        else:
            comparison["paired_wilcoxon"] = None
            comparison["cohen_dz"] = None
            comparison["effect_interpretation"] = (
                "insufficient overlap for paired test "
                f"(need ≥5 subjects, got {len(common)})"
            )

        return {
            "zero_shot": zs,
            "loso": loso_results,
            "comparison": comparison,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _resolve_pool_sizes(self, n_total: int) -> tuple[int, int]:
        """Determine (n_train, n_test) subject counts."""
        if self.n_train_subjects is not None and self.n_test_subjects is not None:
            n_train = min(self.n_train_subjects, n_total - 1)
            n_test = min(self.n_test_subjects, n_total - n_train)
            return max(1, n_train), max(1, n_test)

        if self.n_train_subjects is not None:
            n_train = min(self.n_train_subjects, n_total - 1)
            n_test = max(1, n_total - n_train)
            return n_train, n_test

        if self.n_test_subjects is not None:
            n_test = min(self.n_test_subjects, n_total - 1)
            n_train = max(1, n_total - n_test)
            return n_train, n_test

        # Defaults: 80 / 20 split.
        n_test = max(1, int(round(n_total * 0.2)))
        n_train = max(1, n_total - n_test)
        return n_train, n_test


# ----------------------------------------------------------------------
# Module-level helpers
# ----------------------------------------------------------------------
def _coerce_scalar(v) -> Any:
    """Convert numpy scalars to native Python types (for JSON safety)."""
    if isinstance(v, np.generic):
        return v.item()
    return v


def _subject_summary(entry: Optional[Dict]) -> Optional[Dict]:
    """Reduce a per_subject entry to a compact worst/best summary."""
    if entry is None:
        return None
    return {
        "subject": entry["subject"],
        "accuracy": float(entry["accuracy"]),
        "macro_f1": float(entry["macro_f1"]),
        "n_samples": int(entry["n_samples"]),
    }
