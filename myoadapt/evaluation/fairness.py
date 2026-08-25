"""
fairness.py — Per-subgroup fairness & bias audit
=================================================

Computes the per-subgroup performance breakdowns that health-AI
reviewers (and EU AI Act high-risk-system obligations) increasingly
expect to see alongside aggregate accuracy.

The audit is subgroup-agnostic by design: callers pass a ``groups``
array (subject IDs, demographic labels, acquisition-site IDs, …) and
the module reports the same metric family for every group, plus
aggregate fairness summaries:

- ``per_subgroup_metrics`` — accuracy / macro-F1 / per-class F1 /
  Rest-class recall / ECE for each subgroup.
- ``disparities`` — max–min gap, standard deviation across groups,
  and the worst-performing subgroup for each metric.
- ``equalized_odds`` — per-subgroup TPR and FPR (bias-detection
  standard from Hardt et al. 2016).
- ``subgroup_calibration`` — per-subgroup ECE / Brier.

The output is a JSON-serialisable dict that plugs directly into a
Model Card (see :mod:`myoadapt.evaluation.model_card`) or into a
standalone fairness section of a paper.

References
----------
- Hardt, M. et al. (2016). *Equality of Opportunity in Supervised
  Learning.* NeurIPS.
- Mitchell, M. et al. (2019). *Model Cards for Model Reporting.*
  FAT*.
- Barocas, S. et al. (2019). *Fairness and Machine Learning.*
  fairmlbook.org.

License: Apache 2.0
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, Dict, Optional

import numpy as np

from myoadapt.evaluation.calibration import (
    brier_score,
    expected_calibration_error,
)
from myoadapt.evaluation.metrics import compute_metrics

logger = logging.getLogger(__name__)


def per_subgroup_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    classes: Optional[Sequence] = None,
    rest_class: Optional[str] = None,
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Per-subgroup performance breakdown.

    Parameters
    ----------
    y_true, y_pred : (n,) arrays of integer or string labels.
    groups : (n,) array of subgroup identifiers (subject IDs,
        demographic labels, acquisition sites, …).
    classes : optional list of all known labels (used to fix the
        confusion-matrix shape across subgroups with different label
        coverage).
    rest_class : optional label treated as "Rest" — included in the
        per-subgroup Rest-recall report.
    y_proba : optional (n, n_classes) probabilities, used to compute
        per-subgroup ECE / Brier.

    Returns
    -------
    dict with:
        ``per_subgroup`` : {group_id: {metric: value, "n": int}}
        ``disparities``  : {metric: {max, min, range, std, worst_group}}
        ``n_subgroups``  : int
        ``n_samples``    : int
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    groups = np.asarray(groups)
    if classes is None:
        classes = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist(),
                         key=lambda x: str(x))

    unique_groups = np.unique(groups)
    per_subgroup: Dict[Any, Dict[str, Any]] = {}
    for g in unique_groups:
        mask = groups == g
        n = int(mask.sum())
        if n == 0:
            continue
        m = compute_metrics(y_true[mask], y_pred[mask],
                            classes=list(classes), rest_class=rest_class)
        entry: Dict[str, Any] = {
            "n": n,
            "accuracy": float(m["accuracy"]),
            "macro_f1": float(m["macro_f1"]),
        }
        # Per-class F1.
        per_class = m.get("per_class", {})
        # ``per_class`` is {class_label: {precision, recall, f1, support}}.
        per_class_f1: Dict[str, float] = {}
        for k, v in per_class.items():
            if isinstance(v, dict):
                per_class_f1[str(k)] = float(v.get("f1", 0.0))
            else:
                per_class_f1[str(k)] = float(v)
        entry["per_class_f1"] = per_class_f1
        if rest_class is not None and str(rest_class) in per_class:
            v = per_class[str(rest_class)]
            entry["rest_recall"] = float(v["recall"]) if isinstance(v, dict) else float(v)
        # Calibration if probabilities were provided.
        if y_proba is not None:
            entry["ece"] = expected_calibration_error(y_true[mask], y_proba[mask])
            entry["brier"] = brier_score(y_true[mask], y_proba[mask])
        per_subgroup[str(g)] = entry

    # Aggregate disparities.
    metric_keys = ["accuracy", "macro_f1"]
    if y_proba is not None:
        metric_keys += ["ece", "brier"]
    if rest_class is not None:
        metric_keys += ["rest_recall"]
    disparities: Dict[str, Dict[str, Any]] = {}
    for mk in metric_keys:
        vals = [(g, e[mk]) for g, e in per_subgroup.items() if mk in e]
        if not vals:
            continue
        only_vals = np.array([v for _, v in vals], dtype=float)
        disparities[mk] = {
            "max": float(only_vals.max()),
            "min": float(only_vals.min()),
            "range": float(only_vals.max() - only_vals.min()),
            "std": float(only_vals.std(ddof=0)) if len(only_vals) > 1 else 0.0,
            "mean": float(only_vals.mean()),
            "worst_group": str(vals[int(np.argmin(only_vals))][0]),
            "best_group": str(vals[int(np.argmax(only_vals))][0]),
        }

    return {
        "per_subgroup": per_subgroup,
        "disparities": disparities,
        "n_subgroups": int(len(unique_groups)),
        "n_samples": int(len(y_true)),
        "metric_keys": metric_keys,
    }


def equalized_odds(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    positive_class: Any = 1,
) -> Dict[str, Dict[str, float]]:
    """Per-subgroup TPR and FPR for a binary positive class.

    Equalized odds (Hardt et al. 2016) requires TPR and FPR to be
    equal across subgroups. This helper returns both quantities per
    subgroup so the gap is visible at a glance.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    groups = np.asarray(groups)
    out: Dict[str, Dict[str, float]] = {}
    for g in np.unique(groups):
        mask = groups == g
        yt = y_true[mask] == positive_class
        yp = y_pred[mask] == positive_class
        tp = int(np.sum(yt & yp))
        fn = int(np.sum(yt & ~yp))
        fp = int(np.sum(~yt & yp))
        tn = int(np.sum(~yt & ~yp))
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        out[str(g)] = {
            "tpr": float(tpr),
            "fpr": float(fpr),
            "n": int(mask.sum()),
            "support_positive": int(tp + fn),
            "support_negative": int(fp + tn),
        }
    return out


def fairness_summary(audit: Dict[str, Any]) -> str:
    """Human-readable one-paragraph summary of a fairness audit.

    Suitable for embedding in a Model Card's "Fairness analysis"
    section or in a paper appendix.
    """
    n = audit["n_subgroups"]
    n_samples = audit["n_samples"]
    parts = [f"Fairness audit: {n} subgroups, {n_samples} samples total."]
    for metric, disp in audit["disparities"].items():
        parts.append(
            f"{metric}: mean={disp['mean']:.3f}, "
            f"range=[{disp['min']:.3f}, {disp['max']:.3f}] "
            f"(gap={disp['range']:.3f}), worst='{disp['worst_group']}'."
        )
    return " ".join(parts)
