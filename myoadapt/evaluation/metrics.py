"""
metrics.py — Classification metrics
====================================

Includes the Rest-class inflation analysis:
    overall_accuracy - active_only_accuracy = inflation

On imbalanced amputee populations (e.g. NinaPro DB3), Rest can dominate
the test set; standard accuracy then overestimates true clinical utility.
The ``inflation`` field quantifies this gap.
"""
from __future__ import annotations

from typing import Dict, List, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def _coerce_rest_class(rest_class: Union[str, int, None],
                       classes: List,
                       y_true: np.ndarray):
    """Return a rest-class label that matches the dtype of ``classes``.

    ``classes`` may be strings (e.g. ``['0', '1', ...]``) or integers
    (e.g. ``[0, 1, ...]``). If ``rest_class`` is provided as a string but
    ``classes`` holds integers (or vice versa), attempt to coerce.
    Returns ``None`` if the rest class is absent from ``classes``.
    """
    if rest_class is None:
        return None
    # If rest_class is already in classes, return as-is
    if rest_class in classes:
        return rest_class
    # Try numeric coercion
    try:
        rc_int = int(rest_class)
        if rc_int in classes:
            return rc_int
    except (TypeError, ValueError):
        pass
    # Try string coercion
    rc_str = str(rest_class)
    if rc_str in classes:
        return rc_str
    return None


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    classes: List,
                    rest_class: Union[str, int, None] = "rest") -> Dict:
    """
    Compute comprehensive classification metrics.

    Includes:
    - Overall accuracy, macro-F1, weighted-F1
    - Rest recall and active-only accuracy (when a rest class is present)
    - Accuracy inflation (overall - active_only) — quantifies how much
      of the headline accuracy is driven by Rest-class dominance
    - Per-class F1
    - Confusion matrices (overall + active-only)
    - Cohen's kappa

    The ``rest_class`` argument is matched against ``classes`` with
    dtype coercion so both integer labels (``rest_class=0``) and string
    labels (``rest_class='rest'``) work.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro",
                               zero_division=0, labels=classes))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted",
                                  zero_division=0, labels=classes))
    kappa = float(cohen_kappa_score(y_true, y_pred))

    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0)
    per_class = {classes[i]: {
        "precision": float(p[i]),
        "recall": float(r[i]),
        "f1": float(f1[i]),
        "support": int(support[i]),
    } for i in range(len(classes))}

    rest_recall = 0.0
    matched_rest = _coerce_rest_class(rest_class, classes, y_true)
    if matched_rest is not None:
        rest_idx = classes.index(matched_rest)
        rest_recall = float(r[rest_idx])

    # Active-only (exclude rest)
    if matched_rest is not None:
        active_mask = y_true != matched_rest
    else:
        active_mask = np.ones(len(y_true), dtype=bool)
    if active_mask.sum() > 0:
        active_acc = float(accuracy_score(
            y_true[active_mask], y_pred[active_mask]))
        active_classes = [c for c in classes if c != matched_rest]
        active_f1 = float(f1_score(
            y_true[active_mask], y_pred[active_mask],
            average="macro", zero_division=0, labels=active_classes)
        ) if active_classes else 0.0
    else:
        active_acc = 0.0
        active_f1 = 0.0

    inflation = accuracy - active_acc

    cm_overall = confusion_matrix(y_true, y_pred, labels=classes)
    active_indices = [i for i, c in enumerate(classes) if c != matched_rest]
    if active_indices and matched_rest is not None:
        cm_active = confusion_matrix(
            y_true[active_mask], y_pred[active_mask],
            labels=[c for c in classes if c != matched_rest])
    else:
        cm_active = cm_overall

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "kappa": kappa,
        "rest_recall": rest_recall,
        "active_only_accuracy": active_acc,
        "active_only_macro_f1": active_f1,
        "inflation": inflation,
        "per_class": per_class,
        "confusion_matrix_overall": cm_overall.tolist(),
        "confusion_matrix_active": cm_active.tolist(),
        "n_samples": int(len(y_true)),
    }


def per_class_report(metrics: Dict) -> str:
    """Pretty-print per-class precision/recall/F1."""
    lines = []
    lines.append(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
    lines.append("-" * 65)
    for cls, m in metrics["per_class"].items():
        lines.append(
            f"{cls:<15} {m['precision']:<12.4f} {m['recall']:<12.4f} "
            f"{m['f1']:<12.4f} {m['support']:<10}"
        )
    lines.append("-" * 65)
    lines.append(f"Overall acc:   {metrics['accuracy']:.4f}")
    lines.append(f"Macro F1:      {metrics['macro_f1']:.4f}")
    lines.append(f"Weighted F1:   {metrics['weighted_f1']:.4f}")
    lines.append(f"Active acc:    {metrics['active_only_accuracy']:.4f}")
    lines.append(f"Inflation:     {metrics['inflation']*100:.2f}pp")
    return "\n".join(lines)
