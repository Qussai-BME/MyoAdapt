"""
diagrams.py — Critical Difference diagrams and paper-ready figures
====================================================================

Produces the visual artefacts that academic reviewers expect to see in a
rigorous multi-model benchmark:

- **Critical Difference (CD) diagram** (Demsar 2006) from a Nemenyi
  post-hoc test on Friedman ranks. The CD diagram is the canonical
  figure for "are these K models significantly different across N folds?"
- **Per-fold accuracy box-plot** with paired-line overlay.
- **Per-class F1 heatmap** (gesture × model).
- **Reliability diagram** for calibrated vs uncalibrated probabilities
  (companion to :mod:`myoadapt.evaluation.calibration`).

All figures use a consistent, paper-ready matplotlib style. Figures are
returned as ``matplotlib.figure.Figure`` objects so callers can either
``savefig`` them or display them inline in a notebook / Streamlit page.

References
----------
- Demsar, J. (2006). *Statistical Comparisons of Classifiers over
  Multiple Data Sets.* JMLR 7, 1–30.
- Benavoli, A. et al. (2017). *Time for a Change: a Tutorial for
  Comparing Multiple Classifiers through Bayesian Analysis.* JMLR.

License: Apache 2.0
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Dict, List, Optional, Tuple

import matplotlib
import numpy as np

# Use a non-interactive backend by default so figures render cleanly in
# headless containers (CI, Docker, Streamlit server). Callers can switch
# backends by setting ``matplotlib.use(...)`` before importing this
# module — the import-time guard respects that.
if matplotlib.get_backend().lower() not in {"agg", "module://matplotlib_inline.backend_inline"}:
    try:
        matplotlib.use("Agg")
    except Exception:  # pragma: no cover — backend switch is best-effort
        pass

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# Paper-ready defaults. The 3.5-inch width matches the IEEE / Elsevier
# single-column width; the colourblind-safe palette (Okabe-Ito) is the
# standard recommendation for accessibility.
_PAPER_RC = {
    "figure.figsize": (7.0, 3.5),
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
}


def _apply_paper_style() -> None:
    """Apply the paper-ready rcParams. Idempotent and safe to call repeatedly."""
    plt.rcParams.update(_PAPER_RC)


# Okabe-Ito colourblind-safe palette (8 colours).
OKABE_ITO = [
    "#000000", "#E69F00", "#56B4E9", "#009E73",
    "#F0E442", "#0072B2", "#D55E00", "#CC79A7",
]


# ---------------------------------------------------------------------------
# Critical Difference diagram (Demsar 2006, Nemenyi post-hoc)
# ---------------------------------------------------------------------------
def critical_difference_diagram(
    per_fold_scores: Dict[str, Sequence[float]],
    alpha: float = 0.05,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (8.0, 3.0),
) -> matplotlib.figure.Figure:
    """Render a Critical Difference (CD) diagram from a Nemenyi post-hoc.

    Parameters
    ----------
    per_fold_scores : dict
        ``{model_name: [per-fold scores]}``. All lists must have the
        same length ``N`` (the number of folds / datasets).
    alpha : float
        Family-wise significance level for the Nemenyi CD. Default 0.05.
    title : optional str
        Figure title. Defaults to "Critical Difference diagram (Nemenyi, α=…)".
    figsize : tuple
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object. Caller is responsible for ``plt.savefig`` or
        ``plt.show``.

    Notes
    -----
    The CD value is computed from the studentised-range statistic
    ``q_alpha`` (Demsar 2006, Eq. 5):

        CD = q_alpha * sqrt(k(k+1) / (6N))

    where ``k`` is the number of models and ``N`` the number of folds.
    Models whose average ranks differ by less than CD are not
    significantly different and are joined by a thick horizontal line.
    """
    _apply_paper_style()
    from scipy.stats import rankdata

    # Studentised-range critical values q_alpha for alpha=0.05 and 0.10,
    # indexed by k (number of classifiers). Source: Demsar 2006 Table
    # 5(a) / 5(b) (after Hart 2001). For k > 10 we fall back to the
    # Bonferroni-Dunn approximation (Demsar 2006 §3.2.2) — this is the
    # standard pragmatic choice in the ML literature.
    Q_ALPHA_05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
                  7: 2.948, 8: 3.031, 9: 3.102, 10: 3.164}
    Q_ALPHA_10 = {2: 1.645, 3: 2.052, 4: 2.291, 5: 2.460, 6: 2.586,
                  7: 2.686, 8: 2.770, 9: 2.840, 10: 2.903}

    model_names = list(per_fold_scores.keys())
    k = len(model_names)
    if k < 2:
        raise ValueError("CD diagram requires at least 2 models")
    n = min(len(v) for v in per_fold_scores.values())
    if n < 2:
        raise ValueError("CD diagram requires at least 2 folds per model")

    # Truncate to the common fold count.
    matrix = np.array([per_fold_scores[m][:n] for m in model_names], dtype=float).T  # (n, k)

    # Per-fold ranks: smallest score = rank 1. We rank the *negation*
    # so the best (highest accuracy) model gets rank 1.
    ranks = np.zeros_like(matrix, dtype=float)
    for i in range(n):
        ranks[i] = rankdata(-matrix[i], method="average")
    avg_ranks = ranks.mean(axis=0)

    # Critical difference.
    if k <= 10:
        q = Q_ALPHA_05 if abs(alpha - 0.05) < 1e-3 else Q_ALPHA_10
        q_alpha = q.get(k, q[10])
    else:
        # Bonferroni-Dunn fallback for k > 10.
        from scipy.stats import t as student_t
        # Two-sided Bonferroni-corrected t critical value.
        q_alpha = student_t.ppf(1 - alpha / (k * (k - 1)), df=float("inf"))
    cd = q_alpha * np.sqrt(k * (k + 1) / (6.0 * n))

    # Order models by rank (left = best).
    order = np.argsort(avg_ranks)
    sorted_names = [model_names[i] for i in order]
    sorted_ranks = avg_ranks[order]

    fig, ax = plt.subplots(figsize=figsize)

    # The horizontal axis spans the rank space [0.5, k+0.5].
    ax.set_xlim(0.5, k + 0.5)
    ax.set_ylim(0, 1)

    # Top axis: rank tick marks.
    for i, r in enumerate(range(1, k + 1)):
        ax.plot([r, r], [0.92, 0.99], color="black", lw=0.8)
        ax.text(r, 1.02, str(r), ha="center", va="bottom", fontsize=9)
    ax.plot([0.5, k + 0.5], [0.99, 0.99], color="black", lw=0.8)
    ax.text(0.5, 1.10, "Rank", ha="left", va="bottom", fontsize=9, fontweight="bold")

    # Model labels along the bottom, alternating up/down for legibility.
    for i, (name, rank) in enumerate(zip(sorted_names, sorted_ranks)):
        y_text = 0.30 if i % 2 == 0 else 0.10
        y_line = 0.40 if i % 2 == 0 else 0.20
        ax.plot([rank, rank], [0.45, y_line + 0.02], color="black", lw=0.6)
        ax.text(rank, y_text, name, ha="center", va="center", fontsize=8.5)

    # CD bar at the top-left.
    cd_left = 0.6
    cd_y = 0.78
    ax.plot([cd_left, cd_left + cd], [cd_y, cd_y], color="black", lw=1.5)
    ax.plot([cd_left, cd_left], [cd_y - 0.03, cd_y + 0.03], color="black", lw=1.5)
    ax.plot([cd_left + cd, cd_left + cd], [cd_y - 0.03, cd_y + 0.03], color="black", lw=1.5)
    ax.text(cd_left + cd / 2, cd_y + 0.06, f"CD = {cd:.2f}",
            ha="center", va="bottom", fontsize=9)

    # Significance groups: thick horizontal bars joining models whose
    # rank difference is < CD.
    groups: List[Tuple[int, int]] = []
    for i in range(k):
        for j in range(i + 1, k):
            if abs(sorted_ranks[i] - sorted_ranks[j]) < cd:
                groups.append((i, j))
    # Collapse overlapping groups into single bars (visual simplification).
    if groups:
        merged: List[Tuple[int, int]] = []
        for a, b in groups:
            if merged and a <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], b))
            else:
                merged.append((a, b))
        for y_offset, (a, b) in enumerate(merged):
            x_left = min(sorted_ranks[a], sorted_ranks[b]) - 0.1
            x_right = max(sorted_ranks[a], sorted_ranks[b]) + 0.1
            y = 0.58 - y_offset * 0.04
            ax.plot([x_left, x_right], [y, y], color=OKABE_ITO[2], lw=2.5, solid_capstyle="round")

    ax.set_title(title or f"Critical Difference diagram (Nemenyi, α={alpha:.2f})")
    ax.axis("off")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Per-fold accuracy box plot with paired overlay
# ---------------------------------------------------------------------------
def per_fold_boxplot(
    per_fold_scores: Dict[str, Sequence[float]],
    metric_name: str = "Accuracy",
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (6.0, 3.5),
) -> matplotlib.figure.Figure:
    """Per-fold accuracy box-plot with paired lines connecting the same fold.

    The paired overlay reveals whether per-model differences are
    consistent across folds (parallel lines) or fold-dependent
    (crossing lines) — a visual cue for the Friedman test result.
    """
    _apply_paper_style()
    model_names = list(per_fold_scores.keys())
    if len(model_names) < 2:
        raise ValueError("Box-plot requires at least 2 models")
    data = [np.asarray(per_fold_scores[m], dtype=float) for m in model_names]
    n = min(len(d) for d in data)
    data = [d[:n] for d in data]
    matrix = np.column_stack(data)  # (n, k)

    fig, ax = plt.subplots(figsize=figsize)

    # Paired lines.
    for i in range(n):
        ax.plot(range(1, len(model_names) + 1), matrix[i],
                color="gray", alpha=0.35, lw=0.8, marker=".", markersize=3)

    # Box-plot on top.
    bp = ax.boxplot(
        data, positions=range(1, len(model_names) + 1),
        widths=0.45, patch_artist=True, showfliers=False,
    )
    for patch, color in zip(bp["boxes"], OKABE_ITO[1:]):
        patch.set_facecolor(color)
        patch.set_alpha(0.45)
        patch.set_edgecolor(color)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.5)

    ax.set_xticks(range(1, len(model_names) + 1))
    ax.set_xticklabels(model_names, rotation=15, ha="right")
    ax.set_ylabel(metric_name)
    ax.set_title(title or f"Per-fold {metric_name.lower()}")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Per-class F1 heatmap
# ---------------------------------------------------------------------------
def per_class_heatmap(
    per_class_f1: Dict[str, Dict[str, float]],
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (7.0, 4.0),
    cmap: str = "viridis",
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> matplotlib.figure.Figure:
    """Per-class F1 heatmap (rows = models, cols = classes).

    Parameters
    ----------
    per_class_f1 : dict
        ``{model_name: {class_label: f1_score}}``. Missing entries are
        rendered as NaN (blank cell).
    """
    _apply_paper_style()
    model_names = list(per_class_f1.keys())
    if not model_names:
        raise ValueError("Empty per-class F1 dict")
    # Union of all class labels, sorted.
    classes = sorted({c for m in per_class_f1.values() for c in m.keys()},
                     key=lambda x: str(x))
    matrix = np.full((len(model_names), len(classes)), np.nan, dtype=float)
    for i, m in enumerate(model_names):
        for j, c in enumerate(classes):
            if c in per_class_f1[m]:
                matrix[i, j] = per_class_f1[m][c]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names)
    ax.set_xlabel("Class")
    ax.set_title(title or "Per-class F1 heatmap")

    # Annotate cells with their value.
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if np.isfinite(v):
                color = "white" if v < (vmin + vmax) / 2 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color=color, fontsize=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("F1")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Reliability diagram (companion to calibration module)
# ---------------------------------------------------------------------------
def reliability_diagram(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
    title: Optional[str] = None,
    label: Optional[str] = None,
    color: Optional[str] = None,
    figsize: Tuple[float, float] = (4.5, 4.5),
) -> matplotlib.figure.Figure:
    """Reliability (calibration) diagram.

    Parameters
    ----------
    y_true : (n_samples,) integer class labels
    y_proba : (n_samples, n_classes) predicted probabilities
    n_bins : number of bins for the confidence axis
    label : legend label for the curve
    color : curve color
    """
    _apply_paper_style()
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    if y_proba.ndim == 1:
        # Binary: treat as P(class=1).
        confidences = y_proba
        correct = (y_true == 1).astype(float)
    else:
        confidences = y_proba.max(axis=1)
        preds = y_proba.argmax(axis=1)
        correct = (preds == y_true).astype(float)

    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_acc = np.zeros(n_bins)
    bin_conf = np.zeros(n_bins)
    bin_count = np.zeros(n_bins, dtype=int)
    for i in range(n_bins):
        mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
        if i == n_bins - 1:  # inclusive right edge
            mask = mask | (confidences == bins[i + 1])
        if mask.any():
            bin_acc[i] = correct[mask].mean()
            bin_conf[i] = confidences[mask].mean()
            bin_count[i] = mask.sum()

    fig, ax = plt.subplots(figsize=figsize)
    # Perfect-calibration diagonal.
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Perfect")
    # Reliability bars / line.
    valid = bin_count > 0
    ax.bar(bin_centers[valid], bin_acc[valid], width=0.8 / n_bins,
           alpha=0.6, color=color or OKABE_ITO[2],
           edgecolor=color or OKABE_ITO[2], label=label or "Model")
    ax.plot(bin_centers[valid], bin_conf[valid], "o-",
            color=color or OKABE_ITO[2], lw=1.5)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of correct predictions")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title or "Reliability diagram")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
