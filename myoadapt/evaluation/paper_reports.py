"""
paper_reports.py — LaTeX-ready tables and figure bundles for papers
====================================================================

Produces the artefacts academic reviewers expect in a multi-model
benchmark paper:

- ``main_results_table_tex`` — main results table (mean ± std per
  metric, per model) formatted as a LaTeX ``tabular`` ready to paste
  into a manuscript.
- ``per_fold_table_tex`` — per-fold accuracy / macro-F1 table.
- ``effect_size_table_tex`` — pairwise effect-size table (Cohen's d,
  Hedges' g, rank-biserial r) with qualitative interpretation column.
- ``stats_table_tex`` — Friedman χ² + p, Nemenyi CD, Wilcoxon
  Holm-Šídák corrected p-values, in one compact table.
- ``write_figure_bundle`` — saves a directory of PNG + PDF + LaTeX
  caption files for the four canonical figures (CD diagram, per-fold
  box-plot, per-class heatmap, reliability diagram).

All LaTeX tables use ``booktabs`` (``\\toprule``, ``\\midrule``,
``\\bottomrule``) — the de-facto standard for ICLR / NeurIPS / TBME /
JNER submissions. Tables are returned as strings so callers can write
them to ``.tex`` files or paste them directly into a manuscript.

License: Apache 2.0
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from myoadapt.evaluation.statistics import (
    bootstrap_ci_bca,
    cohen_d,
    friedman_test,
    hedges_g,
    interpret_effect_size,
    paired_wilcoxon_test,
    wilcoxon_pairwise,
)

logger = logging.getLogger(__name__)


def _escape_latex(s: str) -> str:
    """Escape characters that have special meaning in LaTeX."""
    repl = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
            "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}"}
    return "".join(repl.get(c, c) for c in str(s))


def _fmt_pm(mean: float, std: float, digits: int = 3) -> str:
    """Format mean ± std with consistent digits."""
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def main_results_table_tex(
    per_model_results: Dict[str, Dict[str, float]],
    metrics: Sequence[str] = ("accuracy", "macro_f1"),
    caption: str = "Main results (mean $\\pm$ std across folds).",
    label: str = "tab:main_results",
    digits: int = 3,
) -> str:
    """Format a main-results table as a LaTeX ``tabular``.

    Parameters
    ----------
    per_model_results : dict
        ``{model_name: {metric_mean: float, metric_std: float, ...}}``.
        The keys ``f"{metric}_mean"`` and ``f"{metric}_std"`` are
        expected for every metric in ``metrics``.
    metrics : tuple of metric base-names.
    caption, label : LaTeX caption / label for the float.
    """
    n_models = len(per_model_results)
    if n_models == 0:
        return "% (empty main_results_table_tex)"
    cols = "l" + "c" * len(metrics)
    header = " & ".join([_escape_latex(m) for m in metrics])
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\begin{tabular}{" + cols + r"}",
        r"\toprule",
        f"Model & {header} \\\\",
        r"\midrule",
    ]
    for model, results in per_model_results.items():
        cells = [_escape_latex(model)]
        for m in metrics:
            mean = results.get(f"{m}_mean", float("nan"))
            std = results.get(f"{m}_std", 0.0)
            cells.append(_fmt_pm(mean, std, digits))
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def per_fold_table_tex(
    per_model_folds: Dict[str, Sequence[float]],
    metric_name: str = "Accuracy",
    caption: Optional[str] = None,
    label: str = "tab:per_fold",
    digits: int = 3,
) -> str:
    """Per-fold scores table — one column per fold, one row per model.

    Includes a final ``Mean`` column.
    """
    n_models = len(per_model_folds)
    if n_models == 0:
        return "% (empty per_fold_table_tex)"
    n_folds = max(len(v) for v in per_model_folds.values())
    cols = "l" + "c" * (n_folds + 1)
    header = " & ".join([f"F{i+1}" for i in range(n_folds)] + ["Mean"])
    cap = caption or f"Per-fold {metric_name.lower()} for each model."
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{cap}}}",
        f"\\label{{{label}}}",
        r"\begin{tabular}{" + cols + r"}",
        r"\toprule",
        f"Model & {header} \\\\",
        r"\midrule",
    ]
    for model, folds in per_model_folds.items():
        cells = [_escape_latex(model)]
        row = list(folds) + [float("nan")] * (n_folds - len(folds))
        for v in row:
            cells.append(f"{v:.{digits}f}")
        cells.append(f"{np.mean(folds):.{digits}f}")
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def effect_size_table_tex(
    per_model_folds: Dict[str, Sequence[float]],
    caption: str = "Pairwise effect sizes (Cohen's $d$, Hedges' $g$, rank-biserial $r$).",
    label: str = "tab:effect_sizes",
    digits: int = 3,
) -> str:
    """Pairwise effect-size table (Cohen's d, Hedges' g, rank-biserial r).

    The rank-biserial *r* comes from the paired Wilcoxon signed-rank
    test; *d* and *g* are computed on the paired differences.
    """
    names = list(per_model_folds.keys())
    if len(names) < 2:
        return "% (effect_size_table_tex requires >= 2 models)"
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Pair & Cohen's $d$ & Hedges' $g$ & $r$ (Wilcoxon) & $p$ (Wilcoxon) & Interpretation \\",
        r"\midrule",
    ]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = np.asarray(per_model_folds[names[i]], dtype=float)
            b = np.asarray(per_model_folds[names[j]], dtype=float)
            n = min(len(a), len(b))
            if n < 2:
                continue
            a, b = a[:n], b[:n]
            d = cohen_d(a, b)
            g = hedges_g(a, b)
            wil = paired_wilcoxon_test(a, b)
            interp = interpret_effect_size(g)
            pair = f"{_escape_latex(names[i])} vs {_escape_latex(names[j])}"
            lines.append(
                f"{pair} & {d:.{digits}f} & {g:.{digits}f} & "
                f"{wil['effect_size_r']:.{digits}f} & {wil['p_value']:.{digits}f} & "
                f"{_escape_latex(interp)} \\\\"
            )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def stats_table_tex(
    per_model_folds: Dict[str, Sequence[float]],
    caption: str = "Statistical tests: Friedman + Nemenyi CD + Wilcoxon Holm-\\v{S}\\'\\i d\\'ak.",
    label: str = "tab:stats",
    digits: int = 4,
) -> str:
    """Combined statistical-test table.

    Top block: Friedman χ², p, n_models, n_folds.
    Middle block: Nemenyi CD value.
    Bottom block: pairwise Wilcoxon with Holm-Šídák correction.
    """
    names = list(per_model_folds.keys())
    fried = friedman_test(per_model_folds)
    # Nemenyi CD value (same formula as in diagrams.py).
    n = min(len(v) for v in per_model_folds.values())
    k = len(names)
    Q_ALPHA_05 = {2: 2.772, 3: 2.936, 4: 3.078, 5: 3.182, 6: 3.260,
                  7: 3.325, 8: 3.381, 9: 3.433, 10: 3.479}
    q_alpha = Q_ALPHA_05.get(k, Q_ALPHA_05[10]) if k <= 10 else 2.772
    cd = q_alpha * np.sqrt(k * (k + 1) / (6.0 * n))
    wilc = wilcoxon_pairwise(per_model_folds)

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\begin{tabular}{lc}",
        r"\toprule",
        r"Test & Value \\",
        r"\midrule",
        f"Friedman $\\chi^2$ & {fried['chi2']:.{digits}f} \\\\",
        f"Friedman $p$-value & {fried['p_value']:.{digits}f} \\\\",
        f"Models ($k$) & {k} \\\\",
        f"Folds ($N$) & {n} \\\\",
        f"Nemenyi CD ($\\alpha=0.05$) & {cd:.{digits}f} \\\\",
        r"\midrule",
        r"\multicolumn{2}{l}{\textbf{Pairwise Wilcoxon + Holm-\\v{S}\\'\\i d\\'ak}} \\",
        r"\midrule",
        r"Pair & $p_\\mathrm{corr}$ \\",
        r"\midrule",
    ]
    for t in wilc.get("tests", []):
        a, b = t["model_a"], t["model_b"]
        p_corr = t.get("p_value_corrected", t.get("p_value_raw", 1.0))
        lines.append(
            f"{_escape_latex(a)} vs {_escape_latex(b)} & {p_corr:.{digits}f} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def bca_ci_table_tex(
    per_model_folds: Dict[str, Sequence[float]],
    caption: str = "Bootstrap 95\\% confidence intervals (BCa, 2000 resamples).",
    label: str = "tab:bca_ci",
    digits: int = 3,
    n_bootstrap: int = 2000,
    random_state: int = 42,
) -> str:
    """Per-model BCa 95\\% CI for the mean of the per-fold scores.

    The BCa interval is the recommended default over the naive
    percentile bootstrap — see :func:`bootstrap_ci_bca`.
    """
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Model & Mean & 95\\% CI low & 95\\% CI high \\",
        r"\midrule",
    ]
    for model, folds in per_model_folds.items():
        ci = bootstrap_ci_bca(list(folds), n_bootstrap=n_bootstrap,
                              random_state=random_state)
        lines.append(
            f"{_escape_latex(model)} & {ci['point']:.{digits}f} & "
            f"{ci['lower']:.{digits}f} & {ci['upper']:.{digits}f} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Figure bundle
# ---------------------------------------------------------------------------
def write_figure_bundle(
    per_model_folds: Dict[str, Sequence[float]],
    output_dir: Path,
    per_class_f1: Optional[Dict[str, Dict[str, float]]] = None,
    y_true: Optional[np.ndarray] = None,
    y_proba: Optional[np.ndarray] = None,
    alpha: float = 0.05,
    figures: Sequence[str] = ("cd", "boxplot", "heatmap", "reliability"),
) -> Dict[str, Path]:
    """Write a bundle of paper-ready figures + LaTeX captions.

    Each figure is saved as both PNG (300 dpi) and PDF (vector). A
    sidecar ``.tex`` file contains a ready-to-use ``\\begin{figure}``
    block with a sensible caption.

    Returns
    -------
    dict mapping figure name → path of the PNG file.
    """
    from myoadapt.evaluation.diagrams import (
        critical_difference_diagram,
        per_class_heatmap,
        per_fold_boxplot,
        reliability_diagram,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, Path] = {}

    def _save(fig, name: str, caption: str, label: str) -> None:
        png = output_dir / f"{name}.png"
        pdf = output_dir / f"{name}.pdf"
        tex = output_dir / f"{name}.tex"
        fig.savefig(png, dpi=300, bbox_inches="tight")
        fig.savefig(pdf, bbox_inches="tight")
        tex.write_text(
            "\\begin{figure}[htbp]\n"
            "\\centering\n"
            f"\\includegraphics[width=0.9\\linewidth]{{{name}}}\n"
            f"\\caption{{{caption}}}\n"
            f"\\label{{{label}}}\n"
            "\\end{figure}\n",
            encoding="utf-8",
        )
        written[name] = png

    if "cd" in figures:
        fig = critical_difference_diagram(per_model_folds, alpha=alpha)
        _save(fig, "cd_diagram",
              "Critical Difference diagram (Nemenyi post-hoc, $\\alpha=0.05$). "
              "Models joined by a horizontal bar are not significantly different.",
              "fig:cd_diagram")
    if "boxplot" in figures:
        fig = per_fold_boxplot(per_model_folds)
        _save(fig, "per_fold_boxplot",
              "Per-fold accuracy distribution. Grey lines connect the same fold across models.",
              "fig:per_fold_boxplot")
    if "heatmap" in figures and per_class_f1 is not None:
        fig = per_class_heatmap(per_class_f1)
        _save(fig, "per_class_heatmap",
              "Per-class F1 heatmap (rows = models, columns = classes).",
              "fig:per_class_heatmap")
    if "reliability" in figures and y_true is not None and y_proba is not None:
        fig = reliability_diagram(y_true, y_proba)
        _save(fig, "reliability_diagram",
              "Reliability diagram. Bars show the fraction of correct predictions in each "
              "confidence bin; the dashed diagonal is perfect calibration.",
              "fig:reliability_diagram")
    logger.info(f"Wrote {len(written)} figures to {output_dir}")
    return written
