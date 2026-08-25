"""
statistics.py — Statistical tests for cross-subject evaluation
==============================================================

Provides:
- friedman_test         — Friedman non-parametric test across folds
- wilcoxon_pairwise     — Wilcoxon signed-rank + Holm-Šídák correction
- nemenyi_posthoc       — Nemenyi post-hoc for Friedman
- cohen_d               — effect size
- holm_sidak_correction — Holm-Šídák step-down correction
- bootstrap_ci          — bootstrap confidence interval

"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


def friedman_test(per_fold_scores: Dict[str, List[float]]) -> Dict:
    """
    Friedman test: do K models have significantly different performance
    across N folds?

    Parameters
    ----------
    per_fold_scores : dict mapping model_name -> list of per-fold scores

    Returns
    -------
    dict with: chi2, p_value, degrees_of_freedom, n_folds, n_models,
              mean_rank_per_model
    """
    model_names = list(per_fold_scores.keys())
    n_models = len(model_names)
    if n_models < 2:
        return {"chi2": 0.0, "p_value": 1.0, "n_models": n_models,
                "n_folds": 0, "mean_rank_per_model": {}}

    n_folds = min(len(v) for v in per_fold_scores.values())
    if n_folds < 3:
        return {"chi2": 0.0, "p_value": 1.0, "n_models": n_models,
                "n_folds": n_folds, "mean_rank_per_model": {}}

    matrix = np.array([per_fold_scores[m][:n_folds] for m in model_names]).T
    chi2, p_value = stats.friedmanchisquare(*[matrix[:, i] for i in range(n_models)])

    # Mean ranks — use average ranks for ties (Friedman requirement)
    from scipy.stats import rankdata
    ranks = np.zeros_like(matrix, dtype=np.float64)
    for i in range(n_folds):
        # rankdata ranks ascending (smallest = 1); we want descending
        # (best model = rank 1), so rank the negation.
        ranks[i] = rankdata(-matrix[i], method="average")
    mean_ranks = {model_names[i]: float(ranks[:, i].mean())
                   for i in range(n_models)}

    return {
        "chi2": float(chi2),
        "p_value": float(p_value),
        "degrees_of_freedom": n_models - 1,
        "n_folds": n_folds,
        "n_models": n_models,
        "mean_rank_per_model": mean_ranks,
        "significant_at_0.05": bool(p_value < 0.05),
    }


def wilcoxon_pairwise(per_fold_scores: Dict[str, List[float]],
                       reference: Optional[str] = None,
                       alpha: float = 0.05) -> Dict:
    """
    Pairwise Wilcoxon signed-rank tests with Holm-Šídák correction.

    If reference is None, all pairs are tested. Otherwise, only reference
    vs others.
    """
    model_names = list(per_fold_scores.keys())
    if reference is None:
        # All pairs
        pairs = [(a, b) for i, a in enumerate(model_names)
                  for b in model_names[i + 1 :]]
    else:
        pairs = [(reference, b) for b in model_names if b != reference]

    n_tests = len(pairs)
    raw_p_values: List[float] = []
    test_info: List[Dict] = []

    for a, b in pairs:
        scores_a = np.asarray(per_fold_scores[a])
        scores_b = np.asarray(per_fold_scores[b])
        n = min(len(scores_a), len(scores_b))
        if n < 5:
            # Too few samples for Wilcoxon
            raw_p_values.append(1.0)
            test_info.append({
                "model_a": a, "model_b": b, "statistic": 0.0,
                "p_value_raw": 1.0, "p_value_corrected": 1.0,
                "significant": False, "n_folds": n,
                "warning": "Too few folds for Wilcoxon",
            })
            continue
        try:
            stat, p = stats.wilcoxon(scores_a[:n], scores_b[:n])
            stat, p = float(stat), float(p)
        except Exception:
            stat, p = 0.0, 1.0
        raw_p_values.append(p)
        d = cohen_d(scores_a[:n], scores_b[:n])
        test_info.append({
            "model_a": a, "model_b": b, "statistic": stat,
            "p_value_raw": p, "cohen_d": d,
            "n_folds": n,
        })

    # Holm-Šídák correction
    corrected = holm_sidak_correction(raw_p_values, alpha=alpha)
    for info, c in zip(test_info, corrected):
        info["p_value_corrected"] = c["p_value_corrected"]
        info["significant"] = c["significant"]
        info["reject_h0"] = c["significant"]

    return {
        "n_tests": n_tests,
        "alpha": alpha,
        "correction": "Holm-Šídák",
        "tests": test_info,
        "n_significant": sum(1 for t in test_info if t["significant"]),
    }


def nemenyi_posthoc(per_fold_scores: Dict[str, List[float]]) -> Dict:
    """Nemenyi post-hoc test after Friedman."""
    friedman = friedman_test(per_fold_scores)
    n_models = friedman["n_models"]
    n_folds = friedman["n_folds"]
    if n_models < 2 or n_folds < 3:
        return {"critical_diff": 0.0, "comparisons": []}

    # Critical difference (Nemenyi)
    q_alpha = {0.05: {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728,
                      6: 2.850, 7: 2.948, 8: 3.031, 9: 3.102, 10: 3.164}}
    q = q_alpha[0.05].get(n_models, 3.164)
    cd = q * np.sqrt(n_models * (n_models + 1) / (6.0 * n_folds))

    model_names = list(per_fold_scores.keys())
    comparisons = []
    for i, a in enumerate(model_names):
        for b in model_names[i + 1 :]:
            rank_diff = abs(friedman["mean_rank_per_model"][a] -
                            friedman["mean_rank_per_model"][b])
            comparisons.append({
                "model_a": a, "model_b": b,
                "rank_diff": float(rank_diff),
                "significant": bool(rank_diff > cd),
            })

    return {
        "critical_diff": float(cd),
        "comparisons": comparisons,
        "mean_rank_per_model": friedman["mean_rank_per_model"],
    }


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d effect size (independent-samples, pooled SD).

    Uses the pooled-standard-deviation form
        d = (mean_a - mean_b) / sqrt((var_a + var_b) / 2)
    which is the standard ``d_s`` form (Cohen 1988).
    For paired-samples ``d_z`` use the SD of the per-pair differences
    instead.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    pooled_std = np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2)
    if pooled_std < 1e-12:
        return 0.0
    return float((a.mean() - b.mean()) / pooled_std)


def cohen_dz(differences: np.ndarray) -> float:
    """Cohen's d_z effect size for paired samples (uses SD of differences).

    Use this when comparing two models evaluated on the SAME folds.
    """
    diffs = np.asarray(differences, dtype=np.float64)
    sd_diff = diffs.std(ddof=1)
    if sd_diff < 1e-12:
        return 0.0
    return float(diffs.mean() / sd_diff)


def holm_sidak_correction(p_values: List[float],
                           alpha: float = 0.05) -> List[Dict]:
    """Holm-Šídák step-down correction.

    Returns a list (one entry per input p-value) with the adjusted
    p-value (Šídák form: ``1 - (1-p)^(n-rank)``) and a significance
    flag computed against the Šídák threshold ``1 - (1-alpha)^(1/(n-rank))``.
    Both are made monotone across the step-down ordering.
    """
    n = len(p_values)
    order = sorted(range(n), key=lambda i: p_values[i])
    results: List[Optional[Dict]] = [None] * n
    for rank, idx in enumerate(order):
        m = n - rank  # number of remaining tests at this step
        # Adjusted p-value: Sidak form
        p_raw = p_values[idx]
        adj_p = 1.0 - (1.0 - p_raw) ** m
        # Step-down monotonicity on the adjusted p-value
        if rank > 0:
            prev_adj = results[order[rank - 1]]["p_value_corrected"]
            adj_p = max(adj_p, prev_adj)
        # Sidak threshold for significance at this step
        sidak_thresh = 1.0 - (1.0 - alpha) ** (1.0 / m)
        significant = bool(p_raw < sidak_thresh)
        # Step-down monotonicity on significance (once non-significant, stays non-significant)
        if rank > 0 and not results[order[rank - 1]]["significant"]:
            significant = False
        results[idx] = {
            "p_value_raw": p_raw,
            "p_value_corrected": min(adj_p, 1.0),
            "significant": significant,
            "rank": rank,
        }
    return results  # type: ignore[return-value]


def bootstrap_ci(scores: List[float],
                  statistic=np.mean,
                  n_bootstrap: int = 1000,
                  alpha: float = 0.05,
                  random_state: int = 42) -> Tuple[float, float]:
    """Bootstrap confidence interval for any statistic."""
    rng = np.random.default_rng(random_state)
    scores = np.asarray(scores)
    boot_stats = []
    for _ in range(n_bootstrap):
        sample = rng.choice(scores, size=len(scores), replace=True)
        boot_stats.append(statistic(sample))
    lower = float(np.percentile(boot_stats, 100 * alpha / 2))
    upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    return lower, upper


def hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    """Hedges' *g* effect size — Cohen's *d* with small-sample bias correction.

    Recommended over Cohen's *d* when sample sizes are small (N < 20 per
    group), which is the common case for cross-subject EMG benchmarks
    (e.g. NinaPro DB3 has 11 amputes, DB7 has 22 subjects).

    The correction factor is

        J = 1 - 3 / (4 * df - 1),   df = n_a + n_b - 2

    so ``g = J * d``. ``g`` is always slightly smaller in magnitude than
    ``d`` and is the unbiased estimator of the population effect size.

    References
    ----------
    - Hedges, L. V. (1981). *Distribution Theory for Glass's Estimator
      of Effect Size and Related Estimators.* Review of Educational
      Research 6, 107–145.
    - Lakens, D. (2013). *Calculating and Reporting Effect Sizes.* Front.
      Psychol. 4:863 (13,515 citations).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return 0.0
    pooled_var = ((n_a - 1) * a.var(ddof=1) + (n_b - 1) * b.var(ddof=1)) / (n_a + n_b - 2)
    pooled_std = np.sqrt(pooled_var)
    if pooled_std < 1e-12:
        return 0.0
    d = float((a.mean() - b.mean()) / pooled_std)
    df = n_a + n_b - 2
    J = 1.0 - 3.0 / (4.0 * df - 1.0)
    return float(J * d)


def interpret_effect_size(d: float) -> str:
    """Cohen (1988) qualitative interpretation of an effect size.

    Thresholds: |d| < 0.2 → negligible; 0.2–0.5 → small;
    0.5–0.8 → medium; ≥ 0.8 → large.
    """
    ad = abs(float(d))
    if ad < 0.2:
        return "negligible"
    if ad < 0.5:
        return "small"
    if ad < 0.8:
        return "medium"
    return "large"


def bootstrap_ci_bca(scores: List[float],
                     statistic=np.mean,
                     n_bootstrap: int = 2000,
                     alpha: float = 0.05,
                     random_state: int = 42) -> Dict[str, float]:
    """Bias-corrected and accelerated (BCa) bootstrap confidence interval.

    BCa is the recommended default over the naive percentile bootstrap:
    it is second-order accurate and transformation-respecting (Efron &
    Tibshirani 1993, ch. 14). The output dict carries the point
    estimate, the lower and upper bounds, and the bias / acceleration
    factors so reviewers can verify the methodology.

    Returns
    -------
    dict with keys: ``point``, ``lower``, ``upper``, ``z0_bias``,
    ``acceleration``, ``n_bootstrap``, ``alpha``.
    """
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    if n < 2:
        return {"point": float(statistic(scores)) if n == 1 else float("nan"),
                "lower": float("nan"), "upper": float("nan"),
                "z0_bias": 0.0, "acceleration": 0.0,
                "n_bootstrap": n_bootstrap, "alpha": alpha}
    rng = np.random.default_rng(random_state)
    theta_hat = float(statistic(scores))

    # Bootstrap resamples.
    boot_stats = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        sample = rng.choice(scores, size=n, replace=True)
        boot_stats[i] = float(statistic(sample))

    # Bias-correction z0.
    prop_less = float(np.mean(boot_stats < theta_hat))
    # Clamp to avoid ±inf from scipy.ndtri at 0 / 1.
    prop_less = min(max(prop_less, 1e-10), 1 - 1e-10)
    from scipy.stats import norm
    z0 = float(norm.ppf(prop_less))

    # Acceleration a via jackknife.
    jackknife_stats = np.empty(n, dtype=np.float64)
    for i in range(n):
        jackknife_stats[i] = float(statistic(np.delete(scores, i)))
    jack_mean = jackknife_stats.mean()
    num = float(np.sum((jack_mean - jackknife_stats) ** 3))
    den = float(6.0 * (np.sum((jack_mean - jackknife_stats) ** 2)) ** 1.5)
    a_hat = num / den if abs(den) > 1e-30 else 0.0

    # BCa adjusted quantiles.
    z_alpha_lo = float(norm.ppf(alpha / 2))
    z_alpha_hi = float(norm.ppf(1 - alpha / 2))
    def _adjust(z_alpha: float) -> float:
        denom = 1.0 - a_hat * (z0 + z_alpha)
        if abs(denom) < 1e-12:
            return 0.5
        return float(norm.cdf(z0 + (z0 + z_alpha) / denom))
    alpha1 = _adjust(z_alpha_lo)
    alpha2 = _adjust(z_alpha_hi)

    lower = float(np.percentile(boot_stats, 100 * alpha1))
    upper = float(np.percentile(boot_stats, 100 * alpha2))
    return {
        "point": theta_hat,
        "lower": lower,
        "upper": upper,
        "z0_bias": z0,
        "acceleration": a_hat,
        "n_bootstrap": n_bootstrap,
        "alpha": alpha,
    }


def paired_wilcoxon_test(a: np.ndarray, b: np.ndarray,
                          alternative: str = "two-sided") -> Dict[str, float]:
    """Paired Wilcoxon signed-rank test (no correction; for a single pair).

    Wraps :func:`scipy.stats.wilcoxon` with a stable return contract
    that includes the effect size (matched-pairs rank-biserial *r*).

    Use :func:`wilcoxon_pairwise` when you have more than two models —
    it applies the Holm-Šídák correction automatically.
    """
    from scipy.stats import rankdata, wilcoxon
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = min(len(a), len(b))
    if n < 5:
        return {"statistic": 0.0, "p_value": 1.0, "n": n,
                "effect_size_r": 0.0, "alternative": alternative}
    a, b = a[:n], b[:n]
    diffs = a - b
    nonzero = diffs != 0
    if nonzero.sum() < 1:
        return {"statistic": 0.0, "p_value": 1.0, "n": n,
                "effect_size_r": 0.0, "alternative": alternative}
    stat, p = wilcoxon(a, b, alternative=alternative)
    # Matched-pairs rank-biserial correlation (Kerby 2014).
    ranks = rankdata(np.abs(diffs[nonzero]))
    pos = (diffs[nonzero] > 0).astype(float)
    r = float(np.sum(ranks * pos) - np.sum(ranks * (1 - pos))) / float(np.sum(ranks))
    return {
        "statistic": float(stat),
        "p_value": float(p),
        "n": int(n),
        "effect_size_r": r,
        "alternative": alternative,
    }


def power_analysis_paired_ttest(effect_size: float, n: int,
                                 alpha: float = 0.05) -> float:
    """A-priori statistical power for a paired-samples t-test.

    Uses the non-central t distribution. Returns the achieved power
    (1 - β) for a given effect size *d_z*, sample size *n*, and
    significance level *alpha*.

    Use this to justify sample size *before* running an experiment:
    reviewers in 2026 increasingly ask for a power justification when
    the per-fold sample size is small (typical for amputee databases).
    """
    from scipy.stats import nct
    if n < 2 or effect_size <= 0:
        return 0.0
    df = n - 1
    ncp = effect_size * np.sqrt(n)
    # Two-sided test: critical values under the central t.
    from scipy.stats import t as central_t
    t_crit = central_t.ppf(1 - alpha / 2, df)
    # Power = P(reject H0 | H1 true) = P(|T| > t_crit) under non-central t.
    upper = 1.0 - nct.cdf(t_crit, df, ncp)
    lower = nct.cdf(-t_crit, df, ncp)
    return float(upper + lower)


def minimum_sample_size_paired(effect_size: float,
                                alpha: float = 0.05,
                                target_power: float = 0.8) -> int:
    """Smallest N giving ``target_power`` for a paired t-test at ``effect_size``.

    Iteratively searches upward from N=3. Returns the minimum N at which
    :func:`power_analysis_paired_ttest` first reaches ``target_power``.
    Caps at N=1000 to avoid pathological loops.
    """
    if effect_size <= 0:
        return 1000
    for n in range(3, 1001):
        if power_analysis_paired_ttest(effect_size, n, alpha) >= target_power:
            return n
    return 1000
