"""Correctness tests for the statistics module.

These tests verify that ``friedman_test``, ``cohen_d`` and
``holm_sidak_correction`` produce *known* answers on canonical inputs,
not just internally-consistent ones. They guard against silent numerical
regressions (e.g. an off-by-one in the rank formula or a sign flip in
Cohen's d).

Reference values
----------------
- Friedman: the Wikipedia worked example (3 treatments × 6 blocks).
- Cohen's d: hand-computed on two arithmetic sequences.
- Holm-Šídák: hand-computed adjusted p-values + monotonicity invariant.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from myoadapt.evaluation.statistics import (
    cohen_d,
    cohen_dz,
    friedman_test,
    holm_sidak_correction,
    nemenyi_posthoc,
)


# ----------------------------------------------------------------------------
# Friedman test — canonical Wikipedia example.
# ----------------------------------------------------------------------------
def test_friedman_test_wikipedia_example():
    """Reproduce the Wikipedia worked example for the Friedman test.

    Data: 3 treatments × 6 blocks (lower is better → rank 1 = worst).
        A = [7, 8, 6, 7, 10, 6]
        B = [8, 9, 7, 8, 11, 7]
        C = [9, 10, 8, 9, 12, 8]

    Expected (scipy.stats.friedmanchisquare):
        chi2 = 12.0
        p    = 0.002478752176666357
        dof  = 2
        mean ranks: A=3.0, B=2.0, C=1.0  (C is best, A is worst)
    """
    scores = {
        "A": [7, 8, 6, 7, 10, 6],
        "B": [8, 9, 7, 8, 11, 7],
        "C": [9, 10, 8, 9, 12, 8],
    }
    res = friedman_test(scores)

    # Cross-check chi2 against scipy directly (proves our wrapper delegates
    # the test correctly without transforming the data).
    chi2_ref, p_ref = stats.friedmanchisquare(
        scores["A"], scores["B"], scores["C"]
    )
    assert res["chi2"] == pytest.approx(chi2_ref, abs=1e-9)
    assert res["p_value"] == pytest.approx(p_ref, abs=1e-9)

    # Hard-coded expected values from the canonical example.
    assert res["chi2"] == pytest.approx(12.0, abs=1e-9)
    assert res["p_value"] == pytest.approx(0.002478752176666357, rel=1e-6)
    assert res["degrees_of_freedom"] == 2
    assert res["n_models"] == 3
    assert res["n_folds"] == 6
    assert res["significant_at_0.05"] is True

    # Mean ranks: C (best) should be ~1.0, B ~2.0, A (worst) ~3.0.
    # Our function ranks *higher scores as better* (rank 1 = best), which
    # is the convention used in ML benchmark papers. So:
    #   C (highest scores) → mean rank 1.0
    #   B (middle) → mean rank 2.0
    #   A (lowest) → mean rank 3.0
    assert res["mean_rank_per_model"]["C"] == pytest.approx(1.0, abs=1e-9)
    assert res["mean_rank_per_model"]["B"] == pytest.approx(2.0, abs=1e-9)
    assert res["mean_rank_per_model"]["A"] == pytest.approx(3.0, abs=1e-9)


def test_friedman_test_too_few_models_returns_trivial():
    """friedman_test with <2 models must short-circuit (chi2=0, p=1)."""
    res = friedman_test({"only_one": [0.8, 0.9, 0.7]})
    assert res["chi2"] == 0.0
    assert res["p_value"] == 1.0
    assert res["n_models"] == 1


def test_friedman_test_too_few_folds_returns_trivial():
    """friedman_test with <3 folds must short-circuit (chi2=0, p=1)."""
    res = friedman_test({
        "a": [0.8, 0.9],
        "b": [0.7, 0.85],
    })
    assert res["chi2"] == 0.0
    assert res["p_value"] == 1.0
    assert res["n_folds"] == 2


# ----------------------------------------------------------------------------
# Cohen's d — hand-computed reference.
# ----------------------------------------------------------------------------
def test_cohen_d_known_value():
    """Two arithmetic sequences [1..5] vs [3..7] have a known Cohen's d.

    Manual computation:
        mean_a = 3, mean_b = 5  →  Δ = -2
        var_a = var_b = 2.5 (ddof=1)
        pooled_std = sqrt((2.5 + 2.5) / 2) = sqrt(2.5) ≈ 1.5811
        d = Δ / pooled_std = -2 / sqrt(2.5) ≈ -1.2649
    """
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = np.array([3.0, 4.0, 5.0, 6.0, 7.0])
    d = cohen_d(a, b)
    expected = -2.0 / np.sqrt(2.5)
    assert d == pytest.approx(expected, rel=1e-9)
    assert d == pytest.approx(-1.2649110640673518, rel=1e-9)
    # Sign convention: a < b on average → d must be negative.
    assert d < 0


def test_cohen_d_zero_when_identical():
    """If both inputs have zero variance and identical means, d must be 0
    (guard against 0/0 NaNs)."""
    a = np.array([2.0, 2.0, 2.0])
    b = np.array([2.0, 2.0, 2.0])
    assert cohen_d(a, b) == 0.0


def test_cohen_d_symmetric_sign_flip():
    """Swapping a and b must flip the sign of d."""
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = np.array([3.0, 4.0, 5.0, 6.0, 7.0])
    d_ab = cohen_d(a, b)
    d_ba = cohen_d(b, a)
    assert d_ab == pytest.approx(-d_ba, rel=1e-9)


# ----------------------------------------------------------------------------
# Holm-Šídák — known adjusted p-values + monotonicity.
# ----------------------------------------------------------------------------
def test_holm_sidak_known_values():
    """Hand-computed Holm-Šídák adjusted p-values for a known input.

    Input p-values (5 tests):
        p = [0.01, 0.04, 0.03, 0.20, 0.001]

    Sorted ascending (rank → p_raw):
        0 → 0.001   m=5  adj = 1-(1-0.001)^5   ≈ 0.0049900100
        1 → 0.01    m=4  adj = 1-(1-0.01)^4    ≈ 0.0394039900
        2 → 0.03    m=3  adj = 1-(1-0.03)^3    ≈ 0.0873270000
        3 → 0.04    m=2  adj = 1-(1-0.04)^2    ≈ 0.0784000000
        4 → 0.20    m=1  adj = 1-(1-0.20)^1    ≈ 0.2000000000

    Step-down monotonicity on adj_p:
        rank 0: 0.0049900100
        rank 1: 0.0394039900  (≥ rank 0 → keep)
        rank 2: 0.0873270000  (≥ rank 1 → keep)
        rank 3: 0.0784000000  (< rank 2 → raise to 0.0873270000)
        rank 4: 0.2000000000  (≥ rank 3 → keep)

    So the corrected p-values (in input order) are:
        input[0]=0.01  → rank 1 → 0.0394039900
        input[1]=0.04  → rank 3 → 0.0873270000  (monotone-raised)
        input[2]=0.03  → rank 2 → 0.0873270000
        input[3]=0.20  → rank 4 → 0.2000000000
        input[4]=0.001 → rank 0 → 0.0049900100
    """
    pvals = [0.01, 0.04, 0.03, 0.20, 0.001]
    out = holm_sidak_correction(pvals, alpha=0.05)
    assert len(out) == 5

    expected_corrected = [
        0.0394039900,  # input[0]=0.01, rank 1
        0.0873270000,  # input[1]=0.04, rank 3 (monotone-raised)
        0.0873270000,  # input[2]=0.03, rank 2
        0.2000000000,  # input[3]=0.20, rank 4
        0.0049900100,  # input[4]=0.001, rank 0
    ]
    for entry, exp in zip(out, expected_corrected):
        assert entry["p_value_corrected"] == pytest.approx(exp, rel=1e-6), (
            f"expected {exp}, got {entry['p_value_corrected']} "
            f"(raw={entry['p_value_raw']}, rank={entry['rank']})"
        )

    # Significance: rank 0 (p=0.001) and rank 1 (p=0.01) should be
    # significant. After step-down monotonicity, the remaining ones are
    # not significant.
    sig_by_input_index = [entry["significant"] for entry in out]
    assert sig_by_input_index == [True, False, False, False, True], (
        f"significance flags wrong: {sig_by_input_index}"
    )


def test_holm_sidak_monotonicity_invariant():
    """For any input, the corrected p-values MUST be monotonically
    non-decreasing when traversed in sorted (rank) order.

    This is the key correctness invariant of the step-down procedure:
    once an adjusted p-value is X, no subsequent rank can have a smaller
    adjusted p-value.
    """
    rng = np.random.default_rng(0)
    for trial in range(20):
        pvals = (rng.random(8) * 0.5).tolist()  # 8 random p-values in [0, 0.5]
        out = holm_sidak_correction(pvals, alpha=0.05)
        # Sort by rank and check monotonicity of p_value_corrected.
        by_rank = sorted(out, key=lambda e: e["rank"])
        corrected = [e["p_value_corrected"] for e in by_rank]
        for i in range(1, len(corrected)):
            assert corrected[i] >= corrected[i - 1] - 1e-12, (
                f"trial {trial}: non-monotone at rank {i}: "
                f"{corrected[i-1]} -> {corrected[i]} (full: {corrected})"
            )


def test_holm_sidak_significance_step_down_invariant():
    """Once a test is non-significant at rank r, every test at rank > r
    must also be non-significant (step-down property)."""
    rng = np.random.default_rng(123)
    for trial in range(20):
        pvals = (rng.random(6) * 0.5).tolist()
        out = holm_sidak_correction(pvals, alpha=0.05)
        by_rank = sorted(out, key=lambda e: e["rank"])
        seen_non_sig = False
        for entry in by_rank:
            if seen_non_sig:
                assert entry["significant"] is False, (
                    f"trial {trial}: rank {entry['rank']} is significant "
                    f"after a previous non-significant test"
                )
            if not entry["significant"]:
                seen_non_sig = True


def test_holm_sidak_preserves_input_order():
    """The returned list must be in the same order as the input p-values
    (each entry corresponds to the same index)."""
    pvals = [0.01, 0.04, 0.03, 0.20, 0.001]
    out = holm_sidak_correction(pvals, alpha=0.05)
    for entry, p in zip(out, pvals):
        assert entry["p_value_raw"] == p


# ----------------------------------------------------------------------------
# cohen_dz — paired effect size (distinct from the unpaired cohen_d above;
# had zero direct test coverage before, only exercised indirectly through
# ZeroShotEvaluator.compare_with_loso).
# ----------------------------------------------------------------------------
def test_cohen_dz_known_value():
    """differences = [2,4,6,8,10]: mean=6, std(ddof=1)=sqrt(10).
    dz = 6 / sqrt(10) ~= 1.8973665961."""
    diffs = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    dz = cohen_dz(diffs)
    expected = 6.0 / np.sqrt(10.0)
    assert dz == pytest.approx(expected, rel=1e-9)
    assert dz == pytest.approx(1.8973665961010275, rel=1e-9)


def test_cohen_dz_zero_variance_guarded():
    """Constant differences (zero variance) must return 0.0, not NaN/Inf."""
    assert cohen_dz(np.array([3.0, 3.0, 3.0, 3.0])) == 0.0


def test_cohen_dz_sign_matches_direction():
    assert cohen_dz(np.array([1.0, 2.0, 3.0])) > 0
    assert cohen_dz(np.array([-1.0, -2.0, -3.0])) < 0


def test_cohen_dz_differs_from_unpaired_cohen_d():
    """Sanity guard against the two functions being accidentally aliased:
    for the same two samples, the paired and unpaired effect sizes are
    computed differently and need not agree numerically."""
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = np.array([3.0, 4.0, 5.0, 6.0, 7.0])
    d_unpaired = cohen_d(a, b)
    d_paired = cohen_dz(a - b)
    assert d_unpaired != pytest.approx(d_paired)


# ----------------------------------------------------------------------------
# nemenyi_posthoc — post-hoc pairwise comparison after Friedman.
# Had zero direct test coverage before (friedman_test itself is tested
# above via the Wikipedia example; nemenyi_posthoc calls it internally
# but was never itself exercised).
# ----------------------------------------------------------------------------
def test_nemenyi_posthoc_trivial_with_too_few_models():
    scores = {"model_a": [0.8, 0.7, 0.9]}
    result = nemenyi_posthoc(scores)
    assert result == {"critical_diff": 0.0, "comparisons": []}


def test_nemenyi_posthoc_trivial_with_too_few_folds():
    scores = {"model_a": [0.8, 0.7], "model_b": [0.6, 0.5]}
    result = nemenyi_posthoc(scores)
    assert result == {"critical_diff": 0.0, "comparisons": []}


def test_nemenyi_posthoc_structure_and_comparison_count():
    rng = np.random.default_rng(0)
    scores = {f"model_{i}": (0.5 + i * 0.05 + rng.normal(0, 0.02, 10)).tolist()
              for i in range(4)}
    result = nemenyi_posthoc(scores)
    assert result["critical_diff"] > 0
    # C(4,2) = 6 pairwise comparisons
    assert len(result["comparisons"]) == 6
    for c in result["comparisons"]:
        assert set(c.keys()) == {"model_a", "model_b", "rank_diff", "significant"}
        assert c["rank_diff"] >= 0
        assert c["significant"] == (c["rank_diff"] > result["critical_diff"])


def test_nemenyi_posthoc_detects_clearly_separated_models():
    """A model that's always ranked best/worst across every fold should
    show a large, significant rank difference against the others."""
    n_folds = 10
    scores = {
        "always_best": [0.95] * n_folds,
        "always_worst": [0.50] * n_folds,
        "middle": [0.75] * n_folds,
    }
    result = nemenyi_posthoc(scores)
    by_pair = {frozenset((c["model_a"], c["model_b"])): c for c in result["comparisons"]}
    best_vs_worst = by_pair[frozenset(("always_best", "always_worst"))]
    assert best_vs_worst["significant"] is True
    assert best_vs_worst["rank_diff"] == pytest.approx(2.0)  # max possible rank spread for 3 models
