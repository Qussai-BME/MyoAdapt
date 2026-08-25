"""Unit tests for myoadapt.evaluation.demographic_fairness.DemographicFairnessAudit."""
from __future__ import annotations

import numpy as np
import pytest

from myoadapt.evaluation.demographic_fairness import DemographicFairnessAudit


def _biased_predictions(n=200, seed=0):
    """y_pred corrupted specifically for older (>55) subjects — a
    detectable, deliberate fairness gap to check the audit against."""
    rng = np.random.default_rng(seed)
    groups = rng.integers(0, 8, n)
    y_true = rng.integers(0, 5, n)
    y_pred = y_true.copy()
    age = rng.uniform(18, 75, n)
    old_idx = np.where(age > 55)[0]
    corrupt = old_idx[: len(old_idx) // 2]
    y_pred[corrupt] = rng.integers(0, 5, len(corrupt))
    return y_true, y_pred, groups, age


def test_audit_runs_with_no_demographics():
    y_true, y_pred, groups, _ = _biased_predictions()
    result = DemographicFairnessAudit().audit(y_true, y_pred, groups)
    assert result["demographics_audited"] == ["subject"]
    assert result["n_samples"] == 200


def test_audit_flags_the_age_bias_we_injected():
    y_true, y_pred, groups, age = _biased_predictions()
    result = DemographicFairnessAudit(tolerance_pp=10.0).audit(
        y_true, y_pred, groups, demographics={"age": age})
    assert "age" in result["demographics_audited"]
    assert result["summary"]["worst_variable"] == "age"
    assert result["summary"]["fairness_flag"] in ("concerning", "fail")
    assert result["summary"]["worst_disparity_pp"] > 10.0


def test_audit_no_bias_case_stays_within_tolerance():
    rng = np.random.default_rng(1)
    n = 300
    y_true = rng.integers(0, 5, n)
    y_pred = y_true.copy()
    flip = rng.choice(n, size=15, replace=False)  # uniform, small, unrelated to age
    y_pred[flip] = rng.integers(0, 5, len(flip))
    age = rng.uniform(18, 75, n)
    groups = rng.integers(0, 8, n)
    result = DemographicFairnessAudit(tolerance_pp=25.0).audit(
        y_true, y_pred, groups, demographics={"age": age})
    assert result["summary"]["fairness_flag"] == "pass"


def test_audit_handles_categorical_demographics():
    y_true, y_pred, groups, _ = _biased_predictions()
    rng = np.random.default_rng(0)
    hand = rng.choice(["left", "right"], len(y_true))
    result = DemographicFairnessAudit().audit(
        y_true, y_pred, groups, demographics={"dominant_hand": hand})
    assert "dominant_hand" in result["per_variable"]
    subgroups = result["per_variable"]["dominant_hand"]["per_subgroup"]
    assert set(subgroups.keys()) <= {"left", "right"}


def test_narrative_is_nonempty_and_mentions_audited_variables():
    y_true, y_pred, groups, age = _biased_predictions()
    result = DemographicFairnessAudit().audit(y_true, y_pred, groups, demographics={"age": age})
    assert isinstance(result["narrative"], str) and len(result["narrative"]) > 0
    assert "age" in result["narrative"]


def test_custom_age_bins_are_used():
    y_true, y_pred, groups, age = _biased_predictions()
    custom_bins = [(18, 40), (41, 100)]
    audit = DemographicFairnessAudit(age_bins=custom_bins)
    result = audit.audit(y_true, y_pred, groups, demographics={"age": age})
    labels = set(result["per_variable"]["age"]["per_subgroup"].keys())
    assert labels <= {"18-40", "41+", "41-100"}  # exact label format is an implementation
    # detail; just confirm it split into 2 bins, not the default 4
    assert len(labels) <= 2
