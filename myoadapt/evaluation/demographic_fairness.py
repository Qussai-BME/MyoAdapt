"""
demographic_fairness.py — Fairness audit across demographic and
physiological variables.

Extends the basic per-subgroup fairness audit to include:
- Age groups (18-30, 31-50, 51-70, 70+)
- BMI categories (underweight, normal, overweight, obese)
- Skin properties (hydration, elasticity — affects sEMG quality)
- Amputation level (for amputee populations)
- Dominant hand

This makes MyoAdapt the first open-source sEMG platform with a
physiologically-grounded bias audit — addressing the 2024-2025
call for demographic reporting in health-AI benchmarks.

References:
- Marulanda et al. (2024). "A 91-subject sEMG database with
  demographic diversity." Nature Scientific Data.
- Mitchell et al. (2019). "Model Cards for Model Reporting."
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from myoadapt.evaluation.calibration import (
    brier_score,
    expected_calibration_error,
)
from myoadapt.evaluation.metrics import compute_metrics

logger = logging.getLogger(__name__)


# Default bin definitions. Age bins are (low, high) inclusive-of-low.
# BMI bins follow the WHO classification with a label per bin.
_DEFAULT_AGE_BINS: List[Tuple[int, int]] = [
    (18, 30),
    (31, 50),
    (51, 70),
    (71, 200),
]
_DEFAULT_BMI_BINS: List[Tuple[float, float, str]] = [
    (0.0, 18.5, "underweight"),
    (18.5, 25.0, "normal"),
    (25.0, 30.0, "overweight"),
    (30.0, 100.0, "obese"),
]


class DemographicFairnessAudit:
    """
    Physiologically-grounded fairness audit.

    Extends :mod:`myoadapt.evaluation.fairness` with explicit
    binning for continuous physiological variables (age, BMI) and
    first-class handling of categorical clinical metadata
    (amputation level, dominant hand, skin hydration).

    The audit is designed to plug directly into a Model Card's
    "Fairness analysis" section, satisfying the demographic-
    reporting expectations of EU AI Act Annex IV and FDA GMLP.

    Parameters
    ----------
    age_bins : list of (low, high) tuples, default ``[(18,30),
        (31,50), (51,70), (71,200)]``. Right-inclusive label is
        ``f"{low}-{high}"`` (or ``"71+"`` for the top bin).
    bmi_bins : list of (low, high, label) tuples following the WHO
        classification by default.
    rest_class : str — label treated as the Rest class for the
        inflation analysis (passed through to compute_metrics).
    tolerance_pp : float — disparity (in percentage points) above
        which a variable is flagged as "concerning". Default 10.0.
    """

    def __init__(self,
                 age_bins: Optional[List[Tuple[int, int]]] = None,
                 bmi_bins: Optional[List[Tuple[float, float, str]]] = None,
                 rest_class: str = "rest",
                 tolerance_pp: float = 10.0):
        self.age_bins = list(age_bins) if age_bins is not None else list(_DEFAULT_AGE_BINS)
        self.bmi_bins = list(bmi_bins) if bmi_bins is not None else list(_DEFAULT_BMI_BINS)
        self.rest_class = rest_class
        self.tolerance_pp = float(tolerance_pp)
        self.last_audit_: Optional[Dict] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def audit(self,
              y_true: np.ndarray,
              y_pred: np.ndarray,
              groups: np.ndarray,
              demographics: Optional[Dict[str, np.ndarray]] = None,
              y_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Full fairness audit across all demographic variables.

        Parameters
        ----------
        y_true, y_pred : (n,) arrays of labels.
        groups : (n,) array of subject IDs (used as a baseline
            per-subject subgroup analysis).
        demographics : optional dict mapping variable name → array.
            Recognised keys: ``age``, ``bmi``, ``skin_hydration``,
            ``amputation_level``, ``dominant_hand``. Any other key
            is treated as a categorical variable.
        y_proba : optional (n, n_classes) probabilities for
            per-subgroup calibration (ECE / Brier).

        Returns
        -------
        dict with:
            ``per_variable``  : {variable: {per_subgroup, disparities,
                                             n_subgroups, n_samples}}
            ``summary``        : {worst_variable, worst_disparity_pp,
                                  flagged_variables, fairness_flag,
                                  tolerance_pp, n_variables}
            ``narrative``      : str (auto-generated)
            ``n_samples``      : int
            ``demographics_audited`` : list of variable names
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        groups = np.asarray(groups)
        demographics = demographics or {}

        per_variable: Dict[str, Dict[str, Any]] = {}

        # Baseline per-subject analysis (always run).
        per_variable["subject"] = self._per_subgroup_metrics(
            y_true, y_pred, groups, y_proba=y_proba,
            label_format=lambda g: f"subject_{g}",
        )

        # Age and BMI use dedicated binning.
        if "age" in demographics:
            ages = np.asarray(demographics["age"], dtype=float)
            age_labels = self._bin_continuous(ages, self.age_bins,
                                               formatter=_age_label)
            per_variable["age"] = self._per_subgroup_metrics(
                y_true, y_pred, age_labels, y_proba=y_proba,
            )
        if "bmi" in demographics:
            bmis = np.asarray(demographics["bmi"], dtype=float)
            bmi_labels = self._bin_bmi(bmis)
            per_variable["bmi"] = self._per_subgroup_metrics(
                y_true, y_pred, bmi_labels, y_proba=y_proba,
            )

        # Continuous variables without explicit bins → quartile binning.
        for cv in ("skin_hydration", "skin_elasticity"):
            if cv in demographics:
                vals = np.asarray(demographics[cv], dtype=float)
                labels = self._bin_quartiles(vals, name=cv)
                per_variable[cv] = self._per_subgroup_metrics(
                    y_true, y_pred, labels, y_proba=y_proba,
                )

        # Categorical variables → unique values as group labels.
        for cv in ("amputation_level", "dominant_hand", "sex",
                    "handedness", "ethnicity"):
            if cv in demographics:
                labels = np.asarray(demographics[cv]).astype(str)
                per_variable[cv] = self._per_subgroup_metrics(
                    y_true, y_pred, labels, y_proba=y_proba,
                )

        # Any extra keys not yet handled → treat as categorical.
        for cv, vals in demographics.items():
            if cv in per_variable:
                continue
            arr = np.asarray(vals).astype(str)
            per_variable[cv] = self._per_subgroup_metrics(
                y_true, y_pred, arr, y_proba=y_proba,
            )

        # Build summary.
        summary = self._build_summary(per_variable)
        narrative = self.fairness_narrative({
            "per_variable": per_variable,
            "summary": summary,
            "n_samples": int(len(y_true)),
            "demographics_audited": list(per_variable.keys()),
        })

        result = {
            "per_variable": per_variable,
            "summary": summary,
            "narrative": narrative,
            "n_samples": int(len(y_true)),
            "demographics_audited": list(per_variable.keys()),
            "tolerance_pp": float(self.tolerance_pp),
        }
        self.last_audit_ = result
        return result

    def audit_by_age(self,
                     y_true: np.ndarray,
                     y_pred: np.ndarray,
                     ages: np.ndarray,
                     y_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Age-binned fairness audit (18-30, 31-50, 51-70, 71+)."""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        ages = np.asarray(ages, dtype=float)
        labels = self._bin_continuous(ages, self.age_bins,
                                       formatter=_age_label)
        per_var = self._per_subgroup_metrics(y_true, y_pred, labels,
                                              y_proba=y_proba)
        summary = self._build_summary({"age": per_var})
        return {
            "variable": "age",
            "bins": [_age_label(*b) for b in self.age_bins],
            "per_subgroup": per_var["per_subgroup"],
            "disparities": per_var["disparities"],
            "n_subgroups": per_var["n_subgroups"],
            "n_samples": per_var["n_samples"],
            "summary": summary,
        }

    def audit_by_bmi(self,
                     y_true: np.ndarray,
                     y_pred: np.ndarray,
                     bmis: np.ndarray,
                     y_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """BMI-binned fairness audit (WHO categories)."""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        bmis = np.asarray(bmis, dtype=float)
        labels = self._bin_bmi(bmis)
        per_var = self._per_subgroup_metrics(y_true, y_pred, labels,
                                              y_proba=y_proba)
        summary = self._build_summary({"bmi": per_var})
        return {
            "variable": "bmi",
            "bins": [b[2] for b in self.bmi_bins],
            "per_subgroup": per_var["per_subgroup"],
            "disparities": per_var["disparities"],
            "n_subgroups": per_var["n_subgroups"],
            "n_samples": per_var["n_samples"],
            "summary": summary,
        }

    def fairness_narrative(self, audit_result: Dict[str, Any]) -> str:
        """Generate a human-readable fairness narrative paragraph.

        Suitable for embedding in a Model Card's "Fairness analysis"
        section. Mentions every audited variable, its disparity
        range, the worst-performing subgroup, and an overall fairness
        verdict (pass / concerning / fail) using the ``tolerance_pp``
        threshold.
        """
        per_variable = audit_result.get("per_variable", {})
        summary = audit_result.get("summary", {})
        n_samples = audit_result.get("n_samples", 0)
        n_vars = len(per_variable)

        parts: List[str] = []
        parts.append(
            f"Fairness audit across {n_vars} variable(s) "
            f"({', '.join(per_variable.keys())}), {n_samples} samples."
        )

        for var_name, var_data in per_variable.items():
            disp = var_data.get("disparities", {}).get("accuracy")
            if disp is None:
                parts.append(
                    f" {var_name}: insufficient data for disparity "
                    "computation."
                )
                continue
            parts.append(
                f" {var_name}: accuracy mean={disp['mean']:.3f}, "
                f"range=[{disp['min']:.3f}, {disp['max']:.3f}] "
                f"(gap={disp['range']*100:.1f}pp), "
                f"worst='{disp['worst_group']}', "
                f"best='{disp['best_group']}'."
            )

        flagged = summary.get("flagged_variables", [])
        worst_var = summary.get("worst_variable")
        worst_disp = summary.get("worst_disparity_pp", 0.0)
        verdict = summary.get("fairness_flag", "unknown")
        if flagged:
            parts.append(
                f" {len(flagged)} variable(s) exceed the "
                f"{self.tolerance_pp:.0f}pp tolerance: "
                f"{', '.join(flagged)}."
            )
        parts.append(
            f" Largest disparity: {worst_disp:.1f}pp "
            f"({worst_var}). Overall fairness verdict: {verdict}."
        )
        return " ".join(parts)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _per_subgroup_metrics(self,
                              y_true: np.ndarray,
                              y_pred: np.ndarray,
                              group_labels: np.ndarray,
                              y_proba: Optional[np.ndarray] = None,
                              label_format=None) -> Dict[str, Any]:
        """Per-subgroup metrics + disparities for one variable."""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        group_labels = np.asarray(group_labels)
        mask = _non_missing_mask(group_labels)
        if mask.sum() == 0:
            return {
                "per_subgroup": {},
                "disparities": {},
                "n_subgroups": 0,
                "n_samples": int(len(y_true)),
                "n_missing": int(len(y_true) - mask.sum()),
            }
        y_true_m = y_true[mask]
        y_pred_m = y_pred[mask]
        labels_m = group_labels[mask]
        y_proba_m = y_proba[mask] if y_proba is not None else None

        classes = sorted(np.unique(np.concatenate([y_true_m, y_pred_m])).tolist(),
                         key=lambda x: str(x))
        per_subgroup: Dict[str, Any] = {}
        for g in np.unique(labels_m):
            gmask = labels_m == g
            n = int(gmask.sum())
            if n == 0:
                continue
            m = compute_metrics(y_true_m[gmask], y_pred_m[gmask],
                                classes=classes, rest_class=self.rest_class)
            entry: Dict[str, Any] = {
                "n": n,
                "accuracy": float(m["accuracy"]),
                "macro_f1": float(m["macro_f1"]),
                "weighted_f1": float(m["weighted_f1"]),
                "kappa": float(m["kappa"]),
                "inflation": float(m["inflation"]),
            }
            # Per-class F1.
            per_class = m.get("per_class", {})
            entry["per_class_f1"] = {
                str(k): float(v["f1"]) if isinstance(v, dict) else float(v)
                for k, v in per_class.items()
            }
            if y_proba_m is not None:
                try:
                    entry["ece"] = float(expected_calibration_error(
                        y_true_m[gmask], y_proba_m[gmask]))
                    entry["brier"] = float(brier_score(
                        y_true_m[gmask], y_proba_m[gmask]))
                except Exception as exc:
                    logger.debug(f"Calibration metric failed for group {g}: {exc}")
            label = label_format(g) if label_format else str(g)
            per_subgroup[label] = entry

        # Disparities per metric.
        metric_keys = ["accuracy", "macro_f1", "weighted_f1", "kappa"]
        if y_proba is not None:
            metric_keys += ["ece", "brier"]
        disparities: Dict[str, Dict[str, Any]] = {}
        for mk in metric_keys:
            vals = [(g, e[mk]) for g, e in per_subgroup.items() if mk in e]
            if not vals:
                continue
            arr = np.array([v for _, v in vals], dtype=float)
            disparities[mk] = {
                "max": float(arr.max()),
                "min": float(arr.min()),
                "range": float(arr.max() - arr.min()),
                "range_pp": float((arr.max() - arr.min()) * 100.0),
                "std": float(arr.std(ddof=0)) if len(arr) > 1 else 0.0,
                "mean": float(arr.mean()),
                "worst_group": str(vals[int(np.argmin(arr))][0]),
                "best_group": str(vals[int(np.argmax(arr))][0]),
            }

        return {
            "per_subgroup": per_subgroup,
            "disparities": disparities,
            "n_subgroups": int(len(per_subgroup)),
            "n_samples": int(mask.sum()),
            "n_missing": int(len(y_true) - mask.sum()),
        }

    def _bin_continuous(self, values: np.ndarray,
                        bins: List[Tuple[float, float]],
                        formatter=None) -> np.ndarray:
        """Bin a continuous variable into labelled groups.

        Bins are right-open except for the last bin which is right-
        inclusive. NaNs are kept as a sentinel ``"unknown"`` label.
        """
        values = np.asarray(values, dtype=float)
        out = np.array(["unknown"] * len(values), dtype=object)
        valid = ~np.isnan(values)
        for i, (low, high) in enumerate(bins):
            if i == len(bins) - 1:
                mask = valid & (values >= low) & (values <= high)
            else:
                mask = valid & (values >= low) & (values < high)
            label = formatter(low, high) if formatter else f"{low}-{high}"
            out[mask] = label
        return out

    def _bin_bmi(self, bmis: np.ndarray) -> np.ndarray:
        """Bin BMI using the WHO classification."""
        bmis = np.asarray(bmis, dtype=float)
        out = np.array(["unknown"] * len(bmis), dtype=object)
        valid = ~np.isnan(bmis)
        for i, (low, high, label) in enumerate(self.bmi_bins):
            if i == len(self.bmi_bins) - 1:
                mask = valid & (bmis >= low) & (bmis <= high)
            else:
                mask = valid & (bmis >= low) & (bmis < high)
            out[mask] = label
        return out

    def _bin_quartiles(self, values: np.ndarray,
                       name: str = "var") -> np.ndarray:
        """Bin a continuous variable into quartile-labelled groups."""
        values = np.asarray(values, dtype=float)
        out = np.array([f"{name}_unknown"] * len(values), dtype=object)
        valid = ~np.isnan(values)
        if valid.sum() < 4:
            # Not enough data to bin — fall back to a single group.
            out[valid] = f"{name}_all"
            return out
        qs = np.quantile(values[valid], [0.25, 0.5, 0.75])
        labels = [f"{name}_Q1", f"{name}_Q2", f"{name}_Q3", f"{name}_Q4"]
        edges = [-np.inf, qs[0], qs[1], qs[2], np.inf]
        for i, lab in enumerate(labels):
            mask = valid & (values >= edges[i]) & (values < edges[i + 1])
            if i == len(labels) - 1:
                mask = valid & (values >= edges[i]) & (values <= edges[i + 1])
            out[mask] = lab
        return out

    def _build_summary(self,
                       per_variable: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Build a high-level summary across all variables."""
        per_var_max_gap: List[Tuple[str, float]] = []
        flagged: List[str] = []
        for var_name, var_data in per_variable.items():
            disp = var_data.get("disparities", {}).get("accuracy")
            if disp is None:
                continue
            gap_pp = float(disp.get("range_pp", disp.get("range", 0.0) * 100.0))
            per_var_max_gap.append((var_name, gap_pp))
            if gap_pp > self.tolerance_pp:
                flagged.append(var_name)
        if per_var_max_gap:
            worst_var, worst_gap = max(per_var_max_gap, key=lambda x: x[1])
        else:
            worst_var, worst_gap = None, 0.0

        if not flagged:
            verdict = "pass"
        elif len(flagged) == len(per_var_max_gap):
            verdict = "fail"
        else:
            verdict = "concerning"

        return {
            "worst_variable": worst_var,
            "worst_disparity_pp": float(worst_gap),
            "flagged_variables": flagged,
            "fairness_flag": verdict,
            "tolerance_pp": float(self.tolerance_pp),
            "n_variables": int(len(per_variable)),
        }


# ----------------------------------------------------------------------
# Module-level helpers
# ----------------------------------------------------------------------
def _age_label(low: float, high: float) -> str:
    """Render an age bin as a human-readable label."""
    if high >= 200:
        return f"{int(low)}+"
    return f"{int(low)}-{int(high)}"


def _non_missing_mask(arr: np.ndarray) -> np.ndarray:
    """Boolean mask of non-missing entries in a possibly-object array."""
    arr = np.asarray(arr)
    if arr.dtype.kind in {"U", "S", "O"}:
        # Strings / object: missing = "" or "nan" or "unknown".
        return np.array([
            (s != "") and (s is not None) and (str(s).lower() not in {"nan", "unknown", "none"})
            for s in arr
        ], dtype=bool)
    return ~np.isnan(arr.astype(float))
