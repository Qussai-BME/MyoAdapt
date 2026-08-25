"""
drift_detection.py — Detect distribution drift in EMG data.

Detects when the EMG signal distribution changes between training
and deployment — critical for knowing when a model needs retraining.

Drift types detected:
- Feature drift: input feature distribution changes (KS test, MMD)
- Prediction drift: output distribution changes (chi-square)
- Performance drift: accuracy/F1 degrades over time (Page-Hinkley)

References
----------
- evidently AI (2024). "Data drift detection methods."
- NannyML (2024). "Post-deployment drift detection."
- Page (1954). "Continuous inspection schemes." Biometrika.

License: Apache 2.0
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# scipy provides the statistical tests (ks_2samp, chi2_contingency).
try:
    from scipy.stats import chi2_contingency, ks_2samp
    _HAS_SCIPY = True
except ImportError:  # pragma: no cover — scipy is a hard dep
    _HAS_SCIPY = False


def _safe_ks_2samp(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """Two-sample Kolmogorov–Smirnov test with graceful fallbacks.

    Returns ``(statistic, pvalue)``. Falls back to a heuristic
    empirical-CDF distance when scipy is unavailable so the API stays
    usable in environments without scipy.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size == 0 or b.size == 0:
        return 0.0, 1.0
    if _HAS_SCIPY:
        res = ks_2samp(a, b)
        return float(res.statistic), float(res.pvalue)
    # Fallback: empirical CDF distance + crude p-value.
    combined = np.concatenate([a, b])
    cdf_a = np.searchsorted(np.sort(a), combined, side="right") / max(1, a.size)
    cdf_b = np.searchsorted(np.sort(b), combined, side="right") / max(1, b.size)
    stat = float(np.max(np.abs(cdf_a - cdf_b)))
    # Crude p-value approximation via Kolmogorov distribution.
    n_eff = (a.size * b.size) / float(a.size + b.size)
    if n_eff <= 0:
        return stat, 1.0
    lam = (np.sqrt(n_eff) + 0.12 + 0.11 / np.sqrt(n_eff)) * stat
    p = 2.0 * float(np.exp(-2.0 * lam * lam))
    return stat, max(0.0, min(1.0, p))


def _mmd_squared(a: np.ndarray, b: np.ndarray,
                 gamma: Optional[float] = None) -> float:
    """Unbiased Maximum Mean Discrepancy (squared, RBF kernel).

    Used as a complementary multivariate drift statistic to the
    per-feature KS test. Returns 0.0 when either input is empty.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    if b.ndim == 1:
        b = b.reshape(-1, 1)
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    if a.shape[1] != b.shape[1]:
        raise ValueError(
            f"MMD: a and b must have the same number of features "
            f"(got {a.shape[1]} vs {b.shape[1]})"
        )
    m, n = a.shape[0], b.shape[0]
    if m < 2 or n < 2:
        return 0.0
    # Median-heuristic for gamma if not provided.
    if gamma is None:
        all_data = np.vstack([a, b])
        # Subsample to keep the pairwise-distance computation tractable.
        if all_data.shape[0] > 1000:
            rng = np.random.default_rng(0)
            all_data = all_data[rng.choice(all_data.shape[0], 1000, replace=False)]
        d2 = np.sum((all_data[:, None, :] - all_data[None, :, :]) ** 2, axis=-1)
        med = float(np.median(d2[d2 > 0])) if (d2 > 0).any() else 1.0
        gamma = 1.0 / max(med, 1e-12)
    K_aa = np.exp(-gamma * _pairwise_sq_dist(a))
    K_bb = np.exp(-gamma * _pairwise_sq_dist(b))
    K_ab = np.exp(-gamma * _cross_pairwise_sq_dist(a, b))
    # Unbiased estimators (zero diagonal).
    np.fill_diagonal(K_aa, 0.0)
    np.fill_diagonal(K_bb, 0.0)
    mmd = (K_aa.sum() / (m * (m - 1))
           + K_bb.sum() / (n * (n - 1))
           - 2.0 * K_ab.mean())
    return float(max(0.0, mmd))


def _pairwise_sq_dist(x: np.ndarray) -> np.ndarray:
    """(n, n) squared-Euclidean distance matrix."""
    sq = np.sum(x * x, axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (x @ x.T)
    return np.maximum(d2, 0.0)


def _cross_pairwise_sq_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(m, n) squared-Euclidean distance matrix between a and b."""
    sa = np.sum(a * a, axis=1)[:, None]
    sb = np.sum(b * b, axis=1)[None, :]
    d2 = sa + sb - 2.0 * (a @ b.T)
    return np.maximum(d2, 0.0)


def _page_hinkley(scores: np.ndarray,
                  threshold: float = 0.01,
                  drift_threshold: float = 2.0,
                  min_observations: int = 10,
                  mean_ref: Optional[float] = None) -> Dict[str, Any]:
    """Two-sided Page-Hinkley test for mean-shift detection.

    Detects both *increases* and *decreases* in the running mean of
    ``scores`` relative to the in-control mean. The classic one-sided
    PH statistic is:

        U_t = max(0, U_{t-1} + x_t - μ_ref - λ)            (increases)
        L_t = min(0, L_{t-1} + x_t - μ_ref + λ)            (decreases)

    The two-sided test statistic is ``m_t = max(U_t, -L_t)`` and a
    drift is declared when ``m_t`` first crosses ``drift_threshold``.

    Parameters
    ----------
    scores : (n,) array of per-window performance scores (e.g.
        accuracy or macro-F1).
    threshold : cumulative-likelihood reset threshold (PH's λ).
    drift_threshold : cumulative-sum crossing value above which a
        drift is declared.
    min_observations : number of warm-up observations before the test
        can fire (avoids spurious drift at start of stream).
    mean_ref : Optional[float] — the in-control mean. When ``None``,
        falls back to ``mean(scores)`` (appropriate when ``scores`` is
        a pure reference / current series without mixing).

    Returns
    -------
    dict with:
        - ``detected``: bool — whether a drift point was detected.
        - ``change_point``: int or None — index of the drift point.
        - ``mean_change``: float — estimated mean shift magnitude.
        - ``test_stat``: (n,) two-sided PH statistic series.
        - ``cumsum``: (n,) cumulative sum series (raw deviations).
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    n = scores.size
    if n < 2:
        return {"detected": False, "change_point": None, "mean_change": 0.0,
                "test_stat": np.zeros(n), "cumsum": np.zeros(n)}
    # In-control mean: caller may pass the reference-period mean so we
    # don't pollute it with the (potentially drifted) current scores.
    mu_ref = float(mean_ref) if mean_ref is not None else float(np.mean(scores))
    deviations = scores - mu_ref
    cumsum = np.cumsum(deviations)
    # Two-sided PH statistics.
    u_pos = np.zeros(n, dtype=np.float64)  # max(0, …) — detects increases
    u_neg = np.zeros(n, dtype=np.float64)  # min(0, …) — detects decreases
    m_t = np.zeros(n, dtype=np.float64)    # max(u_pos, -u_neg)
    detected = False
    change_point: Optional[int] = None
    for t in range(n):
        u_pos[t] = max(0.0, (u_pos[t - 1] if t > 0 else 0.0)
                       + deviations[t] - threshold)
        u_neg[t] = min(0.0, (u_neg[t - 1] if t > 0 else 0.0)
                       + deviations[t] + threshold)
        m_t[t] = max(u_pos[t], -u_neg[t])
        if (not detected
                and t >= min_observations
                and m_t[t] > drift_threshold):
            detected = True
            change_point = t
    mean_change = float(np.mean(scores[change_point:]) - np.mean(scores[:change_point])
                        if change_point else 0.0)
    return {
        "detected": bool(detected),
        "change_point": int(change_point) if change_point is not None else None,
        "mean_change": mean_change,
        "test_stat": m_t,
        "cumsum": cumsum,
    }


def _severity(ratio: float) -> str:
    """Map a drifted-feature ratio in [0, 1] to a qualitative severity."""
    if ratio < 0.1:
        return "low"
    if ratio < 0.5:
        return "medium"
    return "high"


# ---------------------------------------------------------------------------
# DriftDetector
# ---------------------------------------------------------------------------
class DriftDetector:
    """Production drift detector for EMG features / predictions / scores.

    Three families of drift are detected, each tied to a different
    real-world failure mode:

    - **Feature drift** — electrode shift / sweat / new user changes
      the input feature distribution. Detected with per-feature
      Kolmogorov-Smirnov (plus MMD for the joint distribution).
    - **Prediction drift** — the model's class predictions change
      distribution. Detected with a chi-square test on the label
      histograms.
    - **Performance drift** — accuracy / F1 degrades over time.
      Detected with a two-sided Page-Hinkley test for mean shift.

    Parameters
    ----------
    reference_data : Optional array-like
        Reference (training-time) feature matrix of shape
        ``(n_samples, n_features)``. May also be supplied via
        :meth:`fit`.
    significance_level : float (default 0.05)
        Family-wise significance level α for the per-feature tests.
        Each per-feature p-value is compared against α (no Bonferroni
        correction by default; pass a manually corrected α to apply
        one).
    """

    def __init__(self,
                 reference_data: Optional[np.ndarray] = None,
                 significance_level: float = 0.05):
        if not (0.0 < significance_level < 1.0):
            raise ValueError(
                f"significance_level must be in (0, 1) (got {significance_level})"
            )
        self.significance_level = float(significance_level)
        self.reference_data_: Optional[np.ndarray] = (
            np.asarray(reference_data, dtype=np.float64)
            if reference_data is not None else None
        )
        self.n_features_: Optional[int] = (
            self.reference_data_.shape[1]
            if self.reference_data_ is not None else None
        )

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(self, X_reference: np.ndarray) -> DriftDetector:
        """Store the reference (training-time) feature distribution."""
        X_reference = np.asarray(X_reference, dtype=np.float64)
        if X_reference.ndim != 2:
            raise ValueError(
                f"X_reference must be 2-D (n_samples, n_features) "
                f"(got shape {X_reference.shape})"
            )
        if X_reference.shape[0] < 2:
            raise ValueError(
                f"X_reference needs at least 2 rows (got {X_reference.shape[0]})"
            )
        self.reference_data_ = X_reference.copy()
        self.n_features_ = int(X_reference.shape[1])
        logger.info(
            f"DriftDetector fit: reference shape={X_reference.shape}, "
            f"alpha={self.significance_level}"
        )
        return self

    def _ensure_fitted(self) -> None:
        if self.reference_data_ is None:
            raise RuntimeError(
                "DriftDetector not fitted — call fit(X_reference) first."
            )

    # ------------------------------------------------------------------
    # Feature drift
    # ------------------------------------------------------------------
    def detect_feature_drift(self, X_current: np.ndarray) -> Dict[str, Any]:
        """Per-feature KS test + joint MMD between reference and current.

        Parameters
        ----------
        X_current : (n_samples, n_features) array of current features.

        Returns
        -------
        dict with:
            - ``drift_per_feature``: (n_features,) bool array.
            - ``p_value_per_feature``: (n_features,) float array.
            - ``statistic_per_feature``: (n_features,) float array (KS statistic).
            - ``n_drifted``: int — number of drifted features.
            - ``drift_ratio``: float in [0, 1].
            - ``severity``: 'low' / 'medium' / 'high'.
            - ``mmd``: float — joint-distribution MMD² (RBF kernel).
            - ``any_drift``: bool — at least one feature drifted.
        """
        self._ensure_fitted()
        X_current = np.asarray(X_current, dtype=np.float64)
        if X_current.ndim != 2:
            raise ValueError(
                f"X_current must be 2-D (got shape {X_current.shape})"
            )
        if X_current.shape[1] != self.reference_data_.shape[1]:
            raise ValueError(
                f"Feature mismatch: reference has "
                f"{self.reference_data_.shape[1]} features, current has "
                f"{X_current.shape[1]}"
            )
        n_features = X_current.shape[1]
        drift_per_feature = np.zeros(n_features, dtype=bool)
        p_values = np.zeros(n_features, dtype=np.float64)
        statistics = np.zeros(n_features, dtype=np.float64)
        for j in range(n_features):
            stat, p = _safe_ks_2samp(self.reference_data_[:, j], X_current[:, j])
            statistics[j] = stat
            p_values[j] = p
            drift_per_feature[j] = bool(p < self.significance_level)
        n_drifted = int(drift_per_feature.sum())
        ratio = float(n_drifted / max(1, n_features))
        # Joint MMD — a complementary multivariate view of drift.
        try:
            mmd = float(np.sqrt(max(0.0, _mmd_squared(self.reference_data_, X_current))))
        except Exception as exc:  # pragma: no cover — numerical issues
            logger.warning(f"MMD computation failed: {exc}")
            mmd = float("nan")
        return {
            "drift_per_feature": drift_per_feature,
            "p_value_per_feature": p_values,
            "statistic_per_feature": statistics,
            "n_drifted": n_drifted,
            "n_features": int(n_features),
            "drift_ratio": ratio,
            "severity": _severity(ratio),
            "mmd": mmd,
            "any_drift": bool(n_drifted > 0),
            "significance_level": self.significance_level,
        }

    # ------------------------------------------------------------------
    # Prediction drift (chi-square on label histograms)
    # ------------------------------------------------------------------
    def detect_prediction_drift(self,
                                 y_ref: np.ndarray,
                                 y_current: np.ndarray) -> Dict[str, Any]:
        """Chi-square test on prediction (class) distributions.

        Parameters
        ----------
        y_ref : (n_ref,) integer class labels from the reference period.
        y_current : (n_current,) integer class labels from the current
            period.

        Returns
        -------
        dict with:
            - ``drifted``: bool — whether the chi-square test rejects
              the null hypothesis of equal distributions.
            - ``p_value``: float.
            - ``statistic``: chi-square statistic.
            - ``dof``: degrees of freedom.
            - ``class_distribution_ref`` / ``class_distribution_current``:
              normalised class-histogram dicts.
            - ``js_divergence``: Jensen-Shannon divergence (a symmetric
              drift magnitude bounded in [0, 1]).
        """
        y_ref = np.asarray(y_ref).ravel()
        y_current = np.asarray(y_current).ravel()
        classes = np.unique(np.concatenate([y_ref, y_current]))
        if classes.size < 2:
            # Single-class case — no distribution to drift.
            return {
                "drifted": False,
                "p_value": 1.0,
                "statistic": 0.0,
                "dof": 0,
                "class_distribution_ref": {},
                "class_distribution_current": {},
                "js_divergence": 0.0,
                "significance_level": self.significance_level,
            }
        # Build contingency table of counts per class.
        ref_counts = np.array([int(np.sum(y_ref == c)) for c in classes],
                              dtype=np.float64)
        cur_counts = np.array([int(np.sum(y_current == c)) for c in classes],
                              dtype=np.float64)
        table = np.vstack([ref_counts, cur_counts])
        # Avoid zero rows/columns for chi-square stability.
        table_for_test = table + 0.5
        if _HAS_SCIPY:
            chi2, p, dof, _ = chi2_contingency(table_for_test)
            chi2, p, dof = float(chi2), float(p), int(dof)
        else:
            # Fallback: compute a crude chi-square statistic & a
            # normal-approximation p-value.
            row_sums = table_for_test.sum(axis=1, keepdims=True)
            col_sums = table_for_test.sum(axis=0, keepdims=True)
            grand = table_for_test.sum()
            expected = (row_sums @ col_sums) / max(1e-12, grand)
            chi2 = float(np.sum((table_for_test - expected) ** 2 /
                                 np.maximum(expected, 1e-12)))
            dof = int((table.shape[0] - 1) * (table.shape[1] - 1))
            from math import erfc, sqrt
            # Treat chi2 as a chi-square-distributed variable.
            try:
                # Use scipy's survival function via numpy if available.
                from scipy.stats import chi2 as chi2_dist
                p = float(chi2_dist.sf(chi2, dof))
            except Exception:
                # Crude Wilson–Hilferty normal approximation.
                if dof > 0:
                    z = ((chi2 / dof) ** (1.0 / 3.0)
                         - (1 - 2.0 / (9 * dof))) / np.sqrt(2.0 / (9 * dof))
                    p = 0.5 * float(erfc(z / sqrt(2.0)))
                else:
                    p = 1.0
        ref_dist = {str(c): float(v / max(1, ref_counts.sum()))
                    for c, v in zip(classes, ref_counts)}
        cur_dist = {str(c): float(v / max(1, cur_counts.sum()))
                    for c, v in zip(classes, cur_counts)}
        # Jensen-Shannon divergence — a symmetric drift magnitude.
        p_ref = np.array([v for v in ref_dist.values()], dtype=np.float64)
        p_cur = np.array([v for v in cur_dist.values()], dtype=np.float64)
        js = _js_divergence(p_ref, p_cur)
        return {
            "drifted": bool(p < self.significance_level),
            "p_value": float(p),
            "statistic": float(chi2),
            "dof": int(dof),
            "class_distribution_ref": ref_dist,
            "class_distribution_current": cur_dist,
            "js_divergence": js,
            "significance_level": self.significance_level,
        }

    # ------------------------------------------------------------------
    # Performance drift (Page-Hinkley)
    # ------------------------------------------------------------------
    def detect_performance_drift(self,
                                  scores_ref: np.ndarray,
                                  scores_current: np.ndarray,
                                  drift_threshold: float = 2.0,
                                  min_observations: int = 10) -> Dict[str, Any]:
        """Two-sided Page-Hinkley test for performance degradation.

        Compares the reference mean score against the current series
        and detects a sustained mean shift.

        Parameters
        ----------
        scores_ref : (n_ref,) reference-period scores (e.g. accuracy).
        scores_current : (n_current,) current-period scores.
        drift_threshold : cumulative-sum crossing value above which
            a drift is declared.
        min_observations : number of warm-up observations before the
            test can fire.

        Returns
        -------
        dict with the Page-Hinkley outputs plus a
        ``mean_score_ref`` / ``mean_score_current`` summary.
        """
        scores_ref = np.asarray(scores_ref, dtype=np.float64).ravel()
        scores_current = np.asarray(scores_current, dtype=np.float64).ravel()
        if scores_ref.size < 2 or scores_current.size < 2:
            return {
                "detected": False,
                "change_point": None,
                "mean_change": 0.0,
                "mean_score_ref": float(np.mean(scores_ref)) if scores_ref.size else float("nan"),
                "mean_score_current": float(np.mean(scores_current)) if scores_current.size else float("nan"),
                "test_stat": np.zeros(scores_current.size),
                "cumsum": np.zeros(scores_current.size),
                "drift_threshold": float(drift_threshold),
                "significance_level": self.significance_level,
            }
        # Page-Hinkley on the current scores, with the reference mean
        # as the in-control centre. We embed the reference mean as a
        # prefix so the running-mean PH starts near the reference.
        mean_ref = float(np.mean(scores_ref))
        # Concatenate the reference scores as a warm-up so the test
        # statistic is calibrated to the in-control mean before the
        # (potentially drifted) current series is scanned.
        full = np.concatenate([scores_ref, scores_current])
        ph = _page_hinkley(
            full,
            drift_threshold=drift_threshold,
            min_observations=max(min_observations, scores_ref.size),
            mean_ref=mean_ref,
        )
        # If change_point was found, translate it into the current frame.
        cp = ph["change_point"]
        if cp is not None:
            cp_current = cp - scores_ref.size
            if cp_current < 0:
                cp_current = 0
        else:
            cp_current = None
        return {
            "detected": bool(ph["detected"]),
            "change_point": int(cp_current) if cp_current is not None else None,
            "mean_change": float(ph["mean_change"]),
            "mean_score_ref": mean_ref,
            "mean_score_current": float(np.mean(scores_current)),
            "test_stat": ph["test_stat"],
            "cumsum": ph["cumsum"],
            "drift_threshold": float(drift_threshold),
            "significance_level": self.significance_level,
        }

    # ------------------------------------------------------------------
    # Combined drift summary
    # ------------------------------------------------------------------
    def detect_all(self,
                   X_current: np.ndarray,
                   y_ref: Optional[np.ndarray] = None,
                   y_current: Optional[np.ndarray] = None,
                   scores_ref: Optional[np.ndarray] = None,
                   scores_current: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Run all applicable drift tests and return a unified summary.

        Only the feature drift test is mandatory (requires
        ``X_current``). The prediction / performance drift tests run
        only when their respective inputs are supplied.

        Returns
        -------
        dict with keys ``feature``, ``prediction``, ``performance``
        (each ``None`` if not run) plus an aggregate ``summary`` with:
            - ``any_drift``: bool.
            - ``drift_types``: list of drift type names that fired.
            - ``severity``: overall qualitative severity.
        """
        self._ensure_fitted()
        feature_result = self.detect_feature_drift(X_current)
        prediction_result: Optional[Dict[str, Any]] = None
        performance_result: Optional[Dict[str, Any]] = None
        if y_ref is not None and y_current is not None:
            prediction_result = self.detect_prediction_drift(y_ref, y_current)
        if scores_ref is not None and scores_current is not None:
            performance_result = self.detect_performance_drift(
                scores_ref, scores_current,
            )

        drift_types: List[str] = []
        if feature_result["any_drift"]:
            drift_types.append("feature")
        if prediction_result is not None and prediction_result.get("drifted"):
            drift_types.append("prediction")
        if performance_result is not None and performance_result.get("detected"):
            drift_types.append("performance")

        # Overall severity: take the max of the per-test severities.
        severities = [feature_result.get("severity", "low")]
        if prediction_result is not None and prediction_result.get("drifted"):
            js = float(prediction_result.get("js_divergence", 0.0))
            if js > 0.3:
                severities.append("high")
            elif js > 0.1:
                severities.append("medium")
            else:
                severities.append("low")
        if performance_result is not None and performance_result.get("detected"):
            mc = abs(float(performance_result.get("mean_change", 0.0)))
            if mc > 0.1:
                severities.append("high")
            elif mc > 0.05:
                severities.append("medium")
            else:
                severities.append("low")
        rank = {"low": 0, "medium": 1, "high": 2}
        overall_severity = max(severities, key=lambda s: rank.get(s, 0))

        return {
            "feature": feature_result,
            "prediction": prediction_result,
            "performance": performance_result,
            "summary": {
                "any_drift": bool(len(drift_types) > 0),
                "drift_types": drift_types,
                "severity": overall_severity,
                "n_features": feature_result["n_features"],
                "significance_level": self.significance_level,
            },
        }

    # ------------------------------------------------------------------
    # Narrative
    # ------------------------------------------------------------------
    def drift_narrative(self, results: Dict[str, Any]) -> str:
        """Human-readable drift summary (output of :meth:`detect_all`)."""
        summary = results.get("summary", {})
        lines: List[str] = []
        any_drift = bool(summary.get("any_drift", False))
        if not any_drift:
            lines.append("No drift detected across any monitored dimension.")
            return "\n".join(lines)
        lines.append(
            f"Drift DETECTED — overall severity: {summary.get('severity', 'unknown').upper()}."
        )
        lines.append(
            f"Drift types that fired: {', '.join(summary.get('drift_types', []))}."
        )
        feat = results.get("feature")
        if feat is not None and feat.get("any_drift"):
            lines.append(
                f"  • Feature drift: {feat['n_drifted']}/{feat['n_features']} "
                f"features ({feat['drift_ratio'] * 100:.1f}%) shifted distribution "
                f"({feat['severity']} severity, MMD ≈ {feat['mmd']:.4f})."
            )
        pred = results.get("prediction")
        if pred is not None and pred.get("drifted"):
            lines.append(
                f"  • Prediction drift: chi-square p = {pred['p_value']:.2e}, "
                f"JS divergence = {pred['js_divergence']:.4f}."
            )
        perf = results.get("performance")
        if perf is not None and perf.get("detected"):
            lines.append(
                f"  • Performance drift: Page-Hinkley detected a mean shift of "
                f"{perf['mean_change']:+.4f} at observation "
                f"{perf.get('change_point')} (ref mean = "
                f"{perf['mean_score_ref']:.4f}, current mean = "
                f"{perf['mean_score_current']:.4f})."
            )
        # Recommendation.
        if summary.get("severity") == "high":
            lines.append(
                "Recommendation: retrain the model immediately — drift is severe "
                "and downstream predictions are likely unreliable."
            )
        elif summary.get("severity") == "medium":
            lines.append(
                "Recommendation: schedule a recalibration; consider enabling "
                "domain-adaptation (e.g. CORAL/EA) before the next inference batch."
            )
        else:
            lines.append(
                "Recommendation: monitor; drift is mild and the model may still "
                "be within its operational tolerance."
            )
        return "\n".join(lines)


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Symmetric Jensen-Shannon divergence (base-2), bounded in [0, 1]."""
    p = np.asarray(p, dtype=np.float64).ravel()
    q = np.asarray(q, dtype=np.float64).ravel()
    if p.size != q.size:
        raise ValueError(
            f"JS divergence: p and q must have the same length "
            f"(got {p.size} vs {q.size})"
        )
    p = p / max(1e-12, p.sum())
    q = q / max(1e-12, q.sum())
    m = 0.5 * (p + q)
    eps = 1e-12
    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        a_safe = np.clip(a, eps, 1.0)
        b_safe = np.clip(b, eps, 1.0)
        return float(np.sum(a_safe * np.log2(a_safe / b_safe)))
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


__all__ = ["DriftDetector"]
