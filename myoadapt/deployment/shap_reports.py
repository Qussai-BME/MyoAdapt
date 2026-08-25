"""
shap_reports.py — SHAP-based transparency reports
=================================================

Generates per-prediction transparency reports:
- Predicted class + confidence
- Top contributing features (with SHAP values)
- Per-channel contribution
- Per-feature-group contribution
- Trust score
- Plain-language explanation

The JSON-serialised report is intended to support audit documentation
for research-grade deployments. It is not, by itself, a regulatory
certification — see ``docs/compliance/`` for the current compliance
posture and limitations.

"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    logger.warning("SHAP not installed. Install: pip install shap")


@dataclass
class TransparencyReport:
    """Per-prediction transparency report."""
    predicted_class: str
    confidence: float
    top_features: List[Dict[str, Any]]  # [{name, shap_value, contribution_pct}]
    per_channel_contribution: Dict[str, float]
    per_feature_group_contribution: Dict[str, float]
    trust_score: float  # 0-1
    explanation: str
    warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


class ShapReportGenerator:
    """
    Generate SHAP-based transparency reports for EMG predictions.

    Usage
    -----
    >>> gen = ShapReportGenerator(classifier, feature_names=feats)
    >>> report = gen.explain(X_test[:1])
    >>> print(report.to_json())
    """

    FEATURE_GROUPS = {
        "MAV": "Time-Domain", "RMS": "Time-Domain", "ZCR": "Time-Domain",
        "WL": "Time-Domain", "SSC": "Time-Domain", "WAMP": "Time-Domain",
        "MYOP": "Time-Domain", "STD": "Time-Domain", "VAR": "Time-Domain",
        "SKEW": "Time-Domain", "KURT": "Time-Domain", "MAX": "Time-Domain",
        "MIN": "Time-Domain", "RANGE": "Time-Domain", "IEMG": "Time-Domain",
        "AR_": "Time-Domain",
        "activity": "Hjorth", "mobility": "Hjorth", "complexity": "Hjorth",
        "MNF": "Frequency-Domain", "MDF": "Frequency-Domain",
        "PKF": "Frequency-Domain", "PSR": "Frequency-Domain",
        "SNR": "Frequency-Domain", "SM1": "Frequency-Domain",
        "SM2": "Frequency-Domain", "SM3": "Frequency-Domain",
        "Hist_": "Histogram",
        "ICC_": "Inter-Channel Correlation",
        "WPE_": "Time-Frequency",
        "STFT_": "Time-Frequency",
    }

    def __init__(self, classifier, feature_names: Optional[List[str]] = None,
                 background: Optional[np.ndarray] = None):
        self.classifier = classifier
        if hasattr(classifier, "feature_names") and classifier.feature_names:
            self.feature_names = classifier.feature_names
        elif feature_names:
            self.feature_names = feature_names
        else:
            self.feature_names = [f"f{i}" for i in range(100)]

        self.explainer_ = None
        if HAS_SHAP:
            try:
                # Use TreeExplainer for tree-based models
                model = (classifier.model if hasattr(classifier, "model")
                        else classifier)
                if hasattr(model, "estimators_") or "XGB" in type(model).__name__:
                    self.explainer_ = shap.TreeExplainer(model)
                else:
                    if background is None:
                        background = np.zeros((10, len(self.feature_names)),
                                              dtype=np.float32)
                    self.explainer_ = shap.KernelExplainer(
                        (lambda x: classifier.predict_proba(x)
                         if hasattr(classifier, "predict_proba")
                         else classifier.predict(x)),
                        background,
                    )
            except Exception as e:
                logger.warning(f"Could not build SHAP explainer: {e}")

    def explain(self, X: np.ndarray, max_display: int = 20) -> TransparencyReport:
        """Generate a transparency report for one or more samples."""
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Predict
        if hasattr(self.classifier, "predict_proba"):
            proba = self.classifier.predict_proba(X)
            pred_idx = int(np.argmax(proba[0]))
            confidence = float(proba[0, pred_idx])
        else:
            pred = self.classifier.predict(X)
            pred_idx = int(pred[0])
            confidence = 1.0

        classes = (self.classifier.classes_ if hasattr(self.classifier, "classes_")
                   else np.arange(proba.shape[1] if 'proba' in locals() else 1))
        pred_class = str(classes[pred_idx]) if hasattr(classes, "__getitem__") else str(pred_idx)

        # SHAP values
        if self.explainer_ is not None:
            try:
                shap_values = self.explainer_.shap_values(X)
                if isinstance(shap_values, list):
                    # Multi-class: take the predicted class's SHAP
                    shap_arr = shap_values[pred_idx][0]
                elif shap_values.ndim == 3:
                    shap_arr = shap_values[0, :, pred_idx]
                else:
                    shap_arr = shap_values[0]
            except Exception as e:
                logger.warning(f"SHAP computation failed: {e}")
                shap_arr = np.zeros(len(self.feature_names))
        else:
            shap_arr = np.zeros(len(self.feature_names))

        # Top features
        abs_shap = np.abs(shap_arr)
        order = np.argsort(-abs_shap)[:max_display]
        total = abs_shap.sum() + 1e-12
        top_features = [
            {
                "name": (self.feature_names[i] if i < len(self.feature_names)
                        else f"f{i}"),
                "shap_value": float(shap_arr[i]),
                "contribution_pct": float(abs_shap[i] / total * 100),
            }
            for i in order
        ]

        # Per-channel contribution
        per_channel: Dict[str, float] = {}
        for i, name in enumerate(self.feature_names):
            if i >= len(shap_arr):
                break
            # Extract channel id from name (ch0_MAV, ch1_RMS, ...)
            if name.startswith("ch"):
                ch = name.split("_")[0]
                per_channel[ch] = per_channel.get(ch, 0.0) + float(abs_shap[i])

        # Per-feature-group contribution
        per_group: Dict[str, float] = {}
        for i, name in enumerate(self.feature_names):
            if i >= len(shap_arr):
                break
            group = "Other"
            for prefix, grp in self.FEATURE_GROUPS.items():
                if name.startswith(prefix) or prefix in name:
                    group = grp
                    break
            per_group[group] = per_group.get(group, 0.0) + float(abs_shap[i])

        # Trust score (calibration-aware)
        trust_score = self._compute_trust_score(confidence, per_group)

        # Plain-language explanation
        explanation = self._generate_explanation(
            pred_class, confidence, top_features[:3], per_group
        )

        warnings = []
        if confidence < 0.5:
            warnings.append("Low confidence prediction — consider rejection threshold.")
        if not HAS_SHAP:
            warnings.append("SHAP not installed — feature contributions are zero.")
        if any(g > 0.8 * total for g in per_group.values()):
            warnings.append("Single feature group dominates — check for overfitting.")

        return TransparencyReport(
            predicted_class=pred_class,
            confidence=confidence,
            top_features=top_features,
            per_channel_contribution=per_channel,
            per_feature_group_contribution=per_group,
            trust_score=trust_score,
            explanation=explanation,
            warnings=warnings,
        )

    def _compute_trust_score(self, confidence: float,
                              per_group: Dict[str, float]) -> float:
        """Compute a 0-1 trust score."""
        # Penalize if a single group dominates
        total = sum(per_group.values()) + 1e-12
        max_group_frac = max((g / total for g in per_group.values()), default=0.0)
        dominance_penalty = max(0, max_group_frac - 0.5) * 0.5
        return float(max(0.0, min(1.0, confidence - dominance_penalty)))

    def _generate_explanation(self, pred: str, conf: float,
                                top3: List[Dict], groups: Dict) -> str:
        top_feat = top3[0]["name"] if top3 else "unknown"
        top_contrib = top3[0]["contribution_pct"] if top3 else 0.0
        dominant_group = max(groups, key=groups.get) if groups else "unknown"
        return (
            f"The model predicts '{pred}' with {conf*100:.1f}% confidence. "
            f"The top contributing feature is '{top_feat}' "
            f"({top_contrib:.1f}% of total |SHAP|). "
            f"The dominant feature group is '{dominant_group}'."
        )

    def batch_explain(self, X: np.ndarray,
                       output_dir: Union[str, Path]) -> List[TransparencyReport]:
        """Generate and save reports for many samples."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        reports = []
        for i in range(len(X)):
            r = self.explain(X[i:i+1])
            reports.append(r)
            with open(output_dir / f"report_{i:04d}.json", "w") as f:
                f.write(r.to_json())
        return reports
