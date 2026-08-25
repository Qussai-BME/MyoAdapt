"""
model_card.py — Model Card & Datasheet generator
=================================================

Auto-generates Model Cards (Mitchell et al. 2019) and Data Cards
(Pushkarna et al. 2022) from the structured artefacts MyoAdapt
already produces (training config, evaluation results, fairness audit,
calibration report). The output is a Markdown file ready to ship
alongside a release or paste into a paper appendix.

The generator is deliberately template-driven: every section is a
plain function that returns a Markdown string, so callers can override
individual sections without rewriting the whole card.

References
----------
- Mitchell, M. et al. (2019). *Model Cards for Model Reporting.*
  FAT*.
- Pushkarna, M. et al. (2022). *Data Cards: Purposeful and
  Transparent Dataset Documentation for Responsible AI.*
- EU AI Act Annex IV (technical documentation for high-risk AI).
- FDA GMLP (Good Machine Learning Practice) principles.

License: Apache 2.0
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)


def _safe_get(d: Optional[Dict[str, Any]], key: str, default: str = "N/A") -> str:
    if not d:
        return default
    v = d.get(key, default)
    return "N/A" if v is None else str(v)


def model_card(
    model_name: str,
    model_version: str = "2.0.0",
    model_type: str = "Classifier",
    intended_use: str = "",
    out_of_scope: str = "",
    training_data: Optional[Dict[str, Any]] = None,
    evaluation: Optional[Dict[str, Any]] = None,
    fairness: Optional[Dict[str, Any]] = None,
    calibration: Optional[Dict[str, Any]] = None,
    limitations: Optional[str] = None,
    ethical_considerations: Optional[str] = None,
    citation: Optional[str] = None,
    contact: str = "adlbiqussai@gmail.com",
) -> str:
    """Generate a Model Card in Markdown.

    All optional sections default to a neutral placeholder; pass a
    populated dict (or string) to fill them in. The structure mirrors
    Mitchell et al. (2019) with the additions recommended by the EU
    AI Act Annex IV (intended purpose, out-of-scope uses, known
    limitations, human-oversight measures).
    """
    training_data = training_data or {}
    evaluation = evaluation or {}
    fairness = fairness or {}
    calibration = calibration or {}
    timestamp = datetime.now(timezone.utc).isoformat()

    sections: list[str] = []
    sections.append(f"# Model Card: {model_name}\n")
    sections.append(f"**Version**: {model_version}  ")
    sections.append(f"**Type**: {model_type}  ")
    sections.append(f"**Generated**: {timestamp}  ")
    sections.append(f"**Contact**: {contact}\n")

    sections.append("## 1. Intended use\n")
    sections.append(intended_use or
                    "Research-grade surface EMG pattern recognition. "
                    "Intended for offline evaluation on documented public "
                    "datasets (NinaPro DB1/DB2/DB3/DB7, CapgMyo, UCI). Not "
                    "for unsupervised clinical deployment.\n")

    sections.append("## 2. Out-of-scope uses\n")
    sections.append(out_of_scope or
                    "- Real-time control of a medical device without an "
                    "independent safety layer.\n"
                    "- Deployment on populations not represented in the "
                    "training data without revalidation.\n"
                    "- Use as the sole basis for a clinical decision.\n")

    sections.append("## 3. Training data\n")
    sections.append(f"- **Dataset**: {_safe_get(training_data, 'name')}\n")
    sections.append(f"- **Subjects**: {_safe_get(training_data, 'n_subjects')}\n")
    sections.append(f"- **Windows**: {_safe_get(training_data, 'n_windows')}\n")
    sections.append(f"- **Classes**: {_safe_get(training_data, 'n_classes')}\n")
    sections.append(f"- **Sampling rate (Hz)**: {_safe_get(training_data, 'fs')}\n")
    sections.append(f"- **Channels**: {_safe_get(training_data, 'n_channels')}\n")
    if "demographics" in training_data:
        sections.append(f"- **Demographics**: {training_data['demographics']}\n")
    if "preprocessing" in training_data:
        sections.append(f"- **Preprocessing**: {training_data['preprocessing']}\n")

    sections.append("## 4. Evaluation\n")
    sections.append(f"- **Protocol**: {_safe_get(evaluation, 'protocol')}\n")
    sections.append(f"- **Folds**: {_safe_get(evaluation, 'n_folds')}\n")
    sections.append(f"- **Accuracy (mean ± std)**: {_safe_get(evaluation, 'accuracy_mean')} ± {_safe_get(evaluation, 'accuracy_std')}\n")
    sections.append(f"- **Macro-F1 (mean ± std)**: {_safe_get(evaluation, 'macro_f1_mean')} ± {_safe_get(evaluation, 'macro_f1_std')}\n")
    if "bootstrap_ci" in evaluation:
        ci = evaluation["bootstrap_ci"]
        sections.append(f"- **Bootstrap 95% CI (accuracy)**: [{ci.get('lower', 'N/A')}, {ci.get('upper', 'N/A')}]\n")
    if "statistical_tests" in evaluation:
        st = evaluation["statistical_tests"]
        sections.append(f"- **Friedman χ²**: {_safe_get(st, 'friedman_chi2')}, p = {_safe_get(st, 'friedman_p')}\n")
        sections.append(f"- **Nemenyi CD (α=0.05)**: {_safe_get(st, 'nemenyi_cd')}\n")

    if fairness:
        sections.append("## 5. Fairness / subgroup analysis\n")
        sections.append(f"- **Subgroups audited**: {_safe_get(fairness, 'n_subgroups')}\n")
        if "disparities" in fairness:
            for metric, disp in fairness["disparities"].items():
                sections.append(
                    f"- **{metric}**: mean={disp.get('mean', 'N/A'):.3f}, "
                    f"range=[{disp.get('min', 0):.3f}, {disp.get('max', 0):.3f}], "
                    f"gap={disp.get('range', 0):.3f}, worst='{disp.get('worst_group', 'N/A')}'\n"
                )

    if calibration:
        sections.append("## 6. Calibration & uncertainty\n")
        sections.append(f"- **ECE**: {_safe_get(calibration, 'ece')}\n")
        sections.append(f"- **MCE**: {_safe_get(calibration, 'mce')}\n")
        sections.append(f"- **Brier score**: {_safe_get(calibration, 'brier')}\n")
        sections.append(f"- **NLL**: {_safe_get(calibration, 'nll')}\n")
        if "calibration_method" in calibration:
            sections.append(f"- **Calibration method**: {calibration['calibration_method']}\n")

    sections.append("## 7. Limitations\n")
    sections.append(limitations or
                    "- Performance is dataset-dependent; cross-database "
                    "generalisation typically drops 15–30 percentage "
                    "points relative to within-database LOSO.\n"
                    "- Per-subject variability is high; aggregate metrics "
                    "may mask subgroups with poor performance.\n"
                    "- The model is not a medical device and has not "
                    "received regulatory clearance.\n")

    sections.append("## 8. Ethical considerations\n")
    sections.append(ethical_considerations or
                    "- EMG data is biometric; subject-level data governance "
                    "and consent must be respected (see GDPR notes).\n"
                    "- Prosthetic control applications carry safety "
                    "consequences — a human-oversight / kill-switch layer "
                    "is mandatory for any deployed system.\n"
                    "- Bias audits should be repeated whenever the training "
                    "data or target population changes.\n")

    if citation:
        sections.append("## 9. Citation\n")
        sections.append("```bibtex\n" + citation + "\n```\n")

    sections.append("## 10. Compliance posture (research-grade)\n")
    sections.append(
        "- **EU AI Act**: transparency artefacts (this card, SHAP reports, "
        "audit log) are designed to support Article 13 documentation. "
        "Not a compliance certification.\n"
        "- **FDA SaMD**: ONNX export produces a frozen, hash-verified "
        "artefact suitable for a future 510(k) submission. No submission "
        "has been filed.\n"
        "- **GDPR**: no raw EMG leaves the training environment; ONNX "
        "inference consumes only feature vectors.\n"
    )
    return "\n".join(sections)


def write_model_card(
    output_path: Union[str, Path],
    **kwargs: Any,
) -> Path:
    """Generate and write a Model Card to ``output_path`` (Markdown)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    md = model_card(**kwargs)
    output_path.write_text(md, encoding="utf-8")
    logger.info(f"Model Card written to {output_path}")
    return output_path


def datasheet(
    dataset_name: str,
    n_subjects: int,
    n_windows: int,
    n_classes: int,
    fs: int,
    n_channels: int,
    source_url: str = "",
    license: str = "CC-BY-NC-SA 4.0 (NinaPro)",
    demographics: str = "mixed healthy and amputee participants",
    preprocessing: str = "Butterworth bandpass 20–450 Hz, 50 Hz notch, 200 ms / 50 ms windows",
    known_issues: str = "",
    citation: str = "",
) -> str:
    """Generate a Data Card / datasheet (Pushkarna et al. 2022)."""
    sections: list[str] = []
    sections.append(f"# Datasheet: {dataset_name}\n")
    sections.append("## 1. Motivation\n")
    sections.append(
        f"This datasheet documents the **{dataset_name}** surface EMG dataset as used "
        "by MyoAdapt. The dataset is intended for research in cross-subject "
        "and cross-database myoelectric control pattern recognition.\n"
    )
    sections.append("## 2. Composition\n")
    sections.append(f"- **Subjects**: {n_subjects}\n")
    sections.append(f"- **Windows**: {n_windows}\n")
    sections.append(f"- **Classes**: {n_classes}\n")
    sections.append(f"- **Sampling rate**: {fs} Hz\n")
    sections.append(f"- **Channels**: {n_channels}\n")
    sections.append(f"- **Population**: {demographics}\n")
    sections.append("## 3. Collection process\n")
    sections.append(f"- **Source**: {source_url or 'see dataset homepage'}\n")
    sections.append(f"- **License**: {license}\n")
    sections.append("## 4. Preprocessing\n")
    sections.append(f"- {preprocessing}\n")
    sections.append("## 5. Known issues / caveats\n")
    sections.append(known_issues or
                    "- Class imbalance is common (Rest class dominates).\n"
                    "- Per-subject variability is high; cross-subject "
                    "generalisation is the hard problem.\n"
                    "- Amputee databases (DB3) have smaller N than healthy "
                    "databases — small-sample statistics apply.\n")
    if citation:
        sections.append("## 6. Citation\n")
        sections.append("```bibtex\n" + citation + "\n```\n")
    return "\n".join(sections)


def write_datasheet(
    output_path: Union[str, Path],
    **kwargs: Any,
) -> Path:
    """Generate and write a Datasheet to ``output_path`` (Markdown)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    md = datasheet(**kwargs)
    output_path.write_text(md, encoding="utf-8")
    logger.info(f"Datasheet written to {output_path}")
    return output_path
