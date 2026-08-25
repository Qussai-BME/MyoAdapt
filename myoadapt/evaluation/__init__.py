"""
myoadapt.evaluation — Unified evaluation harness
==================================================

Public API:
    LOSOEvaluator      — Leave-One-Subject-Out
    LODOEvaluator      — Leave-One-Database-Out
    ZeroShotEvaluator  — Zero-Shot Cross-User (large-pool, no adaptation)
    DemographicFairnessAudit — physiologically-grounded fairness audit
    ElectrodeShiftRobustness — electrode-shift robustness sweeps
    compute_metrics    — single-fold metrics
    friedman_test      — Friedman non-parametric test
    wilcoxon_pairwise  — Wilcoxon signed-rank + Holm-Šídák correction
    paired_wilcoxon_test — single-pair Wilcoxon with effect size
    nemenyi_posthoc    — Nemenyi post-hoc for Friedman
    cohen_d, cohen_dz, hedges_g — effect sizes (Hedges' g for small-N)
    interpret_effect_size — Cohen (1988) qualitative interpretation
    bootstrap_ci       — percentile bootstrap confidence interval
    bootstrap_ci_bca   — BCa bootstrap CI (recommended over percentile)
    power_analysis_paired_ttest — a-priori power for paired t-test
    minimum_sample_size_paired — minimum N for target power
    ReportGenerator    — auto-generate paper-ready tables/figures
    critical_difference_diagram — Nemenyi CD diagram (Demsar 2006)
    per_fold_boxplot   — paired per-fold box-plot
    per_class_heatmap  — per-class F1 heatmap
    reliability_diagram — calibration reliability diagram
"""
from myoadapt.evaluation.demographic_fairness import DemographicFairnessAudit
from myoadapt.evaluation.diagrams import (
    critical_difference_diagram,
    per_class_heatmap,
    per_fold_boxplot,
    reliability_diagram,
)
from myoadapt.evaluation.electrode_shift import ElectrodeShiftRobustness
from myoadapt.evaluation.lodo import LODOEvaluator
from myoadapt.evaluation.loso import LOSOEvaluator
from myoadapt.evaluation.metrics import compute_metrics, per_class_report
from myoadapt.evaluation.reports import ReportGenerator
from myoadapt.evaluation.statistics import (
    bootstrap_ci,
    bootstrap_ci_bca,
    cohen_d,
    cohen_dz,
    friedman_test,
    hedges_g,
    holm_sidak_correction,
    interpret_effect_size,
    minimum_sample_size_paired,
    nemenyi_posthoc,
    paired_wilcoxon_test,
    power_analysis_paired_ttest,
    wilcoxon_pairwise,
)
from myoadapt.evaluation.zero_shot import ZeroShotEvaluator

__all__ = [
    "LOSOEvaluator", "LODOEvaluator",
    "ZeroShotEvaluator",
    "DemographicFairnessAudit",
    "ElectrodeShiftRobustness",
    "compute_metrics", "per_class_report",
    "friedman_test", "wilcoxon_pairwise", "paired_wilcoxon_test",
    "nemenyi_posthoc",
    "cohen_d", "cohen_dz", "hedges_g", "interpret_effect_size",
    "holm_sidak_correction", "bootstrap_ci", "bootstrap_ci_bca",
    "power_analysis_paired_ttest", "minimum_sample_size_paired",
    "ReportGenerator",
    "critical_difference_diagram", "per_fold_boxplot",
    "per_class_heatmap", "reliability_diagram",
    # v2.0 gap-closing modules
    "HyperparameterOptimizer",  # Optuna-based HPO wrapper
    "FatigueTracker",           # EMG fatigue monitoring (MNF/MDF/RMS/IMNF/FI)
    "DriftDetector",            # Production drift detection (feature/prediction/performance)
]

# Re-export paper_reports and model_card (added v2.0)
from myoadapt.evaluation.drift_detection import DriftDetector
from myoadapt.evaluation.fatigue import FatigueTracker
from myoadapt.evaluation.hyperopt import HyperparameterOptimizer
from myoadapt.evaluation.model_card import (
    datasheet,
    model_card,
    write_datasheet,
    write_model_card,
)
from myoadapt.evaluation.paper_reports import (
    bca_ci_table_tex,
    effect_size_table_tex,
    main_results_table_tex,
    per_fold_table_tex,
    stats_table_tex,
    write_figure_bundle,
)
