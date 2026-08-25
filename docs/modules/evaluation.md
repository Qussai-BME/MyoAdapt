# `myoadapt.evaluation` — Evaluation, statistics, calibration, fairness

## Sub-modules

| Sub-module | What it provides |
|---|---|
| `metrics` | `compute_metrics`, Rest-class inflation, per-class breakdowns |
| `statistics` | Friedman, Wilcoxon, Nemenyi, Cohen's d / dz, Hedges' g, BCa CI, power analysis |
| `calibration` | Temperature / Platt / Isotonic calibration, ECE / MCE / Brier / NLL, conformal prediction |
| `fairness` | Per-subgroup metrics, equalized odds, fairness summaries |
| `diagrams` | Critical Difference diagram, per-fold box-plot, per-class heatmap, reliability diagram |
| `paper_reports` | LaTeX-ready tables (booktabs) + figure bundle writer |
| `model_card` | Auto Model Card (Mitchell 2019) + Datasheet (Pushkarna 2022) generator |
| `loso` / `lodo` | Cross-subject / cross-database evaluators |
| `reports` | CSV + Markdown table generator |

## Statistical rigor pipeline

The canonical end-to-end statistical comparison for a multi-model
benchmark:

```python
import numpy as np
from myoadapt.evaluation import (
    friedman_test, wilcoxon_pairwise, nemenyi_posthoc,
    cohen_d, hedges_g, interpret_effect_size,
    bootstrap_ci_bca, power_analysis_paired_ttest,
    critical_difference_diagram, write_figure_bundle,
    main_results_table_tex, effect_size_table_tex, stats_table_tex,
)

# per_fold_scores: {model_name: [per-fold accuracy, ...]}
folds = {
    "RF":  [0.71, 0.68, 0.74, 0.69, 0.72, 0.70, 0.73, 0.71, 0.69, 0.72],
    "LDA": [0.65, 0.62, 0.67, 0.63, 0.66, 0.64, 0.68, 0.65, 0.63, 0.66],
    "XGB": [0.73, 0.70, 0.75, 0.71, 0.74, 0.72, 0.76, 0.73, 0.71, 0.74],
}

# 1. Friedman test: are the K models significantly different across N folds?
fried = friedman_test(folds)
print(f"Friedman χ² = {fried['chi2']:.3f}, p = {fried['p_value']:.4f}")

# 2. Nemenyi post-hoc: which pairs differ?
nemenyi = nemenyi_posthoc(folds)

# 3. Pairwise Wilcoxon with Holm-Šídák correction.
wilc = wilcoxon_pairwise(folds)

# 4. Effect sizes.
for a, b in [("RF", "LDA"), ("RF", "XGB"), ("LDA", "XGB")]:
    d = cohen_d(folds[a], folds[b])
    g = hedges_g(folds[a], folds[b])
    print(f"{a} vs {b}: d={d:.3f}, g={g:.3f} ({interpret_effect_size(g)})")

# 5. BCa 95% CI for each model's mean accuracy.
for m, scores in folds.items():
    ci = bootstrap_ci_bca(scores, n_bootstrap=2000)
    print(f"{m}: {ci['point']:.3f} [{ci['lower']:.3f}, {ci['upper']:.3f}]")

# 6. Power analysis: was N=10 enough to detect d=0.5?
power = power_analysis_paired_ttest(0.5, n=10)
print(f"Power at d=0.5, n=10: {power:.3f} (target: 0.8)")

# 7. Critical Difference diagram (paper-ready figure).
fig = critical_difference_diagram(folds)
fig.savefig("cd_diagram.png", dpi=300, bbox_inches="tight")

# 8. Full figure bundle + LaTeX tables.
write_figure_bundle(folds, Path("./paper_figures"))
print(main_results_table_tex(...))
print(effect_size_table_tex(folds))
print(stats_table_tex(folds))
```

## Calibration workflow

```python
from myoadapt.evaluation import (
    TemperatureScaling, PlattCalibration, IsotonicCalibration,
    expected_calibration_error, calibration_report, conformal_prediction_set,
)

# Split data into train / calibration / test.
# Fit calibrator on calibration set, evaluate on test set.
ts = TemperatureScaling().fit(val_logits, val_y)
calibrated_proba = ts.transform(test_logits)

before = calibration_report(test_y, softmax(test_logits))
after  = calibration_report(test_y, calibrated_proba)
print(f"ECE: {before['ece']:.3f} -> {after['ece']:.3f}")

# Conformal prediction sets (distribution-free coverage guarantee).
sets = conformal_prediction_set(
    test_proba,
    y_calib=val_y, y_proba_calib=val_proba,
    alpha=0.1,  # 90% coverage
)
```

## Fairness audit

```python
from myoadapt.evaluation import per_subgroup_metrics, fairness_summary

audit = per_subgroup_metrics(
    y_true, y_pred, groups=subject_ids,
    y_proba=y_proba,  # optional, enables per-subgroup ECE / Brier
)
print(fairness_summary(audit))
# Fairness audit: 22 subgroups, 9020 samples total.
#   accuracy: mean=0.734, range=[0.610, 1.000] (gap=0.390), worst='3'.
#   macro_f1: mean=0.626, range=[0.195, 0.737] (gap=0.542), worst='13'.
#   ece:      mean=0.484, range=[0.365, 0.707] (gap=0.341), worst='3'.
```

## Model Card generation

```python
from myoadapt.evaluation import write_model_card

write_model_card(
    "./reports/model_card.md",
    model_name="XGBoost-DB2",
    intended_use="Cross-subject gesture classification on NinaPro DB2.",
    training_data={"name": "NinaPro-DB2", "n_subjects": 40, ...},
    evaluation={"protocol": "LOSO", "n_folds": 40,
                "accuracy_mean": 0.72, "accuracy_std": 0.08, ...},
    fairness=audit,
    calibration=calibration_report_dict,
)
```
