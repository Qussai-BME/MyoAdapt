# MyoAdapt

**An open-source, CPU-native, research-only Python platform for surface electromyography (sEMG) pattern recognition, evaluation, and controlled deployment experiments.**
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22096284.svg)](https://doi.org/10.5281/zenodo.22096284)

🚀 **[Live Interactive Demo](https://myoadapt-qussai-bme.streamlit.app/)

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-2.0.0-green.svg)](https://github.com/Qussai-BME/MyoAdapt)
[![Release posture](https://img.shields.io/badge/release-research--only-orange.svg)](docs/RELEASE_READINESS.md)
[![EU AI Act](https://img.shields.io/badge/EU%20AI%20Act-research--grade-indigo.svg)](docs/compliance/eu_ai_act.md)
[![Statistical Rigor](https://img.shields.io/badge/stats-Friedman%20%7C%20Nemenyi%20%7C%20BCa-purple.svg)](#statistical-rigor-suite)
[![Calibration](https://img.shields.io/badge/calibration-ECE%20%7C%20conformal-teal.svg)](#calibration--uncertainty)

---

> **Research-use notice.** MyoAdapt is not a medical device and is not validated for clinical diagnosis, treatment, patient management, or autonomous prosthetic control. Do not use its outputs as the sole basis for a safety-critical or health-related decision. Deployment operators remain responsible for data governance, human oversight, model validation, security controls, and regulatory assessment.

> **Release evidence.** The repository ships implemented evaluation utilities; it does not ship a clinical-validation package, regulatory clearance, certified safety case, or independent benchmark evidence for a specific intended use. Read [Release Readiness](docs/RELEASE_READINESS.md), the [Risk Register](docs/RISK_REGISTER.md), and the [Research Use Notice](docs/RESEARCH_USE_NOTICE.md) before installation or deployment.

## What it is

MyoAdapt is research software for surface electromyography (sEMG) pattern recognition. It provides modular signal-processing, evaluation, export, and visualization utilities that can run on CPU-only environments. The project does not, by itself, establish performance on a target population, operational robustness, clinical benefit, or readiness for a regulated deployment.

The platform is built around **six design constraints** that distinguish
it from every other open-source EMG toolkit:

1. **CPU-native operation** — every stage (filter, window, extract, train,
   evaluate, export) runs on a Celeron-class CPU. When a CUDA device is
   available, the PyTorch-based models (CNN1D, LiteDAN, EMGFoundation)
   auto-select it; otherwise they fall back to CPU.
2. **Generalization-evaluation utilities** — LOSO and LODO evaluators, subject-aware preprocessing, Rest-class inflation reporting, and active-only metrics. These tools do not establish zero-calibration performance unless they are run on a documented, representative, held-out protocol.
3. **Deployment experiments** — ONNX export, load/smoke verification, optional numerical-parity checking, an artifact SHA-256 sidecar, and projected hardware benchmarks. A SHA-256 comparison detects mismatch only when the expected hash is obtained from a trusted independent channel; it is not a complete provenance or tamper-prevention system.
4. **Transparency and documentation utilities** — SHAP reports, feature/channel aggregation, and confidence-derived informational scores. These outputs are not safety controls, clinical explanations, regulatory evidence, or a compliance certification.
5. **Statistical rigor suite** — Friedman + Nemenyi CD diagrams, Wilcoxon
   signed-rank with Holm-Šídák correction, Cohen's d / **Hedges' g**
   (small-N corrected), rank-biserial r, **BCa bootstrap confidence
   intervals**, and **a-priori statistical power analysis** — the exact
   artefacts reviewers ask for in 2026 ML/biosignal papers.
6. **Reproducibility utilities** — single-seed propagation across Python / NumPy / PyTorch, environment fingerprinting, and unsigned run manifests that record output hashes, git metadata when available, and configuration. These artifacts support review but do not guarantee deterministic reproduction or satisfy an external checklist without project-specific evidence.

---

## What's new in v2.0

| Module | What it adds | Why it matters |
|---|---|---|
| `evaluation.calibration` | Temperature / Platt / Isotonic calibration, ECE / MCE / Brier / NLL, **conformal prediction sets** | Safety-critical for prosthetics; EU AI Act transparency |
| `evaluation.fairness` | Per-subgroup metrics, equalized odds, fairness summaries | Bias audits now expected in health-AI venues |
| `evaluation.model_card` | Auto Model Cards (Mitchell 2019) + Datasheets (Pushkarna 2022) | NeurIPS D&B / Nature Methods expectation |
| `evaluation.paper_reports` | LaTeX-ready tables (booktabs) + figure bundles (PNG + PDF + .tex) | Kills the tedious part of paper-writing |
| `evaluation.diagrams` | **Nemenyi CD diagram**, per-fold box-plot, per-class heatmap, reliability diagram | The exact figures reviewers demand |
| `evaluation.statistics` | + Hedges' g, BCa CI, paired Wilcoxon with r, **power analysis** | Small-N rigor + a-priori sample-size justification |
| `reproducibility` | `set_global_seed`, env hash, `RunManifest` with SHA-256 verification | NeurIPS 2025/2026 reproducibility checklist |
| `models.base` | Classical-model aliases (`get_model("xgboost")` etc.) | Smoother UX; matches README |
| `deployment.onnx_export` | + SHA-256 hash in sidecar JSON | Tamper detection; FDA GMLP record-keeping |
| CLI | New: `audit`, `model-card`, `calibrate`, `power` | One-command audit / calibration / power |

---

## Quick start

### Install

```bash
git clone https://github.com/Qussai-BME/MyoAdapt.git
cd myoadapt
pip install -e ".[all]"
```

### Verify the environment

```bash
myoadapt info
```

### Launch the interactive UI

```bash
myoadapt ui --port 8501
```

### Run the full pipeline from the CLI

```bash
# 1. Train a model on real NinaPro data (or synthetic fallback)
myoadapt train --db DB2 --model xgboost --output ./models/xgb_db2.pkl

# 2. Evaluate with Leave-One-Subject-Out cross-validation
myoadapt eval --model-path ./models/xgb_db2.pkl --db DB2 --protocol loso

# 3. Export to ONNX (with SHA-256 sidecar for tamper detection)
myoadapt export --model-path ./models/xgb_db2.pkl --output ./models/xgb_db2.onnx

# 4. Run the 5-CPU latency benchmark
myoadapt benchmark --model-path ./models/xgb_db2.pkl --n-runs 100

# 5. Calibrate predicted probabilities (temperature / Platt / isotonic)
myoadapt calibrate --model-path ./models/xgb_db2.pkl --db DB2 --method temperature

# 6. Run a fairness / bias audit across subjects
myoadapt audit --model-path ./models/xgb_db2.pkl --db DB2

# 7. Generate a Model Card (Mitchell et al. 2019)
myoadapt model-card --model-path ./models/xgb_db2.pkl --output ./reports/card.md

# 8. Generate a SHAP transparency report for one sample
myoadapt explain --model-path ./models/xgb_db2.pkl --features ./data/sample.json

# 9. A-priori statistical power analysis
myoadapt power --effect-size 0.5           # minimum N for power=0.8
myoadapt power --effect-size 0.5 --n 12    # achieved power at N=12

# 10. Start the REST API + WebSocket streaming server
myoadapt serve --port 8000
myoadapt stream --model-path ./models/xgb_db2.pkl --port 8001
```

Or via Docker:

```bash
docker-compose up
```

---

## Architecture

```
myoadapt/
├── cli/           Typer CLI: train, eval, export, serve, stream, benchmark,
│                  explain, ui, info, verify, audit, model-card, calibrate, power
├── data/          Loaders · preprocessing · LOSO/LODO splits · Euclidean Alignment
├── features/      Time-domain · frequency-domain · histograms · correlation · TF · MiniROCKET
├── models/        Classical (7) · CNN-1D · Lite-DAN (~53K params) · EMG Foundation Model
├── adaptation/    CORAL · TCA · SA · EA · Adversarial · eigenvalue diagnostics
├── evaluation/    LOSO + LODO · metrics · statistics · calibration · fairness ·
│                  diagrams · paper_reports · model_card · reports
├── deployment/    ONNX (+ SHA-256) · realtime · SHAP reports · trust scores · 5-CPU benchmark
├── tracking/      Local JSON tracker · MLflow adapter
├── api/           FastAPI REST · WebSocket streaming
├── ui/            Streamlit 5-page UI + centralized theme
├── tasks/         Continuous decoding (LSTM + Transformer regression)
└── reproducibility/  Seed propagation · env hashing · run manifests
```

The full dependency graph is acyclic: data → features → models → adaptation →
evaluation → deployment. Each module is independently importable and
covered by unit tests.

---

## Modules at a glance

| Module | What it does | Key classes / functions |
|---|---|---|
| `data` | Loaders for NinaPro DB1/DB2/DB3/DB7, CapgMyo, UCI, Meta emg2pose/emg2qwerty (synthetic fallback when files absent) · Butterworth + Notch · Euclidean Alignment · LOSO + LODO splits · GAN/diffusion synthetic augmentation | `load_dataset`, `load_meta_dataset`, `list_databases`, `preprocess_signal`, `loso_splits`, `compute_subject_alignment`, `EMGGANAugmenter`, `EMGDiffusionAugmenter` — reachable via `myoadapt augment` |
| `features` | Time-domain (22/channel, AR via Yule-Walker) · Frequency-domain (8) · Histograms · Inter-channel correlation · Time-frequency (wavelets) · MiniROCKET (10K kernels) | `extract_features`, `list_features`, `MiniRocketVerifier` |
| `models` | Classical: SVM, RF, XGBoost, LightGBM, ExtraTrees, LDA, Logistic · CNN-1D · Lite-DAN (~53K params, GRL, gradual λ) · EMG Foundation Model | `EMGClassifier`, `CNN1D`, `LiteDAN`, `EMGFoundation`, `get_model`, `list_models` |
| `adaptation` | CORAL · TCA · Subspace Alignment · Euclidean Alignment · Adversarial (GRL) · Eigenvalue diagnostics | `CORAL`, `TCA`, `SubspaceAlignment`, `EuclideanAlignment`, `PerSubjectEA`, `AdversarialDA` |
| `evaluation` | LOSO + LODO · per-fold metrics · Rest inflation · statistics · **calibration** · **fairness** · **demographic fairness** · **drift detection** · **electrode-shift robustness** · **zero-shot evaluation** · **hyperparameter tuning** · **diagrams** · **paper_reports** · **model_card** | `LOSOEvaluator`, `LODOEvaluator`, `compute_metrics`, `friedman_test`, `wilcoxon_pairwise`, `nemenyi_posthoc`, `cohen_d`, `cohen_dz`, `hedges_g`, `bootstrap_ci_bca`, `power_analysis_paired_ttest`, `critical_difference_diagram`, `TemperatureScaling`, `conformal_prediction_set`, `per_subgroup_metrics`, `DemographicFairnessAudit`, `DriftDetector`, `ElectrodeShiftRobustness`, `ZeroShotEvaluator`, `HyperparameterOptimizer`, `write_model_card` — reachable via `myoadapt audit --demographics-csv` / `myoadapt drift` / `myoadapt robustness` / `myoadapt zeroshot` / `myoadapt tune` |
| `deployment` | ONNX export + verification + **SHA-256** · Realtime inference · SHAP reports · Trust scorer · 5-CPU PassMark benchmark | `OnnxExporter`, `RealtimeInference`, `ShapReportGenerator`, `TrustScorer`, `HardwareBenchmark` |
| `tracking` | Local JSON tracker · MLflow adapter | `LocalTracker`, `MLflowTracker`, `get_tracker` |
| `api` | FastAPI REST endpoints · WebSocket streaming server | `create_app`, `create_ws_app` |
| `ui` | Streamlit 5-page UI + centralized theme | `apply_theme`, `hero`, `kpi_card` |
| `tasks` | Continuous/pose decoding (LSTM + Transformer regression, MLP + patch-Transformer pose) · rest/active intent detection (energy-threshold + learned) · EMG↔text semantic retrieval (CLIP-style contrastive) | `LSTMRegressor`, `TransformerRegressor`, `PoseMLPRegressor`, `EMGSemanticRetriever`, `EnergyThresholdDetector`, `LearnedIntentDetector` — reachable via `myoadapt decode` / `myoadapt intent` / `myoadapt retrieve` |
| `reproducibility` | Seed propagation · env fingerprint · run manifests · config snapshots | `set_global_seed`, `environment_hash`, `RunManifest` |

---

## Statistical rigor suite

The evaluation module ships the full statistical-rigor pipeline that
academic reviewers expect in a 2026 multi-model benchmark:

```python
from myoadapt.evaluation import (
    # Effect sizes
    cohen_d, cohen_dz, hedges_g, interpret_effect_size,
    # Bootstrap CIs
    bootstrap_ci, bootstrap_ci_bca,
    # Hypothesis tests
    friedman_test, wilcoxon_pairwise, paired_wilcoxon_test, nemenyi_posthoc,
    # Power analysis
    power_analysis_paired_ttest, minimum_sample_size_paired,
    # Figures (paper-ready)
    critical_difference_diagram, per_fold_boxplot, per_class_heatmap, reliability_diagram,
    # LaTeX tables
    main_results_table_tex, effect_size_table_tex, stats_table_tex, bca_ci_table_tex,
    # Figure bundle (writes PNG + PDF + .tex caption files)
    write_figure_bundle,
)
```

**Critical Difference diagram** (Demsar 2006): one figure that shows
whether your K models are statistically different across N folds. Models
joined by a horizontal bar are NOT significantly different.

**BCa bootstrap** (Efron & Tibshirani 1993): bias-corrected and
accelerated — second-order accurate, transformation-respecting. The
recommended default over the naive percentile bootstrap.

**Hedges' g**: Cohen's d with small-sample bias correction. Use this
when N < 20 per group (typical for amputee databases: NinaPro DB3 has
11 amputes, DB7 has 22 subjects).

**A-priori power analysis**: justify your sample size BEFORE running
the experiment. `myoadapt power --effect-size 0.5 --n 12` tells you
the achieved power; `myoadapt power --effect-size 0.5` tells you the
minimum N needed for power = 0.8.

---

## Calibration & uncertainty

```python
from myoadapt.evaluation import (
    TemperatureScaling, PlattCalibration, IsotonicCalibration,
    expected_calibration_error, brier_score, calibration_report,
    conformal_prediction_set,
)
```

Three calibration strategies, each with a clear use case:
- **Temperature scaling** (Guo et al. 2017): single-parameter, preserves
  the ranking of logits. Recommended default for neural networks.
- **Platt sigmoid** (Platt 1999): logistic regression on logits.
- **Isotonic regression** (Zadrozny & Elkan 2002): non-parametric
  monotonic fit. Most flexible but needs ≥1000 samples per class.

**Conformal prediction** (Vovk 2005; Angelopoulos 2023): distribution-
free finite-sample coverage. Returns a *set* of classes per prediction
with the guarantee that the true class is in the set with probability
≥ 1 - α.

---

## Reproducibility engine

```python
from myoadapt.reproducibility import (
    set_global_seed, environment_fingerprint, environment_hash,
    RunManifest, snapshot_config,
)

set_global_seed(42)  # Seeds Python + NumPy + PyTorch (CUDA included)

manifest = RunManifest(
    run_name="rf_db2_loso",
    dataset="NinaPro-DB2",
    model="random_forest",
    seed=42,
    config={"k_features": 420, "n_estimators": 200},
)
manifest.add_output("./models/xgb_db2.pkl", role="model")
manifest.add_output("./results/loso_db2.csv", role="results")
manifest.write("./results/manifest.json")

# Later — verify no tampering:
RunManifest.verify("./results/manifest.json")
# {'verified': True, 'checked': 2, 'mismatches': []}
```

The manifest is the single artefact reviewers should ask for when
auditing a result. It satisfies the NeurIPS 2025/2026 Reproducibility
Checklist and the FDA GMLP / EU AI Act Annex IV record-keeping
obligations (research-grade; not a compliance certification).

---

## Testing

```bash
pytest tests/ --no-cov -q
```

**344 tests pass by default** with the full `.[all]` extras installed (verified
directly against `audit_evidence/final_fast_after_all_hardening.txt`). 1 is
skipped unless an MLflow tracking server is running; the xgboost-fallback
test is skipped when xgboost is installed. 13 more are marked
`@pytest.mark.slow` (realistic-scale CLI runs — minutes by design, same cost
as `train`/`tune`/`audit` at full data scale) and excluded from the default
run; run them with `pytest -m slow` (all 13 pass — see
`audit_evidence/slow_regression_no_cov_final.txt`). Some individual test
files degrade gracefully without PyTorch (`pytest.importorskip`) — exact
pass count without the `ml` extra hasn't been re-verified since the newest
test files were added; treat 344 as the reference number. The suite covers
feature extraction correctness (including
AR coefficient verification against a synthetic AR(2) process with
known coefficients), MiniROCKET 10K-kernel near-singularity diagnostics,
Euclidean Alignment correctness, Lite-DAN ablation hooks, ONNX export
parity, SHAP report structure, hardware benchmark projection,
calibration, fairness, reproducibility, statistical power, and
end-to-end pipeline integration.

---

## Documentation

Full documentation is in [`docs/`](docs/) and can be served locally with
mkdocs:

```bash
pip install mkdocs mkdocs-material
mkdocs serve
```

| Document | What it covers |
|---|---|
| `docs/quickstart.md` | 5-minute install + first model |
| `docs/architecture.md` | Module map + dependency graph |
| `docs/api_reference.md` | Auto-generated API reference |
| `docs/modules/` | Per-module deep dives (data, features, models, adaptation, evaluation, deployment, tracking, api, ui, tasks) |
| `docs/compliance/eu_ai_act.md` | EU AI Act Article 13 alignment (research-grade; not a compliance certification) |
| `docs/compliance/fda_samd.md` | FDA SaMD pathway notes (research-grade; no submission filed) |
| `docs/compliance/gdpr.md` | GDPR data-governance notes |

---

## Datasets

MyoAdapt provides loaders for several open sEMG databases. The loader
falls back to a synthetic dataset of the correct shape when the real
data files are not present, so the platform can be verified end-to-end
without downloading anything.

| Database | Use case |
|---|---|
| NinaPro DB1 / DB2 / DB3 / DB7 | Healthy + amputee gesture classification |
| CapgMyo-DBa | High-density EMG gestures |
| UCI | Low-channel healthy grasps |

To use real NinaPro data, download it from <https://ninapro.hevs.ch/> and
place the `.mat` files under `./data/<DB>/`.

---

## License

Apache 2.0 — commercial-friendly. See [`LICENSE`](LICENSE).

---

## Acknowledgments

- NinaPro database maintainers (HES-SO Valais-Wallis, Switzerland)
- CapgMyo database (Zhejiang University, China)
- PyTorch, scikit-learn, XGBoost, SHAP, ONNX, FastAPI, Streamlit communities
- Demsar (2006), Guo et al. (2017), Hedges (1981), Lakens (2013),
  Mitchell et al. (2019), Pushkarna et al. (2022), Angelopoulos &
  Bates (2023) — methodological foundations.
