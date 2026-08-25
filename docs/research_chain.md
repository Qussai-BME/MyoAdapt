# Research Capabilities & Paper Chain

MyoAdapt is the software artifact accompanying a **5-paper research chain**
on cross-subject sEMG pattern recognition. Each paper contributes a module
to the platform, and the platform reproduces every result.

## Paper Chain Overview

| # | Title | Venue | Platform Module |
|---|-------|-------|-----------------|
| 1 | Rest-Class Metric Inflation in Zero-Calibration Cross-Subject sEMG | Three-database LOSO benchmark (DB2/DB3/DB7, 73 subjects) | `evaluation.metrics`, `evaluation.loso` |
| 2 | Why MiniROCKET PPV Features Resist Domain Adaptation | Mechanistic study of CORAL failure | `features.minirocket`, `adaptation.coral` |
| 3 | Subject-Invariant EMG via a 52K-Parameter Lite Adversarial Network | CPU-only zero-calibration myocontrol | `models.lite_dan`, `deployment.hardware_bench` |
| 4 | When Does Domain Adaptation Help Cross-Subject sEMG? | Meta-analytic framework | `evaluation.statistics`, `evaluation.diagrams` |
| 5 | EMG-FM: A Foundation Model for Surface Electromyography | Self-supervised pretraining | `models.emg_foundation` |

---

## Paper 1: Rest-Class Metric Inflation

**Full title:** Rest-Class Metric Inflation in Zero-Calibration Cross-Subject sEMG: A Three-Database LOSO Benchmark Across Intact-Limb and Transradial Amputee Populations

**Authors:** Qussai Adlbi, Mohamad Ayham Darwich

**Affiliation:** Al-Andalus University for Medical Sciences (Syria) / Pázmány Péter Catholic University (Hungary)

### Contributions

1. **First simultaneous LOSO benchmark** across NinaPro DB2, DB3, and DB7 (73 subjects total, intact-limb and transradial amputees).
2. **38–44 pp advantage** of hand-crafted features (308-D hybrid TD+FD+Hjorth+Histogram+Correlation) over deep learning (CNN-1D) in the zero-calibration setting.
3. **Rest-class inflation metric** — quantifies how overall accuracy overestimates clinical utility when Rest dominates 30–50% of windows.
4. **XGBoost + SHAP analysis** identifying the most discriminative feature groups across populations.
5. **Amputee-specific failure mode** documented: DB3 macro-F1 drops to ~4% despite 40%+ overall accuracy.

### Key results

| Database | Population | LOSO accuracy | Macro-F1 | Rest inflation |
|----------|-----------|---------------|-----------|----------------|
| DB7 | Intact-limb | 65.96 ± 5.31% | 28.37% | 37.59 pp |
| DB2 | Intact-limb | 54.64 ± 4.89% | 24.92% | 29.72 pp |
| DB3 | Amputee | 43.46 ± 6.12% | 4.01% | 39.45 pp |

### Platform modules

- `myoadapt.evaluation.metrics.compute_metrics` — Rest-class inflation metric
- `myoadapt.evaluation.loso.LOSOEvaluator` — LOSO cross-validation
- `myoadapt.features` — 308-D hybrid feature set

---

## Paper 2: MiniROCKET PPV Near-Singularity

**Full title:** Why MiniROCKET PPV Features Resist Domain Adaptation: A Mechanistic Study of Cross-Subject sEMG Gesture Recognition Failure

### Contributions

1. **First systematic LOSO evaluation** of MiniROCKET on sEMG across three NinaPro databases.
2. **Empirical demonstration** that CORAL significantly *worsens* accuracy by amplifying near-singular PPV covariance.
3. **Mechanistic explanation** grounded in eigenvalue spectrum analysis of the PPV feature covariance matrix (condition number ~10^10–10^12).
4. **Window-size and training-sample ablations** confirming the LOSO ceiling is not an artifact of under-tuned hyperparameters.

### Key results

| Database | MiniROCKET LOSO | CORAL-adapted | Change |
|----------|-----------------|---------------|--------|
| DB7 | 9.15 ± 2.21% | 7.42 ± 1.89% | −1.73 pp (worse) |
| DB2 | 13.08 ± 3.05% | 11.36 ± 2.78% | −1.72 pp (worse) |
| DB3 | 6.63 ± 1.22% | 5.91 ± 1.05% | −0.72 pp (worse) |

### Platform modules

- `myoadapt.features.minirocket.MiniRocketVerifier` — eigenvalue diagnostic
- `myoadapt.adaptation.coral.CORAL` — CORrelation ALignment with mean alignment
- `myoadapt.evaluation.diagrams` — CD diagrams, per-fold boxplots

---

## Paper 3: Lite-DAN (52K-Parameter Adversarial Network)

**Full title:** Subject-Invariant EMG Pattern Recognition via a 52K-Parameter Lite Adversarial Network: CPU-Only Zero-Calibration Myocontrol for Low-Resource Settings

### Contributions

1. **Lite-DAN architecture** — 52K-parameter domain adversarial network with gradient reversal layer (GRL) and gradual λ schedule (0 → 1).
2. **5-CPU hardware benchmark** projecting inference latency to Intel Celeron G5900, i3-10100, i5-10400, i7-10700, AMD Ryzen 5 3600.
3. **Ablation study**: λ fixed=0.1 / fixed=0.5 / gradual 0→1; GRL on/off; Euclidean Alignment on/off.
4. **CPU-only deployment** — no GPU required for any stage, targeting low-resource clinical settings.

### Hypothesis

> Nonlinear adversarial DA on physiologically-motivated 308-D features will outperform linear DA methods (Paper 2) and the unadapted baseline (Paper 1), with the improvement quantified empirically.

### Platform modules

- `myoadapt.models.lite_dan.LiteDAN` — 52K-param adversarial network
- `myoadapt.deployment.hardware_bench.HardwareBenchmark` — 5-CPU projection
- `myoadapt.deployment.onnx_export.OnnxExporter` — ONNX + SHA-256

---

## Paper 4: Meta-Analytic Framework

**Full title:** When Does Domain Adaptation Help Cross-Subject sEMG? A Meta-Analytic Framework Across Feature Spaces, Classifier Families, and Population Types

### Contributions

1. **Meta-analytic framework** synthesizing results from Papers 1–3 across feature spaces (TD, MiniROCKET, raw waveform), classifier families (classical, CNN, adversarial), and populations (intact-limb, amputee).
2. **Novel insights**:
   - Linear DA succeeds on rank-full features but fails on near-singular (Paper 2 + new experiments)
   - Adversarial DA succeeds on physiologically-motivated features (Paper 3) but may fail on raw-waveform (Paper 1 CNN-1D)
   - Amputee data (DB3) requires fundamentally different approaches — DA cannot rescue Rest-class dominance
   - Subject-invariant features (Rest recall 97.5%) coexist with subject-specific features (active gestures) — partial invariance is the norm
3. **Statistical rigor**: Friedman + Nemenyi CD diagrams, Hedges' g effect sizes, BCa bootstrap CIs, a-priori power analysis.

### Platform modules

- `myoadapt.evaluation.statistics` — Friedman, Wilcoxon, Nemenyi, Hedges' g, BCa CI, power analysis
- `myoadapt.evaluation.diagrams` — CD diagrams, per-fold boxplots, per-class heatmaps
- `myoadapt.evaluation.paper_reports` — LaTeX-ready tables + figure bundles

---

## Paper 5: EMG Foundation Model

**Full title:** EMG-FM: A Foundation Model for Surface Electromyography

### Contributions

1. **EMG foundation model** — transformer encoder with masked-patch self-supervised pretraining on the concatenated NinaPro DB1–DB7 corpus.
2. **Self-supervised pretraining** — masked-patch reconstruction objective, no labels required.
3. **Fine-tuning hooks** — classification head added without destroying SSL weights; supports downstream tasks (gesture classification, continuous decoding).
4. **Open-source checkpoint registry** — no pretrained weights shipped; users pretrain on their own data via `scripts/pretrain_emg_foundation.py`.

### Platform modules

- `myoadapt.models.emg_foundation.EMGFoundation` — transformer encoder + SSL + classification head
- `myoadapt.tasks.continuous` — LSTM + Transformer regression for continuous decoding

---

## Research Roadmap

### Completed
- ✓ Paper 1: Rest-class inflation benchmark (DB2/DB3/DB7)
- ✓ Paper 2: MiniROCKET PPV near-singularity diagnostic
- ✓ Platform: MyoAdapt v2.0 (359 tests / 344 passing by default in the base install
  with full `.[all]` deps, 11 modules, 23 CLI commands). Includes
  continuous/pose decoding, rest-active intent detection, EMG-text
  semantic retrieval, and GAN/diffusion data augmentation
  (`myoadapt.tasks`, `data.synthetic_augmentation`) — activated and
  tested; not yet the subject of any paper in this series. Real
  emg2pose/emg2qwerty labels available via `data.meta_loaders` when
  those datasets are present, synthetic fallback otherwise.

### In Progress
- ◷ Paper 3: Lite-DAN LOSO evaluation on DB2/DB3/DB7 + 5-CPU benchmark
- ◷ Paper 4: Meta-analysis synthesis

### Planned
- ○ Paper 5: EMG foundation model pretraining on NinaPro DB1–DB7
- ○ Raspberry Pi edge demo with real Myo armband
- ○ Continuous EMG decoding (joint angle regression) for surgical robotics

---

## How to Cite

If you use MyoAdapt in your research, please cite:

```bibtex
@software{adlbi2025myoadapt,
  author       = {Qussai Adlbi},
  title        = {MyoAdapt: Open-Source CPU-Native sEMG Pattern Recognition Platform},
  year         = {2025},
  url          = {https://github.com/Qussai-BME/MyoAdapt},
  orcid        = {0009-0000-7667-1992},
  affiliation  = {Al-Andalus University for Medical Sciences},
  license      = {Apache-2.0},
}
```

## References

The bibliography associated with these research notes must be supplied with any manuscript or evidence bundle that relies on a result. This software release does not treat the narrative, tables, or cited methods in this page as independently reproduced performance evidence; see [Model Evidence Requirements](MODEL_EVIDENCE_REQUIREMENTS.md) before making a research claim.
