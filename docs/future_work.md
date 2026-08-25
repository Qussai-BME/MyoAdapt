# Future Work

The five-paper research programme described in
[`paper_series.md`](paper_series.md) identifies four directions that
motivate ongoing development of MyoAdapt. These directions are
research-led: each addresses a limitation that emerged from the
preceding papers, and each is supported (or, in the case of the
foundation model, will be supported) by specific modules in the
codebase.

---

## 1. EMG Foundation Model (Paper 5)

### Vision

Self-supervised pretraining has transformed NLP (BERT, GPT) and vision
(ViT, SAM). EMG has no open-source foundation model at the time of
writing; the only comparable effort (Yang et al. 2025) is not
open-source. Paper 5 of the series — *EMG-FM: A Foundation Model for
Surface Electromyography* — proposes the first reproducible open-source
EMG foundation model.

### Technical approach

* **Architecture.** Transformer encoder with patch embedding. Each
  analysis window is split into fixed-length patches and projected to
  the embedding dimension via a linear layer.
* **Pretraining objective.** Masked patch reconstruction. A random
  subset of patches is masked, and the encoder is trained to reconstruct
  the held-out patches from the visible ones.
* **Pretraining corpus.** Concatenated NinaPro DB1–DB7, with channel
  counts reconciled by sub-sampling or padding. The corpus is the
  largest open sEMG dataset currently available.
* **Fine-tuning.** Supervised fine-tuning on the downstream gesture
  classification task; embedding extraction for use as a frozen feature
  extractor.

### MyoAdapt support

* `models.emg_foundation.EMGFoundation` — the model class with
  `pretrain(X)`, `finetune(X, y)`, and `extract_embeddings(X)`
  methods.
* `scripts/pretrain_emg_foundation.py` — runnable pretraining script
  that loads NinaPro DB1/DB2/DB3/DB7 (+ CapgMyo-DBa, UCI) via
  `load_dataset`, concatenates by channel count, and runs
  `EMGFoundation.pretrain(X)`.

The architecture and the pretraining loop ship with v2.0. Pretrained
weights on the full NinaPro corpus are **not** shipped with the
package; users regenerate them by running the pretraining script on
the real `.mat` files. The current implementation has been validated
on synthetic data only.

### Expected impact

* Few-shot cross-subject generalisation exceeding the supervised Paper 1
  baseline (54.64% LOSO accuracy on NinaPro DB2).
* Reduced per-subject calibration data — the principal bottleneck for
  clinical deployment.
* A reference implementation for the sEMG community, addressing a gap
  in the open-source ecosystem.

### Timeline

* **Architecture & pretraining loop:** shipped in v2.0.
* **Pretraining on NinaPro DB1–DB7:** next milestone.
* **Open release of pretrained weights:** following pretraining and
  internal validation.
* **Paper 5 manuscript:** following the open release.

---

## 2. Continuous EMG Decoding

### Vision

Discrete gesture classification captures only the *intent* layer of
myoelectric control: which gesture is being performed. Continuous
decoding captures the full kinematic trajectory — joint angles,
forces, velocities — enabling applications in surgical robotics (where
the surgeon's hand motion is decoded from forearm EMG) and advanced
prosthetics (where the device mimics the patient's intended joint
angles in real time). Paper 4's identification of *partial invariance*
as a graded property of the representation directly motivates the
extension to continuous prediction: the same encoder that supports
discrete classification should support continuous regression if the
representation is sufficiently rich.

### Technical approach

* **Architecture.** Sequence-to-sequence regression with two model
  families — an LSTM-based regressor and a Transformer-based regressor
  — operating on sliding windows of sEMG samples.
* **Targets.** Joint-angle trajectories (kinematics) recorded
  synchronously with the sEMG, as available in NinaPro DB1, DB4, DB5,
  DB6, and DB7.
* **Evaluation.** Per-subject RMSE and Pearson correlation on held-out
  trials; cross-subject generalisation via LOSO on the kinematic
  targets.

### MyoAdapt support

* `tasks.continuous.ContinuousDecoder` — the high-level decoder
  abstraction.
* `tasks.continuous.LSTMRegressor` — the LSTM-based regressor.
* `tasks.continuous.TransformerRegressor` — the Transformer-based
  regressor.

The module is shipped; full benchmark evaluation on real NinaPro
kinematic data is the next milestone.

### Expected impact

* Extends the platform from classification-only to regression-capable,
  covering the two principal myoelectric-control paradigms.
* Provides a baseline for cross-subject continuous decoding — a
  research direction where the literature currently lacks a
  standardised benchmark.
* Enables downstream applications in surgical robotics and
  advanced prosthetics.

### Timeline

* **Module & models:** shipped in v2.0.
* **Cross-subject benchmark on real NinaPro kinematics:** next
  milestone.
* **Extension to force and stiffness decoding:** longer-term.

---

## 3. Real-time Edge Deployment

### Vision

Prosthetic devices have a power budget on the order of 5–10 W; GPUs
are not an option. Paper 3 establishes that Lite-DAN (≈52K parameters)
runs end-to-end on commodity CPUs, including an Intel Celeron G5900.
The next step is a live, real-time demonstration on a constrained-edge
device (Raspberry Pi class), exercising the full
ONNX-export → SHA-256 verification → streaming inference pipeline
on live EMG from a commercial armband.

### Technical approach

* **Hardware.** Raspberry Pi 4 (or equivalent) with a commercial sEMG
  armband providing 8 channels at 200 Hz.
* **Pipeline.**
  1. Live EMG acquisition via the armband's SDK.
  2. Preprocessing (Butterworth bandpass + IIR notch) — already shipped
     in `data.preprocessing`.
  3. Sliding-window feature extraction or direct waveform inference.
  4. ONNX runtime inference — `deployment.realtime.RealtimeInference`
     provides the streaming inference loop.
  5. Per-prediction SHA-256 verification — `deployment.onnx_export`
     writes a sidecar hash; `deployment.trust_scores` augments
     predictions with confidence, calibration, distribution similarity,
     and temporal stability.
* **Targets.** End-to-end latency < 50 ms (the standard threshold for
  myoelectric control), sustained on the constrained device.

### MyoAdapt support

* `deployment.onnx_export.OnnxExporter` — ONNX export with numerical
  parity verification and SHA-256 sidecar.
* `deployment.realtime.RealtimeInference` — sliding-window streaming
  inference.
* `deployment.hardware_bench.HardwareBenchmark` — 5-CPU PassMark
  latency projection.
* `deployment.shap_reports.ShapReportGenerator` — per-prediction
  transparency artefacts.
* `deployment.trust_scores.TrustScorer` — multi-factor trust score.

### Expected impact

* Demonstrates that the methods developed in Papers 1–4 are deployable
  on hardware that meets the power and cost constraints of real
  prosthetic devices.
* Validates the SHA-256 tamper-detection mechanism in a live
  deployment setting.
* Provides a reference real-time implementation for the sEMG community.

### Timeline

* **ONNX export, SHA-256 sidecar, streaming inference, hardware
  benchmark:** shipped in v2.0.
* **Raspberry Pi + armband live demo:** next milestone.
* **Battery-powered wearable prototype:** longer-term.

---

## 4. Community Benchmark Platform

### Vision

Cross-study comparison in cross-subject sEMG is presently obstructed by
three issues identified across Papers 1–4: (i) Rest-class metric
inflation (Paper 2), (ii) inconsistent evaluation protocols across
studies, and (iii) the absence of a standardised statistical-rigor
pipeline (Paper 4). A reproducible benchmark harness — released with
the platform — would address all three and would establish a
community reference for cross-database evaluation.

### Technical approach

* **Protocols.** LOSO (within-database) and LODO (cross-database)
  evaluation, with subject-aware preprocessing to prevent test-data
  leakage into training transforms.
* **Metric suite.** Standard accuracy and macro-F1 *plus* the Rest-
  inflation-aware metrics of Paper 2 (`inflation_pp`, `rest_recall`,
  `active_only_accuracy`, `active_only_macro_f1`).
* **Statistical pipeline.** Friedman + Nemenyi CD diagram, Wilcoxon
  signed-rank with Holm–Šídák correction, Hedges' g (small-N corrected),
  BCa bootstrap 95% CI, a-priori statistical power analysis — all
  shipped in `evaluation.statistics` and `evaluation.diagrams`.
* **Reproducibility.** Signed run manifests (`reproducibility.RunManifest`)
  capturing the git HEAD, environment hash, and output hashes —
  satisfying the NeurIPS 2025/2026 Reproducibility Checklist.
* **Datasets.** NinaPro DB1/DB2/DB3/DB7, CapgMyo-DBa, UCI — all
  loadable via `data.loaders`.

### MyoAdapt support

* `evaluation.loso.LOSOEvaluator` — within-database LOSO.
* `evaluation.lodo.LODOEvaluator` — cross-database LODO.
* `evaluation.metrics` — including the Rest-inflation metric suite.
* `evaluation.statistics` — Friedman, Nemenyi, Wilcoxon + Holm–Šídák,
  Hedges' g, BCa CI, power analysis.
* `evaluation.diagrams` — Nemenyi CD diagram, per-fold box-plot,
  per-class heatmap, reliability diagram.
* `evaluation.paper_reports` — LaTeX-ready tables (booktabs) + figure
  bundles (PNG + PDF + .tex).
* `reproducibility.RunManifest` — signed run manifests with SHA-256
  verification.

### Expected impact

* Establishes a standardised cross-database benchmark for the sEMG
  community, addressing the cross-study comparability problem
  identified in Papers 2 and 4.
* Provides the exact artefacts (CD diagrams, BCa CIs, effect sizes,
  power analyses) that reviewers expect in a 2026 multi-model
  benchmark.
* Reproducible by construction: every benchmark result is captured by
  a signed run manifest that can be verified by a third party.

### Timeline

* **All underlying modules:** shipped in v2.0.
* **Benchmark suite packaging (a curated set of pre-configured
  protocols + scripts):** next milestone.
* **Community contribution guidelines for adding new models and
  datasets:** following the packaging milestone.

---

## Author and contact

**Qussai Adlbi**
Al-Andalus University for Medical Sciences
Email: adlbiqussai@gmail.com
ORCID: [0009-0000-7667-1992](https://orcid.org/0009-0000-7667-1992)
GitHub: <https://github.com/Qussai-BME/MyoAdapt>
