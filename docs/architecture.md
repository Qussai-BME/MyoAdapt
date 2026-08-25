# Architecture

## Overview

MyoAdapt v2.0 is organized into 11 modules with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│ CLI / API / UI │
│ (myoadapt train | REST | WebSocket | Streamlit) │
└──────────────────────┬──────────────────────────────────────┘
 │
 ┌──────────────┴──────────────┐
 │ Evaluation │
 │ (LOSO, LODO, statistics) │
 └──────────────┬──────────────┘
 │
 ┌──────────────┴──────────────┐
 │ Adaptation │
 │ (CORAL, TCA, SA, EA, Adv) │
 └──────────────┬──────────────┘
 │
 ┌──────────────┴──────────────┐
 │ Models │
 │ (XGB, RF, LDA, CNN1D, │
 │ Lite-DAN, EMG-FM) │
 └──────────────┬──────────────┘
 │
 ┌──────────────┴──────────────┐
 │ Features │
 │ (TD, FD, TF, Hist, Corr, │
 │ MiniROCKET) │
 └──────────────┬──────────────┘
 │
 ┌──────────────┴──────────────┐
 │ Data │
 │ (loaders, preprocess, │
 │ splits) │
 └─────────────────────────────┘
```

## Design principles

1. **Single source of truth** — `myoadapt.config.MyoAdaptConfig` holds all parameters
2. **Lazy imports** — `import myoadapt` is fast; heavy modules load on demand
3. **Registry pattern** — models, features, and adaptation methods all use registries
4. **Fold-level preprocessing** — LOSO never leaks test data into training transforms
5. **Reproducibility** — every experiment is fully specified by config + random_state

## Module responsibilities

### `data`
- Loaders for NinaPro DB1/2/3/7, CapgMyo, UCI
- Filtering (Butterworth/Chebyshev/Bessel/Elliptic + IIR notch)
- Windowing with optional majority-vote labels
- Euclidean Alignment (Hahne et al. 2014)
- Splits: LOSO, LODO, k-fold, train/test by subject

### `features`
- Time-domain: MAV, RMS, ZCR, WL, SSC, WAMP, MYOP, AR (Yule-Walker; robust to scipy version), Hjorth, derived (22 per channel)
- Frequency-domain: MNF, MDF, PKF, PSR, SNR, SM1-3 (8 per channel)
- Time-frequency: wavelet packet energy + STFT bands (8 per channel)
- Histogram: 10 normalized bins per channel
- Inter-channel correlation: C(N,2) features
- MiniROCKET: 10,000 PPV features + verifier
- Registry: register custom extractors

### `models`
- Classical: XGBoost, LightGBM, RandomForest, ExtraTrees, LDA, SVM, LogisticRegression
- CNN1D: naive 1D-CNN baseline
- LiteDAN: ~53K-param Domain Adversarial Network with gradual λ
- EMGFoundation: Transformer encoder + masked reconstruction

### `adaptation`
- CORAL: Correlation Alignment (with condition-number warning)
- TCA: Transfer Component Analysis (with eigenvalue scale-gap diagnostic)
- SA: Subspace Alignment
- EA: Euclidean Alignment (per Hahne 2014)
- Adversarial: gradient-reversal-based (delegates to LiteDAN)

### `evaluation`
- LOSOEvaluator: per-fold preprocessing, parallel folds
- LODOEvaluator: cross-database generalization
- Metrics: accuracy, macro-F1, weighted-F1, Rest-class inflation, Cohen's kappa
- Statistics: Friedman, Wilcoxon + Holm-Šídák, Nemenyi, Cohen's d, bootstrap CI
- Reports: auto-generate paper-ready tables (CSV + Markdown)

### `deployment`
- OnnxExporter: sklearn + PyTorch models → ONNX
- RealtimeInference: streaming inference with sliding window
- ShapReportGenerator: per-prediction transparency reports
- TrustScorer: confidence + calibration + distribution + temporal consistency
- HardwareBenchmark: 5-CPU latency projection

### `tracking`
- LocalTracker: JSON-based, no dependencies
- MLflowTracker: MLflow if available, else falls back to Local

### `api`
- REST: train, predict, explain, export, benchmark
- WebSocket: real-time EMG streaming

### `ui`
- 5 Streamlit pages: Train, Evaluate, Compare, Deploy, Explain

### `cli`
- Typer-based: `myoadapt train/eval/export/serve/stream/benchmark/explain/ui/info/verify`

### `tasks`
- ContinuousDecoder: LSTM + Transformer for joint-angle/force regression

## Data flow

```
Raw signal (N, C)
 ↓ filter_signal + notch_filter
Filtered (N, C)
 ↓ compute_alignment_matrix + apply_euclidean_alignment
Aligned (N, C)
 ↓ segment_windows
Windows (W, C, S)
 ↓ extract_features
Features (W, F)
 ↓ StandardScaler + SelectKBest (per fold)
Selected (W, K)
 ↓ model.fit + predict
Predictions (W,)
 ↓ compute_metrics + statistics
Results dict
 ↓ ReportGenerator
CSV + Markdown tables
```
