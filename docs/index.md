# MyoAdapt v2.0

**Open-source sEMG pattern recognition. CPU-native. Edge-deployable. Audit-ready.**

MyoAdapt is a Python research platform for surface electromyography (sEMG)
pattern recognition. It runs end-to-end on commodity CPUs and ships with the
full stack needed to take a myoelectric decoding pipeline from raw signal to
a deployed, audit-ready ONNX artifact.

## Why MyoAdapt

- **CPU-native** (CUDA auto-detected when available): every stage runs on a Celeron-class CPU; no GPU required for any stage. When CUDA is present, the PyTorch-based models auto-select it; otherwise they fall back to CPU.
- **Cross-subject & cross-database**: built-in LOSO and LODO evaluators with subject-aware preprocessing and Rest-class inflation reporting.
- **Edge-deployable**: ONNX export with numerical-parity verification, plus a 5-CPU PassMark latency projection.
- **Audit-ready**: per-prediction SHAP transparency reports with per-channel aggregation; JSON serialization designed to support EU AI Act Article 13 documentation.

## Install

```bash
pip install -e ".[all]"
```

## Quick links

- [Quick start](quickstart.md)
- [Architecture](architecture.md)
- [API reference](api_reference.md)
- [Module docs](modules/data.md)
- [Deployment](deployment.md)
- [Compliance](compliance/eu_ai_act.md)
