# `myoadapt.ui` — Streamlit dashboard

5 modular pages accessible at http://localhost:8501 after running:

```bash
myoadapt ui
```

## Pages

### 1. Train
- Select dataset (DB1/2/3/7, CapgMyo, UCI)
- Select model (classical, CNN1D, Lite-DAN, EMG-FM)
- Configure preprocessing (filter, window, EA)
- Configure feature modules
- One-click CLI command generation

### 2. Evaluate
- Run LOSO or LODO
- View per-fold accuracy distribution
- View existing result tables

### 3. Compare
- Compare multiple models statistically
- Friedman test + Wilcoxon pairwise
- Bar charts of mean accuracy

### 4. Deploy
- Export to ONNX
- Run 5-CPU hardware benchmark
- EU AI Act compliance checklist

### 5. Explain
- Generate SHAP transparency reports
- View per-feature contributions
- View per-channel and per-feature-group breakdowns
