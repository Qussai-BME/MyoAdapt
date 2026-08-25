# Quick start (5 minutes)

## 1. Install

```bash
git clone https://github.com/Qussai-BME/MyoAdapt.git
cd myoadapt
pip install -e ".[all]"
```

Verify:

```bash
myoadapt info
```

You should see a table of installed modules and registered models.

## 2. Get NinaPro data

1. Register at http://ninapro.hevs.ch/
2. Download DB2 (or DB1/DB3/DB7)
3. Extract `.mat` files to `./data/`

Expected file naming: `S01_DB2_E1.mat`, `S02_DB2_E1.mat`, ...

## 3. Train a model

```bash
myoadapt train --db DB2 --model xgboost --data-root ./data --output ./models/xgb_db2.pkl
```

## 4. Evaluate with LOSO

```bash
myoadapt eval --model-path ./models/xgb_db2.pkl --db DB2 --data-root ./data --protocol loso
```

Results saved to `./results/Table2_main_results_DB2.csv`.

## 5. Export to ONNX

```bash
myoadapt export --model-path ./models/xgb_db2.pkl --output ./models/xgb_db2.onnx
```

## 6. Run 5-CPU benchmark

```bash
myoadapt benchmark --model-path ./models/xgb_db2.pkl --n-runs 100
```

## 7. Start the API + UI

```bash
# Terminal 1: API
myoadapt serve --port 8000

# Terminal 2: UI
myoadapt ui --port 8501
```

Open http://localhost:8501 for the Streamlit dashboard.

## 8. Try the Lite-DAN

```bash
myoadapt train --db DB2 --model lite_dan --data-root ./data --output ./models/lite_dan_db2.pkl
```

## 9. Generate a SHAP transparency report

```bash
myoadapt explain --model-path ./models/xgb_db2.pkl --features ./data/sample_features.csv
```

## 10. Verify the MiniROCKET near-singularity

```bash
myoadapt verify --features ./data/per_subject_ppv.npz --n-kernels 10000
```

---

## Next steps

- Read the [Architecture](architecture.md) document
- Follow the [Research chain](research_chain.md) to see how the 4 papers map to modules
- Browse the [API reference](api_reference.md)
- Check the [Deployment guide](deployment.md) for production setup
