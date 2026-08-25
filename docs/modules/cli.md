# `myoadapt.cli` — Command-line interface

## Commands

```bash
# Show platform info
myoadapt info

# Train a model
myoadapt train --db DB2 --model xgboost --output ./models/model.pkl

# Evaluate with LOSO
myoadapt eval --model-path ./models/model.pkl --db DB2 --protocol loso

# Export to ONNX
myoadapt export --model-path ./models/model.pkl --output ./model.onnx

# Start REST API
myoadapt serve --port 8000

# Start WebSocket streaming server
myoadapt stream --model-path ./models/model.pkl --port 8001

# Run 5-CPU hardware benchmark
myoadapt benchmark --model-path ./models/model.pkl --n-runs 100

# Generate SHAP transparency report
myoadapt explain --model-path ./models/model.pkl --features ./data/sample.csv

# Launch Streamlit UI
myoadapt ui --port 8501

# Verify MiniROCKET near-singularity
myoadapt verify --features ./data/per_subject_ppv.npz --n-kernels 10000
```

## Common options

- `--verbose` / `-v`: debug logging
- `--data-root`: path to dataset directory
- `--output`: output path
- `--config`: YAML config file (overrides defaults)

## Example session

```bash
# Full workflow
myoadapt info
myoadapt train --db DB2 --model xgboost
myoadapt eval --model-path ./models/model.pkl --db DB2
myoadapt export --model-path ./models/model.pkl --output ./models/model.onnx
myoadapt benchmark --model-path ./models/model.pkl
myoadapt explain --model-path ./models/model.pkl --features ./data/sample.csv
myoadapt serve &
myoadapt ui
```
