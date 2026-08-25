# Deployment Guide

## Local deployment

### Install
```bash
git clone https://github.com/Qussai-BME/MyoAdapt.git
cd myoadapt
pip install -e ".[all]"
```

### Train + serve
```bash
myoadapt train --db DB2 --model xgboost
myoadapt serve --port 8000
```

### Real-time WebSocket streaming
```bash
myoadapt stream --model-path ./models/model.pkl --port 8001
```

### Streamlit UI
```bash
myoadapt ui --port 8501
```

## Docker deployment

### Single container
```bash
docker build -t myoadapt:v2.0 .
docker run -p 8000:8000 -v $(pwd)/data:/app/data myoadapt:v2.0
```

### Full stack via docker-compose
```bash
docker-compose up # API + UI
docker-compose --profile full up # API + UI + MLflow tracker
```

Services:
- `api` (port 8000) — FastAPI REST + auto-docs at `/docs`
- `ui` (port 8501) — Streamlit dashboard
- `tracker` (port 5000, optional) — MLflow tracking server

## Edge deployment

### Export to ONNX
```bash
myoadapt export --model-path ./models/model.pkl --output ./models/model.onnx
```

### Run 5-CPU benchmark
```bash
myoadapt benchmark --model-path ./models/model.pkl --n-runs 100
```

### Deploy on Raspberry Pi
```python
import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession("model.onnx")
input_name = sess.get_inputs()[0].name

# Real-time loop
def on_emg_window(window):
 features = extract_features(window)
 outputs = sess.run(None, {input_name: features.astype(np.float32)})
 predicted_class = outputs[0].argmax()
 return predicted_class
```

## Production checklist

- [ ] Model trained with LOSO (not random split)
- [ ] Rest-class inflation reported
- [ ] ONNX export verified
- [ ] 5-CPU benchmark passes real-time threshold
- [ ] SHAP transparency report generated for sample predictions
- [ ] Trust threshold configured
- [ ] WebSocket streaming tested
- [ ] Monitoring: prediction log + drift detection
- [ ] Backup: model artifact versioned (Zenodo DOI)

## Compliance

For clinical deployment in the EU, see:
- [EU AI Act compliance](compliance/eu_ai_act.md)
- [FDA SaMD guidance](compliance/fda_samd.md)
- [GDPR data privacy](compliance/gdpr.md)
