# `myoadapt.api` — REST + WebSocket

## REST endpoints

```bash
# Health
curl http://localhost:8000/health

# List models
curl http://localhost:8000/models

# List datasets
curl http://localhost:8000/datasets

# Predict
curl -X POST http://localhost:8000/predict \
 -H "Content-Type: application/json" \
 -d '{"model_path": "./models/model.pkl", "features": [[1.0, 2.0, ...]]}'

# SHAP explain
curl -X POST http://localhost:8000/explain \
 -H "Content-Type: application/json" \
 -d '{"model_path": "./models/model.pkl", "features": [[1.0, 2.0, ...]]}'

# Export to ONNX
curl -X POST http://localhost:8000/export \
 -H "Content-Type: application/json" \
 -d '{"model_path": "./models/model.pkl", "output_path": "./model.onnx"}'

# Hardware benchmark
curl "http://localhost:8000/benchmark?model_path=./models/model.pkl&n_runs=100"
```

## WebSocket streaming

```javascript
const ws = new WebSocket('ws://localhost:8001/ws/stream');

ws.onmessage = (event) => {
 const pred = JSON.parse(event.data);
 console.log(`Gesture: ${pred.prediction} (conf=${pred.confidence.toFixed(2)})`);
};

// Send EMG samples
ws.send(JSON.stringify({
 samples: [[ch1, ch2, ch3, ...], [ch1, ch2, ch3, ...]]
}));
```

## Start servers

```bash
# REST API
myoadapt serve --port 8000

# WebSocket streaming
myoadapt stream --model-path ./models/model.pkl --port 8001

# Or via Docker
docker-compose up
```
