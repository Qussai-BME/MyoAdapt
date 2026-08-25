# `myoadapt.deployment` — Edge deployment

## ONNX export

```python
from myoadapt.deployment import OnnxExporter

info = OnnxExporter.export(clf, './model.onnx',
 feature_names=feature_names)
print(info) # path, size, type, opset

# Verify
verification = OnnxExporter.verify('./model.onnx')
```

## Real-time inference

```python
from myoadapt.deployment import RealtimeInference

rt = RealtimeInference(
 model=clf,
 fs=2000,
 window_ms=200,
 increment_ms=50,
 on_prediction=lambda pred: print(pred),
)

# In your data acquisition loop:
for sample in emg_stream:
 pred = rt.push_samples(sample.reshape(1, -1))
```

## SHAP transparency reports (EU AI Act)

```python
from myoadapt.deployment import ShapReportGenerator

gen = ShapReportGenerator(clf, feature_names=feature_names)
report = gen.explain(X_test[:1])
print(report.to_json())
```

## Trust scoring

```python
from myoadapt.deployment import TrustScorer

scorer = TrustScorer(threshold=0.5)
scorer.fit(X_train)

proba = clf.predict_proba(X_test[:1])
trust = scorer.score(X_test[:1], proba, pred_idx=0)
if scorer.should_reject(trust):
 print("Reject prediction (low trust)")
```

## 5-CPU hardware benchmark

```python
from myoadapt.deployment import HardwareBenchmark

bench = HardwareBenchmark(clf, n_features=308, n_runs=100)
results = bench.run()

# Per-CPU projected latency
for cpu, m in results['per_cpu_projected'].items():
 print(f"{cpu}: {m['projected_p99_ms']:.2f}ms (realtime: {m['realtime_capable']})")

# Markdown table for paper
print(bench.markdown_table(results))
```
