"""Unit tests for deployment modules."""
import numpy as np
import pytest
from pathlib import Path

from myoadapt.deployment.realtime import RealtimeInference
from myoadapt.deployment.trust_scores import TrustScorer
from myoadapt.deployment.hardware_bench import HardwareBenchmark
from myoadapt.deployment.shap_reports import ShapReportGenerator


class DummyModel:
    """Minimal model for testing realtime + benchmark."""
    def __init__(self, n_classes=3, n_features=100):
        self.n_classes = n_classes
        self.n_features = n_features
        self.classes_ = np.arange(n_classes)
        self.feature_names = [f"f{i}" for i in range(n_features)]

    def predict(self, X):
        X = np.asarray(X)
        if X.ndim == 3:
            X_flat = X.reshape(X.shape[0], -1)[:, :self.n_features]
        else:
            X_flat = X[:, :self.n_features]
        return np.random.randint(0, self.n_classes, size=X_flat.shape[0])

    def predict_proba(self, X):
        X = np.asarray(X)
        if X.ndim == 3:
            X_flat = X.reshape(X.shape[0], -1)[:, :self.n_features]
        else:
            X_flat = X[:, :self.n_features]
        preds = self.predict(X_flat)
        onehot = np.zeros((len(preds), self.n_classes), dtype=np.float32)
        onehot[np.arange(len(preds)), preds] = 1.0
        return onehot

    def count_parameters(self):
        return 1000


def test_realtime_basic():
    model = DummyModel()
    rt = RealtimeInference(model, fs=2000, window_ms=200, increment_ms=50)
    # Push 400 samples (one window)
    samples = np.random.randn(400, 12).astype(np.float32)
    preds = rt.push_samples(samples)
    # push_samples returns a list of predictions (may be empty)
    assert isinstance(preds, list)
    assert len(preds) >= 1
    pred = preds[0]
    assert "prediction" in pred
    assert "confidence" in pred


def test_realtime_incremental():
    model = DummyModel()
    rt = RealtimeInference(model, fs=2000, window_ms=200, increment_ms=50)
    # Push samples 50 at a time; push_samples returns a list per call
    samples = np.random.randn(500, 12).astype(np.float32)
    predictions = []
    for i in range(0, 500, 50):
        preds = rt.push_samples(samples[i:i+50])
        predictions.extend(preds)
    assert len(predictions) >= 1


def test_trust_scorer():
    rng = np.random.default_rng(42)
    X_train = rng.standard_normal((100, 50))
    scorer = TrustScorer()
    scorer.fit(X_train)
    X = rng.standard_normal((1, 50))
    proba = np.array([[0.7, 0.2, 0.1]])
    score = scorer.score(X, proba, pred_idx=0)
    assert 0.0 <= score <= 1.0


def test_trust_reject():
    scorer = TrustScorer(threshold=0.5)
    assert scorer.should_reject(0.3) is True
    assert scorer.should_reject(0.7) is False


def test_hardware_benchmark():
    model = DummyModel()
    bench = HardwareBenchmark(model, n_features=100, n_runs=20, warmup=2)
    results = bench.run()
    assert "per_cpu_projected" in results
    assert len(results["per_cpu_projected"]) == 5
    assert results["verdict"] in [
        "ALL CPUs REAL-TIME CAPABLE",
        "SOME CPUs BELOW REALTIME THRESHOLD",
    ]
    md = bench.markdown_table(results)
    assert "5-CPU Hardware Benchmark" in md


def test_shap_report_generator():
    model = DummyModel()
    gen = ShapReportGenerator(model, feature_names=[f"f{i}" for i in range(100)])
    X = np.random.randn(1, 100).astype(np.float32)
    report = gen.explain(X)
    assert hasattr(report, "predicted_class")
    assert hasattr(report, "confidence")
    assert hasattr(report, "top_features")
    assert hasattr(report, "trust_score")
    assert hasattr(report, "explanation")
    # Should be JSON-serializable
    j = report.to_json()
    assert isinstance(j, str)
