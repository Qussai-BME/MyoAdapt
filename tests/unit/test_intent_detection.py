"""Unit tests for rest-vs-active intent detection (myoadapt.tasks.intent_detection)."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from myoadapt.tasks.intent_detection import EnergyThresholdDetector, LearnedIntentDetector
from myoadapt.models.base import MODEL_REGISTRY, get_model


def _rest_active(n_each=40, n_ch=6, n_samples=200, seed=0):
    rng = np.random.default_rng(seed)
    rest = rng.normal(0, 0.02, (n_each, n_samples, n_ch))
    active = rng.normal(0, 0.5, (n_each, n_samples, n_ch))
    X = np.concatenate([rest, active])
    y = np.concatenate([np.zeros(n_each), np.ones(n_each)])
    return X, y, rest, active


# ---------------------------------------------------------------------------
# EnergyThresholdDetector
# ---------------------------------------------------------------------------
def test_energy_threshold_invalid_metric_raises():
    with pytest.raises(ValueError):
        EnergyThresholdDetector(metric="not_rms_or_mav")


def test_energy_threshold_fit_and_separates_rest_from_active():
    X, y, rest, active = _rest_active()
    det = EnergyThresholdDetector(smoothing=1).fit(rest)
    metrics = det.evaluate(X, y)
    assert metrics["accuracy"] == 1.0
    assert metrics["fpr"] == 0.0


def test_energy_threshold_manual_override():
    det = EnergyThresholdDetector().set_threshold(0.1)
    assert det.threshold == 0.1
    with pytest.raises(ValueError):
        det.set_threshold(-1)


def test_energy_threshold_detect_stream_matches_detect():
    _, _, rest, active = _rest_active()
    det = EnergyThresholdDetector(smoothing=1).fit(rest)
    windows = list(active[:5])
    stream_result = det.reset().detect_stream(windows)
    det.reset()
    manual_result = np.array([det.detect(w) for w in windows])
    assert np.array_equal(stream_result, manual_result)


# ---------------------------------------------------------------------------
# LearnedIntentDetector
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("method", ["random_forest", "logistic"])
def test_learned_detector_fit_predict(method):
    X, y, *_ = _rest_active()
    clf = LearnedIntentDetector(method=method).fit(X, y)
    preds = clf.predict(X)
    assert preds.shape == y.shape
    assert set(np.unique(preds)).issubset({0, 1})
    proba = clf.predict_proba(X)
    assert proba.shape == y.shape
    assert (proba >= 0).all() and (proba <= 1).all()


def test_learned_detector_requires_both_classes():
    X, y, *_ = _rest_active()
    with pytest.raises(ValueError):
        LearnedIntentDetector().fit(X, np.zeros(len(y)))


def test_learned_detector_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        LearnedIntentDetector().predict(np.zeros((2, 200, 6)))


def test_learned_detector_save_load_roundtrip():
    X, y, *_ = _rest_active()
    clf = LearnedIntentDetector().fit(X, y)
    preds_before = clf.predict(X)
    with tempfile.TemporaryDirectory() as td:
        path = str(Path(td) / "detector.pkl")
        clf.save(path)
        clf2 = LearnedIntentDetector.load(path)
        preds_after = clf2.predict(X)
    assert np.array_equal(preds_before, preds_after)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
def test_registered_in_model_registry():
    assert "intent_detector" in MODEL_REGISTRY
    m = get_model("intent_detector")
    assert isinstance(m, LearnedIntentDetector)
