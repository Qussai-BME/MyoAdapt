"""
realtime.py — Real-time streaming inference
============================================

Maintains a circular buffer of EMG samples and emits predictions at
a configurable cadence. Designed for:
- WebSocket streaming API
- Raspberry Pi / Edge deployment
- Latency benchmarking

"""
from __future__ import annotations

import logging
import time
from collections import deque
from typing import Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class RealtimeInference:
    """
    Real-time inference engine.

    Parameters
    ----------
    model : trained classifier with .predict()
    fs : sampling rate (Hz)
    window_ms : window length (ms)
    increment_ms : step (ms)
    on_prediction : callback(predictions: List[Dict])
    confidence_threshold : drop predictions below this
    """

    def __init__(self,
                 model,
                 fs: int = 2000,
                 window_ms: int = 200,
                 increment_ms: int = 50,
                 on_prediction: Optional[Callable] = None,
                 confidence_threshold: float = 0.3,
                 feature_extractor: Optional[Callable] = None):
        """
        Parameters
        ----------
        model : trained classifier with ``predict`` / ``predict_proba``.
            If the model is a feature-based classifier (e.g. EMGClassifier),
            you MUST also pass ``feature_extractor`` so raw windows are
            converted to feature vectors before inference.
        fs : sampling rate (Hz).
        window_ms, increment_ms : window geometry.
        on_prediction : optional callback invoked for every emitted prediction.
        confidence_threshold : predictions below this confidence are still
            returned but the ``on_prediction`` callback is not fired.
        feature_extractor : callable(window: np.ndarray) -> np.ndarray
            where ``window`` has shape ``(n_channels, n_samples)`` and the
            return has shape ``(1, n_features)``. If ``None``, the raw
            window is reshaped to ``(1, n_channels * n_samples)`` and passed
            directly — only valid for models that accept raw windows (e.g.
            CNN1D, EMGFoundation).
        """
        self.model = model
        self.fs = fs
        self.window_ms = window_ms
        self.increment_ms = increment_ms
        self.window_samples = int(window_ms * fs / 1000)
        self.increment_samples = int(increment_ms * fs / 1000)
        self.on_prediction = on_prediction
        self.confidence_threshold = confidence_threshold
        self.feature_extractor = feature_extractor

        self.buffer: deque = deque(maxlen=self.window_samples)
        self.samples_since_predict = 0
        self.predictions_emitted = 0
        self.total_latency_ms = 0.0

    def push_samples(self, samples: np.ndarray) -> List[Dict]:
        """
        Push new EMG samples into the buffer.

        Returns the list of predictions emitted during this push (may be
        empty, may contain several if multiple windows were completed).
        """
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim == 1:
            samples = samples.reshape(-1, 1)
        emitted: List[Dict] = []
        for row in samples:
            self.buffer.append(row)
            self.samples_since_predict += 1
            if self.samples_since_predict >= self.increment_samples and \
               len(self.buffer) >= self.window_samples:
                pred = self._predict_window()
                self.samples_since_predict = 0
                if pred is not None:
                    emitted.append(pred)
        return emitted

    def _predict_window(self) -> Optional[Dict]:
        window = np.array(list(self.buffer), dtype=np.float32)  # (window_samples, n_channels)
        t0 = time.perf_counter()
        # Build model input. If a feature_extractor is configured, use it;
        # otherwise reshape the raw window for CNN-style models.
        if self.feature_extractor is not None:
            try:
                x = self.feature_extractor(window.T)  # extractor expects (n_channels, n_samples)
            except Exception as e:
                logger.error(f"Feature extraction failed: {e}")
                return {"error": f"feature_extraction: {e}"}
        else:
            # Reshape for CNN-style models: (1, n_channels, n_samples)
            x = window.T[np.newaxis, :, :]
        try:
            proba = self.model.predict_proba(x)
            pred_idx = int(np.argmax(proba, axis=1)[0])
            confidence = float(proba[0, pred_idx])
            classes = getattr(self.model, "classes_",
                              np.arange(proba.shape[1]))
            pred_label = (classes[pred_idx] if hasattr(classes, "__getitem__")
                          else str(pred_idx))
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {"error": str(e)}

        latency_ms = (time.perf_counter() - t0) * 1000
        self.total_latency_ms += latency_ms
        self.predictions_emitted += 1

        result = {
            "prediction": str(pred_label),
            "confidence": confidence,
            "latency_ms": latency_ms,
            "avg_latency_ms": self.total_latency_ms / self.predictions_emitted,
            "n_emitted": self.predictions_emitted,
            "timestamp": time.time(),
        }

        if self.on_prediction and confidence >= self.confidence_threshold:
            self.on_prediction(result)

        return result

    def reset(self):
        self.buffer.clear()
        self.samples_since_predict = 0
        self.predictions_emitted = 0
        self.total_latency_ms = 0.0

    def stats(self) -> Dict[str, float]:
        return {
            "predictions_emitted": self.predictions_emitted,
            "avg_latency_ms": (self.total_latency_ms / max(1, self.predictions_emitted)),
            "fs": self.fs,
            "window_ms": self.window_ms,
            "increment_ms": self.increment_ms,
        }
