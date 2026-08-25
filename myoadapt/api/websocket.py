"""Bounded, authenticated WebSocket streaming for research sEMG inference.

The endpoint is intended for controlled research deployments. It does not
provide clinical control, safety certification, rate limiting at the network
edge, or transport security; deploy it behind authenticated TLS infrastructure.
"""
from __future__ import annotations

import json
import logging
import os
import secrets
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

MAX_MESSAGE_BYTES = 1_000_000
MAX_SAMPLES_PER_MESSAGE = 4_000
MAX_CHANNELS = 256

try:
    from fastapi import (
        FastAPI,
        Header,
        HTTPException,
        Query,
        WebSocket,
        WebSocketDisconnect,
        status,
    )

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


def _required_api_key(api_key: Optional[str]) -> Optional[str]:
    expected = api_key if api_key is not None else os.environ.get("MYOADAPT_API_KEY")
    required = os.environ.get("MYOADAPT_REQUIRE_API_KEY", "").strip().lower()
    if required in {"1", "true", "yes", "on"} and not expected:
        raise RuntimeError(
            "MYOADAPT_REQUIRE_API_KEY is enabled but no API key was configured. "
            "Set MYOADAPT_API_KEY or pass create_ws_app(api_key=...)."
        )
    return expected


def _coerce_samples(payload: Dict[str, Any]) -> np.ndarray:
    """Validate a bounded, finite, two-dimensional streaming sample matrix."""
    values = payload.get("samples")
    if not isinstance(values, list) or not values:
        raise ValueError("samples must be a non-empty list")
    if len(values) > MAX_SAMPLES_PER_MESSAGE:
        raise ValueError(f"samples exceeds the {MAX_SAMPLES_PER_MESSAGE} row limit")
    if not all(isinstance(row, list) and row for row in values):
        raise ValueError("samples must be a non-empty two-dimensional matrix")
    width = len(values[0])
    if width > MAX_CHANNELS:
        raise ValueError(f"samples exceeds the {MAX_CHANNELS} channel limit")
    if any(len(row) != width for row in values):
        raise ValueError("samples must be a rectangular matrix")
    try:
        samples = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("samples must contain numeric values") from exc
    if not np.isfinite(samples).all():
        raise ValueError("samples must contain finite numeric values only")
    return samples


if HAS_FASTAPI:

    def create_ws_app(
        model_path: Optional[str] = None,
        fs: int = 2000,
        window_ms: int = 200,
        increment_ms: int = 50,
        api_key: Optional[str] = None,
    ) -> FastAPI:
        """Create a research-only WebSocket server for a trusted local model artifact.

        The API key can be supplied directly or through ``MYOADAPT_API_KEY``.
        Use ``MYOADAPT_REQUIRE_API_KEY=true`` to forbid accidental open network
        deployment. Prefer an ``X-API-Key`` header; the optional query parameter
        exists only for clients that cannot set WebSocket headers and should not
        be used through URL-logging infrastructure.
        """
        if not model_path:
            raise ValueError("model_path is required for streaming inference")
        if fs <= 0 or window_ms <= 0 or increment_ms <= 0:
            raise ValueError("fs, window_ms, and increment_ms must be positive")

        from myoadapt.data.preprocessing import preprocess_signal
        from myoadapt.deployment.realtime import RealtimeInference
        from myoadapt.deployment.trust_scores import TrustScorer
        from myoadapt.features import extract_features
        from myoadapt.models.classical import EMGClassifier

        expected_api_key = _required_api_key(api_key)
        app = FastAPI(
            title="MyoAdapt Research WebSocket API",
            description="Research-only sEMG streaming inference; not for clinical or autonomous control.",
            version="2.0.0",
        )

        try:
            model = EMGClassifier.load(
                model_path, expected_sha256=os.environ.get("MYOADAPT_MODEL_SHA256")
            )
        except Exception as exc:
            logger.exception("Streaming model load failed")
            raise RuntimeError("Could not load the configured streaming model") from exc

        def feature_extractor(window: np.ndarray) -> np.ndarray:
            signal = preprocess_signal(window.T, fs=fs)
            return extract_features(signal.T[np.newaxis, :, :], fs=fs)

        def engine_factory() -> Tuple[RealtimeInference, TrustScorer]:
            # Buffers and counters are isolated per connection so one client's
            # samples can never be mixed with another client's stream.
            return (
                RealtimeInference(
                    model,
                    fs=fs,
                    window_ms=window_ms,
                    increment_ms=increment_ms,
                    feature_extractor=feature_extractor,
                ),
                TrustScorer(),
            )

        def valid_key(candidate: Optional[str]) -> bool:
            return expected_api_key is None or (
                candidate is not None and secrets.compare_digest(candidate, expected_api_key)
            )

        def http_key_guard(x_api_key: Optional[str] = Header(None)) -> None:
            if not valid_key(x_api_key):
                raise HTTPException(status_code=401, detail="Invalid or missing API key")

        @app.websocket("/ws/stream")
        async def stream(ws: WebSocket, x_api_key: Optional[str] = Query(default=None)) -> None:
            candidate_key = ws.headers.get("x-api-key") or x_api_key
            if not valid_key(candidate_key):
                await ws.close(code=status.WS_1008_POLICY_VIOLATION, reason="Invalid or missing API key")
                return

            await ws.accept()
            engine, trust_scorer = engine_factory()
            expected_channels: Optional[int] = None
            try:
                while True:
                    message = await ws.receive_text()
                    if len(message.encode("utf-8")) > MAX_MESSAGE_BYTES:
                        await ws.send_json({"error": "Message exceeds the configured size limit"})
                        await ws.close(code=status.WS_1009_MESSAGE_TOO_BIG)
                        return
                    try:
                        payload = json.loads(message)
                        if not isinstance(payload, dict):
                            raise ValueError("message must be a JSON object")
                        samples = _coerce_samples(payload)
                        if expected_channels is None:
                            expected_channels = int(samples.shape[1])
                        elif samples.shape[1] != expected_channels:
                            raise ValueError(
                                f"channel count changed from {expected_channels} to {samples.shape[1]}"
                            )
                        for prediction in engine.push_samples(samples):
                            if "error" in prediction:
                                logger.warning("Streaming prediction failed")
                                await ws.send_json({"error": "Prediction could not be generated"})
                                continue
                            if "confidence" in prediction:
                                prediction["trust_score"] = float(prediction["confidence"])
                            prediction["research_use_only"] = True
                            await ws.send_json(prediction)
                    except json.JSONDecodeError:
                        await ws.send_json({"error": "Invalid JSON"})
                    except ValueError as exc:
                        await ws.send_json({"error": str(exc)})
                    except Exception:
                        logger.exception("Unexpected streaming request failure")
                        await ws.send_json({"error": "Streaming request could not be processed"})
            except WebSocketDisconnect:
                logger.info("Streaming client disconnected")
            except Exception:
                logger.exception("WebSocket connection failure")

        @app.get("/stats", dependencies=[])
        def stats(x_api_key: Optional[str] = Header(None)) -> Dict[str, Any]:
            http_key_guard(x_api_key)
            return {
                "model_loaded": True,
                "fs": fs,
                "window_ms": window_ms,
                "increment_ms": increment_ms,
                "max_message_bytes": MAX_MESSAGE_BYTES,
                "max_samples_per_message": MAX_SAMPLES_PER_MESSAGE,
                "research_use_only": True,
            }

        return app

else:

    def create_ws_app(*args: Any, **kwargs: Any) -> Any:
        raise ImportError("FastAPI not installed. Install: pip install fastapi uvicorn websockets")
