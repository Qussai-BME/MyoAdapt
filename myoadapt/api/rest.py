"""FastAPI REST API for MyoAdapt research deployments.

Security model
--------------
* Models are loaded once from a server-side path; no API endpoint accepts a
  model or output path from a client.
* API-key protection is opt-in for local research use and can be made
  mandatory with ``MYOADAPT_REQUIRE_API_KEY=true`` for network deployments.
* Prediction payloads are bounded, finite, and dimension-checked before a
  model is called.
* HTTP training orchestration is intentionally disabled. Training involves
  local data and compute and must be run through the authenticated operator
  workflow/CLI, not a public request handler.

This module is research software. It is not a clinical service or a medical
control system.
"""
from __future__ import annotations

import logging
import math
import os
import secrets
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

MAX_FEATURES_PER_REQUEST = 100_000
MAX_BATCH_SAMPLES = 256
MAX_BATCH_VALUES = 100_000

try:
    from fastapi import Depends, FastAPI, Header, HTTPException, Request
    from fastapi.exceptions import RequestValidationError
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse, JSONResponse
    from pydantic import BaseModel, Field

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    logger.warning("FastAPI not installed. Install: pip install fastapi uvicorn")


def _required_api_key(api_key: Optional[str]) -> Optional[str]:
    """Resolve explicit then environment API keys and validate strict mode."""
    expected = api_key if api_key is not None else os.environ.get("MYOADAPT_API_KEY")
    required = os.environ.get("MYOADAPT_REQUIRE_API_KEY", "").strip().lower()
    if required in {"1", "true", "yes", "on"} and not expected:
        raise RuntimeError(
            "MYOADAPT_REQUIRE_API_KEY is enabled but no API key was configured. "
            "Set MYOADAPT_API_KEY or pass create_app(api_key=...)."
        )
    return expected


def _finite_vector(values: List[float], label: str = "features") -> List[float]:
    """Reject empty, oversized, NaN, and infinite feature vectors."""
    if not values:
        raise ValueError(f"{label} must not be empty")
    if len(values) > MAX_FEATURES_PER_REQUEST:
        raise ValueError(f"{label} exceeds the {MAX_FEATURES_PER_REQUEST} value limit")
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError(f"{label} must contain finite numeric values only")
    return values


if HAS_FASTAPI:

    class PredictRequest(BaseModel):
        """One feature vector for a loaded model."""

        features: List[float]

    class PredictBatchRequest(BaseModel):
        """A matrix of feature vectors for a loaded model."""

        features: List[List[float]]

    class ExplainRequest(BaseModel):
        """One feature vector for a bounded local explanation."""

        features: List[float]

    def _finite_matrix(rows: List[List[float]]) -> List[List[float]]:
        """Reject empty, oversized, ragged, and non-finite feature matrices."""
        if not rows:
            raise ValueError("features must not be empty")
        if len(rows) > MAX_BATCH_SAMPLES:
            raise ValueError(f"features exceeds the {MAX_BATCH_SAMPLES} row limit")
        width: Optional[int] = None
        total_values = 0
        for row in rows:
            _finite_vector(row)
            width = len(row) if width is None else width
            if len(row) != width:
                raise ValueError("features must be a rectangular matrix")
            total_values += len(row)
        if total_values > MAX_BATCH_VALUES:
            raise ValueError(f"features exceeds the {MAX_BATCH_VALUES} total-value limit")
        return rows

    class TrainRequest(BaseModel):
        """Legacy request shape retained only to return a safe disabled response."""

        dataset: str = Field("DB2", min_length=1, max_length=32)
        model_type: str = Field("xgboost", min_length=1, max_length=64)

    def _expected_feature_count(model: Any) -> Optional[int]:
        """Best-effort input-width extraction without trusting request metadata."""

        scaler = getattr(model, "scaler", None)
        if scaler is not None and hasattr(scaler, "n_features_in_"):
            return int(scaler.n_features_in_)
        if hasattr(model, "n_features") and model.n_features:
            return int(model.n_features)
        raw_model = getattr(model, "model", None)
        if raw_model is not None and hasattr(raw_model, "n_features_in_"):
            return int(raw_model.n_features_in_)
        return None

    def _ensure_model_input_shape(model: Any, X: np.ndarray) -> None:
        expected = _expected_feature_count(model)
        if expected is not None and X.shape[1] != expected:
            raise HTTPException(
                status_code=422,
                detail=f"Feature dimension mismatch: expected {expected}, received {X.shape[1]}",
            )

    def create_app(
        model_path: Optional[str] = None,
        model_type: str = "classical",
        api_key: Optional[str] = None,
    ) -> FastAPI:
        """Create a bounded REST interface for a preloaded research model.

        ``api_key`` overrides ``MYOADAPT_API_KEY`` for programmatic deployments.
        Set ``MYOADAPT_REQUIRE_API_KEY=true`` to reject accidental unauthenticated
        network deployments. TLS, rate limiting, and transport-level request-size
        controls must be provided by the deployment gateway.
        """
        expected_api_key = _required_api_key(api_key)
        app = FastAPI(
            title="MyoAdapt Research API",
            description=(
                "Research-only sEMG inference API. Not for clinical diagnosis, treatment, "
                "or autonomous prosthetic control."
            ),
            version="2.0.0",
            docs_url="/docs",
            redoc_url="/redoc",
        )

        def validate_api_key(x_api_key: Optional[str] = Header(None)) -> None:
            if expected_api_key is None:
                return
            if x_api_key is None or not secrets.compare_digest(x_api_key, expected_api_key):
                raise HTTPException(status_code=401, detail="Invalid or missing API key")

        @app.exception_handler(RequestValidationError)
        async def validation_exception_handler(
            _: Request, exc: RequestValidationError
        ) -> JSONResponse:
            # Validation details can otherwise echo invalid non-finite values
            # (such as NaN) that standard JSON responses cannot serialize.
            logger.warning("Rejected invalid request payload: %s", exc.errors())
            return JSONResponse(status_code=422, content={"detail": "Invalid request payload"})

        cors_origins = os.environ.get("MYOADAPT_CORS_ORIGINS", "")
        if cors_origins:
            origins = [origin.strip() for origin in cors_origins.split(",") if origin.strip()]
            if "*" in origins:
                raise RuntimeError("MYOADAPT_CORS_ORIGINS must name explicit origins; wildcard is refused")
            app.add_middleware(
                CORSMiddleware,
                allow_origins=origins,
                allow_credentials=False,
                allow_methods=["GET", "POST"],
                allow_headers=["Content-Type", "X-API-Key"],
            )

        state: Dict[str, Any] = {"model": None, "model_path": model_path}
        if model_path:
            try:
                if model_type == "classical":
                    from myoadapt.models.classical import EMGClassifier

                    state["model"] = EMGClassifier.load(
                        model_path, expected_sha256=os.environ.get("MYOADAPT_MODEL_SHA256")
                    )
                elif model_type == "cnn1d":
                    from myoadapt.models.cnn1d import CNN1D

                    state["model"] = CNN1D.load(model_path)
                elif model_type == "lite_dan":
                    from myoadapt.models.lite_dan import LiteDAN

                    state["model"] = LiteDAN.load(model_path)
                elif model_type == "emg_foundation":
                    from myoadapt.models.emg_foundation import EMGFoundation

                    state["model"] = EMGFoundation.load(model_path)
                else:
                    raise ValueError(f"Unknown model_type: {model_type}")
                logger.info("Model loaded for REST inference")
            except Exception:
                logger.exception("Model load failed")

        @app.get("/", response_class=HTMLResponse)
        def root() -> str:
            return """
            <html><head><title>MyoAdapt Research API</title></head><body>
            <h1>MyoAdapt Research API</h1>
            <p>Research-only sEMG inference software. Do not use for clinical decisions,
            diagnosis, treatment, or autonomous prosthetic control.</p>
            <ul><li><a href="/docs">Swagger UI</a></li><li><a href="/health">Health Check</a></li></ul>
            </body></html>
            """

        @app.get("/health")
        def health() -> Dict[str, Any]:
            return {
                "status": "healthy",
                "version": "2.0.0",
                "model_loaded": state["model"] is not None,
                "research_use_only": True,
            }

        @app.get("/models", dependencies=[Depends(validate_api_key)])
        def list_models() -> Dict[str, Any]:
            from myoadapt.models.base import MODEL_REGISTRY

            return {"models": list(MODEL_REGISTRY.keys())}

        @app.get("/datasets", dependencies=[Depends(validate_api_key)])
        def list_datasets() -> Dict[str, Any]:
            from myoadapt.data.loaders import DATASET_REGISTRY, list_databases

            return {"datasets": list_databases(), "metadata": DATASET_REGISTRY}

        @app.post("/train", dependencies=[Depends(validate_api_key)])
        def train(_: TrainRequest) -> None:
            raise HTTPException(
                status_code=501,
                detail=(
                    "HTTP training is disabled. Run the operator-controlled CLI with a "
                    "reviewed local dataset and capture a versioned evaluation manifest."
                ),
            )

        @app.post("/predict", dependencies=[Depends(validate_api_key)])
        def predict(req: PredictRequest) -> Dict[str, Any]:
            try:
                _finite_vector(req.features)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc))
            model = state["model"]
            if model is None:
                raise HTTPException(status_code=503, detail="No model loaded")
            X = np.asarray(req.features, dtype=np.float32).reshape(1, -1)
            _ensure_model_input_shape(model, X)
            try:
                predictions = model.predict(X)
                probabilities = model.predict_proba(X) if hasattr(model, "predict_proba") else None
            except Exception:
                logger.exception("Prediction failed")
                raise HTTPException(status_code=422, detail="Model could not process this feature vector")
            return {
                "prediction": predictions.tolist(),
                "probabilities": probabilities.tolist() if probabilities is not None else None,
                "n_features": int(X.shape[1]),
            }

        @app.post("/predict_batch", dependencies=[Depends(validate_api_key)])
        def predict_batch(req: PredictBatchRequest) -> Dict[str, Any]:
            try:
                _finite_matrix(req.features)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc))
            model = state["model"]
            if model is None:
                raise HTTPException(status_code=503, detail="No model loaded")
            X = np.asarray(req.features, dtype=np.float32)
            _ensure_model_input_shape(model, X)
            try:
                predictions = model.predict(X)
                probabilities = model.predict_proba(X) if hasattr(model, "predict_proba") else None
            except Exception:
                logger.exception("Batch prediction failed")
                raise HTTPException(status_code=422, detail="Model could not process this feature matrix")
            return {
                "predictions": predictions.tolist(),
                "probabilities": probabilities.tolist() if probabilities is not None else None,
                "n_samples": int(X.shape[0]),
            }

        @app.post("/explain", dependencies=[Depends(validate_api_key)])
        def explain(req: ExplainRequest) -> Dict[str, Any]:
            try:
                _finite_vector(req.features)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc))
            model = state["model"]
            if model is None:
                raise HTTPException(status_code=503, detail="No model loaded")
            X = np.asarray(req.features, dtype=np.float32).reshape(1, -1)
            _ensure_model_input_shape(model, X)
            try:
                from myoadapt.deployment.shap_reports import ShapReportGenerator

                report = ShapReportGenerator(model).explain(X)
                return report.to_dict() if hasattr(report, "to_dict") else report.__dict__
            except Exception:
                logger.exception("Explanation failed")
                raise HTTPException(status_code=422, detail="Explanation could not be generated")

        @app.get("/experiments", dependencies=[Depends(validate_api_key)])
        def list_experiments() -> Dict[str, Any]:
            from myoadapt.tracking.local_tracker import LocalTracker

            try:
                tracker = LocalTracker()
                return {"experiments": [directory.name for directory in tracker.root_dir.iterdir() if directory.is_dir()]}
            except Exception:
                logger.exception("Experiment listing failed")
                raise HTTPException(status_code=500, detail="Experiment inventory is unavailable")

        return app

else:

    def create_app(*args: Any, **kwargs: Any) -> Any:
        raise ImportError("FastAPI not installed. Install: pip install fastapi uvicorn")
