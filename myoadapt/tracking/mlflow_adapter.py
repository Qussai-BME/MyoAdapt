"""
mlflow_adapter.py — MLflow adapter (optional)
==============================================

Falls back gracefully to LocalTracker if MLflow is not installed.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from myoadapt.tracking.local_tracker import LocalTracker

logger = logging.getLogger(__name__)

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False
    logger.info("MLflow not installed — using LocalTracker fallback")


class MLflowTracker:
    """
    Wrapper that uses MLflow if available, else LocalTracker.
    """

    def __init__(self, tracking_uri: Optional[str] = None,
                 local_root: str = "./experiments"):
        self.local = LocalTracker(local_root)
        if HAS_MLFLOW:
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            self.mlflow = mlflow
        else:
            self.mlflow = None

    def set_experiment(self, name: str):
        self.local.set_experiment(name)
        if self.mlflow:
            self.mlflow.set_experiment(name)

    def start_run(self, run_name: Optional[str] = None):
        self.local.start_run(run_name)
        if self.mlflow:
            self.mlflow.start_run(run_name=run_name)

    def log_params(self, params: Dict[str, Any]):
        self.local.log_params(params)
        if self.mlflow:
            self.mlflow.log_params(params)

    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        self.local.log_metric(key, value, step)
        if self.mlflow:
            self.mlflow.log_metric(key, value, step=step)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        self.local.log_metrics(metrics, step)
        if self.mlflow:
            self.mlflow.log_metrics(metrics, step)

    def log_artifact(self, path: str):
        self.local.log_artifact(path)
        if self.mlflow:
            self.mlflow.log_artifact(path)

    def end_run(self, status: str = "completed"):
        if self.mlflow:
            self.mlflow.end_run()
        return self.local.end_run(status)


def get_tracker(use_mlflow: bool = True,
                tracking_uri: Optional[str] = None) -> MLflowTracker:
    """Factory: get MLflow-backed tracker if available, else local."""
    if use_mlflow and HAS_MLFLOW:
        return MLflowTracker(tracking_uri)
    tracker = MLflowTracker(None)
    tracker.mlflow = None  # force local
    return tracker
