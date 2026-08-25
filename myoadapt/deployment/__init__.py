"""
myoadapt.deployment — Edge deployment
=======================================

Public API:
    OnnxExporter       — export sklearn/torch models to ONNX
    RealtimeInference  — streaming inference with sliding window
    ShapReportGenerator — per-prediction transparency reports
    TrustScorer        — calibration-aware trust scoring
    HardwareBenchmark  — 5-CPU benchmark
"""
from myoadapt.deployment.hardware_bench import HardwareBenchmark
from myoadapt.deployment.onnx_export import OnnxExporter
from myoadapt.deployment.realtime import RealtimeInference
from myoadapt.deployment.shap_reports import ShapReportGenerator, TransparencyReport
from myoadapt.deployment.trust_scores import TrustScorer

__all__ = [
    "OnnxExporter", "RealtimeInference", "ShapReportGenerator",
    "TransparencyReport", "TrustScorer", "HardwareBenchmark",
]
