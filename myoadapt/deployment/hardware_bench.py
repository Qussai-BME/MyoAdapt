"""
hardware_bench.py — 5-CPU latency projection
=============================================

Benchmarks inference latency on the local CPU, then projects to five
representative CPU classes via published PassMark single-thread scores.
This is a *projection*, not a measurement on five real CPUs — the
projection assumes inference latency scales inversely with single-thread
PassMark rating, which is a reasonable first-order approximation for
single-sample CPU-bound inference but breaks down for batched or
multi-threaded workloads.

CPU classes covered (PassMark single-thread scores, 2024 values):
- Intel Celeron G5900 (entry-level desktop)
- Intel Core i3-10100 (entry)
- Intel Core i5-10400 (mid-range)
- Intel Core i7-10700 (high-end desktop)
- AMD Ryzen 5 3600 (alternative mid-range)

The local CPU's PassMark is NOT auto-detected (Python's ``platform``
module exposes only the architecture string, not the CPU model). The
caller must supply ``local_passmark`` if they want accurate projection;
otherwise a default mid-range value is used and the projections should
be treated as illustrative only.
"""
from __future__ import annotations

import json
import logging
import platform
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


# PassMark single-thread scores (public reference values, approx 2024).
# Source: https://www.cpubenchmark.net/singleThread.html (PassMark Software Pty Ltd).
CPU_PASSMARK = {
    "Intel Celeron G5900": 1496,
    "Intel Core i3-10100":  2463,
    "Intel Core i5-10400":  3580,
    "Intel Core i7-10700":  4926,
    "AMD Ryzen 5 3600":     3580,
}

# Default PassMark for the local CPU when the caller does not supply one.
# This is intentionally a mid-range value so the projection is roughly
# correct on a developer laptop. For publication, ALWAYS pass the actual
# local CPU's PassMark via ``HardwareBenchmark(..., local_passmark=...)``.
DEFAULT_LOCAL_PASSMARK = 3500


def _build_input_tensor(model, n_features: int) -> np.ndarray:
    """Build a single-sample dummy input that matches the model's rank."""
    # 3D inputs: CNN1D, EMGFoundation, ContinuousDecoder
    n_channels = getattr(model, "n_channels", None)
    n_samples = getattr(model, "n_samples", None)
    if n_channels is not None and n_samples is not None:
        return np.random.randn(1, int(n_channels), int(n_samples)).astype(np.float32)
    # 2D inputs: EMGClassifier, LiteDAN, sklearn models
    return np.random.randn(1, int(n_features)).astype(np.float32)


def _detect_n_features(model) -> int:
    """Best-effort auto-detection of a model's input feature count."""
    scaler = getattr(model, "scaler", None)
    if scaler is not None and hasattr(scaler, "n_features_in_"):
        return int(scaler.n_features_in_)
    if hasattr(model, "n_features_in_"):
        return int(model.n_features_in_)
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "n_features_in_"):
        return int(inner.n_features_in_)
    for attr in ("n_features", "n_channels"):
        v = getattr(model, attr, None)
        if isinstance(v, int) and v > 0:
            return v
    return 308


class HardwareBenchmark:
    """
    Benchmark inference latency across simulated CPU classes.

    Parameters
    ----------
    model : trained classifier (must have .predict() and .predict_proba())
    n_features : int — feature dimensionality (auto-detected if None)
    n_runs : int — number of inference runs (default 100)
    warmup : int — warmup runs before timing (default 10)

    Auto-detection
    --------------
    If ``n_features`` is None, the benchmark tries to infer it from the model:
      - EMGClassifier.scaler.n_features_in_
      - sklearn models' n_features_in_ attribute
      - LiteDAN/CNN1D .n_features or .n_samples attribute
      - Falls back to 308
    """

    def __init__(self, model, n_features: Optional[int] = None,
                 n_runs: int = 100, warmup: int = 10,
                 local_passmark: Optional[int] = None):
        self.model = model
        self.n_features = n_features if n_features is not None else _detect_n_features(model)
        self.n_runs = n_runs
        self.warmup = warmup
        self.local_passmark = (int(local_passmark) if local_passmark is not None
                               else DEFAULT_LOCAL_PASSMARK)

    def project_to_cpus(self, latency_ms: float) -> Dict[str, float]:
        """Quick projection: given a single measured latency, project to all CPUs.

        Convenience method used when you only have one timing measurement and
        don't want to run the full benchmark.
        """
        return {
            cpu_name: latency_ms * (self.local_passmark / passmark)
            for cpu_name, passmark in CPU_PASSMARK.items()
        }

    def run(self) -> Dict[str, Any]:
        """Run benchmark and return per-CPU projection results."""
        X = _build_input_tensor(self.model, self.n_features)

        # Warmup
        for _ in range(self.warmup):
            try:
                self.model.predict(X)
            except Exception:
                pass

        # Measure
        latencies_ms: List[float] = []
        for _ in range(self.n_runs):
            t0 = time.perf_counter()
            try:
                self.model.predict(X)
            except Exception as e:
                logger.error(f"Inference failed: {e}")
                return {"error": str(e)}
            latencies_ms.append((time.perf_counter() - t0) * 1000)

        local_mean = float(np.mean(latencies_ms))
        local_p95 = float(np.percentile(latencies_ms, 95))
        local_p99 = float(np.percentile(latencies_ms, 99))

        local_cpu = platform.processor() or "Unknown"

        # Project to other CPUs by PassMark ratio
        per_cpu = {}
        for cpu_name, passmark in CPU_PASSMARK.items():
            scale = self.local_passmark / passmark
            per_cpu[cpu_name] = {
                "passmark_score": passmark,
                "scale_factor": float(scale),
                "projected_mean_ms": local_mean * scale,
                "projected_p95_ms": local_p95 * scale,
                "projected_p99_ms": local_p99 * scale,
                "projected_throughput_fps": 1000.0 / max(local_mean * scale, 1e-6),
                "realtime_capable": bool(local_p99 * scale < 50.0),  # <50ms = realtime
            }

        return {
            "local_cpu": local_cpu,
            "local_passmark": self.local_passmark,
            "local_reference_passmark": self.local_passmark,  # alias for legacy callers
            "projection_method": (
                "PassMark single-thread ratio. Projection assumes inference "
                "latency scales inversely with single-thread rating; valid for "
                "single-sample CPU-bound inference only."
            ),
            "n_runs": self.n_runs,
            "n_features": self.n_features,
            "local_mean_ms": local_mean,
            "local_p95_ms": local_p95,
            "local_p99_ms": local_p99,
            "local_throughput_fps": 1000.0 / max(local_mean, 1e-6),
            "per_cpu_projected": per_cpu,
            "model_parameters": (self.model.count_parameters()
                                 if hasattr(self.model, "count_parameters") else 0),
            "verdict": (
                "ALL CPUs REAL-TIME CAPABLE"
                if all(v["realtime_capable"] for v in per_cpu.values())
                else "SOME CPUs BELOW REALTIME THRESHOLD"
            ),
        }

    def save_report(self, results: Dict, path: Union[str, Path]) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Hardware benchmark report → {path}")
        return path

    def markdown_table(self, results: Dict) -> str:
        """Generate a Markdown table for inclusion in papers."""
        lines = [
            "# 5-CPU Hardware Benchmark (PassMark-projected)",
            "",
            f"- Local CPU: {results['local_cpu']}",
            f"- Local PassMark (single-thread): {results['local_passmark']}",
            f"- Model parameters: {results['model_parameters']:,}",
            f"- Number of runs: {results['n_runs']}",
            f"- Projection method: {results.get('projection_method', 'PassMark ratio')}",
            "",
            "| CPU | PassMark | Mean (ms) | P95 (ms) | P99 (ms) | FPS | Realtime |",
            "|-----|----------|-----------|----------|----------|-----|----------|",
        ]
        for cpu, m in results["per_cpu_projected"].items():
            rt = "YES" if m["realtime_capable"] else "NO"
            lines.append(
                f"| {cpu} | {m['passmark_score']} | "
                f"{m['projected_mean_ms']:.2f} | "
                f"{m['projected_p95_ms']:.2f} | "
                f"{m['projected_p99_ms']:.2f} | "
                f"{m['projected_throughput_fps']:.1f} | {rt} |"
            )
        lines.append("")
        lines.append(f"**Verdict**: {results['verdict']}")
        return "\n".join(lines)
