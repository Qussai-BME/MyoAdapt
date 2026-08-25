"""
MyoAdapt v2.0 — Open-Source CPU-Native sEMG Pattern Recognition Platform
========================================================================

An open-source CPU-native research platform for zero-calibration
cross-subject sEMG pattern recognition with edge deployment, adversarial
domain adaptation, and audit-ready explainability.

Modules
-------
data          : NinaPro loaders, preprocessing, LOSO/LODO splits
features      : time/frequency/time-frequency/histogram/correlation/MiniROCKET
models        : classical (XGB/RF/LDA/SVM), CNN-1D, Lite-DAN, EMG foundation
adaptation    : CORAL, TCA, SA, Euclidean Alignment, adversarial (GRL)
evaluation    : LOSO, LODO, metrics, statistics, calibration, fairness,
                paper-ready LaTeX tables, CD diagrams, model cards
deployment    : ONNX export, realtime, SHAP reports, trust scores, HW benchmark
tracking      : MLflow adapter + local JSON fallback
api           : FastAPI REST + WebSocket streaming
ui            : Streamlit modular pages
cli           : typer-based command-line interface
tasks         : continuous decoding (LSTM/Transformer) and other tasks
reproducibility : seed propagation, env hashing, run manifests

Quick start
-----------
>>> from myoadapt.data import NinaProLoader
>>> from myoadapt.models import LiteDAN
>>> from myoadapt.evaluation import LOSOEvaluator
>>> loader = NinaProLoader(root='./data', db='DB2')
>>> X, y, groups, meta = loader.load_split()
>>> model = LiteDAN(n_features=X.shape[1], n_classes=len(set(y)))
>>> evaluator = LOSOEvaluator(model, n_jobs=4)
>>> results = evaluator.run(X, y, groups)

:copyright: (c) 2025 MyoAdapt Contributors.
:license: Apache 2.0, see LICENSE for more details.
"""
from __future__ import annotations

__version__ = "2.0.0"
__author__ = "MyoAdapt Contributors"
__email__ = "adlbiqussai@gmail.com"
__license__ = "Apache-2.0"
__status__ = "Beta"

# Convenience imports (lazy-loaded to keep startup fast)
__all__ = [
    "__version__",
    "data",
    "features",
    "models",
    "adaptation",
    "evaluation",
    "deployment",
    "tracking",
    "api",
    "ui",
    "cli",
    "tasks",
    "config",
    "reproducibility",
]


def _lazy_import(name: str):
    """Lazy import to keep `import myoadapt` fast."""
    import importlib
    return importlib.import_module(f"myoadapt.{name}")


def __getattr__(name: str):
    if name in __all__ and name != "__version__":
        return _lazy_import(name)
    raise AttributeError(f"module 'myoadapt' has no attribute {name!r}")
