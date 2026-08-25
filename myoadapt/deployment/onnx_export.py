"""
onnx_export.py — Export classifiers to ONNX for Edge deployment
================================================================

Supports:
- sklearn pipelines (XGBClassifier, RF, LDA, LinearSVC, LogisticRegression)
- PyTorch models (CNN1D, LiteDAN, EMGFoundation) via torch.onnx.export

Output:
- Single .onnx file
- Sidecar JSON with metadata (feature names, classes, model type, SHA256)

The SHA256 hash on the .onnx file lets the artefact be verified at
deployment time and recorded in a Model Card / Run Manifest — a
prerequisite for FDA GMLP and EU AI Act Annex IV record-keeping.

"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _sha256_of_file(path: Path, chunk_size: int = 1 << 16) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

try:
    import onnx
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False
    logger.warning("ONNX packages not installed. Install: pip install onnx onnxruntime")

try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    HAS_SKL2ONNX = True
except ImportError:
    HAS_SKL2ONNX = False


class OnnxExporter:
    """Export trained EMG classifiers to ONNX format."""

    @staticmethod
    def export(classifier, output_path: str,
               feature_names: Optional[List[str]] = None,
               n_features: Optional[int] = None,
               model_type: str = "auto") -> Dict[str, Any]:
        """
        Export a trained classifier to ONNX.

        Parameters
        ----------
        classifier : EMGClassifier / CNN1D / LiteDAN / EMGFoundation
        output_path : where to save the .onnx file
        feature_names : list of feature names (metadata only)
        n_features : input dimension
        model_type : 'auto' (detect), 'sklearn', 'torch'

        Returns
        -------
        dict with export info (path, size, opset, etc.)
        """
        if not HAS_ONNX:
            raise ImportError("Install: pip install onnx onnxruntime")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Detect type
        if model_type == "auto":
            cls_name = type(classifier).__name__
            if cls_name in ("CNN1D", "LiteDAN", "EMGFoundation"):
                model_type = "torch"
            elif hasattr(classifier, "model"):  # EMGClassifier
                model_type = "sklearn"
            else:
                model_type = "sklearn"

        if model_type == "sklearn":
            return OnnxExporter._export_sklearn(classifier, output_path,
                                                  feature_names, n_features)
        elif model_type == "torch":
            return OnnxExporter._export_torch(classifier, output_path,
                                                feature_names, n_features)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    @staticmethod
    def _export_sklearn(classifier, output_path: Path,
                         feature_names: Optional[List[str]],
                         n_features: Optional[int]) -> Dict:
        if not HAS_SKL2ONNX:
            raise ImportError("Install: pip install skl2onnx")

        if hasattr(classifier, "model"):
            # EMGClassifier wrapper — build a sklearn Pipeline
            from sklearn.pipeline import Pipeline
            steps = []
            if getattr(classifier, "scaler", None) is not None:
                steps.append(("scaler", classifier.scaler))
            if getattr(classifier, "selector", None) is not None:
                steps.append(("selector", classifier.selector))
            steps.append(("classifier", classifier.model))
            pipeline = Pipeline(steps)
            n_features = (classifier.scaler.n_features_in_
                          if getattr(classifier, "scaler", None) is not None
                          else n_features or 100)
            classes = classifier.classes_.tolist() if classifier.classes_ is not None else None
        else:
            pipeline = classifier
            n_features = n_features or 100
            classes = (classifier.classes_.tolist()
                       if hasattr(classifier, "classes_") else None)

        initial_type = [("input", FloatTensorType([None, n_features]))]
        onnx_model = convert_sklearn(pipeline, initial_types=initial_type,
                                       target_opset=17,
                                       options={id(pipeline): {"zipmap": True}})

        with open(output_path, "wb") as f:
            f.write(onnx_model.SerializeToString())

        # Sidecar metadata (incl. SHA256 for tamper detection).
        sha = _sha256_of_file(output_path)
        meta = {
            "model_type": "sklearn",
            "n_features": n_features,
            "classes": classes,
            "feature_names": feature_names,
            "opset": 17,
            "sha256": sha,
        }
        with open(output_path.with_suffix(".json"), "w") as f:
            json.dump(meta, f, indent=2, default=str)

        return {
            "path": str(output_path),
            "size_bytes": output_path.stat().st_size,
            "model_type": "sklearn",
            "opset": 17,
            "sha256": sha,
        }

    @staticmethod
    def _export_torch(classifier, output_path: Path,
                        feature_names: Optional[List[str]],
                        n_features: Optional[int]) -> Dict:
        import torch
        if classifier.net is None:
            raise RuntimeError("Model not trained")

        net = classifier.net
        net.eval()

        if hasattr(classifier, "n_features") and classifier.n_features:
            n_features = classifier.n_features
            dummy = torch.randn(1, n_features)
        elif hasattr(classifier, "n_channels"):
            # CNN-style: input is (batch, n_channels, n_samples)
            dummy = torch.randn(1, classifier.n_channels, classifier.n_samples)
        else:
            dummy = torch.randn(1, n_features or 308)

        torch.onnx.export(
            net, dummy, str(output_path),
            export_params=True, opset_version=17,
            do_constant_folding=True,
            input_names=["input"], output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        )

        sha = _sha256_of_file(output_path)
        meta = {
            "model_type": "torch",
            "n_features": n_features,
            "classes": (classifier.classes_.tolist() if classifier.classes_ is not None else None),
            "feature_names": feature_names,
            "opset": 17,
            "n_parameters": classifier.count_parameters(),
            "sha256": sha,
        }
        with open(output_path.with_suffix(".json"), "w") as f:
            json.dump(meta, f, indent=2, default=str)

        return {
            "path": str(output_path),
            "size_bytes": output_path.stat().st_size,
            "model_type": "torch",
            "opset": 17,
            "n_parameters": classifier.count_parameters(),
            "sha256": sha,
        }

    @staticmethod
    def verify(onnx_path: str) -> Dict[str, Any]:
        """Verify an ONNX model loads and runs."""
        if not HAS_ONNX:
            raise ImportError("Install: pip install onnx onnxruntime")
        sess = ort.InferenceSession(onnx_path)
        input_name = sess.get_inputs()[0].name
        input_shape = sess.get_inputs()[0].shape
        input_dtype = sess.get_inputs()[0].type
        output_names = [o.name for o in sess.get_outputs()]
        # Build a dummy input that matches the model's rank.
        # ONNX shapes use None for dynamic dims; we substitute 2 for the
        # batch axis and 1 for unknown feature axes when determinable.
        def _dim_size(d):
            if isinstance(d, int):
                return d
            return 1  # unknown — use 1 as a safe placeholder
        dims = [_dim_size(d) for d in input_shape]
        # If all dims are ints, use as-is. If any were unknown (now 1) but the
        # axis is a feature axis, that's still safe for a smoke run.
        if len(dims) == 2:
            dims[0] = 2  # batch
        elif len(dims) == 3:
            dims[0] = 2  # batch
        dummy = np.random.randn(*dims).astype(np.float32)
        outputs = sess.run(None, {input_name: dummy})
        return {
            "input_name": input_name,
            "input_shape": input_shape,
            "input_dtype": input_dtype,
            "output_names": output_names,
            "test_passed": True,
            "test_output_shape": [list(o.shape) for o in outputs],
        }


    @staticmethod
    def verify_sha256(onnx_path: str) -> Dict[str, Any]:
        """Verify ONNX file integrity against its SHA-256 sidecar.

        Reads the sidecar JSON (same path with .json extension),
        extracts the stored sha256, recomputes the file's sha256,
        and returns whether they match.
        """
        onnx_path = Path(onnx_path)
        sidecar_path = onnx_path.with_suffix(".json")
        if not sidecar_path.exists():
            return {"verified": False, "expected": None, "actual": None,
                    "error": "No sidecar JSON found"}
        with open(sidecar_path) as f:
            sidecar = json.load(f)
        expected = sidecar.get("sha256")
        actual = _sha256_of_file(onnx_path)
        return {"verified": expected == actual, "expected": expected, "actual": actual}

    @staticmethod
    def verify_parity(onnx_path: str, model, X_sample: np.ndarray,
                       n_samples: int = 100, threshold: float = 1e-5) -> Dict[str, Any]:
        """Verify numerical parity between original model and ONNX session.

        Runs both the original model and the ONNX session on X_sample,
        computes max absolute difference.
        """
        if not HAS_ONNX:
            raise ImportError("Install: pip install onnx onnxruntime")
        sess = ort.InferenceSession(onnx_path)
        input_name = sess.get_inputs()[0].name

        # Original model predictions
        if hasattr(model, "predict_proba"):
            orig_proba = model.predict_proba(X_sample)
        else:
            orig_proba = np.zeros((len(X_sample), 1))

        # ONNX predictions
        onnx_outputs = sess.run(None, {input_name: X_sample.astype(np.float32)})
        onnx_proba = onnx_outputs[-1]
        # skl2onnx zipmap=True produces a list of dicts — convert to array.
        if isinstance(onnx_proba, list):
            onnx_proba = np.array([[v for k, v in sorted(d.items())]
                                    for d in onnx_proba], dtype=np.float32)
        if isinstance(onnx_proba, np.ndarray) and onnx_proba.ndim == 1:
            onnx_proba = np.column_stack([1 - onnx_proba, onnx_proba])  # binary

        # Compare
        max_diff = float(np.max(np.abs(orig_proba - onnx_proba)))
        mean_diff = float(np.mean(np.abs(orig_proba - onnx_proba)))
        return {
            "max_abs_diff": max_diff,
            "mean_abs_diff": mean_diff,
            "parity_passed": max_diff < threshold,
            "threshold": threshold,
            "n_samples": len(X_sample),
        }
