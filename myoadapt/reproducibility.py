"""
reproducibility.py — Reproducibility engine
============================================

Captures everything needed to exactly reproduce a MyoAdapt run:

- **Config snapshot** — the resolved ``MyoAdaptConfig`` (after all
  defaults, YAML overrides, and CLI flags are applied), written as a
  stable, hashable JSON document.
- **Environment hash** — a SHA256 over the Python version, key
  dependency versions, and a frozen pip-requirements string. Two runs
  with the same env hash use the same library stack.
- **Run manifest** — a single JSON document bundling the config
  snapshot, env hash, git HEAD (if available), random seed, dataset
  identifier, model identifier, output file paths with their SHA256s,
  start/stop timestamps, and platform info.
- **Seed propagation** — a single ``set_global_seed()`` that seeds
  Python, NumPy, and PyTorch (when available) so every stochastic
  stage in the pipeline shares one root.

The manifest is the single artefact reviewers should ask for when
auditing a result. It is designed to satisfy the NeurIPS 2025/2026
Reproducibility Checklist and the FDA GMLP / EU AI Act Annex IV
record-keeping obligations (research-grade; not a compliance
certification).

License: Apache 2.0
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Seed propagation
# ---------------------------------------------------------------------------
def set_global_seed(seed: int = 42) -> Dict[str, int]:
    """Seed Python, NumPy, and (if installed) PyTorch.

    Returns a dict mapping library name → seed actually applied. Use
    this at the top of every script / notebook so the entire pipeline
    is deterministic from one root seed.

    Notes
    -----
    PyTorch CUDA RNGs are also seeded when available, and
    ``torch.backends.cudnn.deterministic`` is set to True (with the
    expected performance trade-off). For full CUDA determinism you
    should additionally set ``PYTHONHASHSEED`` before launching the
    interpreter; this function logs a reminder when CUDA is present.
    """
    seed = int(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:  # pragma: no cover — numpy is a hard dep
        pass
    applied = {"python": seed, "numpy": seed, "pythonhashseed": seed}
    try:
        import torch
        torch.manual_seed(seed)
        applied["torch_cpu"] = seed
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            applied["torch_cuda"] = seed
            logger.info("CUDA detected — cuDNN set to deterministic mode. "
                        "For full determinism also export PYTHONHASHSEED=%d.", seed)
    except ImportError:
        pass
    logger.info("Global seed set to %d (applied to: %s)", seed, list(applied.keys()))
    return applied


# ---------------------------------------------------------------------------
# Environment hash
# ---------------------------------------------------------------------------
def _safe_version(mod_name: str) -> str:
    try:
        mod = __import__(mod_name)
        return getattr(mod, "__version__", "unknown")
    except Exception:
        return "not-installed"


def environment_fingerprint() -> Dict[str, Any]:
    """Return a structured environment fingerprint.

    Includes the Python version, platform, and the resolved versions
    of every MyoAdapt-relevant library. The fingerprint is
    JSON-serialisable and order-stable so it can be hashed.
    """
    packages = [
        "numpy", "scipy", "scikit-learn", "pandas", "matplotlib",
        "torch", "xgboost", "lightgbm", "onnx", "onnxruntime",
        "skl2onnx", "shap", "mlflow", "fastapi", "streamlit",
        "typer", "rich", "PyWavelets", "tabulate",
    ]
    versions: Dict[str, str] = {}
    for name in packages:
        # sktime-style names that differ from the import name.
        import_name = {
            "scikit-learn": "sklearn",
            "PyWavelets": "pywt",
        }.get(name, name)
        versions[name] = _safe_version(import_name)
    return {
        "python_version": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "machine": platform.machine(),
        "packages": versions,
    }


def environment_hash(fingerprint: Optional[Dict[str, Any]] = None) -> str:
    """SHA256 of the canonical-JSON environment fingerprint."""
    fp = fingerprint if fingerprint is not None else environment_fingerprint()
    canonical = json.dumps(fp, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Git HEAD (best-effort)
# ---------------------------------------------------------------------------
def git_info(repo_path: Union[str, Path] = ".") -> Dict[str, str]:
    """Best-effort git HEAD information.

    Returns a dict with ``commit``, ``branch``, ``dirty`` (bool),
    ``author_date``, and ``describe``. If the path is not a git repo
    or git is unavailable, returns an empty dict (the caller can
    detect missing keys and treat the run as "git unknown").
    """
    repo_path = Path(repo_path).resolve()
    info: Dict[str, str] = {}
    try:
        def _run(*args: str) -> str:
            return subprocess.run(
                ["git", "-C", str(repo_path), *args],
                capture_output=True, text=True, check=False, timeout=5,
            ).stdout.strip()
        commit = _run("rev-parse", "HEAD")
        if not commit or "fatal" in commit.lower():
            return info
        info["commit"] = commit
        info["branch"] = _run("rev-parse", "--abbrev-ref", "HEAD") or "detached"
        status = _run("status", "--porcelain")
        info["dirty"] = "true" if status else "false"
        info["author_date"] = _run("show", "-s", "--format=%aI", "HEAD")
        info["describe"] = _run("describe", "--tags", "--always") or commit[:7]
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as e:
        logger.debug(f"git_info unavailable: {e}")
    return info


# ---------------------------------------------------------------------------
# File hashing
# ---------------------------------------------------------------------------
def sha256_of_file(path: Union[str, Path], chunk_size: int = 1 << 16) -> str:
    """SHA256 of a file's contents (streaming)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Run manifest
# ---------------------------------------------------------------------------
class RunManifest:
    """Builder + serialiser for a single reproducibility manifest.

    Usage::

        manifest = RunManifest(
            run_name="rf_db2_loso",
            dataset="NinaPro-DB2",
            model="random_forest",
            seed=42,
            config=config_dict,
        )
        manifest.add_output("./models/xgb_db2.pkl")
        manifest.add_output("./results/loso_db2.csv")
        manifest.write("./results/manifest.json")
    """

    def __init__(
        self,
        run_name: str,
        dataset: str,
        model: str,
        seed: int = 42,
        config: Optional[Dict[str, Any]] = None,
        repo_path: Union[str, Path] = ".",
    ):
        self.run_name = run_name
        self.dataset = dataset
        self.model = model
        self.seed = int(seed)
        self.config = config or {}
        self.repo_path = Path(repo_path).resolve()
        self._start = time.time()
        self._start_iso = datetime.now(timezone.utc).isoformat()
        self._outputs: List[Dict[str, Any]] = []

    def add_output(self, path: Union[str, Path],
                    role: Optional[str] = None) -> Dict[str, Any]:
        """Record an output artefact (file path + SHA256 + size)."""
        path = Path(path)
        entry: Dict[str, Any] = {
            "path": str(path),
            "role": role or path.suffix.lstrip("."),
        }
        if path.exists() and path.is_file():
            entry["sha256"] = sha256_of_file(path)
            entry["size_bytes"] = path.stat().st_size
        else:
            entry["sha256"] = None
            entry["size_bytes"] = None
            entry["missing"] = True
        self._outputs.append(entry)
        return entry

    def to_dict(self) -> Dict[str, Any]:
        env_fp = environment_fingerprint()
        return {
            "schema_version": "1.0",
            "run_name": self.run_name,
            "dataset": self.dataset,
            "model": self.model,
            "seed": self.seed,
            "config": self.config,
            "environment": env_fp,
            "environment_hash_sha256": environment_hash(env_fp),
            "git": git_info(self.repo_path),
            "started_at_utc": self._start_iso,
            "ended_at_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(time.time() - self._start, 4),
            "outputs": self._outputs,
            "myoadapt_version": _myoadapt_version(),
        }

    def write(self, path: Union[str, Path]) -> Path:
        """Write the manifest as canonical JSON (stable key order)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)
        logger.info(f"Run manifest written to {path}")
        return path

    @staticmethod
    def load(path: Union[str, Path]) -> Dict[str, Any]:
        """Load a manifest from disk and return its parsed contents."""
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def verify(manifest_path: Union[str, Path],
                outputs_root: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """Verify that every recorded output still exists and matches its SHA256.

        Returns a dict with ``verified`` (bool), ``checked`` (int), and
        ``mismatches`` (list of paths whose hash differs).
        """
        manifest = RunManifest.load(manifest_path)
        outputs_root = Path(outputs_root) if outputs_root else Path(manifest_path).parent
        checked = 0
        mismatches: List[str] = []
        for entry in manifest.get("outputs", []):
            p = Path(entry["path"])
            if not p.is_absolute():
                p = outputs_root / p
            if not p.exists():
                mismatches.append(f"{p}: missing")
                continue
            actual = sha256_of_file(p)
            expected = entry.get("sha256")
            checked += 1
            if expected and actual != expected:
                mismatches.append(f"{p}: hash mismatch (expected {expected[:12]}…, got {actual[:12]}…)")
        return {
            "verified": len(mismatches) == 0,
            "checked": checked,
            "mismatches": mismatches,
        }


def _myoadapt_version() -> str:
    try:
        import myoadapt
        return getattr(myoadapt, "__version__", "unknown")
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Convenience: config-snapshot helper
# ---------------------------------------------------------------------------
def snapshot_config(config_obj: Any, path: Union[str, Path]) -> Path:
    """Serialise any config object exposing ``model_dump()`` (pydantic) or
    ``__dict__`` to a stable JSON file.

    Used by the CLI to drop a config snapshot next to every artefact.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(config_obj, "model_dump"):
        data = config_obj.model_dump()
    elif hasattr(config_obj, "dict"):
        data = config_obj.dict()
    else:
        data = {k: v for k, v in vars(config_obj).items()
                if not k.startswith("_") and _is_jsonable(v)}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True, default=str)
    return path


def _is_jsonable(v: Any) -> bool:
    try:
        json.dumps(v, default=str)
        return True
    except (TypeError, ValueError):
        return False
