# `myoadapt.reproducibility` — Reproducibility engine

Captures everything needed to exactly reproduce a MyoAdapt run.

## Public API

```python
from myoadapt.reproducibility import (
    set_global_seed,           # seeds Python + NumPy + PyTorch (CUDA included)
    environment_fingerprint,   # structured dict of versions / platform
    environment_hash,          # SHA-256 of the fingerprint
    git_info,                  # best-effort git HEAD (commit, branch, dirty)
    sha256_of_file,            # streaming SHA-256 of a file
    RunManifest,               # builder + serialiser for a single run
    snapshot_config,           # dump a config object to stable JSON
)
```

## Seed propagation

```python
from myoadapt.reproducibility import set_global_seed
applied = set_global_seed(42)
# {'python': 42, 'numpy': 42, 'pythonhashseed': 42, 'torch_cpu': 42}
```

`set_global_seed` seeds Python's `random`, NumPy, and PyTorch (CPU +
CUDA, with `cudnn.deterministic = True`). Use it at the top of every
script / notebook so the entire pipeline shares one root seed.

## Environment fingerprint

```python
from myoadapt.reproducibility import environment_fingerprint, environment_hash
fp = environment_fingerprint()
# {
#   "python_version": "3.12.13",
#   "platform": "Linux-6.8.0-...",
#   "packages": {"numpy": "2.5.2", "scipy": "1.18.0", ...},
# }
h = environment_hash(fp)
# SHA-256, 64 hex chars
```

Two runs with the same env hash use the same library stack.

## Run manifest

```python
from myoadapt.reproducibility import RunManifest

manifest = RunManifest(
    run_name="rf_db2_loso",
    dataset="NinaPro-DB2",
    model="random_forest",
    seed=42,
    config={"k_features": 420, "n_estimators": 200},
)
manifest.add_output("./models/xgb_db2.pkl", role="model")
manifest.add_output("./results/loso_db2.csv", role="results")
manifest.write("./results/manifest.json")
```

The manifest JSON records:

- The resolved config (after all defaults + overrides).
- The environment fingerprint + hash.
- The git HEAD (commit, branch, dirty flag, author date).
- Start / stop UTC timestamps + elapsed seconds.
- Every output artefact's path, SHA-256, size, and role.
- The MyoAdapt version that produced the manifest.

## Verification

```python
from myoadapt.reproducibility import RunManifest

v = RunManifest.verify("./results/manifest.json")
# {'verified': True, 'checked': 2, 'mismatches': []}
```

`verify` re-hashes every recorded output and reports any mismatch —
detecting tampering, accidental overwrites, or environment drift
between the original run and the audit.

## NeurIPS / FDA GMLP alignment

The manifest satisfies the NeurIPS 2025/2026 Reproducibility Checklist
requirements for code + data + environment documentation, and supports
the FDA GMLP / EU AI Act Annex IV record-keeping obligations
(research-grade; not a compliance certification).
