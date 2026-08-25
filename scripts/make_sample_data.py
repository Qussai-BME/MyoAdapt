#!/usr/bin/env python
"""
make_sample_data.py — generate data/sample/smoke_test.npz
=========================================================

Tiny synthetic EMG-like dataset for quickstart smoke tests and integration
tests. Procedurally generated, deterministic (random_state=42), so the file
is byte-reproducible.

Schema (saved to the .npz):
    windows   : (N, 12, 400)  float32  — 200-ms windows @ 2000 Hz
    labels    : (N,)         int32     — gesture label 0..n_gestures-1
    groups    : (N,)         int32     — subject id 1..n_subjects
    databases : (N,)         int32     — synthetic database id 1..n_dbs
    fs        : ()            int32     — sampling rate (2000)
    n_channels: ()            int32     — number of channels (12)

Usage:
    python scripts/make_sample_data.py
    # → writes data/sample/smoke_test.npz
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


# ---- Deterministic configuration -------------------------------------------------
N_CHANNELS = 12
FS = 2000                 # Hz
WINDOW_MS = 200           # 200-ms windows → 400 samples (MyoAdapt default)
WIN_SAMPLES = FS * WINDOW_MS // 1000
N_SUBJECTS = 5
N_GESTURES = 5
WINDOWS_PER_GESTURE = 10
N_DATABASES = 3
RANDOM_STATE = 42

# Default output path (relative to repo root).
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "data" / "sample" / "smoke_test.npz"


def _generate() -> dict:
    """Generate the synthetic payload as a dict of numpy arrays."""
    rng = np.random.default_rng(RANDOM_STATE)
    n_per_subject = N_GESTURES * WINDOWS_PER_GESTURE
    n_total = N_SUBJECTS * n_per_subject

    windows = np.zeros((n_total, N_CHANNELS, WIN_SAMPLES), dtype=np.float32)
    labels = np.zeros(n_total, dtype=np.int32)
    groups = np.zeros(n_total, dtype=np.int32)
    databases = np.zeros(n_total, dtype=np.int32)

    # Distribute subjects round-robin across the N_DATABASES synthetic dbs.
    db_assignment = np.array(
        [(s % N_DATABASES) + 1 for s in range(1, N_SUBJECTS + 1)],
        dtype=np.int32,
    )

    t = np.arange(WIN_SAMPLES) / FS
    idx = 0
    for subj in range(1, N_SUBJECTS + 1):
        # Per-subject DC offset to make the data subject-distinguishable.
        subj_offset = (subj - 1) * 0.05
        db_id = int(db_assignment[subj - 1])
        for gesture in range(N_GESTURES):
            for _ in range(WINDOWS_PER_GESTURE):
                # Background noise + a per-gesture sinusoidal activation
                # that is slightly different per channel so the windows
                # are not trivially identical.
                base = rng.standard_normal((N_CHANNELS, WIN_SAMPLES)).astype(np.float32) * 0.1
                freq = 5 + gesture  # Hz, gesture-specific
                activation = 0.5 * np.sin(2 * np.pi * freq * t)
                base += activation[np.newaxis, :]
                # Per-channel phase shift so channels are not identical.
                for ch in range(N_CHANNELS):
                    base[ch] += 0.05 * np.sin(2 * np.pi * freq * t + 0.3 * ch)
                base += subj_offset
                windows[idx] = base
                labels[idx] = gesture
                groups[idx] = subj
                databases[idx] = db_id
                idx += 1

    return {
        "windows": windows,
        "labels": labels,
        "groups": groups,
        "databases": databases,
        "fs": np.int32(FS),
        "n_channels": np.int32(N_CHANNELS),
    }


def main(out_path: Path | None = None) -> Path:
    out_path = out_path or DEFAULT_OUT
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _generate()
    np.savez(out_path, **payload)
    print(f"Wrote {out_path}")
    print(f"  windows: shape={payload['windows'].shape} dtype={payload['windows'].dtype}")
    print(f"  labels:  shape={payload['labels'].shape} "
          f"unique={sorted(np.unique(payload['labels']).tolist())}")
    print(f"  groups:  shape={payload['groups'].shape} "
          f"unique={sorted(np.unique(payload['groups']).tolist())}")
    print(f"  databases: unique={sorted(np.unique(payload['databases']).tolist())}")
    print(f"  fs={int(payload['fs'])}  n_channels={int(payload['n_channels'])}")
    return out_path


if __name__ == "__main__":
    main()
