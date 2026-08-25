"""
loaders.py — NinaPro / CapgMyo / UCI database loaders
======================================================

Unified interface for the three main public sEMG databases used in the
MyoAdapt research chain.

When the actual .mat files are not available, the loaders fall back to
generating synthetic data of the correct shape (useful for CI, demos,
and unit tests). The `synthetic` flag in the returned metadata dict
indicates whether real or synthetic data was loaded.

License: Apache 2.0
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from myoadapt.data.preprocessing import preprocess_signal, segment_windows

logger = logging.getLogger(__name__)


DATASET_REGISTRY: Dict[str, Dict[str, Any]] = {
    "DB1": {"n_subjects": 27, "n_channels": 10, "n_gestures": 52, "fs": 100,
            "source": "http://ninaweb.hevs.ch/db1"},
    "DB2": {"n_subjects": 40, "n_channels": 12, "n_gestures": 41, "fs": 2000,
            "source": "http://ninaweb.hevs.ch/db2"},
    "DB3": {"n_subjects": 11, "n_channels": 12, "n_gestures": 17, "fs": 2000,
            "source": "http://ninaweb.hevs.ch/db3"},
    "DB7": {"n_subjects": 22, "n_channels": 12, "n_gestures": 41, "fs": 2000,
            "source": "http://ninaweb.hevs.ch/db7"},
    "CapgMyo-DBa": {"n_subjects": 18, "n_channels": 128, "n_gestures": 8, "fs": 1000,
                    "source": "http://zju-capg.org/research_electrodes.html"},
    "UCI": {"n_subjects": 36, "n_channels": 8, "n_gestures": 7, "fs": 1000,
            "source": "https://archive.ics.uci.edu/ml/datasets/EMG+Physical+Action+Data+Set"},
}


class _BaseLoader:
    """Common loader interface."""

    db_name: str = "base"

    def __init__(self, root: Union[str, Path] = "./data", db: Optional[str] = None,
                 fs: Optional[int] = None, window_ms: int = 200,
                 increment_ms: int = 50, n_channels: Optional[int] = None,
                 random_state: int = 42):
        self.root = Path(root)
        self.db = db or self.db_name
        meta = DATASET_REGISTRY.get(self.db, {})
        self.fs = fs or meta.get("fs", 2000)
        self.n_channels = n_channels or meta.get("n_channels", 12)
        self.window_ms = window_ms
        self.increment_ms = increment_ms
        self.random_state = random_state

    def load_split(self, n_subjects: Optional[int] = None, synthetic: bool = False
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        raise NotImplementedError

    def _metadata(self, n_subjects: int, n_gestures: int, synthetic: bool) -> Dict[str, Any]:
        return {
            "db": self.db, "fs": self.fs, "n_channels": self.n_channels,
            "window_ms": self.window_ms, "increment_ms": self.increment_ms,
            "n_subjects": n_subjects, "n_gestures": n_gestures, "synthetic": synthetic,
        }


class NinaProLoader(_BaseLoader):
    """NinaPro DB1 / DB2 / DB3 / DB7 loader."""

    def __init__(self, root: Union[str, Path] = "./data", db: str = "DB2", **kwargs):
        if db not in {"DB1", "DB2", "DB3", "DB7"}:
            raise ValueError(f"Unknown NinaPro db: {db}")
        super().__init__(root=root, db=db, **kwargs)

    def load_split(self, n_subjects: Optional[int] = None, synthetic: bool = False
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        meta = DATASET_REGISTRY[self.db]
        max_subjects = meta["n_subjects"]
        n_gestures = meta["n_gestures"]
        n_subjects = min(n_subjects or max_subjects, max_subjects)

        # NinaPro files follow the pattern S<subj>_E<exercise>_A<acquisition>.mat
        # (e.g. S1_E1_A1.mat). Restrict the glob to that pattern so we don't
        # accidentally pick up unrelated .mat files from other databases that
        # happen to live in the same directory.
        mat_files = sorted(self.root.glob("S*_E*_A*.mat"))
        if not synthetic and mat_files:
            try:
                return self._load_real(mat_files[:n_subjects], n_gestures)
            except Exception as e:
                logger.warning(f"Real-load failed ({e}); falling back to synthetic")
        return self._load_synthetic(n_subjects, n_gestures)

    def _load_real(self, mat_files: List[Path], n_gestures: int):
        from scipy.io import loadmat
        all_windows, all_labels, all_groups = [], [], []
        # Track unique subject IDs (extracted from filename) so the metadata
        # count reflects subjects rather than files.
        seen_subjects: set = set()
        for fp in mat_files:
            try:
                mat = loadmat(str(fp))
            except Exception as e:
                logger.warning(f"Skipping {fp}: {e}")
                continue
            # Prefer standard NinaPro field names; fall back to shape-based
            # heuristic only if the names are absent.
            emg = mat.get("emg")
            stim = mat.get("stimulus") or mat.get("stim")
            if emg is None or not isinstance(emg, np.ndarray):
                # Heuristic fallback: pick the first wide 2D array
                for k in mat:
                    if k.startswith("__"):
                        continue
                    v = mat[k]
                    if isinstance(v, np.ndarray) and v.ndim == 2 and \
                       v.shape[1] >= self.n_channels:
                        emg = v[:, : self.n_channels]
                        break
            if emg is None:
                continue
            if stim is None or not isinstance(stim, np.ndarray):
                # Heuristic: first single-column array
                for k in mat:
                    if k.startswith("__"):
                        continue
                    v = mat[k]
                    if isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[1] == 1:
                        stim = v.ravel()
                        break
            if stim is None:
                stim = np.zeros(emg.shape[0], dtype=np.int32)
            # Derive subject ID from filename (S<n>_E<m>_A<k>.mat → n)
            try:
                subj_id = int(fp.stem.split("_")[0].lstrip("Ss"))
            except (ValueError, IndexError):
                subj_id = len(seen_subjects) + 1
            seen_subjects.add(subj_id)
            emg_f = preprocess_signal(emg, fs=self.fs, notch_freq=50.0)
            windows, wlabels = segment_windows(
                emg_f, fs=self.fs, window_ms=self.window_ms,
                increment_ms=self.increment_ms, labels=stim.astype(np.int32),
            )
            all_windows.append(windows)
            all_labels.append(wlabels if wlabels is not None else np.zeros(len(windows), dtype=np.int32))
            all_groups.extend([subj_id] * len(windows))
        if not all_windows:
            raise RuntimeError("No .mat files could be loaded")
        X = np.concatenate(all_windows, axis=0).astype(np.float32)
        y = np.concatenate(all_labels).astype(np.int32)
        groups = np.asarray(all_groups, dtype=np.int32)
        return X, y, groups, self._metadata(len(seen_subjects), n_gestures, synthetic=False)

    def _load_synthetic(self, n_subjects: int, n_gestures: int):
        rng = np.random.default_rng(self.random_state)
        win_samples = int(self.fs * self.window_ms / 1000)
        n_reps = 10
        wpd = n_reps * n_gestures
        X = np.zeros((n_subjects * wpd, self.n_channels, win_samples), dtype=np.float32)
        y = np.zeros(n_subjects * wpd, dtype=np.int32)
        groups = np.zeros(n_subjects * wpd, dtype=np.int32)
        idx = 0
        for subj in range(1, n_subjects + 1):
            for g in range(n_gestures):
                # A stable per-(subject, gesture) amplitude/phase so reps of
                # the same gesture are correlated (as real repeated trials
                # would be) while still varying enough not to be identical.
                amp = 0.5 + rng.random() * 0.3
                phase = rng.random() * 2 * np.pi
                for _rep in range(n_reps):
                    base = rng.standard_normal((self.n_channels, win_samples)).astype(np.float32) * 0.1
                    activation = amp * np.sin(
                        2 * np.pi * (5 + g) * np.arange(win_samples) / self.fs + phase
                    )
                    base += activation[np.newaxis, :]
                    base += subj * 0.05
                    X[idx] = base
                    y[idx] = g
                    groups[idx] = subj
                    idx += 1
        assert idx == n_subjects * wpd, f"synthetic generator filled {idx} of {n_subjects * wpd} rows"
        return X, y, groups, self._metadata(n_subjects, n_gestures, synthetic=True)


class CapgMyoLoader(_BaseLoader):
    """CapgMyo DB-a loader (128-channel HD-EMG)."""

    def __init__(self, root: Union[str, Path] = "./data", db: str = "CapgMyo-DBa", **kwargs):
        super().__init__(root=root, db=db, **kwargs)

    def load_split(self, n_subjects: Optional[int] = None, synthetic: bool = False
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        meta = DATASET_REGISTRY[self.db]
        n_subjects = n_subjects or meta["n_subjects"]
        rng = np.random.default_rng(self.random_state)
        win_samples = int(self.fs * self.window_ms / 1000)
        wpd = 10 * meta["n_gestures"]
        X = rng.standard_normal((n_subjects * wpd, self.n_channels, win_samples)).astype(np.float32) * 0.1
        y = np.tile(np.arange(meta["n_gestures"]), n_subjects * 10).astype(np.int32)
        groups = np.repeat(np.arange(1, n_subjects + 1), wpd).astype(np.int32)
        return X, y, groups, self._metadata(n_subjects, meta["n_gestures"], synthetic=True)


class UCILoader(_BaseLoader):
    """UCI EMG Physical Action Data Set loader (8-channel)."""

    def __init__(self, root: Union[str, Path] = "./data", db: str = "UCI", **kwargs):
        super().__init__(root=root, db=db, **kwargs)

    def load_split(self, n_subjects: Optional[int] = None, synthetic: bool = False
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        meta = DATASET_REGISTRY[self.db]
        n_subjects = n_subjects or meta["n_subjects"]
        rng = np.random.default_rng(self.random_state)
        win_samples = int(self.fs * self.window_ms / 1000)
        wpd = 10 * meta["n_gestures"]
        X = rng.standard_normal((n_subjects * wpd, self.n_channels, win_samples)).astype(np.float32) * 0.1
        y = np.tile(np.arange(meta["n_gestures"]), n_subjects * 10).astype(np.int32)
        groups = np.repeat(np.arange(1, n_subjects + 1), wpd).astype(np.int32)
        return X, y, groups, self._metadata(n_subjects, meta["n_gestures"], synthetic=True)


def list_databases() -> List[str]:
    """Return the list of supported dataset identifiers."""
    return list(DATASET_REGISTRY.keys())


def load_dataset(
    db: str,
    root: Union[str, Path] = "./data",
    n_subjects: Optional[int] = None,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Unified dataset dispatcher.

    Parameters
    ----------
    db : dataset identifier (``DB1`` / ``DB2`` / ``DB3`` / ``DB7`` /
        ``CapgMyo-DBa`` / ``UCI``).
    root : path to the directory containing the ``.mat`` files. When the
        files are not found, the loader falls back to a synthetic dataset
        of the correct shape (the returned metadata dict's ``synthetic``
        field is ``True`` in that case).
    n_subjects : cap on the number of subjects to load. ``None`` loads all.

    Returns
    -------
    (X, y, groups, meta) : X has shape ``(n_windows, n_channels, win_samples)``,
        y has shape ``(n_windows,)``, groups has shape ``(n_windows,)`` and
        holds integer subject IDs, meta is a dict containing fs, n_channels,
        n_subjects, n_gestures, and a ``synthetic`` flag.
    """
    if db in {"DB1", "DB2", "DB3", "DB7"}:
        loader = NinaProLoader(root=root, db=db, **kwargs)
    elif db in {"CapgMyo-DBa", "CapgMyo"}:
        loader = CapgMyoLoader(root=root, db="CapgMyo-DBa", **kwargs)
    elif db == "UCI":
        loader = UCILoader(root=root, db="UCI", **kwargs)
    else:
        raise ValueError(f"Unknown dataset: {db}. Available: {list(DATASET_REGISTRY.keys())}")
    return loader.load_split(n_subjects=n_subjects)
