"""
meta_loaders.py — Loaders for Meta Reality Labs EMG datasets.

Supports:
- EMG-to-Pose (emg2pose): 23 subjects, wrist sEMG → 21-DOF hand pose
- EMG-to-Qwerty (emg2qwerty): 17 subjects, sEMG → keyboard typing

Both datasets use Meta's open format (HDF5 or parquet). The loader
falls back to synthetic data of the correct shape when real files
are absent — same pattern as NinaProLoader.

References:
- Meta Reality Labs (2024). "A wearable EMG system for hand gesture
  recognition." NeurIPS Datasets & Benchmarks.
- emg2pose: github.com/facebookresearch/emg2pose
- emg2qwerty: github.com/facebookresearch/emg2qwerty

License: Apache 2.0
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from myoadapt.data.preprocessing import preprocess_signal, segment_windows

logger = logging.getLogger(__name__)


DATASET_META: Dict[str, Dict[str, Any]] = {
    "EMG2Pose": {
        "n_subjects": 23,
        "n_channels": 16,
        "fs": 2000,
        "n_outputs": 21,
        "task": "regression",
        "source": "https://github.com/facebookresearch/emg2pose",
    },
    "EMG2Qwerty": {
        "n_subjects": 17,
        "n_channels": 16,
        "fs": 2000,
        "n_outputs": None,  # variable number of keystrokes
        "task": "classification",
        "source": "https://github.com/facebookresearch/emg2qwerty",
    },
}


# Synthetic windows-per-subject — keeps in-memory demos lightweight while
# exercising the full pipeline shape (n_windows, n_channels, win_samples).
_WINDOWS_PER_SUBJECT_POSE = 50
_WINDOWS_PER_SUBJECT_QWERTY = 60
_DEFAULT_KEYSTROKES = 10  # digits 0..9 when no real label set is present


class MetaEMGLoader:
    """Loader for Meta Reality Labs EMG datasets (emg2pose, emg2qwerty).

    Parameters
    ----------
    root : path to the directory containing Meta HDF5 files. When the
        files are not found, the loader falls back to a synthetic
        dataset of the correct shape (the returned metadata dict's
        ``synthetic`` field is ``True`` in that case).
    dataset : ``"EMG2Pose"`` or ``"EMG2Qwerty"``.
    fs : sampling rate in Hz.
    n_channels : number of EMG channels.
    window_ms : window length in milliseconds.
    increment_ms : window step in milliseconds.
    random_state : seed for the synthetic fallback generator.
    """

    def __init__(
        self,
        root: Union[str, Path] = "./data",
        dataset: str = "EMG2Pose",
        fs: int = 2000,
        n_channels: int = 16,
        window_ms: int = 200,
        increment_ms: int = 50,
        random_state: int = 42,
    ) -> None:
        if dataset not in DATASET_META:
            raise ValueError(
                f"Unknown Meta dataset: {dataset!r}. "
                f"Available: {list(DATASET_META.keys())}"
            )
        self.root = Path(root)
        self.dataset = dataset
        meta = DATASET_META[dataset]
        # Fall back to the dataset's canonical fs / n_channels when the
        # caller leaves them at the loader defaults but the dataset
        # specifies something different. (Both Meta datasets are 16 / 2000
        # so this is mostly defensive.)
        self.fs = int(fs or meta["fs"])
        self.n_channels = int(n_channels or meta["n_channels"])
        self.window_ms = int(window_ms)
        self.increment_ms = int(increment_ms)
        self.random_state = int(random_state)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def load_split(
        self,
        n_subjects: Optional[int] = None,
        synthetic: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """Load the dataset (real HDF5 or synthetic fallback).

        Returns
        -------
        (X, y_or_pose, groups, meta)
            X : float32 array of shape ``(n_windows, n_channels, win_samples)``.
            y_or_pose :
                - For EMG2Pose: float32 ``(n_windows, 21)`` continuous
                  joint angles in radians.
                - For EMG2Qwerty: int32 ``(n_windows,)`` keystroke labels.
            groups : int32 array of subject IDs (1-indexed).
            meta : dict containing dataset metadata and a ``synthetic``
                flag.
        """
        meta = DATASET_META[self.dataset]
        max_subjects = meta["n_subjects"]
        n_subjects = min(n_subjects or max_subjects, max_subjects)

        if not synthetic:
            try:
                return self._load_real(n_subjects)
            except Exception as e:  # pragma: no cover — defensive
                logger.warning(
                    "Real-load failed for %s (%s); falling back to synthetic.",
                    self.dataset, e,
                )
        return self._load_synthetic(n_subjects)

    # ------------------------------------------------------------------
    # Real loader (HDF5)
    # ------------------------------------------------------------------
    def _load_real(
        self, n_subjects: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """Load from Meta-format HDF5 files.

        Meta distributes emg2pose / emg2qwerty as ``.h5`` files where
        each file holds an ``emg`` dataset ``(T, n_channels)`` and either
        a ``pose`` dataset ``(T, 21)`` (emg2pose) or a ``keystrokes``
        dataset ``(T,)`` (emg2qwerty). Subject IDs are encoded in the
        filename as ``subject_<id>...h5`` / ``subj_<id>.h5``. We glob
        for these and load each subject in turn.
        """
        try:
            import h5py  # local import — only needed for real loads
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("h5py required to load real Meta data") from e

        h5_files = sorted(
            list(self.root.glob("*.h5")) + list(self.root.glob("*.hdf5"))
        )
        if not h5_files:
            raise RuntimeError(
                f"No .h5 files found under {self.root} for {self.dataset}"
            )

        meta = DATASET_META[self.dataset]
        all_X: List[np.ndarray] = []
        all_y: List[np.ndarray] = []
        all_groups: List[int] = []
        seen_subjects: set = set()

        for fp in h5_files:
            if len(seen_subjects) >= n_subjects:
                break
            try:
                with h5py.File(str(fp), "r") as f:
                    if "emg" not in f:
                        logger.warning("Skipping %s: no 'emg' dataset", fp.name)
                        continue
                    emg = np.asarray(f["emg"][:], dtype=np.float32)
                    if emg.ndim != 2 or emg.shape[1] < self.n_channels:
                        logger.warning(
                            "Skipping %s: emg shape %s unexpected",
                            fp.name, getattr(emg, "shape", None),
                        )
                        continue
                    emg = emg[:, : self.n_channels]

                    targets: Optional[np.ndarray] = None
                    if self.dataset == "EMG2Pose":
                        pose_ds = f.get("pose") or f.get("joint_angles")
                        if pose_ds is None:
                            logger.warning(
                                "Skipping %s: no pose dataset found", fp.name,
                            )
                            continue
                        targets = np.asarray(pose_ds[:], dtype=np.float32)
                        if targets.ndim == 1:
                            targets = targets.reshape(-1, 1)
                        elif targets.shape[1] != meta["n_outputs"]:
                            # Truncate / pad to the canonical DOF count
                            targets = targets[:, : meta["n_outputs"]]
                            if targets.shape[1] < meta["n_outputs"]:
                                pad = np.zeros(
                                    (targets.shape[0],
                                     meta["n_outputs"] - targets.shape[1]),
                                    dtype=np.float32,
                                )
                                targets = np.concatenate([targets, pad], axis=1)
                    else:  # EMG2Qwerty
                        ks_ds = (
                            f.get("keystrokes") or f.get("labels")
                            or f.get("key_labels")
                        )
                        if ks_ds is None:
                            logger.warning(
                                "Skipping %s: no keystrokes dataset found",
                                fp.name,
                            )
                            continue
                        ks = np.asarray(ks_ds[:])
                        if ks.ndim > 1:
                            ks = ks.ravel()
                        targets = ks.astype(np.int32)
            except Exception as e:  # pragma: no cover — defensive
                logger.warning("Skipping %s: %s", fp.name, e)
                continue

            # Align EMG and target lengths
            n = min(len(emg), len(targets))
            emg = emg[:n]
            targets = targets[:n]

            subj_id = self._subject_id_from_filename(
                fp.name, fallback=len(seen_subjects) + 1,
            )
            if subj_id in seen_subjects:
                continue
            seen_subjects.add(subj_id)

            emg_f = preprocess_signal(emg, fs=self.fs, notch_freq=50.0)

            if self.dataset == "EMG2Qwerty":
                # 1-D labels → segment_windows can label directly
                windows, wlabels = segment_windows(
                    emg_f, fs=self.fs, window_ms=self.window_ms,
                    increment_ms=self.increment_ms, labels=targets,
                )
                if windows is None or len(windows) == 0:
                    continue
                if wlabels is None:
                    wlabels = np.zeros(len(windows), dtype=np.int32)
                wlabels = np.asarray(wlabels, dtype=np.int32)
            else:  # EMG2Pose — multi-output continuous targets
                windows, _ = segment_windows(
                    emg_f, fs=self.fs, window_ms=self.window_ms,
                    increment_ms=self.increment_ms, labels=None,
                )
                if windows is None or len(windows) == 0:
                    continue
                # Compute per-window mean pose (mean pooling over samples
                # in the window — the standard emg2pose aggregation).
                wlabels = self._window_means(
                    targets, fs=self.fs, window_ms=self.window_ms,
                    increment_ms=self.increment_ms, n_windows=len(windows),
                )

            all_X.append(windows.astype(np.float32))
            all_y.append(np.asarray(wlabels))
            all_groups.extend([subj_id] * len(windows))

        if not all_X:
            raise RuntimeError(
                f"No usable HDF5 files were loaded from {self.root}"
            )

        X = np.concatenate(all_X, axis=0).astype(np.float32)
        if self.dataset == "EMG2Pose":
            y = np.concatenate(all_y, axis=0).astype(np.float32)
        else:
            y = np.concatenate(all_y, axis=0).astype(np.int32)
        groups = np.asarray(all_groups, dtype=np.int32)
        return X, y, groups, self._metadata(len(seen_subjects), synthetic=False)

    # ------------------------------------------------------------------
    # Synthetic fallback
    # ------------------------------------------------------------------
    def _load_synthetic(
        self, n_subjects: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """Generate deterministic synthetic data of the correct shape."""
        rng = np.random.default_rng(self.random_state)
        meta = DATASET_META[self.dataset]
        win_samples = int(self.fs * self.window_ms / 1000)
        t = np.arange(win_samples) / self.fs

        if self.dataset == "EMG2Pose":
            n_outputs = meta["n_outputs"] or 21
            wpd = _WINDOWS_PER_SUBJECT_POSE
            n_total = n_subjects * wpd
            X = np.zeros((n_total, self.n_channels, win_samples), dtype=np.float32)
            y = np.zeros((n_total, n_outputs), dtype=np.float32)
            groups = np.zeros(n_total, dtype=np.int32)
            idx = 0
            for subj in range(1, n_subjects + 1):
                for w in range(wpd):
                    base = rng.standard_normal(
                        (self.n_channels, win_samples),
                    ).astype(np.float32) * 0.05
                    # 5 characteristic muscle-activation frequencies
                    pattern = (0.4 + rng.random() * 0.4) * np.sin(
                        2 * np.pi * (4 + (w % 5)) * t
                    )
                    base += pattern[np.newaxis, :]
                    base += subj * 0.02  # subject-specific offset

                    # Pose: smooth low-frequency joint-angle signal (radians)
                    # mapped from a subset of channels so the data is
                    # learnable (not pure noise).
                    pose = np.zeros(n_outputs, dtype=np.float32)
                    for d in range(n_outputs):
                        ch = d % self.n_channels
                        phase = 0.1 * d + rng.random() * 0.5
                        amp = 0.5 * base[ch].mean()
                        pose[d] = float(
                            amp * np.sin(0.5 + phase) + subj * 0.01
                        )
                    X[idx] = base
                    y[idx] = pose
                    groups[idx] = subj
                    idx += 1
        else:  # EMG2Qwerty
            n_keystrokes = meta["n_outputs"] or _DEFAULT_KEYSTROKES
            wpd = _WINDOWS_PER_SUBJECT_QWERTY
            n_total = n_subjects * wpd
            X = np.zeros((n_total, self.n_channels, win_samples), dtype=np.float32)
            y = np.zeros(n_total, dtype=np.int32)
            groups = np.zeros(n_total, dtype=np.int32)
            idx = 0
            for subj in range(1, n_subjects + 1):
                for w in range(wpd):
                    k = w % n_keystrokes
                    base = rng.standard_normal(
                        (self.n_channels, win_samples),
                    ).astype(np.float32) * 0.05
                    # Each keystroke class has a characteristic temporal envelope
                    envelope = (0.4 + rng.random() * 0.4) * np.exp(
                        -((t - 0.1) ** 2) / (2 * (0.04 + 0.01 * k) ** 2)
                    )
                    base += envelope[np.newaxis, :] + k * 0.05
                    base += subj * 0.02
                    X[idx] = base
                    y[idx] = k
                    groups[idx] = subj
                    idx += 1

        return X, y, groups, self._metadata(n_subjects, synthetic=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _subject_id_from_filename(name: str, fallback: int) -> int:
        """Best-effort subject-ID extraction from an HDF5 filename."""
        m = re.search(r"(?:subject|subj|s)[_\-\s]?(\d+)", name, re.IGNORECASE)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                pass
        return fallback

    @staticmethod
    def _window_means(
        targets: np.ndarray,
        fs: int,
        window_ms: int,
        increment_ms: int,
        n_windows: int,
    ) -> np.ndarray:
        """Mean-pool multi-output targets over each window.

        Mirrors the windowing logic of ``segment_windows`` so that
        the i-th row of the returned array aligns with the i-th window.
        """
        win_samples = int(fs * window_ms / 1000)
        step_samples = int(fs * increment_ms / 1000)
        n_total = targets.shape[0]
        starts = np.arange(0, n_total - win_samples + 1, step_samples)
        # ``segment_windows`` pads short signals, so we may need extra windows
        if len(starts) < n_windows:
            extra = n_windows - len(starts)
            last = starts[-1] + step_samples if len(starts) else 0
            starts = np.concatenate([
                starts,
                np.arange(last, last + extra * step_samples, step_samples),
            ])
        starts = starts[:n_windows]
        out = np.zeros((n_windows, targets.shape[1]), dtype=np.float32)
        for i, s in enumerate(starts):
            end = min(s + win_samples, n_total)
            out[i] = targets[s:end].mean(axis=0) if end > s else 0.0
        return out

    def _metadata(self, n_subjects: int, synthetic: bool) -> Dict[str, Any]:
        meta = DATASET_META[self.dataset]
        return {
            "dataset": self.dataset,
            "fs": self.fs,
            "n_channels": self.n_channels,
            "window_ms": self.window_ms,
            "increment_ms": self.increment_ms,
            "n_subjects": n_subjects,
            "n_outputs": meta["n_outputs"],
            "task": meta["task"],
            "synthetic": synthetic,
            "win_samples": int(self.fs * self.window_ms / 1000),
        }


def list_meta_datasets() -> List[str]:
    """Return the list of supported Meta dataset identifiers."""
    return list(DATASET_META.keys())


def load_meta_dataset(
    dataset: str = "EMG2Pose",
    root: Union[str, Path] = "./data",
    n_subjects: Optional[int] = None,
    synthetic: bool = False,
    **kwargs: Any,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Unified dispatcher for Meta EMG datasets.

    Parameters
    ----------
    dataset : ``"EMG2Pose"`` or ``"EMG2Qwerty"``.
    root : path to the directory containing the ``.h5`` files. When the
        files are not found, the loader falls back to a synthetic
        dataset of the correct shape (the returned metadata dict's
        ``synthetic`` field is ``True`` in that case).
    n_subjects : cap on the number of subjects to load. ``None`` loads
        all available.
    synthetic : if ``True``, skip the on-disk lookup and go straight to
        the synthetic generator (useful for CI / unit tests / demos).
    **kwargs : forwarded to the :class:`MetaEMGLoader` constructor
        (e.g. ``fs``, ``n_channels``, ``window_ms``, ``increment_ms``,
        ``random_state``).

    Returns
    -------
    (X, y, groups, meta) : X has shape ``(n_windows, n_channels, win_samples)``.
        ``y`` is float32 ``(n_windows, 21)`` for EMG2Pose (continuous
        joint angles) and int32 ``(n_windows,)`` for EMG2Qwerty
        (keystroke labels). ``groups`` holds integer subject IDs. The
        meta dict includes a ``synthetic`` flag.
    """
    loader = MetaEMGLoader(root=root, dataset=dataset, **kwargs)
    return loader.load_split(n_subjects=n_subjects, synthetic=synthetic)
