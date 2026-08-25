"""Unit tests for myoadapt.data.meta_loaders (emg2pose / emg2qwerty)."""
from __future__ import annotations

import numpy as np
import pytest

from myoadapt.data.meta_loaders import MetaEMGLoader, list_meta_datasets, load_meta_dataset, DATASET_META


def test_list_meta_datasets():
    names = list_meta_datasets()
    assert "EMG2Pose" in names and "EMG2Qwerty" in names


def test_unknown_dataset_raises():
    with pytest.raises(ValueError):
        MetaEMGLoader(dataset="NotARealDataset")


@pytest.mark.parametrize("name", ["EMG2Pose", "EMG2Qwerty"])
def test_synthetic_fallback_shapes(name):
    X, y, groups, meta = load_meta_dataset(dataset=name, root="/nonexistent-path",
                                            n_subjects=2, synthetic=True)
    n_ch = DATASET_META[name]["n_channels"]
    assert X.ndim == 3 and X.shape[1] == n_ch
    assert X.shape[0] == y.shape[0]
    assert not np.isnan(X).any()
    assert meta["synthetic"] is True
    assert set(np.unique(groups).tolist()) == {1, 2}


def test_emg2pose_labels_are_continuous_21dof():
    X, y, groups, meta = load_meta_dataset(dataset="EMG2Pose", root="/nonexistent",
                                            n_subjects=1, synthetic=True)
    assert y.shape[1] == 21
    assert y.dtype == np.float32


def test_emg2qwerty_labels_are_discrete_keystrokes():
    X, y, groups, meta = load_meta_dataset(dataset="EMG2Qwerty", root="/nonexistent",
                                            n_subjects=1, synthetic=True)
    assert y.ndim == 1
    assert np.issubdtype(y.dtype, np.integer)


def test_n_subjects_cap_is_respected():
    X, y, groups, meta = load_meta_dataset(dataset="EMG2Pose", root="/nonexistent",
                                            n_subjects=1, synthetic=True)
    assert set(np.unique(groups).tolist()) == {1}


def test_real_load_falls_back_to_synthetic_when_no_files_present(tmp_path):
    """synthetic=False (the default) should still degrade gracefully to
    synthetic data when no .h5 files exist, rather than raising."""
    X, y, groups, meta = load_meta_dataset(dataset="EMG2Pose", root=str(tmp_path), n_subjects=1)
    assert meta["synthetic"] is True
    assert X.shape[0] > 0
