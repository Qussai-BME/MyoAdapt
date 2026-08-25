"""Unit tests for splits and config."""
import numpy as np
import pytest
from pathlib import Path

from myoadapt.data.splits import (
    loso_splits, lodo_splits, kfold_splits, train_test_split_subject,
)
from myoadapt.config import MyoAdaptConfig, FilterConfig, WindowConfig


def test_loso_splits():
    groups = np.array([1, 1, 2, 2, 3, 3])
    splits = list(loso_splits(groups))
    assert len(splits) == 3
    for train_idx, test_idx in splits:
        assert len(train_idx) == 4
        assert len(test_idx) == 2


def test_lodo_splits():
    dbs = np.array([1, 1, 2, 2, 3, 3])
    splits = list(lodo_splits(dbs))
    assert len(splits) == 3


def test_train_test_split_subject():
    groups = np.array([1, 1, 2, 2, 3, 3, 4, 4])
    train, test = train_test_split_subject(groups, test_size=0.5, random_state=42)
    assert len(train) + len(test) == 8
    # No leakage
    train_subjects = set(groups[train])
    test_subjects = set(groups[test])
    assert train_subjects.isdisjoint(test_subjects)


def test_config_validation():
    cfg = MyoAdaptConfig()
    assert cfg.validate() is True


def test_config_yaml_roundtrip(tmp_path):
    cfg = MyoAdaptConfig()
    cfg.experiment_name = "test_exp"
    path = tmp_path / "config.yaml"
    cfg.to_yaml(path)
    loaded = MyoAdaptConfig.from_yaml(path)
    assert loaded.experiment_name == "test_exp"


def test_window_config_overlap():
    w = WindowConfig(window_ms=200, increment_ms=50)
    assert w.overlap == pytest.approx(0.75)


def test_filter_config_invalid():
    with pytest.raises(ValueError):
        FilterConfig(cutoff_low=600, cutoff_high=450).validate()
