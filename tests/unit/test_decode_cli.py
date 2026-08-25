"""Functional tests for `myoadapt decode` — verifies it routes on the
dataset's declared task (regression vs classification) instead of always
forcing a regression head. Regression test for a real bug: EMG2Qwerty
(keystroke classification) used to be trained as continuous regression,
which ran without error but was semantically wrong.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from myoadapt.cli.main import app

runner = CliRunner()
pytestmark = pytest.mark.slow


def test_decode_emg2pose_defaults_to_regression(tmp_path):
    out = tmp_path / "pose.pt"
    result = runner.invoke(app, [
        "decode", "--dataset", "EMG2Pose", "--data-root", "/nonexistent-path",
        "--n-subjects", "2", "--n-epochs", "2", "--output", str(out),
    ])
    assert result.exit_code == 0, result.output
    assert "(regression)" in result.output
    assert "r2" in result.output or "mse" in result.output
    assert out.exists()


def test_decode_emg2qwerty_defaults_to_classification(tmp_path):
    out = tmp_path / "qwerty.pt"
    result = runner.invoke(app, [
        "decode", "--dataset", "EMG2Qwerty", "--data-root", "/nonexistent-path",
        "--n-subjects", "2", "--output", str(out),
    ])
    assert result.exit_code == 0, result.output
    assert "(classification)" in result.output
    assert "accuracy" in result.output
    assert "macro_f1" in result.output
    # classification saves .pkl (EMGClassifier), not the raw .pt path
    assert Path(str(out).replace(".pt", ".pkl")).exists()


def test_decode_rejects_regression_model_for_classification_dataset():
    result = runner.invoke(app, [
        "decode", "--dataset", "EMG2Qwerty", "--data-root", "/nonexistent-path",
        "--n-subjects", "2", "--model", "pose_transformer",
    ])
    assert result.exit_code != 0
    normalized = " ".join(result.output.split())
    assert "classification task" in normalized


def test_decode_rejects_classification_model_for_regression_dataset():
    result = runner.invoke(app, [
        "decode", "--dataset", "EMG2Pose", "--data-root", "/nonexistent-path",
        "--n-subjects", "2", "--model", "xgboost",
    ])
    assert result.exit_code != 0
    normalized = " ".join(result.output.split())
    assert "regression task" in normalized
