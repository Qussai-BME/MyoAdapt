"""Functional tests for the `myoadapt retrieve` and `myoadapt augment`
commands — these wire semantic_retrieval.py and synthetic_augmentation.py
into the CLI for the first time (previously tested but Python-API only).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from myoadapt.cli.main import app

runner = CliRunner()
pytestmark = pytest.mark.slow


def test_retrieve_end_to_end(tmp_path):
    out = tmp_path / "retriever.pt"
    result = runner.invoke(app, [
        "retrieve", "--db", "DB2", "--data-root", "/nonexistent-path",
        "--n-subjects", "3", "--n-epochs", "5", "--output", str(out),
    ])
    assert result.exit_code == 0, result.output
    assert "recall@1" in result.output
    assert out.exists()


def test_retrieve_with_labels_csv_and_query(tmp_path):
    csv_path = tmp_path / "labels.csv"
    csv_path.write_text("gesture_id,description\n0,rest\n1,open hand\n2,close fist\n")
    out = tmp_path / "retriever.pt"
    result = runner.invoke(app, [
        "retrieve", "--db", "DB2", "--data-root", "/nonexistent-path",
        "--n-subjects", "2", "--n-epochs", "5", "--labels-csv", str(csv_path),
        "--query", "open hand", "--output", str(out),
    ])
    assert result.exit_code == 0, result.output
    assert "open hand" in result.output
    assert "Top-5 EMG windows" in result.output


def test_augment_gan_single_gesture(tmp_path):
    out = tmp_path / "augmented.npz"
    result = runner.invoke(app, [
        "augment", "--db", "DB2", "--data-root", "/nonexistent-path",
        "--method", "gan", "--gesture", "3", "--n-generate", "10",
        "--n-epochs", "5", "--output", str(out),
    ])
    assert result.exit_code == 0, result.output
    assert out.exists()
    data = np.load(out)
    assert set(data.keys()) == {"real", "generated", "real_labels", "generated_labels"}
    assert data["generated"].shape[0] == 10
    assert not np.isnan(data["generated"]).any()
    assert (data["real_labels"] == 3).all()


def test_augment_unknown_gesture_errors_cleanly():
    result = runner.invoke(app, [
        "augment", "--db", "DB2", "--data-root", "/nonexistent-path",
        "--method", "gan", "--gesture", "9999", "--n-generate", "5", "--n-epochs", "2",
    ])
    assert result.exit_code != 0
    assert "No windows found" in result.output


def test_augment_unknown_method_errors_cleanly():
    result = runner.invoke(app, [
        "augment", "--db", "DB2", "--data-root", "/nonexistent-path",
        "--method", "not_a_real_method", "--n-generate", "5", "--n-epochs", "2",
    ])
    assert result.exit_code != 0
