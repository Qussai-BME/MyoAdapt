"""Functional tests for `myoadapt zeroshot` and `myoadapt robustness` —
wire ZeroShotEvaluator and ElectrodeShiftRobustness into the CLI for the
first time (previously tested but Python-API only, per the punch list).
"""
from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from myoadapt.cli.main import app

runner = CliRunner()
pytestmark = pytest.mark.slow


def test_zeroshot_end_to_end(tmp_path):
    out = tmp_path / "zeroshot.json"
    result = runner.invoke(app, [
        "zeroshot", "--db", "DB2", "--data-root", "/nonexistent-path",
        "--n-train-subjects", "4", "--n-test-subjects", "2",
        "--max-windows-per-subject", "82",
        "--output", str(out),
    ])
    assert result.exit_code == 0, result.output
    assert "Worst subject" in result.output
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["n_train_subjects"] == 4
    assert data["n_test_subjects"] == 2


def test_zeroshot_compare_loso(tmp_path):
    out = tmp_path / "zeroshot_loso.json"
    result = runner.invoke(app, [
        "zeroshot", "--db", "DB2", "--data-root", "/nonexistent-path",
        "--model", "logistic",
        "--n-train-subjects", "3", "--n-test-subjects", "3",
        "--max-windows-per-subject", "82", "--compare-loso", "--output", str(out),
    ])
    assert result.exit_code == 0, result.output
    assert "Gap:" in result.output


def test_zeroshot_only_processes_the_requested_subject_pool(tmp_path):
    """Regression test: the command used to extract features on the full
    default dataset (40 subjects) regardless of pool size, making
    --n-train-subjects/--n-test-subjects have no effect on runtime."""
    out = tmp_path / "zeroshot.json"
    result = runner.invoke(app, [
        "zeroshot", "--db", "DB2", "--data-root", "/nonexistent-path",
        "--n-train-subjects", "3", "--n-test-subjects", "2",
        "--max-windows-per-subject", "82",
        "--output", str(out),
    ])
    assert result.exit_code == 0, result.output
    assert "using 5 (3 train pool / 2 test pool)" in result.output


def test_robustness_end_to_end(tmp_path):
    from typer.testing import CliRunner as _CR
    train_out = tmp_path / "model.pkl"
    train_result = runner.invoke(app, [
        "train", "--db", "DB2", "--data-root", "/nonexistent-path",
        "--model", "random_forest", "--n-subjects", "2",
        "--output", str(train_out),
    ])
    assert train_result.exit_code == 0, train_result.output

    report_out = tmp_path / "robustness.json"
    result = runner.invoke(app, [
        "robustness", "--model-path", str(train_out), "--db", "DB2",
        "--data-root", "/nonexistent-path", "--n-subjects", "1",
        "--max-shift", "1", "--output", str(report_out),
    ])
    assert result.exit_code == 0, result.output
    assert "worst_shift_type" in result.output
    assert report_out.exists()
    data = json.loads(report_out.read_text())
    assert set(data.keys()) == {"circular", "dropout", "swap", "noise", "summary"}
