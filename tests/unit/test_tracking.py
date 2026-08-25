"""Unit tests for the tracking module.

Covers LocalTracker (JSON-backed) end-to-end: set_experiment → start_run →
log_params / log_metric / log_artifact → end_run → list_runs → get_best_run.

The MLflow adapter test is skipped because MLflow requires a running server
(or at minimum a non-trivial local store setup) and we don't want that as
a hard dependency in CI.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from myoadapt.tracking.local_tracker import LocalTracker


def test_local_tracker_full_round_trip(tmp_path: Path) -> None:
    """Exercise the full LocalTracker lifecycle and verify the JSON file
    on disk contains exactly what we logged."""
    tracker = LocalTracker(root_dir=tmp_path)
    tracker.set_experiment("exp_round_trip")
    tracker.start_run(run_name="run_alpha")
    tracker.log_params({"db": "DB2", "model": "rf", "n_features": 16})
    tracker.log_metric("accuracy", 0.85)
    tracker.log_metric("accuracy", 0.88, step=1)  # second value → time series
    tracker.log_metric("macro_f1", 0.82)
    tracker.log_artifact("/tmp/some-model.pkl")
    out_path = tracker.end_run(status="completed")

    # --- The JSON file should exist and be valid --------------------------
    assert out_path.exists()
    with open(out_path) as f:
        payload = json.load(f)

    assert payload["run_id"] == "run_alpha"
    assert payload["experiment"] == "exp_round_trip"
    assert payload["status"] == "completed"
    assert payload["params"] == {"db": "DB2", "model": "rf", "n_features": 16}
    assert payload["metrics"]["accuracy"][0]["value"] == 0.85
    assert payload["metrics"]["accuracy"][1]["value"] == 0.88
    assert payload["metrics"]["macro_f1"][0]["value"] == 0.82
    assert "/tmp/some-model.pkl" in payload["artifacts"]
    assert "start_time" in payload
    assert "end_time" in payload
    assert payload["elapsed_seconds"] >= 0.0

    # --- list_runs should pick up the run we just wrote -------------------
    runs = tracker.list_runs()
    assert len(runs) == 1
    assert runs[0]["run_id"] == "run_alpha"

    # --- get_best_run should return this run for the accuracy metric -----
    best = tracker.get_best_run("accuracy", mode="max")
    assert best is not None
    assert best["run_id"] == "run_alpha"


def test_local_tracker_multiple_runs(tmp_path: Path) -> None:
    """Two runs in the same experiment must both be visible via list_runs."""
    tracker = LocalTracker(root_dir=tmp_path)
    tracker.set_experiment("exp_multi")

    for name, acc in [("run1", 0.7), ("run2", 0.9), ("run3", 0.5)]:
        tracker.start_run(run_name=name)
        tracker.log_metric("accuracy", acc)
        tracker.end_run()

    runs = tracker.list_runs()
    assert len(runs) == 3
    best = tracker.get_best_run("accuracy", mode="max")
    assert best is not None
    assert best["run_id"] == "run2"  # highest accuracy

    worst = tracker.get_best_run("accuracy", mode="min")
    assert worst is not None
    assert worst["run_id"] == "run3"  # lowest accuracy


def test_local_tracker_default_experiment(tmp_path: Path) -> None:
    """If set_experiment is not called, start_run must auto-create 'default'."""
    tracker = LocalTracker(root_dir=tmp_path)
    tracker.start_run(run_name="auto")
    tracker.log_metric("loss", 1.23)
    tracker.end_run()

    runs = tracker.list_runs()
    assert len(runs) == 1
    assert runs[0]["experiment"] == "default"
    # And the file should live under ./default/
    assert (tmp_path / "default" / "auto.json").exists()


def test_local_tracker_list_runs_empty(tmp_path: Path) -> None:
    """list_runs on an experiment with no runs returns an empty list."""
    tracker = LocalTracker(root_dir=tmp_path)
    tracker.set_experiment("empty_exp")
    assert tracker.list_runs() == []
    # And when no experiment has been set at all, returns empty list too.
    fresh = LocalTracker(root_dir=tmp_path / "fresh")
    assert fresh.list_runs() == []


@pytest.mark.skip(reason="MLflow adapter requires a running MLflow server; "
                  "covered in a separate integration environment.")
def test_mlflow_tracker_smoke(tmp_path: Path) -> None:
    """Smoke test for MLflowTracker. Skipped by default — set the
    ``MYOCONTROL_MLFLOW_URI`` env var to enable in CI."""
    from myoadapt.tracking.mlflow_adapter import MLflowTracker

    tracker = MLflowTracker(tracking_uri=f"file://{tmp_path}/mlruns")
    tracker.set_experiment("ci_smoke")
    tracker.start_run(run_name="ci_run")
    tracker.log_params({"db": "DB2"})
    tracker.log_metric("accuracy", 0.5)
    tracker.end_run()
