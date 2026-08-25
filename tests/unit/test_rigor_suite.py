"""
Unit tests for reproducibility engine, fairness audit, and model card.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from myoadapt.reproducibility import (
    set_global_seed, environment_fingerprint, environment_hash,
    git_info, sha256_of_file, RunManifest, snapshot_config,
)
from myoadapt.evaluation.fairness import (
    per_subgroup_metrics, equalized_odds, fairness_summary,
)
from myoadapt.evaluation.model_card import model_card, write_model_card, datasheet
from myoadapt.evaluation.paper_reports import (
    main_results_table_tex, per_fold_table_tex, effect_size_table_tex,
    stats_table_tex, bca_ci_table_tex, write_figure_bundle,
)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def test_set_global_seed_returns_applied_dict():
    applied = set_global_seed(123)
    assert "python" in applied
    assert applied["python"] == 123
    assert "numpy" in applied


def test_environment_fingerprint_stable():
    fp1 = environment_fingerprint()
    fp2 = environment_fingerprint()
    assert fp1 == fp2


def test_environment_hash_is_sha256():
    h = environment_hash()
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


def test_environment_hash_changes_with_fingerprint():
    fp = environment_fingerprint()
    h1 = environment_hash(fp)
    fp_modified = dict(fp)
    fp_modified["python_version"] = "9.9.9"
    h2 = environment_hash(fp_modified)
    assert h1 != h2


def test_sha256_of_file_matches_hashlib():
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        f.write("hello world")
        path = f.name
    try:
        import hashlib
        expected = hashlib.sha256(b"hello world").hexdigest()
        assert sha256_of_file(path) == expected
    finally:
        os.unlink(path)


def test_run_manifest_write_and_verify():
    with tempfile.TemporaryDirectory() as d:
        # Create a fake output file.
        out_file = Path(d) / "model.pkl"
        out_file.write_bytes(b"fake model bytes")
        manifest = RunManifest(
            run_name="test_run",
            dataset="synthetic",
            model="random_forest",
            seed=42,
            config={"k_features": 420, "n_estimators": 100},
            repo_path=d,
        )
        manifest.add_output(out_file, role="model")
        manifest_path = Path(d) / "manifest.json"
        manifest.write(manifest_path)
        # File written.
        assert manifest_path.exists()
        # Load and verify round-trips.
        loaded = RunManifest.load(manifest_path)
        assert loaded["run_name"] == "test_run"
        assert loaded["dataset"] == "synthetic"
        assert loaded["outputs"][0]["sha256"] is not None
        # Verify hashes match.
        v = RunManifest.verify(manifest_path)
        assert v["verified"] is True
        assert v["checked"] == 1


def test_run_manifest_detects_tampering():
    with tempfile.TemporaryDirectory() as d:
        out_file = Path(d) / "model.pkl"
        out_file.write_bytes(b"original")
        manifest = RunManifest("test", "synthetic", "rf", seed=0, repo_path=d)
        manifest.add_output(out_file)
        manifest_path = Path(d) / "manifest.json"
        manifest.write(manifest_path)
        # Tamper with the file.
        out_file.write_bytes(b"tampered content")
        v = RunManifest.verify(manifest_path)
        assert v["verified"] is False
        assert len(v["mismatches"]) >= 1


def test_snapshot_config_writes_json():
    class FakeConfig:
        def __init__(self):
            self.k_features = 420
            self.ar_order = 4
            self._private = "should_not_appear"
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "config.json"
        snapshot_config(FakeConfig(), path)
        with open(path) as f:
            data = json.load(f)
        assert data["k_features"] == 420
        assert data["ar_order"] == 4
        assert "_private" not in data


# ---------------------------------------------------------------------------
# Fairness
# ---------------------------------------------------------------------------
def test_per_subgroup_metrics_returns_all_groups():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 3, 300)
    y_pred = rng.integers(0, 3, 300)
    groups = np.array(["A"] * 100 + ["B"] * 100 + ["C"] * 100)
    audit = per_subgroup_metrics(y_true, y_pred, groups)
    assert audit["n_subgroups"] == 3
    assert audit["n_samples"] == 300
    assert set(audit["per_subgroup"].keys()) == {"A", "B", "C"}


def test_per_subgroup_metrics_includes_calibration_when_proba_given():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 3, 100)
    y_pred = rng.integers(0, 3, 100)
    groups = rng.choice(["A", "B"], 100)
    y_proba = rng.dirichlet([1, 1, 1], 100)
    audit = per_subgroup_metrics(y_true, y_pred, groups, y_proba=y_proba)
    for entry in audit["per_subgroup"].values():
        assert "ece" in entry
        assert "brier" in entry


def test_disparities_report_worst_group():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 0, 1, 1, 0, 0])  # A: 100% acc, B: 50% acc
    groups = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])
    audit = per_subgroup_metrics(y_true, y_pred, groups, classes=[0, 1])
    assert audit["disparities"]["accuracy"]["worst_group"] == "B"
    assert audit["disparities"]["accuracy"]["best_group"] == "A"
    assert audit["disparities"]["accuracy"]["range"] == 0.5


def test_equalized_odds_returns_tpr_fpr():
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1, 1, 0, 1])
    groups = np.array(["A"] * 4 + ["B"] * 4)
    eo = equalized_odds(y_true, y_pred, groups, positive_class=1)
    assert "A" in eo and "B" in eo
    assert "tpr" in eo["A"] and "fpr" in eo["A"]
    assert 0.0 <= eo["A"]["tpr"] <= 1.0


def test_fairness_summary_returns_string():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 3, 100)
    y_pred = rng.integers(0, 3, 100)
    groups = rng.choice(["A", "B"], 100)
    audit = per_subgroup_metrics(y_true, y_pred, groups)
    s = fairness_summary(audit)
    assert isinstance(s, str)
    assert "Fairness audit" in s


# ---------------------------------------------------------------------------
# Model Card
# ---------------------------------------------------------------------------
def test_model_card_has_all_sections():
    md = model_card(
        "TestModel",
        intended_use="Test",
        evaluation={"protocol": "LOSO", "n_folds": 10,
                    "accuracy_mean": 0.7, "accuracy_std": 0.05,
                    "macro_f1_mean": 0.65, "macro_f1_std": 0.06},
    )
    for section in ("Intended use", "Out-of-scope", "Training data",
                    "Evaluation", "Limitations", "Ethical",
                    "Compliance"):
        assert section in md


def test_write_model_card_creates_file():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "model_card.md"
        write_model_card(path, model_name="X")
        assert path.exists()
        content = path.read_text()
        assert "# Model Card" in content


def test_datasheet_has_basic_fields():
    md = datasheet("NinaPro-DB2", n_subjects=40, n_windows=12000,
                    n_classes=50, fs=2000, n_channels=12)
    assert "NinaPro-DB2" in md
    assert "40" in md
    assert "Composition" in md


# ---------------------------------------------------------------------------
# Paper reports
# ---------------------------------------------------------------------------
def test_main_results_table_tex_renders():
    res = {
        "RF": {"accuracy_mean": 0.7, "accuracy_std": 0.05,
                "macro_f1_mean": 0.65, "macro_f1_std": 0.06},
        "LDA": {"accuracy_mean": 0.65, "accuracy_std": 0.06,
                 "macro_f1_mean": 0.60, "macro_f1_std": 0.07},
    }
    tex = main_results_table_tex(res)
    assert r"\begin{table}" in tex
    assert r"\end{table}" in tex
    assert "RF" in tex and "LDA" in tex
    assert r"\toprule" in tex and r"\bottomrule" in tex


def test_per_fold_table_tex_includes_mean_column():
    folds = {"RF": [0.7, 0.6, 0.8], "LDA": [0.65, 0.55, 0.7]}
    tex = per_fold_table_tex(folds)
    assert "Mean" in tex
    assert "RF" in tex


def test_effect_size_table_tex_includes_hedges_g():
    rng = np.random.default_rng(0)
    folds = {
        "RF": rng.normal(0.7, 0.05, 15).tolist(),
        "LDA": rng.normal(0.65, 0.06, 15).tolist(),
    }
    tex = effect_size_table_tex(folds)
    assert "Hedges" in tex
    assert "Cohen" in tex


def test_stats_table_tex_includes_friedman():
    rng = np.random.default_rng(0)
    folds = {
        "RF": rng.normal(0.7, 0.05, 15).tolist(),
        "LDA": rng.normal(0.65, 0.06, 15).tolist(),
        "XGB": rng.normal(0.71, 0.04, 15).tolist(),
    }
    tex = stats_table_tex(folds)
    assert "Friedman" in tex
    assert "Nemenyi" in tex


def test_bca_ci_table_tex_includes_bounds():
    rng = np.random.default_rng(0)
    folds = {"RF": rng.normal(0.7, 0.05, 30).tolist()}
    tex = bca_ci_table_tex(folds, n_bootstrap=100)
    assert "CI low" in tex or "95" in tex


def test_write_figure_bundle_creates_png_pdf_tex():
    import matplotlib
    matplotlib.use("Agg")
    rng = np.random.default_rng(0)
    folds = {
        "RF": rng.normal(0.7, 0.05, 12).tolist(),
        "LDA": rng.normal(0.65, 0.06, 12).tolist(),
    }
    with tempfile.TemporaryDirectory() as d:
        written = write_figure_bundle(folds, Path(d),
                                       figures=("cd", "boxplot"))
        assert "cd_diagram" in written
        assert "per_fold_boxplot" in written
        cd_png = Path(d) / "cd_diagram.png"
        cd_pdf = Path(d) / "cd_diagram.pdf"
        cd_tex = Path(d) / "cd_diagram.tex"
        assert cd_png.exists() and cd_pdf.exists() and cd_tex.exists()
        assert cd_png.stat().st_size > 0


# ---------------------------------------------------------------------------
# Classical model aliases
# ---------------------------------------------------------------------------
def test_get_model_accepts_classical_aliases():
    from myoadapt.models import get_model
    for alias in ("xgboost", "random_forest", "lda", "svm",
                  "logistic", "extra_trees", "lightgbm"):
        m = get_model(alias)
        assert m is not None
        assert m.model_type == alias


def test_list_models_includes_aliases():
    from myoadapt.models import list_models
    names = list_models()
    for alias in ("xgboost", "random_forest", "lda", "svm",
                  "logistic", "extra_trees", "lightgbm",
                  "classical", "cnn1d", "lite_dan", "emg_foundation"):
        assert alias in names
