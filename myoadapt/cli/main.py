"""
main.py — Typer CLI for MyoAdapt v2.0
========================================

Commands:
    myoadapt train       — train a model
    myoadapt eval        — evaluate a model with LOSO/LODO
    myoadapt export      — export to ONNX
    myoadapt serve       — start REST API
    myoadapt stream      — start WebSocket stream server
    myoadapt benchmark   — run 5-CPU hardware benchmark
    myoadapt explain     — generate SHAP report for one sample
    myoadapt ui          — launch Streamlit UI
    myoadapt info        — show platform info
    myoadapt verify      — verify a MiniROCKET diagnostic

"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from myoadapt import __version__

app = typer.Typer(
    name="myoadapt",
    help="Open-Source CPU-Native sEMG Pattern Recognition Platform",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
console = Console()


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )


@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", "-v")):
    """MyoAdapt v2.0 CLI."""
    setup_logging(verbose)


@app.command()
def info():
    """Show platform information."""
    console.print(f"[bold cyan]MyoAdapt v{__version__}[/bold cyan]")
    console.print("Open-Source CPU-Native sEMG Pattern Recognition Platform")
    console.print()

    table = Table(title="Available Modules")
    table.add_column("Module", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Description")

    checks = [
        ("numpy", "Core arrays"),
        ("scipy", "Filtering / statistics"),
        ("sklearn", "Classical ML"),
        ("torch", "Deep learning (CNN1D, LiteDAN, EMG-FM)"),
        ("xgboost", "XGBoost classifier"),
        ("onnx", "ONNX export"),
        ("onnxruntime", "ONNX inference"),
        ("shap", "Explainability"),
        ("fastapi", "REST API"),
        ("streamlit", "Web UI"),
        ("mlflow", "Experiment tracking"),
    ]
    for mod, desc in checks:
        try:
            __import__(mod)
            status = "[green]✓ installed[/green]"
        except ImportError:
            status = "[red]✗ not installed[/red]"
        table.add_row(mod, status, desc)
    console.print(table)

    # Models
    from myoadapt.models.base import MODEL_REGISTRY
    console.print(f"\n[bold]Registered models[/bold]: {list(MODEL_REGISTRY.keys())}")

    # Datasets
    from myoadapt.data.loaders import list_databases
    console.print(f"[bold]Supported datasets[/bold]: {list_databases()}")


@app.command()
def train(
    db: str = typer.Option("DB2", "--db", help="Dataset (DB1/DB2/DB3/DB7/CapgMyo-DBa/UCI)"),
    exercise: str = typer.Option("E1", "--exercise", help="NinaPro exercise selector (E1/E2/E3)"),
    model: str = typer.Option("xgboost", "--model", help="Model name"),
    data_root: str = typer.Option("./data", "--data-root"),
    output: str = typer.Option("./models/model.pkl", "--output"),
    k_features: int = typer.Option(420, "--k-features"),
    n_epochs: int = typer.Option(50, "--epochs"),
    device: str = typer.Option("auto", "--device"),
    config: Optional[str] = typer.Option(None, "--config", help="YAML config file"),
    n_subjects: Optional[int] = typer.Option(
        None, "--n-subjects",
        help="Limit to the first N subjects (useful for a quick smoke test — "
             "full default-scale synthetic data is ~16k windows and feature "
             "extraction alone takes several minutes)."),
):
    """Train a model on a dataset.

    Loads data from ``--data-root`` (falls back to synthetic data of the
correct shape when the real .mat files are not present), preprocesses
    each window, extracts the standard feature set, and trains the
    selected classifier. The trained model is saved to ``--output`` and
    a tracking record is logged via LocalTracker.
    """
    from myoadapt.config import MyoAdaptConfig
    from myoadapt.data.loaders import load_dataset
    from myoadapt.data.preprocessing import preprocess_signal
    from myoadapt.features import extract_features
    from myoadapt.models import get_model
    from myoadapt.tracking.local_tracker import LocalTracker

    console.print(f"[bold cyan]Training {model} on {db}[/bold cyan]")
    cfg = MyoAdaptConfig.from_yaml(config) if config else MyoAdaptConfig()
    cfg.validate()

    # Load data. ``load_dataset`` returns (X, y, groups, meta) where X has
    # shape (n_windows, n_channels, win_samples). When the real .mat files
    # are not present, a synthetic dataset of the correct shape is returned
    # and ``meta["synthetic"]`` is True.
    try:
        windows, labels, groups_arr, meta = load_dataset(db=db, root=data_root)
    except FileNotFoundError as e:
        console.print(f"[red]Data not found:[/red] {e}")
        raise typer.Exit(1)
    except ValueError as e:
        console.print(f"[red]Dataset error:[/red] {e}")
        raise typer.Exit(1)

    synthetic_note = " [yellow](synthetic fallback)[/yellow]" if meta.get("synthetic") else ""
    console.print(f"Loaded {meta['n_subjects']} subjects, "
                  f"{windows.shape[0]} windows{synthetic_note}")

    if n_subjects is not None:
        keep_ids = sorted(set(np.asarray(groups_arr).tolist()))[:n_subjects]
        mask = np.isin(groups_arr, keep_ids)
        windows, labels, groups_arr = windows[mask], np.asarray(labels)[mask], np.asarray(groups_arr)[mask]
        console.print(f"--n-subjects {n_subjects}: reduced to {windows.shape[0]} windows")
    elif windows.shape[0] > 3000:
        console.print(f"[dim]{windows.shape[0]} windows — feature extraction "
                       f"(MiniROCKET-based, ~20ms/window) will take a few minutes. "
                       f"Use --n-subjects for a quick smoke test.[/dim]")

    # Preprocess + (optional) feature-extract. Classical and Lite-DAN models
    # operate on hand-crafted features; CNN1D and EMG-FM consume raw 3-D
    # windows directly. We branch on the model family so each receives the
    # input shape it expects.
    y = np.asarray(labels)
    groups_arr = np.asarray(groups_arr)
    n_classes = int(len(np.unique(y)))

    needs_features = model in (
        "xgboost", "random_forest", "lda", "svm", "logistic",
        "extra_trees", "lightgbm", "lite_dan",
    )
    needs_raw_windows = model in ("cnn1d", "emg_foundation")

    if needs_features:
        filtered = np.empty_like(windows, dtype=np.float64)
        for i in range(windows.shape[0]):
            # windows[i] has shape (n_channels, win_samples) — preprocess_signal
            # expects (n_samples, n_channels), so transpose.
            sig = windows[i].T
            sig_filtered = preprocess_signal(
                sig, fs=cfg.filter.sampling_rate,
                low=cfg.filter.cutoff_low,
                high=cfg.filter.cutoff_high,
                order=cfg.filter.filter_order,
                notch_freq=cfg.filter.notch_freq,
            )
            filtered[i] = sig_filtered.T  # back to (n_channels, win_samples)
        # One batched call instead of one per window — extract_features
        # already accepts (n_windows, n_channels, win_samples) directly;
        # calling it per-window was pure avoidable overhead.
        X = extract_features(filtered,
                              modules=["time_domain", "frequency_domain",
                                       "histogram", "correlation"],
                              fs=cfg.filter.sampling_rate,
                              ar_order=cfg.features.ar_order)
        console.print(f"Features: {X.shape}, classes: {n_classes}")
    elif needs_raw_windows:
        # Re-filter each window but keep the 3-D (n_windows, n_channels,
        # win_samples) layout the deep models expect.
        filtered = np.empty_like(windows, dtype=np.float32)
        for i in range(windows.shape[0]):
            sig = windows[i].T
            sig_f = preprocess_signal(
                sig, fs=cfg.filter.sampling_rate,
                low=cfg.filter.cutoff_low,
                high=cfg.filter.cutoff_high,
                order=cfg.filter.filter_order,
                notch_freq=cfg.filter.notch_freq,
            )
            filtered[i] = sig_f.T.astype(np.float32)
        X = filtered
        console.print(f"Raw windows: {X.shape}, classes: {n_classes}")
    else:
        console.print(f"[red]Unknown model: {model}[/red]")
        raise typer.Exit(1)

    # Train model
    if model in ("xgboost", "random_forest", "lda", "svm", "logistic", "extra_trees", "lightgbm"):
        from myoadapt.models.classical import EMGClassifier
        clf = EMGClassifier(model_type=model, k_features=k_features)
        clf.fit(X, y)
        clf.save(output)
    elif model == "lite_dan":
        n_features = int(X.shape[1])
        n_domains = int(len(np.unique(groups_arr)))
        m = get_model(
            model,
            n_features=n_features,
            n_classes=n_classes,
            n_domains=n_domains,
            n_epochs=n_epochs,
            device=device,
        )
        m.fit(X, y, groups=groups_arr)
        m.save(output)
    elif model in ("cnn1d", "emg_foundation"):
        n_channels = int(X.shape[1])
        n_samples = int(X.shape[2])
        kwargs = {"n_channels": n_channels, "n_classes": n_classes,
                  "n_samples": n_samples, "n_epochs": n_epochs, "device": device}
        if model == "emg_foundation":
            # EMGFoundation exposes ssl_pretrain_epochs / finetune_epochs,
            # not the single n_epochs kwarg. Map n_epochs onto fine-tuning
            # (the supervised stage that produces a usable classifier).
            kwargs = {
                "n_channels": n_channels,
                "n_samples": n_samples,
                "n_classes": n_classes,
                "finetune_epochs": n_epochs,
                "device": device,
            }
        m = get_model(model, **kwargs)
        m.fit(X, y)
        m.save(output)
    else:
        console.print(f"[red]Unknown model: {model}[/red]")
        raise typer.Exit(1)

    console.print(f"[green]✓ Model saved to {output}[/green]")

    # Log
    tracker = LocalTracker()
    tracker.set_experiment(f"{db}_{exercise}")
    tracker.start_run(run_name=f"{model}_{db}")
    tracker.log_params({"db": db, "model": model, "n_samples": X.shape[0],
                          "n_features": X.shape[1]})
    tracker.end_run()


@app.command()
def eval(
    model_path: str = typer.Option(..., "--model-path"),
    db: str = typer.Option("DB2", "--db"),
    exercise: str = typer.Option("E1", "--exercise"),
    data_root: str = typer.Option("./data", "--data-root"),
    protocol: str = typer.Option("loso", "--protocol", help="loso or lodo"),
    output: str = typer.Option("./results", "--output"),
):
    """Evaluate a trained model with LOSO/LODO."""
    from myoadapt.data.loaders import load_dataset
    from myoadapt.data.preprocessing import preprocess_signal
    from myoadapt.evaluation import LOSOEvaluator, ReportGenerator
    from myoadapt.features import extract_features
    from myoadapt.models.classical import EMGClassifier

    console.print(f"[bold cyan]Evaluating {model_path} on {db} ({protocol})[/bold cyan]")
    clf = EMGClassifier.load(model_path)

    windows, labels, groups_arr, meta = load_dataset(db=db, root=data_root)
    filtered = np.empty_like(windows, dtype=np.float64)
    for i in range(windows.shape[0]):
        sig = windows[i].T
        sig_filtered = preprocess_signal(sig, fs=meta["fs"])
        filtered[i] = sig_filtered.T
    X = extract_features(filtered, fs=meta["fs"])
    y = np.asarray(labels)
    groups_arr = np.asarray(groups_arr)

    def factory():
        return EMGClassifier.load(model_path)

    evaluator = LOSOEvaluator(model_factory=factory, verbose=True)
    results = evaluator.run(X, y, groups_arr)

    gen = ReportGenerator(output)
    gen.generate_tables({model_path: results}, dataset_name=db)
    gen.generate_summary_markdown({model_path: results}, dataset_name=db)

    console.print(f"[green]✓ Results saved to {output}/[/green]")
    agg = results["aggregate"]
    console.print(f"Accuracy: {agg['accuracy_mean']:.4f} ± {agg['accuracy_std']:.4f}")
    console.print(f"Macro-F1: {agg['macro_f1_mean']:.4f} ± {agg['macro_f1_std']:.4f}")


@app.command()
def export(
    model_path: str = typer.Option(..., "--model-path"),
    output: str = typer.Option("model.onnx", "--output"),
    feature_names: Optional[str] = typer.Option(None, "--feature-names"),
):
    """Export a trained model to ONNX."""
    from myoadapt.deployment.onnx_export import OnnxExporter
    from myoadapt.models.classical import EMGClassifier

    clf = EMGClassifier.load(model_path)
    names = None
    if feature_names:
        with open(feature_names) as f:
            names = [line.strip() for line in f]
    info = OnnxExporter.export(clf, output, feature_names=names)
    console.print(f"[green]✓ Exported to {info['path']}[/green]")
    console.print(f"  Size: {info['size_bytes']:,} bytes")
    console.print(f"  Type: {info['model_type']}")
    # Verify
    verification = OnnxExporter.verify(output)
    console.print(f"  Verification: {'passed' if verification['test_passed'] else 'failed'}")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind address; use a gateway for public exposure."),
    port: int = typer.Option(8000, "--port"),
):
    """Start the REST API server."""
    import uvicorn

    from myoadapt.api.rest import create_app
    app_obj = create_app()
    console.print(f"[bold cyan]Starting API on {host}:{port}[/bold cyan]")
    console.print(f"  Docs: http://{host}:{port}/docs")
    uvicorn.run(app_obj, host=host, port=port)


@app.command()
def stream(
    model_path: str = typer.Option(..., "--model-path"),
    host: str = typer.Option("127.0.0.1", "--host", help="Bind address; use a gateway for public exposure."),
    port: int = typer.Option(8001, "--port"),
):
    """Start the WebSocket streaming server."""
    import uvicorn

    from myoadapt.api.websocket import create_ws_app
    app_obj = create_ws_app(model_path=model_path)
    console.print(f"[bold cyan]Starting WebSocket on {host}:{port}[/bold cyan]")
    console.print(f"  Endpoint: ws://{host}:{port}/ws/stream")
    uvicorn.run(app_obj, host=host, port=port)


@app.command()
def benchmark(
    model_path: str = typer.Option(..., "--model-path"),
    n_runs: int = typer.Option(100, "--n-runs"),
    output: Optional[str] = typer.Option(None, "--output"),
):
    """Run the 5-CPU hardware benchmark."""
    from myoadapt.deployment.hardware_bench import HardwareBenchmark
    from myoadapt.models.classical import EMGClassifier

    clf = EMGClassifier.load(model_path)
    bench = HardwareBenchmark(clf, n_runs=n_runs)
    results = bench.run()

    console.print(bench.markdown_table(results))
    if output:
        bench.save_report(results, output)
        console.print(f"[green]✓ Report saved to {output}[/green]")


@app.command()
def explain(
    model_path: str = typer.Option(..., "--model-path"),
    features_file: str = typer.Option(..., "--features", help="JSON or CSV with features"),
    output: str = typer.Option("./reports", "--output"),
):
    """Generate a SHAP transparency report for one sample."""
    import json

    import numpy as np

    from myoadapt.deployment.shap_reports import ShapReportGenerator
    from myoadapt.models.classical import EMGClassifier

    clf = EMGClassifier.load(model_path)
    gen = ShapReportGenerator(clf)

    # Load features
    if features_file.endswith(".json"):
        with open(features_file) as f:
            data = json.load(f)
        features = np.asarray(data["features"], dtype=np.float32)
    else:
        import pandas as pd
        df = pd.read_csv(features_file)
        features = df.values.astype(np.float32)

    if features.ndim == 1:
        features = features.reshape(1, -1)

    reports = gen.batch_explain(features, output)
    console.print(f"[green]✓ Generated {len(reports)} reports in {output}/[/green]")


@app.command()
def ui(
    port: int = typer.Option(8501, "--port"),
):
    """Launch the Streamlit UI."""
    import subprocess
    from pathlib import Path
    ui_path = Path(__file__).parent.parent / "ui" / "app.py"
    console.print(f"[bold cyan]Launching Streamlit UI on port {port}[/bold cyan]")
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(ui_path),
                    "--server.port", str(port), "--server.address", "127.0.0.1"], check=True)


@app.command()
def verify(
    features_file: str = typer.Option(..., "--features", help="NPZ with per-subject PPV features"),
    n_kernels: int = typer.Option(10_000, "--n-kernels"),
):
    """Verify the MiniROCKET near-singularity claim."""
    import numpy as np

    from myoadapt.features.minirocket import MiniRocketVerifier

    data = np.load(features_file, allow_pickle=True)
    features_per_subject = [data[k] for k in data.files]

    verifier = MiniRocketVerifier(n_kernels=n_kernels)
    diag = verifier.diagnose(features_per_subject)
    console.print("[bold]MiniROCKET Verification[/bold]")
    console.print(f"  Kernels: {diag['n_kernels']}")
    console.print(f"  Subjects: {diag['n_subjects']}")
    console.print(f"  Features: {diag['n_features']}")
    console.print(f"  Condition number: {diag['condition_number_global']:.2e}")
    console.print(f"  Fraction near-singular: {diag['fraction_near_singular_global']:.4f}")
    console.print(f"  Median per-subject cond: {diag['median_condition_number']:.2e}")
    console.print(f"  [bold]Verdict: {diag['verdict']}[/bold]")


@app.command()
def audit(
    model_path: str = typer.Option(..., "--model-path", help="Trained model .pkl"),
    db: str = typer.Option("DB2", "--db"),
    data_root: str = typer.Option("./data", "--data-root"),
    n_subjects: Optional[int] = typer.Option(None, "--n-subjects"),
    demographics_csv: Optional[str] = typer.Option(
        None, "--demographics-csv",
        help="Optional CSV with a 'subject' column matching group IDs, plus "
             "any demographic columns (e.g. age, dominant_hand) to additionally "
             "audit with DemographicFairnessAudit. Skipped if not provided."),
    output: str = typer.Option("./reports/audit", "--output"),
):
    """Run a fairness / bias audit on a trained model.

    Produces a JSON file with per-subgroup (per-subject) metrics,
    disparities, equalized-odds, and a fairness summary — plus, when
    --demographics-csv is given, a second per-demographic-variable audit
    (age bands, dominant hand, etc.), which per-subject metrics alone
    can't surface.
    """
    import json
    from pathlib import Path

    import numpy as np

    from myoadapt.data.loaders import load_dataset
    from myoadapt.data.preprocessing import preprocess_signal
    from myoadapt.evaluation.fairness import (
        fairness_summary,
        per_subgroup_metrics,
    )
    from myoadapt.features import extract_features

    console.print(f"[bold cyan]Auditing {model_path} on {db}[/bold cyan]")
    from myoadapt.models.classical import EMGClassifier
    clf = EMGClassifier.load(model_path)
    windows, labels, groups_arr, meta = load_dataset(db=db, root=data_root)
    if n_subjects is not None:
        keep_ids = sorted(set(np.asarray(groups_arr).tolist()))[:n_subjects]
        mask = np.isin(groups_arr, keep_ids)
        windows, labels, groups_arr = windows[mask], np.asarray(labels)[mask], np.asarray(groups_arr)[mask]

    filtered = np.empty_like(windows, dtype=np.float64)
    for i in range(windows.shape[0]):
        sig = windows[i].T
        sig_f = preprocess_signal(sig, fs=meta["fs"])
        filtered[i] = sig_f.T
    X = extract_features(filtered, fs=meta["fs"])
    y = np.asarray(labels)
    g = np.asarray(groups_arr)

    y_pred = clf.predict(X)
    try:
        y_proba = clf.predict_proba(X)
    except Exception:
        y_proba = None

    audit_result = per_subgroup_metrics(y, y_pred, g, y_proba=y_proba)
    summary = fairness_summary(audit_result)
    audit_result["summary"] = summary

    demographic_result = None
    if demographics_csv:
        import csv

        from myoadapt.evaluation.demographic_fairness import DemographicFairnessAudit
        with open(demographics_csv) as f:
            rows = list(csv.DictReader(f))
        by_subject = {row["subject"]: row for row in rows}
        demo_cols = [c for c in (rows[0].keys() if rows else []) if c != "subject"]
        demographics = {}
        for col in demo_cols:
            values = [by_subject.get(str(sid), {}).get(col) for sid in g.tolist()]
            try:
                demographics[col] = np.array([float(v) for v in values], dtype=np.float64)
            except (TypeError, ValueError):
                demographics[col] = np.array(values, dtype=object)
        demographic_result = DemographicFairnessAudit().audit(y, y_pred, g, demographics=demographics)
        console.print(f"  Demographic audit: {demographic_result['narrative']}")
    else:
        console.print("  [dim]No --demographics-csv given — skipping demographic-variable audit "
                       "(per-subject audit above still runs).[/dim]")

    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"fairness_audit_{db}.json"
    with open(out_file, "w") as f:
        json.dump({"per_subject": audit_result, "per_demographic": demographic_result},
                   f, indent=2, default=str)

    console.print(f"[green]✓ Audit written to {out_file}[/green]")
    console.print(f"  Subgroups audited: {audit_result['n_subgroups']}")
    console.print(f"  Summary: {summary}")


@app.command()
def model_card(
    model_path: str = typer.Option(..., "--model-path", help="Trained model .pkl"),
    output: str = typer.Option("./reports/model_card.md", "--output"),
    intended_use: str = typer.Option(
        "Research-only gesture classification; not for clinical or autonomous control.",
        "--intended-use",
    ),
    dataset: str = typer.Option("NinaPro-DB2", "--dataset"),
):
    """Generate a Model Card (Mitchell et al. 2019) for a trained model."""
    from pathlib import Path

    from myoadapt.evaluation.model_card import write_model_card
    from myoadapt.models.classical import EMGClassifier

    clf = EMGClassifier.load(model_path)
    md = write_model_card(
        Path(output),
        model_name=type(clf).__name__ + "-" + clf.model_type,
        model_type="Classifier",
        intended_use=intended_use,
        training_data={"name": dataset, "n_features": getattr(clf, "k_features", "N/A")},
        evaluation={
            "status": "INCOMPLETE DRAFT — no evaluation artifact was supplied",
            "required": "Attach a versioned held-out evaluation report and run manifest before release",
        },
        contact="adlbiqussai@gmail.com",
    )
    console.print(f"[yellow]✓ Incomplete research model-card draft written to {md}[/yellow]")
    console.print("  Attach a versioned evaluation artifact before releasing or citing this card.")


@app.command()
def calibrate(
    model_path: str = typer.Option(..., "--model-path", help="Trained model .pkl"),
    db: str = typer.Option("DB2", "--db"),
    data_root: str = typer.Option("./data", "--data-root"),
    method: str = typer.Option("temperature", "--method",
                                help="temperature | platt | isotonic"),
    split_strategy: str = typer.Option(
        "subject", "--split-strategy", help="subject | order-experimental"
    ),
    allow_order_split: bool = typer.Option(
        False,
        "--allow-order-split",
        help="Acknowledge that order-based splitting is experimental and may leak session structure.",
    ),
    output: str = typer.Option("./reports/calibration.json", "--output"),
):
    """Produce an exploratory calibration report on an explicit holdout split.

    The default split holds out participants rather than relying on sample
    order. The report is not clinical or deployment calibration evidence.
    """
    import json
    from pathlib import Path

    import numpy as np

    from myoadapt.data.loaders import load_dataset
    from myoadapt.data.preprocessing import preprocess_signal
    from myoadapt.evaluation.calibration import (
        IsotonicCalibration,
        PlattCalibration,
        TemperatureScaling,
        calibration_report,
    )
    from myoadapt.features import extract_features
    from myoadapt.models.classical import EMGClassifier

    console.print(f"[bold cyan]Calibrating {model_path} ({method})[/bold cyan]")
    clf = EMGClassifier.load(model_path)
    windows, labels, groups_arr, meta = load_dataset(db=db, root=data_root)
    filtered = np.empty_like(windows, dtype=np.float64)
    for i in range(windows.shape[0]):
        sig = windows[i].T
        sig_f = preprocess_signal(sig, fs=meta["fs"])
        filtered[i] = sig_f.T
    X = extract_features(filtered, fs=meta["fs"])
    y = np.asarray(labels)

    groups = np.asarray(groups_arr)
    if split_strategy == "subject":
        subject_ids = np.unique(groups)
        if len(subject_ids) < 2:
            console.print("[red]Subject holdout requires at least two subjects.[/red]")
            raise typer.Exit(1)
        split = max(1, int(np.ceil(len(subject_ids) * 0.6)))
        split = min(split, len(subject_ids) - 1)
        calibration_subjects = subject_ids[:split]
        evaluation_subjects = subject_ids[split:]
        calibration_mask = np.isin(groups, calibration_subjects)
        evaluation_mask = np.isin(groups, evaluation_subjects)
        X_cal, X_eval = X[calibration_mask], X[evaluation_mask]
        y_cal, y_eval = y[calibration_mask], y[evaluation_mask]
        split_metadata = {
            "strategy": "subject_holdout",
            "calibration_subjects": [str(value) for value in calibration_subjects.tolist()],
            "evaluation_subjects": [str(value) for value in evaluation_subjects.tolist()],
        }
    elif split_strategy == "order-experimental":
        if not allow_order_split:
            console.print(
                "[red]Order-based splitting is experimental. Pass --allow-order-split "
                "to acknowledge its leakage risk.[/red]"
            )
            raise typer.Exit(1)
        split = int(len(y) * 0.6)
        X_cal, X_eval = X[:split], X[split:]
        y_cal, y_eval = y[:split], y[split:]
        split_metadata = {"strategy": "order_experimental", "warning": "May leak recording or session structure"}
    else:
        console.print("[red]Unknown split strategy. Use subject or order-experimental.[/red]")
        raise typer.Exit(1)

    raw_proba_eval = clf.predict_proba(X_eval)
    raw_proba_cal = clf.predict_proba(X_cal)
    eps = 1e-12
    logits_cal = np.log(np.clip(raw_proba_cal, eps, 1.0))
    logits_eval = np.log(np.clip(raw_proba_eval, eps, 1.0))

    if method == "temperature":
        calibrator = TemperatureScaling().fit(logits_cal, y_cal)
        calibrated_proba = calibrator.transform(logits_eval)
    elif method == "platt":
        calibrator = PlattCalibration().fit(logits_cal, y_cal)
        calibrated_proba = calibrator.transform(logits_eval)
    elif method == "isotonic":
        calibrator = IsotonicCalibration().fit(logits_cal, y_cal)
        calibrated_proba = calibrator.transform(logits_eval)
    else:
        console.print(f"[red]Unknown method: {method}[/red]")
        raise typer.Exit(1)

    before = calibration_report(y_eval, raw_proba_eval)
    after = calibration_report(y_eval, calibrated_proba)

    out_dir = Path(output).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "method": method,
        "status": "EXPLORATORY — not clinical or deployment calibration evidence",
        "split": split_metadata,
        "n_calibration_samples": int(len(y_cal)),
        "n_evaluation_samples": int(len(y_eval)),
        "before": before,
        "after": after,
        "delta_ece": float(before["ece"] - after["ece"]),
    }
    if method == "temperature":
        report["temperature"] = float(calibrator.temperature_)
    with open(output, "w") as f:
        json.dump(report, f, indent=2, default=str)

    console.print(f"[yellow]✓ Exploratory calibration report written to {output}[/yellow]")
    console.print(f"  ECE: {before['ece']:.3f} -> {after['ece']:.3f}")
    console.print(f"  Brier: {before['brier']:.3f} -> {after['brier']:.3f}")
    if method == "temperature":
        console.print(f"  Temperature: {calibrator.temperature_:.4f}")


@app.command()
def power(
    effect_size: float = typer.Option(0.5, "--effect-size",
                                       help="Cohen's d / Hedges' g effect size"),
    n: int = typer.Option(0, "--n",
                          help="Sample size (0 = compute minimum N for power=0.8)"),
    alpha: float = typer.Option(0.05, "--alpha"),
    target_power: float = typer.Option(0.8, "--target-power"),
):
    """Statistical-power analysis for a paired t-test.

    Use this to justify sample size BEFORE running an experiment:
    reviewers in 2026 increasingly ask for a power justification when
    the per-fold sample size is small.

    Two modes:
    - With --n: report achieved power at that (effect_size, n).
    - Without --n (or --n 0): report the minimum N needed to reach
      ``--target-power``.
    """
    from myoadapt.evaluation.statistics import (
        interpret_effect_size,
        minimum_sample_size_paired,
        power_analysis_paired_ttest,
    )

    console.print("[bold cyan]Power analysis[/bold cyan]")
    console.print(f"  Effect size (d): {effect_size:.3f} ({interpret_effect_size(effect_size)})")
    console.print(f"  Alpha: {alpha}")
    if n > 0:
        p = power_analysis_paired_ttest(effect_size, n, alpha)
        verdict = "✓ adequate (≥ 0.8)" if p >= 0.8 else "✗ underpowered (< 0.8)"
        console.print(f"  Sample size: n = {n}")
        console.print(f"  Achieved power: {p:.3f}  {verdict}")
    else:
        n_min = minimum_sample_size_paired(effect_size, alpha, target_power)
        console.print(f"  Target power: {target_power:.2f}")
        console.print(f"  Minimum N required: {n_min}")


@app.command()
def fatigue(
    db: str = typer.Option("DB2", "--db"),
    data_root: str = typer.Option("./data", "--data-root"),
    subject: int = typer.Option(0, "--subject", help="Subject index to analyze (0-based)"),
    channel: int = typer.Option(0, "--channel", help="Channel index to analyze"),
    window_ms: int = typer.Option(200, "--window-ms"),
    increment_ms: int = typer.Option(200, "--increment-ms"),
    output: str = typer.Option("./reports/fatigue.json", "--output"),
):
    """Run EMG fatigue analysis (MNF/MDF slope trends, composite fatigue index)
    for one subject/channel.

    Concatenates that subject's windows in acquisition order as a proxy
    continuous trace. For a real deployment, feed FatigueTracker a
    genuinely continuous recording instead of windowed/shuffled data.
    """
    import json
    from pathlib import Path

    import numpy as np

    from myoadapt.data.loaders import load_dataset
    from myoadapt.evaluation.fatigue import FatigueTracker

    console.print(f"[bold cyan]Fatigue analysis: {db}, subject {subject}, channel {channel}[/bold cyan]")
    windows, labels, groups_arr, meta = load_dataset(db=db, root=data_root)
    mask = np.asarray(groups_arr) == subject
    if not mask.any():
        console.print(f"[red]No windows found for subject {subject}[/red]")
        raise typer.Exit(1)
    subj_windows = windows[mask]
    if channel >= subj_windows.shape[1]:
        console.print(f"[red]Channel {channel} out of range (0-{subj_windows.shape[1] - 1})[/red]")
        raise typer.Exit(1)
    signal = subj_windows[:, channel, :].reshape(-1)

    tracker = FatigueTracker(fs=int(meta["fs"]), window_ms=window_ms, increment_ms=increment_ms)
    result = tracker.analyze(signal)

    out_dir = Path(output).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    serializable = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in result.items()}
    with open(output, "w") as f:
        json.dump(serializable, f, indent=2, default=str)

    console.print(f"[green]✓ Fatigue report written to {output}[/green]")
    console.print(f"  Fatigue index: {result['fatigue_index']:.3f} ({result['fatigue_level']})")
    console.print(f"  MNF slope: {result['mnf_slope']:.2f} Hz/min")
    console.print()
    console.print(tracker.fatigue_narrative(result))


@app.command()
def intent(
    db: str = typer.Option("DB2", "--db"),
    data_root: str = typer.Option("./data", "--data-root"),
    rest_label: int = typer.Option(0, "--rest-label",
                                    help="Gesture label treated as 'rest' (NinaPro convention: 0)"),
    method: str = typer.Option("energy", "--method",
                                help="'energy' (EnergyThresholdDetector, no training) "
                                     "or 'learned' (LearnedIntentDetector, random_forest/logistic)"),
    learned_method: str = typer.Option("random_forest", "--learned-method"),
    test_size: float = typer.Option(0.3, "--test-size"),
    output: str = typer.Option("./models/intent_detector.pkl", "--output"),
):
    """Train/evaluate a rest-vs-active intent detector — the gatekeeper
    that decides WHEN a prosthetic should act, independent of WHAT
    gesture. Reuses the standard gesture-labeled dataset: windows with
    ``label == rest_label`` become the rest (0) class, everything else
    becomes active (1).
    """
    from myoadapt.data.loaders import load_dataset
    from myoadapt.tasks.intent_detection import EnergyThresholdDetector, LearnedIntentDetector

    windows, labels, groups_arr, meta = load_dataset(db=db, root=data_root)
    labels = np.asarray(labels)
    y_binary = (labels != rest_label).astype(int)
    n_rest, n_active = int((y_binary == 0).sum()), int((y_binary == 1).sum())
    console.print(f"[bold cyan]Intent detection: {db} — {n_rest} rest / {n_active} active windows[/bold cyan]")
    if n_rest == 0 or n_active == 0:
        console.print("[red]Need both classes — check --rest-label (got all-rest or all-active)[/red]")
        raise typer.Exit(1)

    # windows come back as (n_windows, n_channels, n_samples); the task
    # modules expect (n_windows, n_samples, n_channels).
    X = np.transpose(windows, (0, 2, 1))
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X))
    n_test = int(len(X) * test_size)
    test_idx, train_idx = idx[:n_test], idx[n_test:]

    if method == "energy":
        det = EnergyThresholdDetector(fs=int(meta.get("fs", 2000)))
        det.fit(X[train_idx][y_binary[train_idx] == 0])
        metrics = det.evaluate(X[test_idx], y_binary[test_idx])
        console.print(f"Calibrated threshold: {det.threshold:.6f}")
    elif method == "learned":
        det = LearnedIntentDetector(n_channels=X.shape[2], n_samples=X.shape[1],
                                     method=learned_method, fs=int(meta.get("fs", 2000)))
        det.fit(X[train_idx], y_binary[train_idx])
        metrics = det.evaluate(X[test_idx], y_binary[test_idx])
        det.save(output)
        console.print(f"[green]✓ Model saved to {output}[/green]")
    else:
        console.print(f"[red]Unknown method: {method!r} (use 'energy' or 'learned')[/red]")
        raise typer.Exit(1)

    table = Table(title="Intent Detection — Test Set")
    for k in ("accuracy", "precision", "tpr", "fpr", "f1", "auc"):
        if k in metrics:
            table.add_row(k, f"{metrics[k]:.4f}")
    console.print(table)


_DECODE_REGRESSION_MODELS = ("pose_mlp", "pose_transformer", "lstm_regressor", "transformer_regressor")
_DECODE_CLASSIFICATION_MODELS = ("xgboost", "random_forest", "lda", "svm", "logistic", "extra_trees", "lightgbm")


@app.command()
def decode(
    dataset: str = typer.Option("EMG2Pose", "--dataset", help="EMG2Pose or EMG2Qwerty"),
    data_root: str = typer.Option("./data", "--data-root"),
    model: Optional[str] = typer.Option(
        None, "--model",
        help="Regression (EMG2Pose): pose_mlp | pose_transformer | lstm_regressor | "
             "transformer_regressor (default pose_transformer). "
             "Classification (EMG2Qwerty): xgboost | random_forest | lda | svm | "
             "logistic | extra_trees | lightgbm (default xgboost)."),
    k_features: int = typer.Option(420, "--k-features", help="Classification only"),
    n_subjects: Optional[int] = typer.Option(None, "--n-subjects"),
    test_size: float = typer.Option(0.3, "--test-size"),
    n_epochs: int = typer.Option(20, "--n-epochs", help="Regression only"),
    output: str = typer.Option("./models/decoder.pt", "--output"),
):
    """Decode from EMG on Meta's emg2pose / emg2qwerty datasets, falling
    back to synthetic data of the correct shape when no real files are
    present under --data-root.

    Routes on the dataset's own declared task: EMG2Pose is continuous
    21-DOF pose regression; EMG2Qwerty is discrete keystroke
    classification, trained the same way `myoadapt train` trains any
    classifier (feature extraction + EMGClassifier), not forced through
    a regression head.
    """
    from myoadapt.data.meta_loaders import load_meta_dataset

    X, y, groups_arr, meta = load_meta_dataset(dataset=dataset, root=data_root, n_subjects=n_subjects)
    task = meta.get("task", "regression")
    console.print(f"[bold cyan]Decode: {dataset} ({'synthetic' if meta['synthetic'] else 'real'} data, "
                  f"task={task}), {X.shape[0]} windows, {len(set(groups_arr.tolist()))} subjects[/bold cyan]")

    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X))
    n_test = int(len(X) * test_size)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    n_ch, n_samp = X.shape[1], X.shape[2]

    if task == "classification":
        if model is None:
            model = "xgboost"
        elif model not in _DECODE_CLASSIFICATION_MODELS:
            console.print(f"[red]--model {model!r} is a regression model; {dataset} is a "
                           f"classification task. Use one of: {', '.join(_DECODE_CLASSIFICATION_MODELS)}[/red]")
            raise typer.Exit(1)

        from sklearn.metrics import accuracy_score, f1_score

        from myoadapt.data.preprocessing import preprocess_signal
        from myoadapt.features import extract_features
        from myoadapt.models.classical import EMGClassifier

        filtered = np.empty_like(X, dtype=np.float64)
        for i in range(X.shape[0]):
            filtered[i] = preprocess_signal(X[i].T, fs=meta["fs"]).T
        Xf = extract_features(filtered, fs=meta["fs"])

        clf = EMGClassifier(model_type=model, k_features=k_features)
        clf.fit(Xf[train_idx], y[train_idx])
        y_pred = clf.predict(Xf[test_idx])
        metrics = {
            "accuracy": float(accuracy_score(y[test_idx], y_pred)),
            "macro_f1": float(f1_score(y[test_idx], y_pred, average="macro", zero_division=0)),
            "n_classes": int(len(np.unique(y))),
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
        }
        clf.save(output.replace(".pt", ".pkl") if output.endswith(".pt") else output)
        output_used = output.replace(".pt", ".pkl") if output.endswith(".pt") else output

    else:  # regression
        from myoadapt.tasks.continuous import LSTMRegressor, TransformerRegressor
        from myoadapt.tasks.pose_regression import PoseMLPRegressor
        from myoadapt.tasks.pose_regression import TransformerRegressor as PoseTransformerRegressor

        if model is None:
            model = "pose_transformer"
        elif model not in _DECODE_REGRESSION_MODELS:
            console.print(f"[red]--model {model!r} is a classification model; {dataset} is a "
                           f"regression task. Use one of: {', '.join(_DECODE_REGRESSION_MODELS)}[/red]")
            raise typer.Exit(1)

        n_out = y.shape[1] if y.ndim > 1 else int(y.max()) + 1
        if model == "pose_mlp":
            Xflat = X.reshape(len(X), -1)  # flatten raw window to feature vector
            m = PoseMLPRegressor(n_features=Xflat.shape[1], n_outputs=n_out, n_epochs=n_epochs)
            m.fit(Xflat[train_idx], y[train_idx])
            metrics = m.evaluate(Xflat[test_idx], y[test_idx]) if hasattr(m, "evaluate") else {}
        elif model == "pose_transformer":
            m = PoseTransformerRegressor(n_channels=n_ch, n_samples=n_samp, n_outputs=n_out,
                                          n_epochs=n_epochs, device="cpu")
            m.fit(X[train_idx], y[train_idx])
            metrics = m.evaluate(X[test_idx], y[test_idx])
        else:
            Xr = np.transpose(X, (0, 2, 1))  # -> (n, seq_len, n_channels) for continuous.py's convention
            cls = LSTMRegressor if model == "lstm_regressor" else TransformerRegressor
            m = cls(n_channels=n_ch, n_outputs=n_out, n_epochs=n_epochs, device="cpu")
            m.fit(Xr[train_idx], y[train_idx])
            metrics = m.evaluate(Xr[test_idx], y[test_idx])
        m.save(output)
        output_used = output

    console.print(f"[green]✓ Model saved to {output_used}[/green]")
    table = Table(title=f"Decode — {dataset} / {model} ({task})")
    for k, v in metrics.items():
        table.add_row(k, f"{v:.4f}" if isinstance(v, float) else str(v))
    console.print(table)


@app.command()
def retrieve(
    db: str = typer.Option("DB2", "--db"),
    data_root: str = typer.Option("./data", "--data-root"),
    n_subjects: Optional[int] = typer.Option(3, "--n-subjects"),
    labels_csv: Optional[str] = typer.Option(
        None, "--labels-csv",
        help="Optional CSV with 'gesture_id,description' columns mapping "
             "numeric gesture labels to text (e.g. '1,open hand fully'). "
             "Without it, gestures are auto-labeled 'gesture N'."),
    query: Optional[str] = typer.Option(
        None, "--query", help="After training, retrieve EMG windows matching this text."),
    n_epochs: int = typer.Option(15, "--n-epochs"),
    output: str = typer.Option("./models/retriever.pt", "--output"),
):
    """Train EMG<->text semantic retrieval (CLIP-style contrastive) and,
    optionally, run one demo query against it.
    """
    import csv

    from myoadapt.data.loaders import load_dataset
    from myoadapt.tasks.semantic_retrieval import EMGSemanticRetriever

    windows, labels, groups_arr, meta = load_dataset(db=db, root=data_root)
    if n_subjects is not None:
        keep_ids = sorted(set(np.asarray(groups_arr).tolist()))[:n_subjects]
        mask = np.isin(groups_arr, keep_ids)
        windows, labels = windows[mask], np.asarray(labels)[mask]

    label_map = {}
    if labels_csv:
        with open(labels_csv) as f:
            for row in csv.DictReader(f):
                label_map[int(row["gesture_id"])] = row["description"]
    descriptions = [label_map.get(int(g), f"gesture {int(g)}") for g in labels]

    console.print(f"[bold cyan]Semantic retrieval: {db}, {windows.shape[0]} windows, "
                  f"{len(set(descriptions))} unique descriptions[/bold cyan]")

    n_test = max(1, int(0.2 * len(windows)))
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(windows))
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    train_descs = [descriptions[i] for i in train_idx]
    test_descs = [descriptions[i] for i in test_idx]

    retriever = EMGSemanticRetriever(n_channels=windows.shape[1], n_samples=windows.shape[2],
                                      n_epochs=n_epochs, device="cpu")
    retriever.fit(windows[train_idx], train_descs)
    metrics = retriever.evaluate_retrieval(windows[test_idx], test_descs, k_values=(1, 5))
    retriever.save(output)

    console.print(f"[green]✓ Model saved to {output}[/green]")
    table = Table(title=f"Semantic Retrieval — {db}")
    for k, v in metrics.items():
        table.add_row(k, f"{v:.4f}" if isinstance(v, float) else str(v))
    console.print(table)

    if query:
        top_idx = retriever.retrieve_emg_by_text(query, windows[test_idx], top_k=5)
        matched_descs = [test_descs[i] for i in top_idx]
        console.print(f"\n[bold]Top-5 EMG windows for query [/bold]'{query}': {matched_descs}")


@app.command()
def augment(
    db: str = typer.Option("DB2", "--db"),
    data_root: str = typer.Option("./data", "--data-root"),
    method: str = typer.Option("gan", "--method", help="'gan' or 'diffusion'"),
    gesture: Optional[int] = typer.Option(
        None, "--gesture", help="Augment only this gesture label (e.g. an "
                                 "underrepresented class). Default: all windows."),
    n_generate: int = typer.Option(100, "--n-generate"),
    n_epochs: int = typer.Option(50, "--n-epochs"),
    output: str = typer.Option("./data/augmented.npz", "--output"),
):
    """Fit a GAN or diffusion generator on real EMG windows and generate
    synthetic augmented samples — useful for expanding a small dataset or
    balancing an underrepresented gesture class.
    """
    from myoadapt.data.loaders import load_dataset
    from myoadapt.data.synthetic_augmentation import EMGDiffusionAugmenter, EMGGANAugmenter

    windows, labels, groups_arr, meta = load_dataset(db=db, root=data_root)
    labels = np.asarray(labels)
    if gesture is not None:
        mask = labels == gesture
        if not mask.any():
            console.print(f"[red]No windows found for gesture {gesture}[/red]")
            raise typer.Exit(1)
        real = windows[mask]
        console.print(f"[bold cyan]Augmenting gesture {gesture}: {real.shape[0]} real windows[/bold cyan]")
    else:
        real = windows
        console.print(f"[bold cyan]Augmenting all gestures: {real.shape[0]} real windows[/bold cyan]")

    cls = EMGGANAugmenter if method == "gan" else EMGDiffusionAugmenter
    if method not in ("gan", "diffusion"):
        console.print(f"[red]Unknown method: {method!r} (use 'gan' or 'diffusion')[/red]")
        raise typer.Exit(1)

    aug = cls(n_channels=real.shape[1], n_samples=real.shape[2], n_epochs=n_epochs, device="cpu")
    aug.fit(real)
    generated = aug.generate(n_generate)

    out_dir = Path(output).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    synth_labels = np.full(n_generate, gesture if gesture is not None else -1, dtype=np.int32)
    np.savez(output, real=real, generated=generated,
             real_labels=labels[labels == gesture] if gesture is not None else labels,
             generated_labels=synth_labels)

    console.print(f"[green]✓ {n_generate} synthetic windows saved to {output}[/green]")
    console.print(f"  Real: {real.shape} | Generated: {generated.shape} | "
                  f"has_nan: {bool(np.isnan(generated).any())}")


@app.command()
def zeroshot(
    db: str = typer.Option("DB2", "--db"),
    data_root: str = typer.Option("./data", "--data-root"),
    model: str = typer.Option("xgboost", "--model"),
    n_train_subjects: Optional[int] = typer.Option(None, "--n-train-subjects"),
    n_test_subjects: Optional[int] = typer.Option(None, "--n-test-subjects"),
    max_windows_per_subject: Optional[int] = typer.Option(
        None,
        "--max-windows-per-subject",
        min=1,
        help="Optional evenly spaced cap per subject for bounded smoke experiments; changes the protocol.",
    ),
    compare_loso: bool = typer.Option(False, "--compare-loso",
                                       help="Also run LOSO and report the gap vs zero-shot"),
    output: str = typer.Option("./reports/zeroshot.json", "--output"),
):
    """Zero-shot cross-user evaluation: train on one subject pool, test
    on a completely disjoint pool of unseen subjects — the strictest
    generalization test available, vs LOSO's single-subject holdout.
    """
    import json

    from myoadapt.data.loaders import load_dataset
    from myoadapt.data.preprocessing import preprocess_signal
    from myoadapt.evaluation.zero_shot import ZeroShotEvaluator
    from myoadapt.features import extract_features
    from myoadapt.models.classical import EMGClassifier

    windows, labels, groups_arr, meta = load_dataset(db=db, root=data_root)
    g_all = np.asarray(groups_arr)
    n_subjects_total = len(set(g_all.tolist()))
    n_train = n_train_subjects or max(3, int(n_subjects_total * 0.8))
    n_test = n_test_subjects or (n_subjects_total - n_train)

    # Filter to just the subjects this run will actually use BEFORE the
    # expensive feature-extraction step — extracting on the full default
    # dataset regardless of pool size defeats the point of these flags.
    pool_ids = sorted(set(g_all.tolist()))[: n_train + n_test]
    mask = np.isin(g_all, pool_ids)
    windows = windows[mask]
    labels = np.asarray(labels)[mask]
    groups_arr = g_all[mask]

    if max_windows_per_subject is not None:
        selected: list[np.ndarray] = []
        for subject in pool_ids:
            subject_indices = np.flatnonzero(groups_arr == subject)
            if len(subject_indices) > max_windows_per_subject:
                evenly_spaced = np.linspace(
                    0, len(subject_indices) - 1, max_windows_per_subject, dtype=int
                )
                subject_indices = subject_indices[evenly_spaced]
            selected.append(subject_indices)
        selected_indices = np.concatenate(selected) if selected else np.array([], dtype=int)
        windows = windows[selected_indices]
        labels = labels[selected_indices]
        groups_arr = groups_arr[selected_indices]

    filtered = np.empty_like(windows, dtype=np.float64)
    for i in range(windows.shape[0]):
        filtered[i] = preprocess_signal(windows[i].T, fs=meta["fs"]).T
    X = extract_features(filtered, fs=meta["fs"])
    y = np.asarray(labels)
    g = np.asarray(groups_arr)

    console.print(f"[bold cyan]Zero-shot: {db}, {n_subjects_total} subjects total, using "
                  f"{n_train + n_test} ({n_train} train pool / {n_test} test pool)[/bold cyan]")
    if max_windows_per_subject is not None:
        console.print(
            "  [yellow]Window cap enabled for a bounded smoke experiment; "
            "do not use its result as a full evaluation.[/yellow]"
        )

    ev = ZeroShotEvaluator(model_factory=lambda: EMGClassifier(model_type=model, k_features=None),
                            n_train_subjects=n_train, n_test_subjects=n_test,
                            random_state=42, verbose=False, rest_class=None)

    if compare_loso:
        result = ev.compare_with_loso(X, y, g)
        console.print(f"  Zero-shot accuracy: {result['zero_shot']['aggregate']['accuracy_mean']:.4f}")
        console.print(f"  LOSO accuracy:      {result['loso']['aggregate']['accuracy_mean']:.4f}")
        console.print(f"  Gap: {result['comparison']['accuracy_gap_pp']:.2f}pp "
                       f"({result['comparison']['effect_interpretation']})")
    else:
        result = ev.run(X, y, g)
        table = Table(title=f"Zero-Shot — {db}")
        for k, v in result["aggregate"].items():
            if "per_subject" not in k:
                table.add_row(k, f"{v:.4f}" if isinstance(v, float) else str(v))
        console.print(table)
        console.print(f"  Worst subject: {result['worst_subject']['subject']} "
                       f"(acc={result['worst_subject']['accuracy']:.3f})")

    out_dir = Path(output).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(result, f, indent=2, default=str)
    console.print(f"[green]✓ Report written to {output}[/green]")


@app.command()
def robustness(
    model_path: str = typer.Option(..., "--model-path", help="Trained model .pkl"),
    db: str = typer.Option("DB2", "--db"),
    data_root: str = typer.Option("./data", "--data-root"),
    n_subjects: Optional[int] = typer.Option(2, "--n-subjects"),
    max_shift: int = typer.Option(2, "--max-shift"),
    output: str = typer.Option("./reports/robustness.json", "--output"),
):
    """Electrode-shift robustness audit: perturb raw EMG windows
    (circular channel shift, dropout, swap, noise) and re-run the full
    preprocess -> feature-extract -> predict pipeline to measure how
    much accuracy degrades — simulates an armband that's rotated or
    partially lost contact after donning.

    Expensive by design: 4 shift types x (max_shift+1) magnitude steps,
    each re-running full feature extraction on every window (this is
    what makes it a *pipeline* robustness test, not just a perturbation
    check). Defaults are kept small; scale --n-subjects/--max-shift up
    for a more thorough audit at the cost of a longer run.
    """
    import json

    from myoadapt.data.loaders import load_dataset
    from myoadapt.data.preprocessing import preprocess_signal
    from myoadapt.evaluation.electrode_shift import ElectrodeShiftRobustness
    from myoadapt.features import extract_features
    from myoadapt.models.classical import EMGClassifier

    windows, labels, groups_arr, meta = load_dataset(db=db, root=data_root)
    if n_subjects is not None:
        keep_ids = sorted(set(np.asarray(groups_arr).tolist()))[:n_subjects]
        mask = np.isin(groups_arr, keep_ids)
        windows, labels = windows[mask], np.asarray(labels)[mask]

    clf = EMGClassifier.load(model_path)

    class _PipelinePredictor:
        """Adapts the trained classifier to ElectrodeShiftRobustness's
        expected interface: takes raw (n, n_channels, n_samples) windows
        (post-perturbation) and returns predictions, running them through
        the same preprocess+feature-extract steps the model was trained on."""
        def predict(self, X_raw):
            filt = np.empty_like(X_raw, dtype=np.float64)
            for i in range(X_raw.shape[0]):
                filt[i] = preprocess_signal(X_raw[i].T, fs=meta["fs"]).T
            feats = extract_features(filt, fs=meta["fs"])
            return clf.predict(feats)

    console.print(f"[bold cyan]Robustness audit: {model_path} on {db} "
                  f"({windows.shape[0]} windows)[/bold cyan]")
    console.print(f"[dim]4 shift types x {max_shift + 1} magnitudes, each re-running "
                   f"preprocess+feature-extract on all windows — this is minutes, not "
                   f"seconds, at realistic scale. Use --n-subjects/--max-shift to shrink "
                   f"it for a quick check.[/dim]")
    ev = ElectrodeShiftRobustness(model=_PipelinePredictor(), n_channels=windows.shape[1], random_state=0)
    result = ev.evaluate_all_shifts(windows, labels, max_shift=max_shift)

    table = Table(title=f"Electrode-Shift Robustness — {db}")
    table.add_row("baseline_accuracy", f"{result['summary']['baseline_accuracy']:.4f}")
    for shift_type, degradation in result["summary"]["per_type_max_degradation_pp"].items():
        table.add_row(f"{shift_type}_max_degradation_pp", f"{degradation:.2f}")
    table.add_row("worst_shift_type", result["summary"]["worst_shift_type"])
    console.print(table)

    out_dir = Path(output).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(result, f, indent=2, default=str)
    console.print(f"[green]✓ Report written to {output}[/green]")


@app.command()
def drift(
    db: str = typer.Option("DB2", "--db"),
    data_root: str = typer.Option("./data", "--data-root"),
    reference_subjects: str = typer.Option(
        "", "--reference-subjects",
        help="Comma-separated subject indices for the reference period. "
             "Default: first half of subjects."),
    current_subjects: str = typer.Option(
        "", "--current-subjects",
        help="Comma-separated subject indices for the current period. "
             "Default: second half of subjects."),
    max_windows_per_group: int = typer.Option(
        300, "--max-windows-per-group",
        help="Subsample each group to at most this many windows before "
             "feature extraction. Drift detection doesn't need — and "
             "feature extraction plus MMD's O(n^2) cost make it "
             "impractical to run on — a full multi-thousand-window "
             "subject group."),
    significance_level: float = typer.Option(0.05, "--alpha"),
):
    """Detect feature-distribution drift between two subject groups —
    e.g. the subjects a model was validated on vs. a new deployment
    population. Only feature drift is covered here (no trained-model
    predictions/scores available at the CLI level for prediction/
    performance drift — use DriftDetector directly in a monitoring loop
    for those).
    """
    from myoadapt.data.loaders import load_dataset
    from myoadapt.evaluation.drift_detection import DriftDetector
    from myoadapt.features.registry import extract_features

    windows, labels, groups_arr, meta = load_dataset(db=db, root=data_root)
    groups_arr = np.asarray(groups_arr)
    subjects = sorted(set(groups_arr.tolist()))
    if len(subjects) < 2:
        console.print("[red]Need at least 2 subjects to compare reference vs. current[/red]")
        raise typer.Exit(1)
    mid = len(subjects) // 2
    ref_ids = [int(s) for s in reference_subjects.split(",") if s] or subjects[:mid]
    cur_ids = [int(s) for s in current_subjects.split(",") if s] or subjects[mid:]

    rng = np.random.default_rng(42)

    def _subsample(mask_ids):
        idx = np.where(np.isin(groups_arr, mask_ids))[0]
        if len(idx) > max_windows_per_group:
            idx = rng.choice(idx, size=max_windows_per_group, replace=False)
        return windows[idx]

    ref_windows = _subsample(ref_ids)
    cur_windows = _subsample(cur_ids)
    console.print(f"[bold cyan]Drift check: {db} — reference subjects {ref_ids} "
                  f"({ref_windows.shape[0]} windows), current subjects {cur_ids} "
                  f"({cur_windows.shape[0]} windows)[/bold cyan]")
    X_ref = extract_features(ref_windows, fs=int(meta.get("fs", 2000)))
    X_cur = extract_features(cur_windows, fs=int(meta.get("fs", 2000)))

    det = DriftDetector(significance_level=significance_level).fit(X_ref)
    result = det.detect_all(X_current=X_cur)

    table = Table(title="Feature Drift")
    fd = result["feature"]
    table.add_row("any_drift", str(fd["any_drift"]))
    table.add_row("n_drifted / n_features", f"{fd['n_drifted']} / {fd['n_features']}")
    table.add_row("severity", fd["severity"])
    table.add_row("MMD²", f"{fd['mmd']:.4f}")
    console.print(table)
    console.print()
    console.print(det.drift_narrative(result))


@app.command()
def tune(
    db: str = typer.Option("DB2", "--db"),
    data_root: str = typer.Option("./data", "--data-root"),
    model: str = typer.Option("random_forest", "--model",
                               help="random_forest | xgboost | lightgbm | extra_trees | svm | logistic | lda"),
    protocol: str = typer.Option("loso", "--protocol", help="loso | kfold"),
    n_trials: int = typer.Option(20, "--n-trials"),
    sampler: str = typer.Option("tpe", "--sampler", help="tpe | random | cmaes"),
    pruner: str = typer.Option("median", "--pruner", help="median | halving | none"),
    max_windows: int = typer.Option(600, "--max-windows",
                                     help="Subsample the dataset to at most this many "
                                          "windows before feature extraction (see the "
                                          "`drift` command for why this matters)."),
    output: str = typer.Option("./reports/tune.json", "--output"),
):
    """Hyperparameter search (Optuna TPE/Random/CMA-ES) over a classical
    model's hyperparameters, cross-validated LOSO or k-fold.
    """
    import json
    from pathlib import Path

    from myoadapt.data.loaders import load_dataset
    from myoadapt.evaluation.hyperopt import HyperparameterOptimizer
    from myoadapt.features.registry import extract_features

    windows, labels, groups_arr, meta = load_dataset(db=db, root=data_root)
    labels, groups_arr = np.asarray(labels), np.asarray(groups_arr)
    if len(windows) > max_windows:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(windows), size=max_windows, replace=False)
        windows, labels, groups_arr = windows[idx], labels[idx], groups_arr[idx]

    console.print(f"[bold cyan]Tuning {model} on {db} ({windows.shape[0]} windows, "
                  f"{protocol} protocol, {n_trials} trials, {sampler} sampler)[/bold cyan]")
    X = extract_features(windows, fs=int(meta.get("fs", 2000)))

    opt = HyperparameterOptimizer(model_type=model, protocol=protocol, n_trials=n_trials,
                                   sampler=sampler, pruner=pruner, random_state=42)
    result = opt.optimize(X, labels, groups=groups_arr if protocol == "loso" else None)
    if "error" in result:
        console.print(f"[red]{result['error']} — {result.get('install', '')}[/red]")
        raise typer.Exit(1)

    out_dir = Path(output).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump({"best_params": result["best_params"], "best_score": result["best_score"],
                   "n_trials": result["n_trials"], "model": model, "protocol": protocol}, f, indent=2)

    table = Table(title=f"HPO — {model} / {protocol}")
    table.add_row("best_score", f"{result['best_score']:.4f}")
    table.add_row("n_trials", str(result["n_trials"]))
    for k, v in result["best_params"].items():
        table.add_row(f"  {k}", str(v))
    console.print(table)
    console.print(f"[green]✓ Saved to {output}[/green]")


if __name__ == "__main__":
    app()
