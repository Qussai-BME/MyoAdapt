"""Integration test: full pipeline end-to-end (small scale)."""
import numpy as np
import pytest

from myoadapt.config import MyoAdaptConfig
from myoadapt.data.preprocessing import preprocess_signal, segment_windows
from myoadapt.features import extract_features
from myoadapt.models.classical import EMGClassifier
from myoadapt.evaluation import LOSOEvaluator, ReportGenerator
from myoadapt.evaluation.metrics import compute_metrics


def _generate_synthetic_data(n_subjects=5, n_channels=12, fs=2000,
                              duration_sec=2, n_classes=3):
    """Generate synthetic EMG-like data for testing."""
    rng = np.random.default_rng(42)
    signals = []
    labels_per_subject = []
    groups = []
    for subj in range(n_subjects):
        n_samples = int(fs * duration_sec)
        sig = rng.standard_normal((n_samples, n_channels)).astype(np.float32) * 0.1
        # Add a low-frequency component (gesture-like)
        t = np.arange(n_samples) / fs
        for ch in range(n_channels):
            sig[:, ch] += 0.5 * np.sin(2 * np.pi * (10 + ch) * t)
        signals.append(sig)
        labels = rng.integers(0, n_classes, size=n_samples)
        labels_per_subject.append(labels)
    return signals, labels_per_subject


def test_full_pipeline_small():
    """End-to-end: preprocess → window → features → train → predict."""
    signals, labels_per_subject = _generate_synthetic_data(
        n_subjects=4, n_channels=12, fs=2000, duration_sec=2
    )

    # Preprocess + window + features per subject
    all_features = []
    all_labels = []
    all_groups = []
    for subj_idx, (sig, labels) in enumerate(zip(signals, labels_per_subject)):
        sig_f = preprocess_signal(sig, fs=2000, notch_freq=50.0)
        windows, window_labels = segment_windows(
            sig_f, fs=2000, window_ms=200, increment_ms=50, labels=labels,
        )
        feats = extract_features(windows, modules=["time_domain", "histogram"])
        all_features.append(feats)
        all_labels.append(window_labels)
        all_groups.extend([subj_idx + 1] * len(windows))

    X = np.concatenate(all_features)
    y = np.concatenate(all_labels)
    groups = np.asarray(all_groups)

    assert X.ndim == 2
    assert X.shape[0] == len(y) == len(groups)
    assert X.shape[1] > 0

    # Train/test split
    from myoadapt.data.splits import train_test_split_subject
    train_idx, test_idx = train_test_split_subject(groups, test_subject=4)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Train
    clf = EMGClassifier(model_type="random_forest", n_estimators=10)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    assert len(preds) == len(y_test)

    # Metrics
    classes = sorted(np.unique(y).tolist())
    metrics = compute_metrics(y_test, preds, classes=classes)
    assert 0 <= metrics["accuracy"] <= 1


def test_loso_evaluator_synthetic():
    """Run full LOSO on synthetic data."""
    signals, labels_per_subject = _generate_synthetic_data(
        n_subjects=4, n_channels=12, fs=2000, duration_sec=1.5, n_classes=3,
    )
    all_features = []
    all_labels = []
    all_groups = []
    for subj_idx, (sig, labels) in enumerate(zip(signals, labels_per_subject)):
        sig_f = preprocess_signal(sig, fs=2000)
        windows, window_labels = segment_windows(
            sig_f, fs=2000, window_ms=200, increment_ms=50, labels=labels,
        )
        feats = extract_features(windows, modules=["time_domain"])
        all_features.append(feats)
        all_labels.append(window_labels)
        all_groups.extend([subj_idx + 1] * len(windows))

    X = np.concatenate(all_features)
    y = np.concatenate(all_labels)
    groups = np.asarray(all_groups)

    def factory():
        return EMGClassifier(model_type="random_forest", n_estimators=10)

    evaluator = LOSOEvaluator(model_factory=factory, verbose=False)
    results = evaluator.run(X, y, groups)

    assert results["n_folds"] == 4
    assert "aggregate" in results
    assert "per_fold" in results
    assert len(results["aggregate"]["per_fold_accuracies"]) == 4


def test_report_generation(tmp_path):
    """Test that ReportGenerator produces tables."""
    signals, labels_per_subject = _generate_synthetic_data(
        n_subjects=3, n_channels=12, fs=2000, duration_sec=1, n_classes=3,
    )
    all_features = []
    all_labels = []
    all_groups = []
    for subj_idx, (sig, labels) in enumerate(zip(signals, labels_per_subject)):
        sig_f = preprocess_signal(sig, fs=2000)
        windows, window_labels = segment_windows(
            sig_f, fs=2000, window_ms=200, increment_ms=50, labels=labels,
        )
        feats = extract_features(windows, modules=["time_domain"])
        all_features.append(feats)
        all_labels.append(window_labels)
        all_groups.extend([subj_idx + 1] * len(windows))

    X = np.concatenate(all_features)
    y = np.concatenate(all_labels)
    groups = np.asarray(all_groups)

    def factory():
        return EMGClassifier(model_type="random_forest", n_estimators=5)

    evaluator = LOSOEvaluator(model_factory=factory, verbose=False)
    results = evaluator.run(X, y, groups)

    gen = ReportGenerator(tmp_path)
    paths = gen.generate_tables({"RF": results}, dataset_name="synthetic")
    assert "main_results" in paths
    assert paths["main_results"].exists()
