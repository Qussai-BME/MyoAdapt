"""
MyoAdapt v2.0 — Quickstart Tutorial
======================================

This notebook demonstrates the full MyoAdapt pipeline on synthetic data:
  1. Generate synthetic multi-channel EMG
  2. Filter + window the signal
  3. Extract time-domain features (AR coefficients via Yule-Walker)
  4. Apply Euclidean Alignment (per-subject, Hahne 2014)
  5. Train a classical classifier
  6. Evaluate with LOSO
  7. Export to ONNX
  8. Generate SHAP transparency report

Run cells top to bottom. Total runtime: ~30 seconds on a laptop CPU.

"""

# ============================================================
# CELL 1 — Setup
# ============================================================
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

import myoadapt
print(f"MyoAdapt v{myoadapt.__version__}")
print(f"Modules: {list(myoadapt.__all__)[:5]}...")

# ============================================================
# CELL 2 — Generate synthetic multi-subject EMG
# ============================================================
# 5 subjects × 4 gestures × 10 repetitions × 12 channels × 400 samples
N_SUBJECTS = 5
N_GESTURES = 4
N_REPS = 10
N_CHANNELS = 12
N_SAMPLES = 400  # 200ms @ 2kHz
FS = 2000  # sampling rate

rng = np.random.default_rng(42)
signals = []  # list of (subject_id, gesture, signal)
for s in range(N_SUBJECTS):
    # Each subject has their own "impedance" → different signal scale
    subject_scale = rng.uniform(0.5, 2.0)
    for g in range(N_GESTURES):
        for r in range(N_REPS):
            # Base signal: filtered noise
            sig = rng.standard_normal((N_SAMPLES, N_CHANNELS)) * subject_scale
            # Add gesture-specific pattern (different channels per gesture)
            sig[:, g*3:(g+1)*3] += np.sin(np.linspace(0, 10*np.pi, N_SAMPLES))[:, None] * 0.5
            signals.append((s, g, sig))

print(f"Generated {len(signals)} EMG windows from {N_SUBJECTS} subjects")

# ============================================================
# CELL 3 — Filter and visualize one window
# ============================================================
from myoadapt.data.preprocessing import filter_signal, notch_filter

# Pick one window
s, g, sig = signals[0]
sig_filtered = filter_signal(sig, fs=FS, low=20, high=500, order=4)
sig_notched = notch_filter(sig_filtered, fs=FS, freq=50.0, quality=30.0)

fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
for ch in range(4):
    axes[0].plot(sig[:, ch], alpha=0.7, label=f"CH{ch}")
    axes[1].plot(sig_filtered[:, ch], alpha=0.7)
    axes[2].plot(sig_notched[:, ch], alpha=0.7)
axes[0].set_title("Raw EMG (4 channels)")
axes[1].set_title("After Butterworth bandpass (20-500 Hz)")
axes[2].set_title("After 50 Hz notch (power-line removal)")
axes[0].legend(loc="upper right", fontsize=8)
plt.tight_layout()
plt.show()

# ============================================================
# CELL 4 — Extract features (AR coefficients via Yule-Walker)
# ============================================================
from myoadapt.features.time_domain import extract_per_channel, AR_COEFFICIENTS_YULE_WALKER

print(f"AR coefficients (Yule-Walker): {AR_COEFFICIENTS_YULE_WALKER}")

# Extract features from all windows
features = []
labels = []
subjects = []
for s, g, sig in signals:
    sig_f = notch_filter(filter_signal(sig, fs=FS, low=20, high=500, order=4),
                         fs=FS, freq=50.0, quality=30.0)
    # Extract per-channel features
    ch_features = {}
    for ch in range(N_CHANNELS):
        ch_feats = extract_per_channel(sig_f[:, ch], ar_order=4)
        for k, v in ch_feats.items():
            ch_features[f"{k}_CH{ch}"] = v
    features.append(ch_features)
    labels.append(g)
    subjects.append(s)

import pandas as pd
df = pd.DataFrame(features)
df["gesture"] = labels
df["subject"] = subjects
print(f"Feature matrix: {df.shape}")
print(f"AR features present: {[c for c in df.columns if c.startswith('AR_')][:8]}...")
print(f"AR feature values (subject 0, channel 0): {df[df.subject==0][['AR_0_CH0', 'AR_1_CH0', 'AR_2_CH0', 'AR_3_CH0']].iloc[0].tolist()}")

# ============================================================
# CELL 5 — Apply Per-Subject Euclidean Alignment (Hahne 2014)
# ============================================================
from myoadapt.adaptation.ea import PerSubjectEA

# Build per-subject raw signals (concatenate all windows per subject)
signals_by_subject = {}
for s, g, sig in signals:
    if s not in signals_by_subject:
        signals_by_subject[s] = []
    signals_by_subject[s].append(sig)
signals_by_subject = {s: np.vstack(sig_list) for s, sig_list in signals_by_subject.items()}

ea = PerSubjectEA()
ea.fit_subjects(signals_by_subject)

# Verify whitening
print("Per-subject EA verification:")
for s, sig in signals_by_subject.items():
    aligned = ea.transform_subject(s, sig)
    cov = np.cov(aligned.T)
    identity_err = np.linalg.norm(cov - np.eye(N_CHANNELS))
    print(f"  Subject {s}: ||cov - I|| = {identity_err:.4f}")

# ============================================================
# CELL 6 — Train a classifier
# ============================================================
from myoadapt.models.classical import EMGClassifier

X = df.drop(columns=["gesture", "subject"]).values.astype(np.float32)
y = df["gesture"].values
groups = df["subject"].values

# Split: subjects 0-3 train, subject 4 test
train_mask = groups < 4
test_mask = groups == 4

clf = EMGClassifier(model_type="random_forest", n_estimators=100, random_state=42)
clf.fit(X[train_mask], y[train_mask])
preds = clf.predict(X[test_mask])
acc = (preds == y[test_mask]).mean()
print(f"Test accuracy on unseen subject: {acc:.3f}")
print(f"Chance level: {1/N_GESTURES:.3f}")

# ============================================================
# CELL 7 — LOSO cross-validation
# ============================================================
from myoadapt.evaluation.loso import LOSOEvaluator
from myoadapt.evaluation.metrics import compute_metrics

# LOSO using a manual loop (compute_metrics requires `classes`)
all_classes = sorted(np.unique(y).tolist())
results = []
for test_subject in range(N_SUBJECTS):
    train_mask = groups != test_subject
    test_mask = groups == test_subject
    clf = EMGClassifier(model_type="random_forest", n_estimators=50, random_state=42)
    clf.fit(X[train_mask], y[train_mask])
    preds = clf.predict(X[test_mask])
    m = compute_metrics(y[test_mask], preds, classes=all_classes, rest_class=None)
    results.append({"subject": test_subject, **m})

results_df = pd.DataFrame(results)
print("LOSO results per subject:")
print(results_df[["subject", "accuracy", "macro_f1"]].to_string(index=False))
print(f"\nMean LOSO accuracy: {results_df['accuracy'].mean():.3f} ± {results_df['accuracy'].std():.3f}")

# ============================================================
# CELL 8 — Export to ONNX + benchmark
# ============================================================
from myoadapt.deployment.onnx_export import OnnxExporter
from myoadapt.deployment.hardware_bench import HardwareBenchmark
import tempfile, os

# Train final model on all data
final_clf = EMGClassifier(model_type="random_forest", n_estimators=100, random_state=42)
final_clf.fit(X, y)

# Export to ONNX (uses the unified OnnxExporter.export dispatcher)
onnx_path = os.path.join(tempfile.mkdtemp(), "model.onnx")
feature_names = [c for c in df.columns if c not in ("gesture", "subject")]
info = OnnxExporter.export(final_clf, onnx_path, feature_names=feature_names)
print(f"ONNX model saved: {info['path']} ({info['size_bytes']:,} bytes)")

# Run 5-CPU benchmark
bench = HardwareBenchmark(final_clf, n_runs=50)
bench_results = bench.run()
print(f"\n5-CPU Benchmark (n_features={bench_results['n_features']}):")
print(f"Local CPU: {bench_results['local_cpu']}")
print(f"Local mean latency: {bench_results['local_mean_ms']:.2f} ms")
print(f"\nPer-CPU projection:")
for cpu, r in bench_results['per_cpu_projected'].items():
    rt = "real-time" if r['realtime_capable'] else "too slow"
    print(f"  {cpu}: {r['projected_mean_ms']:.2f} ms — {rt}")
print(f"\nVerdict: {bench_results['verdict']}")

# ============================================================
# CELL 9 — SHAP transparency report
# ============================================================
from myoadapt.deployment.shap_reports import ShapReportGenerator

shap_gen = ShapReportGenerator(final_clf)
report = shap_gen.explain(X[test_mask][:1])
# TransparencyReport is a dataclass — convert to dict for easy printing.
report_dict = report.to_dict() if hasattr(report, "to_dict") else report.__dict__

print("SHAP Transparency Report")
print("=" * 60)
print(f"Predicted class: {report_dict.get('predicted_class')}")
print(f"Confidence: {report_dict.get('confidence', 0):.3f}")
print(f"Trust score: {report_dict.get('trust_score', 0):.3f}")
print(f"\nTop contributing features:")
for feat in report_dict.get('top_features', [])[:5]:
    print(f"  - {feat}")
print(f"\nPer-channel contribution:")
for ch, val in list(report_dict.get('per_channel_contribution', {}).items())[:5]:
    print(f"  Channel {ch}: {val:+.4f}")
print(f"\nExplanation: {report_dict.get('explanation', 'N/A')}")

print("\n" + "=" * 60)
print("Tutorial complete. Next steps:")
print("  - Try the Lite-DAN model: from myoadapt.models import LiteDAN")
print("  - Run the cross-subject benchmark: python scripts/run_cross_subject_loso.py")
print("  - Explore the docs: https://github.com/Qussai-BME/MyoAdapt")
