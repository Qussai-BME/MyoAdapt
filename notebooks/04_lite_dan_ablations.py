"""
Ablations — Lite-DAN: Lightweight Adversarial Network
==============================================================

This notebook runs the ablation experiments:

  Ablation 1: λ schedule (gradual vs fixed vs none)
  Ablation 2: GRL on/off
  Ablation 3: With/without Euclidean Alignment preprocessing
  Ablation 4: Feature space (7 TD vs 308 hybrid vs raw)

Each ablation isolates one design choice. The full study runs all
4 ablations × 5 databases × LOSO × 5 random seeds = 100 experiments.

This notebook demonstrates the code path on synthetic data. Real
experiments use scripts/run_lite_dan_ablations.py.

"""

# ============================================================
# CELL 1 — Setup: synthetic data with subject domain shift
# ============================================================
import numpy as np
from myoadapt.models.lite_dan import LiteDAN

# Synthetic EMG features with subject domain shift
N_SUBJECTS = 8
N_CLASSES = 6
N_FEATURES = 308  # hybrid feature space
N_SAMPLES_PER_SUBJECT = 100

rng = np.random.default_rng(42)
X_list, y_list, groups_list = [], [], []
for s in range(N_SUBJECTS):
    # Each subject has a different "bias" → domain shift
    subject_bias = rng.standard_normal(N_FEATURES) * 0.5
    for _ in range(N_SAMPLES_PER_SUBJECT):
        # Class-correlated signal
        y = rng.integers(0, N_CLASSES)
        x = rng.standard_normal(N_FEATURES) * 0.3 + subject_bias
        x[:50] += y * 0.4  # class signal in first 50 features
        X_list.append(x)
        y_list.append(y)
        groups_list.append(s)

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list)
groups = np.array(groups_list)
print(f"Synthetic data: {X.shape=}, {len(np.unique(y))=} classes, {len(np.unique(groups))=} subjects")

# ============================================================
# CELL 2 — Ablation 1: λ schedule (gradual vs fixed vs none)
# ============================================================
print("\n" + "=" * 60)
print("ABLATION 1: λ schedule")
print("=" * 60)

results = {}
for schedule in ['gradual', 'fixed', 'none']:
    model = LiteDAN(
        n_features=N_FEATURES, n_classes=N_CLASSES, n_domains=N_SUBJECTS,
        n_epochs=10, batch_size=32,
        lambda_schedule=schedule, lambda_max=1.0, use_grl=True,
    )
    # LOSO: train on 7 subjects, test on 1
    test_subj = 0
    train_mask = groups != test_subj
    test_mask = groups == test_subj
    model.fit(X[train_mask], y[train_mask], groups=groups[train_mask])
    preds = model.predict(X[test_mask])
    acc = (preds == y[test_mask]).mean()
    hist = model.training_history()
    final_lambda = hist['lambda'][-1] if hist['lambda'] else 0
    results[schedule] = {'acc': acc, 'final_lambda': final_lambda}
    print(f"  λ={schedule:8s}: test acc = {acc:.3f}  (final λ = {final_lambda:.3f})")

print()
print("Expected pattern (on real data):")
print("  gradual > fixed > none")
print("  (gradual schedule avoids early-training instability)")

# ============================================================
# CELL 3 — Ablation 2: GRL on/off
# ============================================================
print("\n" + "=" * 60)
print("ABLATION 2: Gradient Reversal Layer (GRL)")
print("=" * 60)

for use_grl in [True, False]:
    model = LiteDAN(
        n_features=N_FEATURES, n_classes=N_CLASSES, n_domains=N_SUBJECTS,
        n_epochs=10, batch_size=32,
        lambda_schedule='gradual', use_grl=use_grl,
    )
    test_subj = 0
    train_mask = groups != test_subj
    test_mask = groups == test_subj
    model.fit(X[train_mask], y[train_mask], groups=groups[train_mask])
    preds = model.predict(X[test_mask])
    acc = (preds == y[test_mask]).mean()
    print(f"  use_grl={use_grl}: test acc = {acc:.3f}")

print()
print("Expected pattern (on real data):")
print("  GRL on > GRL off  (adversarial DA reduces domain shift)")

# ============================================================
# CELL 4 — Ablation 3: With/without Euclidean Alignment
# ============================================================
print("\n" + "=" * 60)
print("ABLATION 3: Euclidean Alignment preprocessing")
print("=" * 60)

from myoadapt.adaptation.ea import PerSubjectEA

# Build per-subject signals (pretend X is already feature-extracted)
# For real EA: apply on raw signals BEFORE feature extraction
# Here we approximate by whitening the feature space per subject
for use_ea in [True, False]:
    if use_ea:
        # Apply per-subject whitening (approximation)
        ea = PerSubjectEA()
        X_aligned = X.copy()
        for s in range(N_SUBJECTS):
            mask = groups == s
            sig_s = X[mask]
            # Treat feature vectors as "signals"
            R = ea.fit_subject(s, sig_s)
            X_aligned[mask] = ea.transform_subject(s, sig_s)
        X_use = X_aligned
    else:
        X_use = X

    model = LiteDAN(
        n_features=N_FEATURES, n_classes=N_CLASSES, n_domains=N_SUBJECTS,
        n_epochs=10, batch_size=32,
        lambda_schedule='gradual', use_grl=True,
    )
    test_subj = 0
    train_mask = groups != test_subj
    test_mask = groups == test_subj
    model.fit(X_use[train_mask], y[train_mask], groups=groups[train_mask])
    preds = model.predict(X_use[test_mask])
    acc = (preds == y[test_mask]).mean()
    print(f"  EA={'on' if use_ea else 'off'}: test acc = {acc:.3f}")

# ============================================================
# CELL 5 — Ablation 4: Feature space comparison
# ============================================================
print("\n" + "=" * 60)
print("ABLATION 4: Feature space")
print("=" * 60)

for n_feats, name in [(7, '7 TD only'), (308, '308 hybrid'), (N_FEATURES, 'raw')]:
    # Use subset of features
    if n_feats == 7:
        X_sub = X[:, :7]
    elif n_feats == 308:
        X_sub = X
    else:
        X_sub = X

    model = LiteDAN(
        n_features=n_feats, n_classes=N_CLASSES, n_domains=N_SUBJECTS,
        n_epochs=10, batch_size=32,
        lambda_schedule='gradual', use_grl=True,
    )
    test_subj = 0
    train_mask = groups != test_subj
    test_mask = groups == test_subj
    model.fit(X_sub[train_mask], y[train_mask], groups=groups[train_mask])
    preds = model.predict(X_sub[test_mask])
    acc = (preds == y[test_mask]).mean()
    n_params = model.count_parameters()
    print(f"  {name:15s} ({n_feats:3d} feats, {n_params:,} params): test acc = {acc:.3f}")

# ============================================================
# CELL 6 — Export and benchmark
# ============================================================
print("\n" + "=" * 60)
print("EXPORT + 5-CPU BENCHMARK")
print("=" * 60)

from myoadapt.deployment.onnx_export import OnnxExporter
from myoadapt.deployment.hardware_bench import HardwareBenchmark
import tempfile, os

# Train final Lite-DAN model
final_model = LiteDAN(
    n_features=N_FEATURES, n_classes=N_CLASSES, n_domains=N_SUBJECTS,
    n_epochs=20, batch_size=32,
    lambda_schedule='gradual', use_grl=True,
)
final_model.fit(X, y, groups=groups)

# Hardware benchmark (Lite-DAN inference)
bench = HardwareBenchmark(final_model, n_runs=50)
results = bench.run()
print(f"Lite-DAN parameters: {results['model_parameters']:,}")
print(f"Local mean latency: {results['local_mean_ms']:.2f} ms")
print(f"\n5-CPU projection:")
for cpu, r in results['per_cpu_projected'].items():
    rt = "✓ real-time" if r['realtime_capable'] else "✗ too slow"
    print(f"  {cpu}: {r['projected_mean_ms']:.2f} ms — {rt}")
print(f"\nVerdict: {results['verdict']}")

print("\n" + "=" * 60)
print("ablations complete!")
print("=" * 60)
print("Next steps:")
print("  1. Run on real NinaPro data: scripts/run_lite_dan_ablations.py")
print("  2. Aggregate results with: evaluation.statistics.friedman_test() and wilcoxon_pairwise()")
print("  3. Generate paper-ready tables: evaluation.reports.ReportGenerator()")
