"""
Cross-Database Meta-Analysis (LODO)
==============================================

This notebook demonstrates the Leave-One-Database-Out (LODO) protocol.
LODO evaluates cross-database generalization:
train on N-1 databases, test on the held-out database.

This is the strictest generalization test in the EMG literature:
  - LOSO: cross-subject within one database
  - LODO: cross-database (different subjects, different sensors,
          different acquisition protocols, different gesture sets)

"""

# ============================================================
# CELL 1 — Simulate multiple databases with different characteristics
# ============================================================
import numpy as np

# Each "database" has different characteristics
DATABASES = {
    "NinaPro-DB2": {"n_channels": 12, "n_subjects": 5, "n_classes": 8,
                    "scale": 1.0, "noise": 0.3, "sampling": 2000},
    "NinaPro-DB1": {"n_channels": 10, "n_subjects": 4, "n_classes": 6,
                    "scale": 1.5, "noise": 0.4, "sampling": 100},
    "CapgMyo-DBa": {"n_channels": 16, "n_subjects": 6, "n_classes": 8,
                    "scale": 0.8, "noise": 0.2, "sampling": 1000},
    "UCI-EMG":     {"n_channels": 8,  "n_subjects": 4, "n_classes": 4,
                    "scale": 1.2, "noise": 0.5, "sampling": 500},
}

def simulate_database(name, params, n_samples_per_subject=30, n_features=308):
    rng = np.random.default_rng(hash(name) % 2**32)
    X_list, y_list, subj_list = [], [], []
    for s in range(params["n_subjects"]):
        subj_bias = rng.standard_normal(n_features) * params["scale"]
        for _ in range(n_samples_per_subject):
            y = rng.integers(0, params["n_classes"])
            x = rng.standard_normal(n_features) * params["noise"] + subj_bias
            x[:50] += y * 0.5  # class signal
            X_list.append(x)
            y_list.append(y)
            subj_list.append(f"{name}_s{s}")
    return np.array(X_list, dtype=np.float32), np.array(y_list), np.array(subj_list)

# Simulate all databases
print("Simulated databases:")
data = {}
for name, params in DATABASES.items():
    X, y, s = simulate_database(name, params)
    data[name] = {"X": X, "y": y, "subjects": s}
    print(f"  {name}: {X.shape}, {len(np.unique(y))} classes, "
          f"{len(np.unique(s))} subjects, scale={params['scale']}")

# ============================================================
# CELL 2 — LODO evaluation
# ============================================================
from myoadapt.models.classical import EMGClassifier
from myoadapt.evaluation.metrics import compute_metrics

print("\n" + "=" * 70)
print("LODO: Leave-One-Database-Out Cross-Validation")
print("=" * 70)

lodo_results = []
db_names = list(data.keys())
for test_db in db_names:
    # Train on all other databases
    train_X = np.vstack([data[d]["X"] for d in db_names if d != test_db])
    train_y = np.concatenate([data[d]["y"] for d in db_names if d != test_db])

    # Test on the held-out database
    test_X = data[test_db]["X"]
    test_y = data[test_db]["y"]

    # Train classifier
    clf = EMGClassifier(model_type="random_forest", n_estimators=100, random_state=42)
    clf.fit(train_X, train_y)
    preds = clf.predict(test_X)

    # Metrics — `classes` is required so the confusion matrix is shaped
    # over the full label set even when a single fold only sees a subset.
    classes = sorted(np.unique(np.concatenate([train_y, test_y])).tolist())
    m = compute_metrics(test_y, preds, classes=classes, rest_class=None)
    lodo_results.append({
        "test_db": test_db,
        "n_train_samples": len(train_X),
        "n_test_samples": len(test_X),
        "n_classes": len(np.unique(test_y)),
        **m,
    })
    print(f"\n  Test on {test_db}:")
    print(f"    accuracy:  {m['accuracy']:.3f}")
    print(f"    macro_f1:  {m.get('macro_f1', 0):.3f}")

# ============================================================
# CELL 3 — Compare LOSO vs LODO
# ============================================================
print("\n" + "=" * 70)
print("Comparison: LOSO (within-database) vs LODO (cross-database)")
print("=" * 70)

# LOSO on each database
loso_results = []
for db_name in db_names:
    X = data[db_name]["X"]
    y = data[db_name]["y"]
    subjects = data[db_name]["subjects"]
    unique_subj = np.unique(subjects)

    fold_accs = []
    for test_subj in unique_subj:
        train_mask = subjects != test_subj
        test_mask = subjects == test_subj
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue
        clf = EMGClassifier(model_type="random_forest", n_estimators=50, random_state=42)
        clf.fit(X[train_mask], y[train_mask])
        preds = clf.predict(X[test_mask])
        fold_accs.append((preds == y[test_mask]).mean())

    loso_results.append({
        "db": db_name,
        "loso_acc_mean": np.mean(fold_accs),
        "loso_acc_std": np.std(fold_accs),
        "lodo_acc": next(r["accuracy"] for r in lodo_results if r["test_db"] == db_name),
    })

print(f"\n{'Database':<15} {'LOSO acc':>15} {'LODO acc':>15} {'Gap':>10}")
print("-" * 60)
for r in loso_results:
    gap = r["loso_acc_mean"] - r["lodo_acc"]
    print(f"{r['db']:<15} {r['loso_acc_mean']:.3f} ± {r['loso_acc_std']:.3f}   "
          f"{r['lodo_acc']:.3f}        {gap:+.3f}")

print()
print("KEY FINDINGS:")
print("  - LODO accuracy is typically 15-30pp lower than LOSO")
print("  - The gap quantifies cross-database domain shift")
print("  - Domain adaptation methods (CORAL, EA, Lite-DAN) can help close this gap")

# ============================================================
# CELL 4 — Statistical analysis
# ============================================================
print("\n" + "=" * 70)
print("Statistical analysis (Friedman + Wilcoxon + Holm-Šídák)")
print("=" * 70)

from myoadapt.evaluation.statistics import (
    friedman_test, wilcoxon_pairwise, cohen_d,
)

# Compare 3 models on the 4 databases (LODO)
# Model A: RandomForest (baseline)
# Model B: RandomForest + Euclidean Alignment
# Model C: Lite-DAN
# (Here we just simulate the comparison)

rng = np.random.default_rng(0)
model_names = ["RF (baseline)", "RF + EA", "Lite-DAN"]
model_results = {m: [] for m in model_names}
for db_name in db_names:
    model_results["RF (baseline)"].append(float(rng.uniform(0.4, 0.55)))
    model_results["RF + EA"].append(float(rng.uniform(0.5, 0.65)))
    model_results["Lite-DAN"].append(float(rng.uniform(0.55, 0.7)))

print("\nPer-database LODO accuracy:")
print(f"{'Database':<15}", end="")
for m in model_names:
    print(f"  {m:<18}", end="")
print()
for i, db in enumerate(db_names):
    print(f"{db:<15}", end="")
    for m in model_names:
        print(f"  {model_results[m][i]:.3f}            ", end="")
    print()

# Friedman test (non-parametric comparison across databases)
fried = friedman_test(model_results)
print(f"\nFriedman test: χ² = {fried['chi2']:.3f}, p = {fried['p_value']:.4f}")
if fried["p_value"] < 0.05:
    print("  → Significant difference between models (p < 0.05)")
else:
    print("  → No significant difference")

# Pairwise Wilcoxon tests with Holm-Šídák correction
print("\nPairwise Wilcoxon tests (with Holm-Šídák correction):")
wilc = wilcoxon_pairwise(model_results)
for test in wilc.get("tests", []):
    a, b = test["model_a"], test["model_b"]
    p_raw = test.get("p_value_raw", test.get("p_value", 1.0))
    p_corr = test.get("p_value_corrected", p_raw)
    print(f"  {a} vs {b}: p_raw = {p_raw:.4f}, p_corrected = {p_corr:.4f}")

# Cohen's d for effect size
print("\nEffect sizes (Cohen's d):")
pairs = [(0, 1), (0, 2), (1, 2)]
for i, j in pairs:
    a = np.asarray(model_results[model_names[i]])
    b = np.asarray(model_results[model_names[j]])
    d = cohen_d(a, b)
    interpretation = ("small" if abs(d) < 0.5
                      else "medium" if abs(d) < 0.8
                      else "large")
    print(f"  {model_names[i]} vs {model_names[j]}: d = {d:.3f} ({interpretation})")

# ============================================================
# CELL 5 — Generate paper-ready report
# ============================================================
print("\n" + "=" * 70)
print("Paper-ready report")
print("=" * 70)

# Aggregate all results
all_results = []
for r in loso_results:
    all_results.append({
        "database": r["db"],
        "protocol": "LOSO",
        "model": "RF (baseline)",
        "accuracy": r["loso_acc_mean"],
        "std": r["loso_acc_std"],
    })
for r in lodo_results:
    all_results.append({
        "database": r["test_db"],
        "protocol": "LODO",
        "model": "RF (baseline)",
        "accuracy": r["accuracy"],
        "std": 0.0,
    })

print(f"\nResults summary ({len(all_results)} experiments):")
print(f"  LOSO mean accuracy: {np.mean([r['accuracy'] for r in all_results if r['protocol']=='LOSO']):.3f}")
print(f"  LODO mean accuracy: {np.mean([r['accuracy'] for r in all_results if r['protocol']=='LODO']):.3f}")
print(f"  LOSO-LODO gap:      {np.mean([r['accuracy'] for r in all_results if r['protocol']=='LOSO']) - np.mean([r['accuracy'] for r in all_results if r['protocol']=='LODO']):.3f}")

print("\n" + "=" * 70)
print("Meta-analysis complete.")
print("=" * 70)
print("Next steps:")
print("  1. Run on real NinaPro DB1/2/3/7 + CapgMyo + UCI")
print("  2. Add domain adaptation methods (CORAL, EA, Lite-DAN) to the comparison")
print("  3. Contribute the cross-database benchmark to the community")
