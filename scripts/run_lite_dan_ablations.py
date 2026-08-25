"""
run_lite_dan_ablations.py — Lite-DAN ablation study.

Ablations:
1. λ schedule: gradual vs fixed vs none
2. Gradient Reversal Layer: with vs without
3. Feature space: 7 TD vs 308 hybrid vs raw windows
4. With/without Euclidean Alignment preprocessing

Usage:
    python scripts/run_lite_dan_ablations.py --data-root ./data --db DB7
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from myoadapt.data.loaders import load_dataset
from myoadapt.data.preprocessing import preprocess_signal
from myoadapt.features import extract_features
from myoadapt.models.lite_dan import LiteDAN
from myoadapt.evaluation import ReportGenerator


def run_ablation(name, X, y, groups, **model_kwargs):
    print(f"\n--- Ablation: {name} ---")
    print(f"  Config: {model_kwargs}")
    from myoadapt.evaluation import LOSOEvaluator
    evaluator = LOSOEvaluator(
        model_factory=lambda kw=model_kwargs: LiteDAN(
            n_features=X.shape[1],
            n_classes=len(np.unique(y)),
            n_domains=len(np.unique(groups)),
            n_epochs=20,  # short for ablation speed
            **kw,
        ),
        verbose=True,
    )
    return evaluator.run(X, y, groups)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="./data",
                        help="Directory containing the NinaPro .mat files.")
    parser.add_argument("--db", default="DB7")
    parser.add_argument("--output", default="./benchmarks/lite_dan_ablations")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # load_dataset returns a single windowed array of shape
    # (n_windows, n_channels, win_samples).
    print(f"Loading {args.db}...")
    windows, labels, groups_arr, meta = load_dataset(db=args.db, root=args.data_root)
    print(f"Loaded {meta['n_subjects']} subjects, {windows.shape[0]} windows "
          f"(synthetic={meta.get('synthetic', False)})")

    print("Extracting 308-hybrid features...")
    all_features = []
    for i in range(windows.shape[0]):
        sig = windows[i].T  # (n_samples, n_channels)
        sig_f = preprocess_signal(sig, fs=meta["fs"])
        feats = extract_features(
            sig_f.T[np.newaxis, :, :],
            modules=["time_domain", "frequency_domain", "histogram", "correlation"],
            fs=meta["fs"],
        )
        all_features.append(feats)
    X = np.concatenate(all_features, axis=0)
    y = np.asarray(labels)
    g = np.asarray(groups_arr)

    # Run ablations
    ablations = {
        "lambda_gradual": {"lambda_schedule": "gradual", "use_grl": True},
        "lambda_fixed":   {"lambda_schedule": "fixed",   "use_grl": True},
        "lambda_none":    {"lambda_schedule": "none",    "use_grl": True},
        "no_grl":         {"lambda_schedule": "gradual", "use_grl": False},
    }
    results = {}
    for name, kw in ablations.items():
        try:
            results[name] = run_ablation(name, X, y, g, **kw)
        except Exception as e:
            print(f"  Ablation {name} failed: {e}")
            results[name] = {"error": str(e)}

    # Drop ablations that failed before generating tables — the report
    # generator expects every entry to have an `aggregate` field.
    clean_results = {n: r for n, r in results.items() if "aggregate" in r}
    if not clean_results:
        print("\nAll ablations failed — no tables generated.")
        return
    gen = ReportGenerator(output_dir)
    gen.generate_tables(clean_results, dataset_name=args.db)
    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
