"""
run_cross_subject_loso.py — Cross-subject LOSO benchmark.

Usage:
    python scripts/run_cross_subject_loso.py --data-root ./data --db DB2

Generates:
    benchmarks/nina_pro_db2/<model>/Table2_main_results_DB2.csv
    benchmarks/nina_pro_db2/<model>/per_fold_DB2.csv
    benchmarks/nina_pro_db2/<model>/statistics_DB2.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np

# Ensure myoadapt is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from myoadapt.config import MyoAdaptConfig
from myoadapt.data.loaders import load_dataset
from myoadapt.data.preprocessing import preprocess_signal
from myoadapt.features import extract_features
from myoadapt.models.classical import EMGClassifier
from myoadapt.evaluation import LOSOEvaluator, ReportGenerator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="./data",
                        help="Directory containing the NinaPro .mat files.")
    parser.add_argument("--db", default="DB2",
                        help="NinaPro database identifier (DB1/DB2/DB3/DB7).")
    parser.add_argument("--exercise", default="E1",
                        help="NinaPro exercise selector (kept for backwards "
                             "compatibility; the loader ignores it when the "
                             "real .mat files are not present).")
    parser.add_argument("--output", default="./benchmarks")
    parser.add_argument("--config", default=None,
                        help="Optional YAML config. Falls back to the default "
                             "MyoAdaptConfig when not provided.")
    args = parser.parse_args()

    if args.config:
        cfg = MyoAdaptConfig.from_yaml(args.config)
    else:
        cfg = MyoAdaptConfig()
    cfg.validate()
    output_dir = Path(args.output) / f"nina_pro_{args.db.lower()}"

    # load_dataset returns (X, y, groups, meta) where X is a windowed array
    # of shape (n_windows, n_channels, win_samples). When the real .mat
    # files are absent, a synthetic dataset of the correct shape is
    # returned so the script can be exercised end-to-end.
    print(f"Loading {args.db} from {args.data_root}...")
    windows, labels, groups_arr, meta = load_dataset(db=args.db, root=args.data_root)
    print(f"Loaded {meta['n_subjects']} subjects, {windows.shape[0]} windows "
          f"(synthetic={meta.get('synthetic', False)})")

    # Re-filter each window and extract the standard feature set.
    all_features = []
    for i in range(windows.shape[0]):
        sig = windows[i].T  # preprocess_signal expects (n_samples, n_channels)
        sig_f = preprocess_signal(
            sig, fs=cfg.filter.sampling_rate,
            low=cfg.filter.cutoff_low,
            high=cfg.filter.cutoff_high,
            order=cfg.filter.filter_order,
            notch_freq=cfg.filter.notch_freq,
        )
        feats = extract_features(
            sig_f.T[np.newaxis, :, :],
            modules=["time_domain", "frequency_domain", "histogram", "correlation"],
            fs=cfg.filter.sampling_rate,
            ar_order=cfg.features.ar_order,
        )
        all_features.append(feats)

    X = np.concatenate(all_features, axis=0)
    y = np.asarray(labels)
    g = np.asarray(groups_arr)
    print(f"Features: {X.shape}")

    results_all = {}
    for model_name in ["random_forest", "lda"]:
        print(f"\n--- {model_name} ---")
        evaluator = LOSOEvaluator(
            model_factory=lambda mn=model_name: EMGClassifier(model_type=mn, n_estimators=50),
            use_feature_selection=True,
            k_features=cfg.features.k_best,
            verbose=True,
        )
        results = evaluator.run(X, y, g)
        results_all[model_name] = results
        gen = ReportGenerator(output_dir / model_name)
        gen.generate_tables({model_name: results}, dataset_name=args.db)
        gen.generate_summary_markdown({model_name: results}, dataset_name=args.db)

    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
