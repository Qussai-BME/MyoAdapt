"""
run_minirocket_verification.py — MiniROCKET diagnostic at 10K kernels.

Usage:
    python scripts/run_minirocket_verification.py --data-root ./data --db DB2

Generates:
    benchmarks/nina_pro_db2/minirocket/eigenvalue_diagnostic.json
    benchmarks/nina_pro_db2/minirocket/verification_report.md
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from myoadapt.data.loaders import load_dataset
from myoadapt.features.minirocket import MiniRocketFeatures, MiniRocketVerifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="./data",
                        help="Directory containing the NinaPro .mat files.")
    parser.add_argument("--db", default="DB2")
    parser.add_argument("--n-kernels", type=int, default=10_000)
    parser.add_argument("--n-subjects", type=int, default=5,
                        help="Number of subjects to use (for speed)")
    parser.add_argument("--output", default="./benchmarks")
    args = parser.parse_args()

    output_dir = Path(args.output) / f"nina_pro_{args.db.lower()}" / "minirocket"
    output_dir.mkdir(parents=True, exist_ok=True)

    # load_dataset returns a single windowed array of shape
    # (n_windows, n_channels, win_samples). When the real .mat files are
    # absent, a synthetic dataset of the correct shape is returned.
    print(f"Loading {args.db}...")
    windows, labels, groups, meta = load_dataset(
        db=args.db, root=args.data_root, n_subjects=args.n_subjects,
    )
    print(f"Loaded {windows.shape[0]} windows from {meta['n_subjects']} subjects")

    print(f"Extracting MiniROCKET PPV features ({args.n_kernels} kernels)...")
    mr = MiniRocketFeatures(n_kernels=args.n_kernels)
    # Group windows by subject so the per-subject diagnostic is meaningful.
    groups_arr = np.asarray(groups)
    features_per_subject = []
    for i, sid in enumerate(np.unique(groups_arr)):
        subj_windows = windows[groups_arr == sid]
        # Cap at 50 windows per subject for speed.
        if len(subj_windows) > 50:
            subj_windows = subj_windows[:50]
        feats = mr.fit_transform(subj_windows)
        features_per_subject.append(feats)
        print(f"  Subject {sid}: {feats.shape}")

    print(f"\nRunning verifier at {args.n_kernels} kernels...")
    verifier = MiniRocketVerifier(n_kernels=args.n_kernels)
    diag = verifier.diagnose(features_per_subject)

    # Save diagnostic
    diag_path = output_dir / "eigenvalue_diagnostic.json"
    with open(diag_path, "w") as f:
        json.dump(diag, f, indent=2, default=str)
    print(f"\nDiagnostic saved to {diag_path}")

    # Print summary
    print(f"\n{'='*60}")
    print(f" Verification — MiniROCKET PPV Near-Singularity")
    print(f"{'='*60}")
    print(f"Kernels:            {diag['n_kernels']}")
    print(f"Subjects:           {diag['n_subjects']}")
    print(f"Features:           {diag['n_features']}")
    print(f"Total windows:      {diag['n_total_windows']}")
    print(f"Condition number:   {diag['condition_number_global']:.2e}")
    print(f"Fraction near-sing: {diag['fraction_near_singular_global']:.4f}")
    print(f"Median per-subject: {diag['median_condition_number']:.2e}")
    print(f"Verdict:            {diag['verdict']}")

    # Markdown report
    md_path = output_dir / "verification_report.md"
    with open(md_path, "w") as f:
        f.write("#  Verification Report\n\n")
        f.write(f"## MiniROCKET PPV Near-Singularity Diagnostic\n\n")
        f.write(f"- **Database**: {args.db}\n")
        f.write(f"- **Subjects used**: {diag['n_subjects']}\n")
        f.write(f"- **Kernels**: {diag['n_kernels']:,}\n")
        f.write(f"- **Features per window**: {diag['n_features']:,}\n")
        f.write(f"- **Total windows**: {diag['n_total_windows']:,}\n\n")
        f.write(f"## Results\n\n")
        f.write(f"- **Condition number (global)**: `{diag['condition_number_global']:.2e}`\n")
        f.write(f"- **Fraction near-singular**: `{diag['fraction_near_singular_global']:.4f}`\n")
        f.write(f"- **Median per-subject condition**: `{diag['median_condition_number']:.2e}`\n\n")
        f.write(f"## Verdict\n\n")
        f.write(f"**{diag['verdict']}**\n")
    print(f"Report saved to {md_path}")


if __name__ == "__main__":
    main()
