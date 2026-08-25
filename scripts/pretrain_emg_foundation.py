"""pretrain_emg_foundation.py — Pretrain the EMG Foundation Model on NinaPro DB1-DB7.

Usage:
    python scripts/pretrain_emg_foundation.py --data-root ./data --output ./models/emg_foundation.pt

This script loads NinaPro DB1, DB2, DB3, DB7 (and optionally CapgMyo-DBa, UCI),
concatenates the raw signals, and runs self-supervised masked-patch reconstruction
pretraining on the EMGFoundation model.

NOTE: MyoAdapt does not ship a pretrained checkpoint. This script is the
documented way to produce one. When the real NinaPro ``.mat`` files are not
present under ``--data-root``, the loaders fall back to synthetic data of the
correct shape (see ``myoadapt.data.loaders``); inspect the ``synthetic``
flag in each loader's metadata to confirm whether real or synthetic data was
used. The resulting checkpoint is only as good as the data it saw.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Make the package importable when running the script directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from myoadapt.data.loaders import load_dataset  # noqa: E402
from myoadapt.models.emg_foundation import EMGFoundation  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Pretrain the EMG Foundation Model on NinaPro DB1-DB7 (and "
            "optionally CapgMyo-DBa / UCI) via masked-patch reconstruction."
        )
    )
    parser.add_argument("--data-root", default="./data",
                        help="Directory containing the NinaPro .mat files.")
    parser.add_argument("--output", default="./models/emg_foundation.pt",
                        help="Where to save the pretrained checkpoint.")
    parser.add_argument("--n-epochs", type=int, default=50,
                        help="Number of SSL pretraining epochs.")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Mini-batch size for SSL pretraining.")
    parser.add_argument(
        "--databases", nargs="+",
        default=["DB1", "DB2", "DB3", "DB7", "CapgMyo-DBa", "UCI"],
        help="Databases to concatenate (in order). Defaults to all six "
             "supported databases.",
    )
    args = parser.parse_args()

    # Load and concatenate all requested databases. Databases with
    # different channel counts cannot be concatenated along the window
    # axis, so we keep only those matching the first successfully loaded
    # database's channel count (and warn about the rest).
    all_signals = []
    n_channels_target = None
    for db in args.databases:
        try:
            X, _, _, meta = load_dataset(db=db, root=args.data_root)
        except Exception as e:
            print(f"Skipped {db}: {e}")
            continue
        synthetic = bool(meta.get("synthetic", False))
        n_ch = int(meta.get("n_channels", X.shape[1]))
        if n_channels_target is None:
            n_channels_target = n_ch
        if n_ch != n_channels_target:
            print(
                f"Skipped {db}: {n_ch} channels (expected "
                f"{n_channels_target}); cannot concatenate with prior "
                f"databases. Re-run with --databases {db} ... to "
                f"pretrain on a different channel count."
            )
            continue
        print(f"Loaded {db}: {X.shape} (synthetic={synthetic})")
        all_signals.append(X)

    if not all_signals:
        raise RuntimeError(
            "No data loaded. Place NinaPro .mat files under "
            f"{args.data_root!r}, or accept the synthetic fallback by "
            "ensuring at least one supported database name resolves."
        )

    X_all = np.concatenate(all_signals, axis=0)
    print(f"Total concatenated: {X_all.shape}")

    model = EMGFoundation(n_channels=X_all.shape[1])
    model.pretrain(
        X_all,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
    )
    model.save(args.output)
    print(f"Saved pretrained model to {args.output}")

    hist = model.training_history()
    if hist.get("ssl_loss"):
        first, last = hist["ssl_loss"][0], hist["ssl_loss"][-1]
        print(f"SSL MSE loss: epoch 1 = {first:.6f}, "
              f"epoch {len(hist['ssl_loss'])} = {last:.6f}")


if __name__ == "__main__":
    main()
