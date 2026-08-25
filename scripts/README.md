# MyoAdapt scripts

Helper scripts for running benchmarks and verification studies.

| Script | What it does |
|---|---|
| `run_cross_subject_loso.py` | Full cross-subject LOSO benchmark on a NinaPro database |
| `run_minirocket_verification.py` | MiniROCKET PPV near-singularity diagnostic at 10K kernels |
| `run_lite_dan_ablations.py` | Lite-DAN ablation study (λ schedule, GRL on/off) |
| `make_sample_data.py` | Generates the synthetic sample dataset shipped under `data/sample/` |
| `pretrain_emg_foundation.py` | Self-supervised pretraining of the EMG Foundation Model |

## Usage

```bash
# Cross-subject LOSO on NinaPro DB2 (or synthetic fallback if no .mat files)
python scripts/run_cross_subject_loso.py --data-root ./data --db DB2

# MiniROCKET PPV diagnostic
python scripts/run_minirocket_verification.py --data-root ./data --db DB2 --n-kernels 10000

# Lite-DAN ablation table
python scripts/run_lite_dan_ablations.py --data-root ./data --db DB7

# Regenerate the synthetic sample dataset
python scripts/make_sample_data.py

# Pretrain the EMG Foundation Model on a concatenated corpus
python scripts/pretrain_emg_foundation.py --data-root ./data --output ./models/emg_foundation.pt
```

Each script writes its results to a CSV/JSON in the output directory and
prints a summary table to stdout. When the real NinaPro `.mat` files are
not present under `--data-root`, the loaders fall back to a synthetic
dataset of the correct shape so the script can be exercised end-to-end.
