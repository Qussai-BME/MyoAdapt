# Benchmarks

Reserved for published benchmark results. Use the scripts under
`scripts/` to populate this directory with your own runs.

## Structure

```
benchmarks/
├── nina_pro_db2/ # LOSO results on NinaPro DB2 (40 subjects)
│ ├── xgboost/
│ ├── random_forest/
│ ├── lda/
│ ├── cnn1d/
│ └── lite_dan/
├── nina_pro_db3/ # LOSO results on NinaPro DB3 (11 amputees)
│ └── ...
├── nina_pro_db7/ # LOSO results on NinaPro DB7 (22 subjects)
│ └── ...
├── lodo/ # Cross-database results
│ └── ...
└── hardware/ # 5-CPU benchmark results
 └── ...
```

## Format

Each model directory contains:
- `Table2_main_results.csv` — aggregate metrics
- `per_fold.csv` — per-fold metrics
- `per_class.csv` — per-class F1
- `statistics.csv` — Friedman + Wilcoxon
- `summary.md` — Markdown report

## Reproducing

```bash
# Replicate a benchmark
myoadapt train --db DB2 --model xgboost
myoadapt eval --model-path ./models/xgb_db2.pkl --db DB2 \
 --output ./benchmarks/nina_pro_db2/xgboost/

# Generate a leaderboard
myoadapt eval --model-path ./models/lite_dan_db2.pkl --db DB2 \
 --output ./benchmarks/nina_pro_db2/lite_dan/
```
