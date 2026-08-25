# `myoadapt.data` — Data layer

Unified loaders and preprocessing for sEMG datasets.

## Loaders

```python
from myoadapt.data import NinaProLoader, load_dataset, list_databases

print(list_databases())
# ['DB1', 'DB2', 'DB3', 'DB7', 'CapgMyo-DBa', 'UCI']

loader = NinaProLoader(root='./data', db='DB2')
X, y, groups, meta = loader.load_split()
```

`load_split()` returns four values:

- `X` — `(n_windows, n_channels, win_samples)` float32 array of windows
- `y` — `(n_windows,)` int32 gesture labels per window
- `groups` — `(n_windows,)` int32 subject IDs per window
- `meta` — dict with `fs`, `n_channels`, `window_ms`, `increment_ms`,
  `n_subjects`, `n_gestures`, and `synthetic` (True when the real
  NinaPro `.mat` files are not present and a synthetic dataset of the
  correct shape is returned).

When the real `.mat` files are not found, the loader transparently
returns a synthetic dataset of the correct shape so the rest of the
pipeline can be exercised end-to-end. Set `synthetic=False` to disable
the fallback.

## Unified dispatcher

```python
from myoadapt.data import load_dataset

X, y, groups, meta = load_dataset(db='DB2', root='./data')
```

`load_dataset(db, root='./data', n_subjects=None, **kwargs)` is the
single entry point used by the CLI and the evaluators.

## Preprocessing

```python
from myoadapt.data import preprocess_signal, segment_windows

# Butterworth band-pass + IIR notch
filtered = preprocess_signal(signal, fs=2000,
                             low=20, high=450,
                             notch_freq=50)

# Sliding window
windows, window_labels = segment_windows(
    filtered, fs=2000, window_ms=200, increment_ms=50, labels=labels,
)
```

The individual filters are also available as `filter_signal` (band-pass)
and `notch_filter` (IIR notch).

## Splits

```python
from myoadapt.data import loso_splits, lodo_splits, train_test_split_subject

# Leave-One-Subject-Out
for train_idx, test_idx in loso_splits(groups):
    ...

# Leave-One-Database-Out
for train_idx, test_idx in lodo_splits(database_ids):
    ...

# Single train/test split by subject
train_idx, test_idx = train_test_split_subject(groups, test_size=0.3)
```

## Euclidean Alignment

```python
from myoadapt.data import compute_alignment_matrix, compute_subject_alignment, apply_euclidean_alignment

# Per-subject whitening (Hahne 2014)
R = compute_subject_alignment(signal)
aligned = apply_euclidean_alignment(signal, R)
```

`compute_subject_alignment` is the per-subject whitening matrix used by
`PerSubjectEA` in `myoadapt.adaptation`. `compute_alignment_matrix` is
the legacy average-matrix variant kept for backward compatibility.
