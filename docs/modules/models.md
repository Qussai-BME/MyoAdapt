# `myoadapt.models` — Model registry

## Available models

| Name | Type | Parameters | CPU-native? |
|------|------|-----------|-----------|
| `classical` | XGBoost, RF, LDA, SVM, Logistic, ExtraTrees, LightGBM | varies | ✓ |
| `cnn1d` | 1D-CNN | ~15K (default config) | ✓ |
| `lite_dan` | Domain Adversarial Network | ~53K (default config; varies with n_features/n_classes/n_domains) | ✓ |
| `emg_foundation` | Transformer encoder | ~200K | ✓ |
| `lstm_regressor` | LSTM for continuous decoding | ~50K | ✓ |
| `transformer_regressor` | Transformer for continuous decoding | ~100K | ✓ |

## Quick usage

```python
from myoadapt.models import EMGClassifier, LiteDAN, get_model

# Classical
clf = EMGClassifier(model_type='xgboost', k_features=420)
clf.fit(X_train, y_train)
clf.save('./model.pkl')

# Lite-DAN (requires subject IDs for domain branch)
model = LiteDAN(n_features=308, n_classes=12, n_domains=12,
 lambda_schedule='gradual', use_grl=True)
model.fit(X, y, groups=subject_ids)
```

## Ablation hooks (Lite-DAN)

For ablation study:

```python
# λ schedule ablation
for schedule in ['gradual', 'fixed', 'none']:
 model = LiteDAN(lambda_schedule=schedule, ...)
 model.fit(X, y, groups=groups)

# GRL ablation
for use_grl in [True, False]:
 model = LiteDAN(use_grl=use_grl, ...)
```

## EMG Foundation Model

```python
from myoadapt.models import EMGFoundation

fm = EMGFoundation(n_channels=12, n_samples=400, n_classes=12)
fm.pretrain(X_unlabeled, n_epochs=50) # self-supervised
fm.fit(X_labeled, y) # supervised fine-tune
features = fm.extract_features(X) # 64-D learned features
```
