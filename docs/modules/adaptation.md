# `myoadapt.adaptation` — Domain adaptation

## Available methods

| Method | Type | Suitable for |
|--------|------|-------------|
| `CORAL` | Linear (covariance alignment) | Well-conditioned features (TD) |
| `TCA` | Linear (kernel MMD) | Well-conditioned features |
| `SubspaceAlignment` | Linear (PCA alignment) | Most features |
| `EuclideanAlignment` | Linear (per Hahne 2014) | Raw signals (before windowing) |
| `AdversarialDA` | Adversarial (GRL) | All (especially MiniROCKET PPV) |

## Quick usage

```python
from myoadapt.adaptation import CORAL, TCA, EuclideanAlignment, AdversarialDA

# Linear DA
coral = CORAL()
coral.fit(X_source, X_target)
X_source_aligned = coral.transform(X_source)

# Check if features are well-conditioned
print(coral.condition_number_) # >1e8 = WARNING (MiniROCKET PPV)

# Adversarial DA (jointly trained with classifier)
model = AdversarialDA(n_classes=12, n_domains=12, lambda_schedule='gradual')
model.fit(X_source, y_source, groups_source, X_target, groups_target)
```

## Diagnostics

CORAL and TCA emit warnings when the source covariance is near-singular:

```
WARNING: CORAL: source covariance is near-singular (condition number=9.51e+11).
This typically INCREASES domain shift on MiniROCKET PPV features.
Consider Euclidean Alignment or use a different feature space.
```

This is the **mechanistic explanation** for why linear DA fails on MiniROCKET PPV features.
