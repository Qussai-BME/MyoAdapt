# `myoadapt.features` — Feature extraction

## Available modules

| Module | Per-channel features | Total (12 channels) |
|--------|---------------------|---------------------|
| time_domain | 22 (MAV, RMS, ZCR, WL, SSC, WAMP, MYOP, AR, Hjorth, derived) | 264 |
| frequency_domain | 8 (MNF, MDF, PKF, PSR, SNR, SM1-3) | 96 |
| time_frequency | 8 (wavelet packets + STFT bands) | 96 |
| histogram | 10 (amplitude bins) | 120 |
| correlation | 66 (inter-channel, fixed) | 66 |
| minirocket | 10,000 PPV | 120,000 |

## Quick usage

```python
from myoadapt.features import extract_features

# Default: TD + FD + histogram + correlation (264 + 96 + 120 + 66 = 546)
features, names = extract_features(windows, return_names=True)

# With MiniROCKET
features = extract_features(windows, modules=['minirocket'])
```

## AR coefficients

AR coefficients are estimated with the Yule-Walker (autocorrelation)
method. The autocorrelation sequence is built from the mean-removed
signal, the Toeplitz normal-equations matrix is assembled, and the
system is solved with `scipy.linalg.solve`. The estimator is
dependency-free (no `scipy.signal.lpc` requirement) and is validated
against a synthetic AR(2) process with known coefficients.

```python
from myoadapt.features.time_domain import ar_coefficients, AR_COEFFICIENTS_YULE_WALKER

assert AR_COEFFICIENTS_YULE_WALKER is True
coeffs = ar_coefficients(seg, order=4)  # validated against synthetic AR(2)
```

## MiniROCKET verifier

```python
from myoadapt.features import MiniRocketVerifier

verifier = MiniRocketVerifier(n_kernels=10_000)
diag = verifier.diagnose(features_per_subject)
print(diag['condition_number_global']) # ~1e10-1e12 (varies with kernel seed)
print(diag['verdict']) # CONFIRMED — near-singular covariance
```

NOTE: the feature extractor behind this verifier is a MiniROCKET-style
random convolution, not the exact Dempster 2021 algorithm — see the
module-level NOTES in `myoadapt.features.minirocket` for the three
concrete differences (weight distribution, dilation schedule, padding).
The near-singularity diagnostic is robust to this approximation. To
aggregate the per-subject condition numbers into a single CONFIRMED /
PARTIAL / NOT CONFIRMED verdict, call
`verifier.verify_eigenvalue_claim(*features_per_subject)`.

## Custom feature registration

```python
from myoadapt.features import register_feature, extract_features

class MyCustom:
 name = "custom"
 def fit(self, X, y=None): return self
 def transform(self, X): return X[:, :5]

register_feature("custom", MyCustom)
features = extract_features(windows, modules=["custom"])
```
