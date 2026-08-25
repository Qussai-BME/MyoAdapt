"""
Cross-Subject LOSO Benchmark
============================

This notebook demonstrates the cross-subject Leave-One-Subject-Out (LOSO)
evaluation pipeline:

1. AR coefficient estimator validation against a synthetic AR(2) process
2. Hand-crafted features (420-D) vs naive CNN-1D baseline
3. Rest-class inflation reporting (relevant for amputee populations)
4. Euclidean Alignment ablation (Hahne 2014)
5. SHAP analysis of feature importance

To run on real NinaPro data:
  1. Download NinaPro DB2 from http://ninapro.hevs.ch/db2
  2. Place .mat files in ./data/DB2/
  3. Set USE_REAL_DATA = True below

Without real data, the notebook uses synthetic data to demonstrate the
methodology. The synthetic results will not match real-data benchmarks
but the code path is identical.

"""

USE_REAL_DATA = False  # Set to True after downloading NinaPro DB2

# ============================================================
# CELL 1 — AR coefficient estimator validation
# ============================================================
"""
AR coefficients are estimated with the Yule-Walker (autocorrelation)
method: the autocorrelation sequence is built from the mean-removed
signal, the Toeplitz normal-equations matrix is assembled, and the
system is solved with scipy.linalg.solve. The estimator is validated
against a synthetic AR(2) process with known coefficients.
"""
import numpy as np
from myoadapt.features.time_domain import ar_coefficients, AR_COEFFICIENTS_YULE_WALKER

print(f"AR_COEFFICIENTS_YULE_WALKER = {AR_COEFFICIENTS_YULE_WALKER}")
assert AR_COEFFICIENTS_YULE_WALKER, "Yule-Walker implementation flag must be True"

# Synthetic AR(2) process: x[t] = 0.6*x[t-1] - 0.3*x[t-2] + noise
# Yule-Walker recovers coefficients in LPC convention: [-0.6, 0.3]
rng = np.random.default_rng(42)
n = 5000
x = np.zeros(n)
for t in range(2, n):
    x[t] = 0.6 * x[t-1] - 0.3 * x[t-2] + rng.standard_normal() * 0.1

coeffs = ar_coefficients(x, order=2)
print(f"True generating coefficients: [0.6, -0.3]")
print(f"Recovered (LPC convention):  {coeffs}")
