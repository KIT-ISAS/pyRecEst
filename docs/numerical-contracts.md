# Numerical Contracts

Numerical algorithms in PyRecEst should make their repair and validation policy
explicit. A covariance-like matrix should not be silently accepted when it is the
wrong shape, asymmetric beyond tolerance, or not positive semidefinite.

Use `pyrecest.numerics` for shared checks:

```python
from pyrecest.numerics import assert_covariance_matrix, jittered_cholesky

cov = assert_covariance_matrix(cov, dim=2)
factor, jitter = jittered_cholesky(cov)
```

## Recommended Policy

| Situation                      | Preferred behavior                                               |
|--------------------------------|------------------------------------------------------------------|
| Bad shape                      | Raise `ShapeError` or `DimensionMismatchError`.                  |
| Asymmetric covariance          | Raise unless the API explicitly documents symmetrization.        |
| Slightly indefinite covariance | Raise, or use documented jitter/projection in a diagnostic path. |
| Ill-conditioned Cholesky       | Return the jitter used, or raise `NumericalStabilityError`.      |

Repair helpers such as `nearest_symmetric_psd` are useful for diagnostics and
controlled experiments. They should not hide modeling errors in default filter
updates.
