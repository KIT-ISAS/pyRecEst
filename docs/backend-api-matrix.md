# Backend API Matrix

PyRecEst has two related backend contracts:

1. the facade-level contract for functions exposed through `pyrecest.backend`;
2. the public API contract for distributions, filters, trackers, and utilities.

The machine-readable source for both contracts is
`src/pyrecest/_backend/capabilities.py`.

## Support Levels

| Level         | Meaning                                                                                                                                |
|---------------|----------------------------------------------------------------------------------------------------------------------------------------|
| `supported`   | Intended to preserve backend semantics for the listed API.                                                                             |
| `bridged`     | Works by crossing into another numerical stack, usually NumPy/SciPy; do not assume device, dtype, or gradient preservation.            |
| `partial`     | Numerically useful, but with documented limitations such as missing modes or mixed native/bridged implementation.                       |
| `unsupported` | Should raise a clear `NotImplementedError` or be documented as unavailable for the backend.                                            |

## Initial Public API Rows

| API                            | NumPy     | PyTorch     | JAX         | Notes                                                                    |
|--------------------------------|-----------|-------------|-------------|--------------------------------------------------------------------------|
| `KalmanFilter`                 | supported | supported   | supported   | Linear Gaussian operations are the portable baseline.                    |
| `UKFOnManifolds`               | supported | partial     | unsupported | JAX exclusions are currently explicit.                                   |
| `SphericalHarmonicsEOTTracker` | supported | unsupported | unsupported | Depends on spherical harmonics and SciPy-adjacent functionality.         |
| `GaussianDistribution`         | supported | supported   | supported   | Basic construction and portable operations should stay backend portable. |
| `LinearDiracDistribution`      | supported | supported   | supported   | Used by conversion and particle-style workflows.                         |
| `EvaluationUtilities`          | supported | bridged     | bridged     | Plotting, assignment, and summaries remain partly NumPy/SciPy oriented.  |
| `BackendFacade`                | supported | partial     | partial     | Facade names are importable, but some functions are bridged or unsupported. |

When adding a new public API, add a row to the matrix, update docs if the row is
user-facing, and add a focused backend test if the API is expected to be
portable.

## Runtime Access

Use the public helper when examples or downstream packages need to inspect
backend support without duplicating the table:

```python
from pyrecest import get_backend_support

assert get_backend_support("KalmanFilter", backend="jax") == "supported"
```

The CLI can also render the matrix:

```bash
pyrecest backends --format markdown
```
