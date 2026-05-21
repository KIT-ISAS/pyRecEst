# Public API Registry

This registry records public APIs that are currently tracked for release
stability and backend-portability decisions. The machine-readable source is
`src/pyrecest/api_registry.py`.

Run this check after adding, removing, stabilizing, deprecating, or reclassifying
user-facing APIs:

```bash
PYTHONPATH=src python scripts/check_public_api_registry.py --check docs/public-api-registry.md
```

<!-- public-api-registry:start -->
| API | Module | Category | Backend contract | Notes |
|-----|--------|----------|------------------|-------|
| `DistributionConversion` | `pyrecest.distributions.conversion` | backend-specific | `DistributionConversion` | Euclidean Gaussian/particle routes are portable; grid, Fourier, and manifold routes are route-specific. |
| `EuclideanParticleFilter` | `pyrecest.filters` | backend-specific | `EuclideanParticleFilter` | Particle behavior depends on sampler and resampling support in the active backend. |
| `EvaluationUtilities` | `pyrecest.evaluation` | backend-specific | `EvaluationUtilities` | Plotting, assignment, summaries, and result helpers are only partly backend-portable. |
| `GaussianDistribution` | `pyrecest.distributions` | stable | `GaussianDistribution` | Basic construction, moment access, and portable operations are part of the core distribution API. |
| `KalmanFilter` | `pyrecest.filters` | stable | `KalmanFilter` | Linear Gaussian filtering is part of the portable baseline. |
| `LinearDiracDistribution` | `pyrecest.distributions` | stable | `LinearDiracDistribution` | Core particle-style representation used by conversion and filtering workflows. |
| `MultiBernoulliTracker` | `pyrecest.filters` | backend-specific | `MultiBernoulliTracker` | Tracking workflows rely on assignment and measurement-set utilities with NumPy-oriented paths. |
| `PointSetRegistration` | `pyrecest.utils` | backend-specific | `PointSetRegistration` | Registration helpers may bridge through NumPy/SciPy and are not guaranteed differentiable. |
| `SphericalHarmonicsEOTTracker` | `pyrecest.filters` | backend-specific | `SphericalHarmonicsEOTTracker` | Depends on spherical-harmonics and SciPy-adjacent functionality. |
| `UKFOnManifolds` | `pyrecest.filters` | backend-specific | `UKFOnManifolds` | Current predict/update paths explicitly exclude JAX. |
| `UnscentedKalmanFilter` | `pyrecest.filters` | backend-specific | `UnscentedKalmanFilter` | Portable for backend-compatible model functions; advanced paths may bridge through NumPy/SciPy. |
<!-- public-api-registry:end -->
