# Installation Footprint

The default `pyrecest` installation should remain the reliable baseline for most
users. Optional extras enable heavier backend or domain-specific stacks:

```bash
python -m pip install "pyrecest[healpy-support]"
python -m pip install "pyrecest[pytorch-support]"
python -m pip install "pyrecest[jax-support]"
python -m pip install "pyrecest[all-support]"
```

The package workflow now includes a default-wheel smoke test for the baseline
public API. Keep that job green before moving any dependency behind an optional
extra. It catches accidental import-time coupling such as a Euclidean filter
requiring a plotting or spherical-harmonics package.

Recommended staged shape:

| Extra             | Intended contents                                                     | Migration check |
|-------------------|-----------------------------------------------------------------------|-----------------|
| default           | Minimal Euclidean distributions, filters, models, and backend facade. | Default-wheel smoke test passes without optional extras. |
| `plot`            | Matplotlib-dependent plotting helpers.                                | Plotting imports are local to plotting methods. |
| `manifolds`       | Domain-specific manifold and spherical-harmonics dependencies.        | Manifold-heavy public APIs have explicit backend/extra docs. |
| `pytorch-support` | PyTorch backend.                                                      | Backend matrix tests pass with `PYRECEST_BACKEND=pytorch`. |
| `jax-support`     | JAX backend and autodiff support.                                     | Backend matrix tests pass with `PYRECEST_BACKEND=jax` and x64 enabled. |
| `all-support`     | Full feature set for development and exploration.                     | Full scheduled matrix passes. |

Use hyphenated extra names in public installation snippets. Pip normalizes
underscores and hyphens, but using one public spelling improves searchability and
matches the form displayed by package indexes.

When an existing required dependency is moved to an extra, update the public API
registry and add a focused import test for every symbol whose dependency changed.
