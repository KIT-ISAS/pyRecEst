# Installation Footprint

The default `pyrecest` installation should remain the reliable baseline for most
users. Optional extras enable heavier backend or domain-specific stacks:

```bash
python -m pip install "pyrecest[healpy-support]"
python -m pip install "pyrecest[pytorch-support]"
python -m pip install "pyrecest[jax-support]"
python -m pip install "pyrecest[all-support]"
```

CI includes a default-install smoke test so that the package remains importable
and basic filtering remains usable without optional backend extras.


Until that split is complete, the default installation remains the supported
baseline for plotting, evaluation, and most NumPy workflows.

Recommended future shape:

| Extra             | Intended contents                                                     |
|-------------------|-----------------------------------------------------------------------|
| default           | Minimal Euclidean distributions, filters, models, and backend facade. |
| `plot`            | Matplotlib-dependent plotting helpers.                                |
| `manifolds`       | Domain-specific manifold and spherical-harmonics dependencies.        |
| `pytorch_support` | PyTorch backend.                                                      |
| `jax_support`     | JAX backend and autodiff support.                                     |
| `all_support`     | Full feature set for development and exploration.                     |
