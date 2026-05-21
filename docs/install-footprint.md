# Installation Footprint

The default `pyrecest` installation should remain the reliable baseline for most
users. Optional extras enable heavier backend or domain-specific stacks:

```bash
python -m pip install "pyrecest[healpy_support]"
python -m pip install "pyrecest[pytorch_support]"
python -m pip install "pyrecest[jax_support]"
python -m pip install "pyrecest[all_support]"
```

Package indexes and installers normalize extra names. The hyphenated forms are
therefore equivalent in modern pip metadata displays:

```bash
python -m pip install "pyrecest[healpy-support]"
python -m pip install "pyrecest[pytorch-support]"
python -m pip install "pyrecest[jax-support]"
python -m pip install "pyrecest[all-support]"
```

Longer term, package footprint can be reduced by making plotting, manifold, and
heavy spherical-harmonics dependencies optional as well. That split should be
done only when import paths and tests clearly identify which APIs depend on each
extra.

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
