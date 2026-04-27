# Getting Started

This guide covers the shortest path from installation to running a PyRecEst
example.

## Requirements

PyRecEst supports Python 3.11 or newer and earlier than Python 3.15.

## Install From PyPI

Install the default NumPy-backed package with:

```bash
python -m pip install pyrecest
```

Optional extras install additional backend or domain-specific dependencies:

```bash
python -m pip install "pyrecest[pytorch_support]"
python -m pip install "pyrecest[jax_support]"
python -m pip install "pyrecest[healpy_support]"
```

## Install For Development

From a source checkout, install all optional dependencies with Poetry:

```bash
poetry install --with dev --all-extras
```

The repository also provides a conda environment:

```bash
conda env create -f environment.yml
```

Run the test suite from the repository root:

```bash
python -m pytest
```

Build the documentation site, including generated API reference pages, with:

```bash
poetry install --with docs --without dev
poetry run mkdocs build --strict
```

## Run A First Example

The Kalman filter example is the smallest end-to-end filter workflow:

```bash
python examples/basic/kalman_filter.py
```

It creates a one-dimensional constant-velocity model, predicts with a linear
system matrix, updates with scalar position measurements, and prints the current
position and velocity estimate after each measurement.

## Choose A Backend

PyRecEst imports `pyrecest.backend` dynamically. The default backend is NumPy.
Set `PYRECEST_BACKEND` before Python imports `pyrecest` to use another backend.

On bash-compatible shells:

```bash
PYRECEST_BACKEND=pytorch python examples/basic/kalman_filter.py
PYRECEST_BACKEND=jax python examples/basic/kalman_filter.py
```

On PowerShell:

```powershell
$env:PYRECEST_BACKEND = "pytorch"
python examples/basic/kalman_filter.py
Remove-Item Env:PYRECEST_BACKEND
```

Install the matching optional extra before selecting a non-default backend.

Some examples and APIs are backend-specific. For example,
`examples/basic/multi_target_tracking.py` currently checks for the NumPy backend
at runtime.

## Explore More Usage

- Start with [the examples guide](examples.md) for runnable scripts.
- Look at [the API overview](api-overview.md) to find the package that owns a
  concept.
- Use [the API reference](reference/index.md) for generated public package
  reference pages.
- Use `tests/` as executable reference material for advanced APIs that do not
  have dedicated tutorials yet.
