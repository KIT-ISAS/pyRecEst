"""Local development sessions mirroring the main CI entry points."""

from __future__ import annotations

import nox

PYTHON_DEFAULT = "3.13"


def _poetry_install(session: nox.Session, *args: str) -> None:
    session.run("poetry", "install", *args, external=True)


@nox.session(python=PYTHON_DEFAULT)
def tests_numpy(session: nox.Session) -> None:
    """Run the default NumPy test suite."""
    _poetry_install(session, "--with", "dev", "--extras", "healpy_support")
    session.env["PYRECEST_BACKEND"] = "numpy"
    session.run(
        "poetry",
        "run",
        "python",
        "-m",
        "pytest",
        "--rootdir",
        ".",
        "-v",
        "./tests",
        external=True,
    )


@nox.session(python=PYTHON_DEFAULT)
def tests_pytorch(session: nox.Session) -> None:
    """Run the PyTorch backend test suite."""
    _poetry_install(
        session, "--with", "dev", "--extras", "healpy_support pytorch_support"
    )
    session.env["PYRECEST_BACKEND"] = "pytorch"
    session.run(
        "poetry",
        "run",
        "python",
        "-m",
        "pytest",
        "--rootdir",
        ".",
        "-v",
        "./tests",
        external=True,
    )


@nox.session(python=PYTHON_DEFAULT)
def tests_jax(session: nox.Session) -> None:
    """Run the JAX backend test suite."""
    _poetry_install(session, "--with", "dev", "--extras", "healpy_support jax_support")
    session.env["PYRECEST_BACKEND"] = "jax"
    session.env["JAX_ENABLE_X64"] = "True"
    session.run(
        "poetry",
        "run",
        "python",
        "-m",
        "pytest",
        "--rootdir",
        ".",
        "-v",
        "./tests",
        external=True,
    )


@nox.session(python=PYTHON_DEFAULT)
def docs(session: nox.Session) -> None:
    """Build documentation in strict mode."""
    _poetry_install(session, "--with", "docs", "--without", "dev")
    session.env["PYTHONPATH"] = "src"
    session.run("poetry", "run", "mkdocs", "build", "--strict", external=True)


@nox.session(python=PYTHON_DEFAULT)
def numerical_stress(session: nox.Session) -> None:
    """Run slower numerical-stability tests."""
    _poetry_install(session, "--with", "dev", "--extras", "healpy_support")
    session.env["PYRECEST_BACKEND"] = "numpy"
    session.run(
        "poetry",
        "run",
        "python",
        "-m",
        "pytest",
        "--rootdir",
        ".",
        "-v",
        "-m",
        "numerical_stress",
        "./tests",
        external=True,
    )


@nox.session(python=PYTHON_DEFAULT)
def benchmarks(session: nox.Session) -> None:
    """Run optional benchmark tests."""
    _poetry_install(session, "--with", "dev")
    session.env["PYRECEST_BACKEND"] = "numpy"
    session.run(
        "poetry",
        "run",
        "python",
        "-m",
        "pytest",
        "-m",
        "benchmark",
        "./benchmarks",
        external=True,
    )


@nox.session(python=PYTHON_DEFAULT)
def lint(session: nox.Session) -> None:
    """Run focused static checks that are expected to stay clean."""
    session.install("ruff>=0.8,<1.0")
    session.run("ruff", "check", ".")


@nox.session(python=PYTHON_DEFAULT)
def package(session: nox.Session) -> None:
    """Build distributions and validate local release metadata."""
    session.install("build", "twine")
    session.run("python", "scripts/check_release_consistency.py", "--local-only")
    session.run("python", "-m", "build")
    session.run("sh", "-c", "python -m twine check dist/*", external=True)


@nox.session(python=PYTHON_DEFAULT)
def minimal_install(session: nox.Session) -> None:
    """Install the default package only and run a public API smoke check."""
    session.run("python", "-m", "pip", "install", "--upgrade", "pip")
    session.run("python", "-m", "pip", "install", ".")
    session.run(
        "python",
        "-c",
        (
            "import pyrecest; "
            "from pyrecest.backend import array, diag; "
            "from pyrecest.filters import KalmanFilter; "
            "kf = KalmanFilter((array([0.0, 1.0]), diag(array([1.0, 1.0])))); "
            "print(pyrecest.__version__, kf.get_point_estimate())"
        ),
    )


@nox.session(python=PYTHON_DEFAULT)
def benchmark_regressions(session: nox.Session) -> None:
    """Run deterministic benchmark scenarios and compare numerical outputs."""
    _poetry_install(session, "--with", "dev")
    session.env["PYRECEST_BACKEND"] = "numpy"
    session.run(
        "poetry",
        "run",
        "python",
        "benchmarks/basic_regressions.py",
        "--output",
        "benchmark-results.json",
        external=True,
    )
    session.run(
        "poetry",
        "run",
        "python",
        "scripts/check_benchmark_results.py",
        "benchmark-results.json",
        "--baseline",
        "benchmarks/baselines/basic_regressions.json",
        external=True,
    )
