import importlib.util
import os
import subprocess
import sys

import pytest


def _env_with_backend(backend_name):
    env = os.environ.copy()
    env["PYRECEST_BACKEND"] = backend_name
    src_path = os.path.abspath("src")
    env["PYTHONPATH"] = (
        src_path
        if not env.get("PYTHONPATH")
        else os.pathsep.join([src_path, env["PYTHONPATH"]])
    )
    return env


@pytest.mark.backend_portable
def test_raw_pytorch_unary_predicates_accept_array_like_with_numpy_backend():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    code = """
import numpy as np

import pyrecest  # noqa: F401
import pyrecest.backend as backend
import pyrecest._backend.pytorch as raw_pytorch

assert getattr(backend, "__backend_name__", None) == "numpy"

assert raw_pytorch.isfinite([1.0, float("inf"), float("nan")]).tolist() == [True, False, False]
assert raw_pytorch.isinf([-float("inf"), 0.0, float("inf"), float("nan")]).tolist() == [True, False, True, False]
assert raw_pytorch.isnan([1.0, float("nan")]).tolist() == [False, True]
assert raw_pytorch.isreal(np.array([1.0 + 0.0j, 1.0 + 2.0j])).tolist() == [True, False]

numpy_result = backend.isfinite([1.0, float("nan")])
assert numpy_result.tolist() == [True, False]
assert type(numpy_result).__module__.startswith("numpy")
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=_env_with_backend("numpy"))


@pytest.mark.backend_portable
def test_public_pytorch_unary_predicates_accept_array_like_inputs():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    code = """
import pyrecest  # noqa: F401
import pyrecest.backend as backend

assert getattr(backend, "__backend_name__", None) == "pytorch"

assert backend.isfinite([1.0, float("nan")]).tolist() == [True, False]
assert backend.isinf([-float("inf"), 0.0, float("inf")]).tolist() == [True, False, True]
assert backend.isnan([1.0, float("nan")]).tolist() == [False, True]
assert backend.isreal([1.0 + 0.0j, 1.0 + 2.0j]).tolist() == [True, False]
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=_env_with_backend("pytorch"))
