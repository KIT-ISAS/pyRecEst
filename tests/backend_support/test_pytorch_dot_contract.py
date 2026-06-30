import importlib.util
import os
import subprocess
import sys

import pytest


def _backend_subprocess_env(backend_name):
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
def test_pytorch_dot_matches_pyrecest_contract_for_batched_inputs():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    env = _backend_subprocess_env("pytorch")

    code = """
import numpy.testing as npt
import pyrecest.backend as backend

cases = [
    ([1.0, 2.0], [[5.0, 6.0], [7.0, 8.0]], [17.0, 23.0]),
    ([[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]], [17.0, 53.0]),
    (2.0, [[5.0, 6.0], [7.0, 8.0]], [[10.0, 12.0], [14.0, 16.0]]),
]

for left, right, expected in cases:
    result = backend.dot(backend.array(left), backend.array(right))
    npt.assert_allclose(backend.to_numpy(result), expected)
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


@pytest.mark.backend_portable
def test_raw_pytorch_dot_matches_numpy_when_public_backend_is_numpy():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    env = _backend_subprocess_env("numpy")

    code = """
import numpy.testing as npt
import pyrecest.backend as public_backend
import pyrecest._backend.pytorch as raw_pytorch

assert public_backend.__backend_name__ == "numpy"

cases = [
    ([1.0, 2.0], [[5.0, 6.0], [7.0, 8.0]], [17.0, 23.0]),
    ([[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]], [17.0, 53.0]),
    (2.0, [[5.0, 6.0], [7.0, 8.0]], [[10.0, 12.0], [14.0, 16.0]]),
]

for left, right, expected in cases:
    result = raw_pytorch.dot(raw_pytorch.array(left), raw_pytorch.array(right))
    npt.assert_allclose(raw_pytorch.to_numpy(result), expected)
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
