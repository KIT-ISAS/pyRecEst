import importlib.util
import math
import os
import subprocess
import sys

import pytest


@pytest.mark.backend_portable
def test_pytorch_special_functions_accept_array_like_inputs():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    env = os.environ.copy()
    env["PYRECEST_BACKEND"] = "pytorch"
    src_path = os.path.abspath("src")
    env["PYTHONPATH"] = (
        src_path
        if not env.get("PYTHONPATH")
        else os.pathsep.join([src_path, env["PYTHONPATH"]])
    )

    code = """
import math

import pyrecest.backend as backend
import pyrecest._backend.pytorch as raw_backend


def assert_close(values, expected):
    actual = backend.to_numpy(values).tolist()
    if not isinstance(actual, list):
        actual = [actual]
    assert len(actual) == len(expected)
    for actual_value, expected_value in zip(actual, expected):
        assert abs(actual_value - expected_value) < 1e-6


for special_backend in (backend, raw_backend):
    assert_close(special_backend.erf([0.0, 1.0]), [0.0, math.erf(1.0)])
    assert_close(special_backend.gammaln([1.0, 2.0, 4.0]), [0.0, 0.0, math.log(6.0)])
    assert_close(special_backend.gamma([1.0, 2.0, 4.0]), [1.0, 1.0, 6.0])
    assert_close(
        special_backend.polygamma(1, [1.0, 2.0]),
        [math.pi**2 / 6.0, math.pi**2 / 6.0 - 1.0],
    )
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
