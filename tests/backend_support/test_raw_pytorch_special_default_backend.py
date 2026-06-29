import importlib.util
import math
import os
import subprocess
import sys

import pytest


@pytest.mark.backend_portable
def test_raw_pytorch_special_functions_accept_array_like_inputs_with_numpy_backend():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    env = os.environ.copy()
    env["PYRECEST_BACKEND"] = "numpy"
    src_path = os.path.abspath("src")
    env["PYTHONPATH"] = (
        src_path
        if not env.get("PYTHONPATH")
        else os.pathsep.join([src_path, env["PYTHONPATH"]])
    )

    code = """
import math

import pyrecest  # noqa: F401
import pyrecest.backend as backend
import pyrecest._backend.pytorch as raw_pytorch

assert getattr(backend, "__backend_name__", None) == "numpy"


def assert_close(values, expected):
    actual = raw_pytorch.to_numpy(values).tolist()
    if not isinstance(actual, list):
        actual = [actual]
    assert len(actual) == len(expected)
    for actual_value, expected_value in zip(actual, expected):
        assert abs(actual_value - expected_value) < 1e-6


assert_close(raw_pytorch.erf([0.0, 1.0]), [0.0, math.erf(1.0)])
assert_close(raw_pytorch.gammaln([1.0, 2.0, 4.0]), [0.0, 0.0, math.log(6.0)])
assert_close(raw_pytorch.gamma([1.0, 2.0, 4.0]), [1.0, 1.0, 6.0])
assert_close(
    raw_pytorch.polygamma(1, [1.0, 2.0]),
    [math.pi**2 / 6.0, math.pi**2 / 6.0 - 1.0],
)

out = raw_pytorch.zeros(3, dtype=raw_pytorch.float64)
returned = raw_pytorch.gammaln([1.0, 2.0, 4.0], out=out)
assert returned is out
assert_close(out, [0.0, 0.0, math.log(6.0)])
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
