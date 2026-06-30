import importlib.util
import os
import subprocess
import sys

import pytest


def _backend_test_env(backend_name):
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
def test_pytorch_diagonal_accepts_numpy_scalar_array_indices():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    code = """
import numpy as np

import pyrecest.backend as backend
import pyrecest._backend.pytorch as pytorch_backend

values_np = np.arange(8).reshape(2, 2, 2)
expected = np.diagonal(
    values_np,
    offset=np.array(0),
    axis1=np.array(1),
    axis2=np.array(2),
).tolist()
values = backend.array(values_np)

for diagonal in (backend.diagonal, pytorch_backend.diagonal):
    result = diagonal(
        values,
        offset=np.array(0),
        axis1=np.array(1),
        axis2=np.array(2),
    )
    assert pytorch_backend.to_numpy(result).tolist() == expected

    tensor_axis_result = diagonal(
        values,
        offset=backend.array(0),
        axis1=backend.array(1),
        axis2=backend.array(2),
    )
    assert pytorch_backend.to_numpy(tensor_axis_result).tolist() == expected

try:
    backend.diagonal(values, axis1=np.array([1]), axis2=2)
except TypeError:
    pass
else:
    raise AssertionError("diagonal accepted a non-scalar axis array")
"""
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        env=_backend_test_env("pytorch"),
    )


@pytest.mark.backend_portable
def test_raw_pytorch_diagonal_accepts_numpy_scalar_array_indices_with_numpy_backend():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    code = """
import numpy as np

import pyrecest.backend as backend
import pyrecest._backend.pytorch as pytorch_backend

assert getattr(backend, "__backend_name__", None) == "numpy"
values_np = np.arange(8).reshape(2, 2, 2)
expected = np.diagonal(
    values_np,
    offset=np.array(0),
    axis1=np.array(1),
    axis2=np.array(2),
).tolist()
values = pytorch_backend.array(values_np)

result = pytorch_backend.diagonal(
    values,
    offset=np.array(0),
    axis1=np.array(1),
    axis2=np.array(2),
)
assert pytorch_backend.to_numpy(result).tolist() == expected

try:
    pytorch_backend.diagonal(values, axis1=np.array([1]), axis2=2)
except TypeError:
    pass
else:
    raise AssertionError("raw diagonal accepted a non-scalar axis array")
"""
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        env=_backend_test_env("numpy"),
    )
