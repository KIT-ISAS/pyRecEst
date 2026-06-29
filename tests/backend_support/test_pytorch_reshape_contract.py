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
def test_pytorch_reshape_array_like_inputs_match_numpy_contract():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    code = """
import numpy as np

import pyrecest.backend as backend
import pyrecest._backend.pytorch as pytorch_backend

matrix = backend.reshape([1, 2, 3, 4], (2, 2))
assert tuple(matrix.shape) == (2, 2)
assert backend.to_numpy(matrix).tolist() == [[1, 2], [3, 4]]

shape_from_numpy = backend.reshape([1, 2, 3, 4], np.array([4, 1]))
assert tuple(shape_from_numpy.shape) == (4, 1)
assert backend.to_numpy(shape_from_numpy).tolist() == [[1], [2], [3], [4]]

shape_from_tensor = backend.reshape([1, 2, 3, 4], backend.array([2, 2]))
assert backend.to_numpy(shape_from_tensor).tolist() == [[1, 2], [3, 4]]

raw_result = pytorch_backend.reshape([1, 2, 3], 3)
assert pytorch_backend.to_numpy(raw_result).tolist() == [1, 2, 3]

try:
    backend.reshape([1, 2], [1.5, 2])
except TypeError:
    pass
else:
    raise AssertionError("reshape accepted a non-integer target shape")
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=_backend_test_env("pytorch"))


@pytest.mark.backend_portable
def test_pytorch_squeeze_accepts_tuple_axes_for_public_and_raw_backend():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    code = """
import pyrecest.backend as backend
from pyrecest._backend import pytorch as pytorch_backend

values = backend.array([[[1], [2]]])

for squeeze_func, to_numpy in (
    (backend.squeeze, backend.to_numpy),
    (pytorch_backend.squeeze, pytorch_backend.to_numpy),
):
    result = squeeze_func(values, axis=(0, 2))
    assert tuple(result.shape) == (2,)
    assert to_numpy(result).tolist() == [1, 2]

    negative_axis_result = squeeze_func(values, axis=(-3, -1))
    assert tuple(negative_axis_result.shape) == (2,)
    assert to_numpy(negative_axis_result).tolist() == [1, 2]

    tensor_axis_result = squeeze_func(values, axis=backend.array([0, 2]))
    assert tuple(tensor_axis_result.shape) == (2,)
    assert to_numpy(tensor_axis_result).tolist() == [1, 2]

    empty_axis_result = squeeze_func(values, axis=())
    assert tuple(empty_axis_result.shape) == (1, 2, 1)
    assert to_numpy(empty_axis_result).tolist() == [[[1], [2]]]

    try:
        squeeze_func(values, axis=(0, -3))
    except ValueError:
        pass
    else:
        raise AssertionError("squeeze accepted duplicate axes")
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=_backend_test_env("pytorch"))


@pytest.mark.backend_portable
def test_raw_pytorch_squeeze_tuple_axes_with_numpy_public_backend():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    code = """
import pyrecest.backend as backend
from pyrecest._backend import pytorch as pytorch_backend

assert getattr(backend, "__backend_name__", None) == "numpy"
values = pytorch_backend.array([[[1], [2]]])

result = pytorch_backend.squeeze(values, axis=(0, 2))
assert tuple(result.shape) == (2,)
assert pytorch_backend.to_numpy(result).tolist() == [1, 2]

axis_tensor_result = pytorch_backend.squeeze(
    values,
    axis=pytorch_backend.array([0, 2]),
)
assert tuple(axis_tensor_result.shape) == (2,)
assert pytorch_backend.to_numpy(axis_tensor_result).tolist() == [1, 2]
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=_backend_test_env("numpy"))
