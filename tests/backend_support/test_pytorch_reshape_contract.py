import importlib.util
import os
import subprocess
import sys

import pytest


@pytest.mark.backend_portable
def test_pytorch_reshape_array_like_inputs_match_numpy_contract():
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
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
