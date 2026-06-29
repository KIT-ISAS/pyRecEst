import importlib.util
import os
import subprocess
import sys

import pytest


def _subprocess_env(backend_name):
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
def test_pytorch_roll_accepts_array_like_inputs_and_numpy_axis_keyword():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    code = """
import pyrecest.backend as backend

assert getattr(backend, "__backend_name__", None) == "pytorch"

axis_result = backend.roll([[1, 2, 3], [4, 5, 6]], 1, axis=1)
assert tuple(axis_result.shape) == (2, 3)
assert backend.to_numpy(axis_result).tolist() == [[3, 1, 2], [6, 4, 5]]

flat_result = backend.roll([[1, 2, 3], [4, 5, 6]], 2)
assert tuple(flat_result.shape) == (2, 3)
assert backend.to_numpy(flat_result).tolist() == [[5, 6, 1], [2, 3, 4]]

multi_axis_result = backend.roll([[1, 2, 3], [4, 5, 6]], (1, -1), axis=(0, 1))
assert backend.to_numpy(multi_axis_result).tolist() == [[5, 6, 4], [2, 3, 1]]
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=_subprocess_env("pytorch"))


@pytest.mark.backend_portable
def test_raw_pytorch_roll_matches_numpy_contract_with_numpy_public_backend():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    code = """
import pyrecest.backend as backend
from pyrecest._backend import pytorch as pytorch_backend

assert getattr(backend, "__backend_name__", None) == "numpy"

axis_result = pytorch_backend.roll([[1, 2, 3], [4, 5, 6]], 1, axis=1)
assert tuple(axis_result.shape) == (2, 3)
assert pytorch_backend.to_numpy(axis_result).tolist() == [[3, 1, 2], [6, 4, 5]]

flat_result = pytorch_backend.roll([[1, 2, 3], [4, 5, 6]], 2)
assert tuple(flat_result.shape) == (2, 3)
assert pytorch_backend.to_numpy(flat_result).tolist() == [[5, 6, 1], [2, 3, 4]]

multi_axis_result = pytorch_backend.roll([[1, 2, 3], [4, 5, 6]], (1, -1), axis=(0, 1))
assert pytorch_backend.to_numpy(multi_axis_result).tolist() == [[5, 6, 4], [2, 3, 1]]
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=_subprocess_env("numpy"))
