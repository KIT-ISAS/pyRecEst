import importlib.util
import os
import subprocess
import sys

import pytest


@pytest.mark.backend_portable
def test_pytorch_tile_scalar_and_array_repetitions_match_numpy_contract():
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
import pyrecest.backend as backend
from pyrecest._backend import pytorch as pytorch_backend

values = backend.array([[1, 2], [3, 4]])

for tile_func, to_numpy in (
    (backend.tile, backend.to_numpy),
    (pytorch_backend.tile, pytorch_backend.to_numpy),
):
    scalar_result = tile_func(values, 2)
    assert tuple(scalar_result.shape) == (2, 4)
    assert to_numpy(scalar_result).tolist() == [[1, 2, 1, 2], [3, 4, 3, 4]]

    array_result = tile_func(values, backend.array([2, 1]))
    assert tuple(array_result.shape) == (4, 2)
    assert to_numpy(array_result).tolist() == [[1, 2], [3, 4], [1, 2], [3, 4]]

    empty_result = tile_func(values, ())
    assert tuple(empty_result.shape) == (2, 2)
    assert to_numpy(empty_result).tolist() == [[1, 2], [3, 4]]
    assert empty_result is not values

    for bad_reps in (1.5, [2.5, 1], "2", backend.array([2.5, 1.0])):
        try:
            tile_func(values, bad_reps)
        except TypeError:
            pass
        else:
            raise AssertionError(f"tile accepted non-integer repetitions {bad_reps!r}")
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
