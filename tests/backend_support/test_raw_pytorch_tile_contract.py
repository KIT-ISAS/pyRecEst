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
def test_raw_pytorch_tile_matches_numpy_when_public_backend_is_numpy():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    env = _backend_subprocess_env("numpy")

    code = """
import numpy as np
import numpy.testing as npt
import pyrecest.backend as public_backend
import pyrecest._backend.pytorch as raw_pytorch

assert public_backend.__backend_name__ == "numpy"

cases = [
    ([[1, 2], [3, 4]], 2),
    ([[1, 2], [3, 4]], [2, 1]),
    ([[1, 2], [3, 4]], ()),
    ([1, 2, 3], [2, 3]),
]

for values, repetitions in cases:
    expected = np.tile(np.asarray(values), repetitions)
    result = raw_pytorch.tile(raw_pytorch.array(values), repetitions)
    npt.assert_array_equal(raw_pytorch.to_numpy(result), expected)
    assert tuple(result.shape) == expected.shape

values = raw_pytorch.array([[1, 2], [3, 4]])
empty_result = raw_pytorch.tile(values, ())
assert empty_result is not values

for bad_reps in (1.5, [2.5, 1], "2", raw_pytorch.array([2.5, 1.0])):
    try:
        raw_pytorch.tile(values, bad_reps)
    except TypeError:
        pass
    else:
        raise AssertionError(f"tile accepted non-integer repetitions {bad_reps!r}")
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
