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
def test_pytorch_outer_matches_numpy_for_flattened_inputs():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    env = _backend_subprocess_env("pytorch")

    code = """
import numpy as np
import numpy.testing as npt
import pyrecest.backend as backend
import pyrecest._backend.pytorch as raw_pytorch

cases = [
    (np.arange(6.0).reshape(2, 3), np.arange(4.0).reshape(2, 2)),
    (2.0, [3.0, 4.0]),
    ([[1, 2], [3, 4]], [[5, 6], [7, 8]]),
]

for left, right in cases:
    expected = np.outer(np.asarray(left), np.asarray(right))
    result = backend.outer(backend.array(left), backend.array(right))
    raw_result = raw_pytorch.outer(raw_pytorch.array(left), raw_pytorch.array(right))

    npt.assert_allclose(backend.to_numpy(result), expected)
    npt.assert_allclose(raw_pytorch.to_numpy(raw_result), expected)
    assert tuple(result.shape) == expected.shape
    assert tuple(raw_result.shape) == expected.shape
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


@pytest.mark.backend_portable
def test_raw_pytorch_outer_matches_numpy_when_public_backend_is_numpy():
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
    (np.arange(6.0).reshape(2, 3), np.arange(4.0).reshape(2, 2)),
    (2.0, [3.0, 4.0]),
    ([[1, 2], [3, 4]], [[5, 6], [7, 8]]),
]

for left, right in cases:
    expected = np.outer(np.asarray(left), np.asarray(right))
    raw_result = raw_pytorch.outer(raw_pytorch.array(left), raw_pytorch.array(right))

    npt.assert_allclose(raw_pytorch.to_numpy(raw_result), expected)
    assert tuple(raw_result.shape) == expected.shape
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
