import importlib.util
import os
import subprocess
import sys

import pytest


@pytest.mark.backend_portable
def test_pytorch_dot_matches_numpy_for_vector_matrix_and_high_rank_inputs():
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
import numpy.testing as npt
import pyrecest.backend as backend

cases = [
    ([1.0, 2.0], [[5.0, 6.0], [7.0, 8.0]]),
    ([[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]),
    (np.arange(24, dtype=float).reshape(2, 3, 4), np.arange(120, dtype=float).reshape(5, 4, 6)),
    (2.0, [[5.0, 6.0], [7.0, 8.0]]),
]

for left, right in cases:
    expected = np.dot(np.asarray(left), np.asarray(right))
    result = backend.dot(backend.array(left), backend.array(right))
    npt.assert_allclose(backend.to_numpy(result), expected)
    assert tuple(result.shape) == expected.shape
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
