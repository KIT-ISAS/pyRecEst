import importlib.util
import os
import subprocess
import sys

import pytest


@pytest.mark.backend_portable
def test_pytorch_fftn_empty_axes_is_noop():
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

values_np = np.array([[0, 1], [2, 3]])
values = backend.array(values_np)

for axes in ((), []):
    transformed = backend.fft.fftn(values, axes=axes)
    inverse_transformed = backend.fft.ifftn(values, axes=axes)

    expected_transformed = np.fft.fftn(values_np, axes=axes)
    expected_inverse = np.fft.ifftn(values_np, axes=axes)

    transformed_np = backend.to_numpy(transformed)
    inverse_np = backend.to_numpy(inverse_transformed)

    npt.assert_array_equal(transformed_np, expected_transformed)
    npt.assert_array_equal(inverse_np, expected_inverse)
    assert transformed_np.dtype == expected_transformed.dtype
    assert inverse_np.dtype == expected_inverse.dtype
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
