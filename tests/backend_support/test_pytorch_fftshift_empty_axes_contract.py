import importlib.util
import os
import subprocess
import sys

import pytest


@pytest.mark.backend_portable
def test_pytorch_fftshift_empty_axes_is_noop():
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
    shifted = backend.fft.fftshift(values, axes=axes)
    unshifted = backend.fft.ifftshift(values, axes=axes)

    npt.assert_array_equal(
        backend.to_numpy(shifted),
        np.fft.fftshift(values_np, axes=axes),
    )
    npt.assert_array_equal(
        backend.to_numpy(unshifted),
        np.fft.ifftshift(values_np, axes=axes),
    )
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
