import importlib.util
import os
import subprocess
import sys

import pytest


@pytest.mark.backend_portable
def test_pytorch_fft_wrappers_accept_numpy_array_axes():
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

values_np = np.arange(6.0).reshape(2, 3)
values = backend.array(values_np)

for axes in (np.array([0]), np.array([0, 1]), np.array([], dtype=int)):
    npt.assert_allclose(
        backend.to_numpy(backend.fft.fftn(values, axes=axes)),
        np.fft.fftn(values_np, axes=axes),
    )
    npt.assert_allclose(
        backend.to_numpy(backend.fft.ifftn(values, axes=axes)),
        np.fft.ifftn(values_np, axes=axes),
    )
    npt.assert_array_equal(
        backend.to_numpy(backend.fft.fftshift(values, axes=axes)),
        np.fft.fftshift(values_np, axes=axes),
    )
    npt.assert_array_equal(
        backend.to_numpy(backend.fft.ifftshift(values, axes=axes)),
        np.fft.ifftshift(values_np, axes=axes),
    )

npt.assert_allclose(
    backend.to_numpy(backend.fft.fftn(values, dim=np.array([0, 1]))),
    np.fft.fftn(values_np, axes=(0, 1)),
)
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
