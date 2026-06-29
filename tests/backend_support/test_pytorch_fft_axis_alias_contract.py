import importlib.util
import os
import subprocess
import sys

import pytest


@pytest.mark.backend_portable
def test_pytorch_fft_helpers_accept_numpy_axis_aliases():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

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

matrix = [[1.0, 2.0], [3.0, 4.0]]
assert np.allclose(
    backend.to_numpy(backend.fft.fftn(matrix, axes=(0, 1))),
    np.fft.fftn(matrix, axes=(0, 1)),
)
assert np.allclose(
    backend.to_numpy(backend.fft.ifftn(matrix, axes=(0, 1))),
    np.fft.ifftn(matrix, axes=(0, 1)),
)
assert np.allclose(
    backend.to_numpy(backend.fft.fftn(matrix, dim=0, axes=None)),
    np.fft.fftn(matrix, axes=(0,)),
)

vector = [1.0, 2.0, 3.0, 4.0]
spectrum = np.fft.rfft(vector, axis=0)
assert np.allclose(
    backend.to_numpy(backend.fft.rfft(vector, axis=0)),
    spectrum,
)
assert np.allclose(
    backend.to_numpy(backend.fft.irfft(spectrum, axis=0)),
    np.fft.irfft(spectrum, axis=0),
)

shift_source = [1, 2, 3, 4]
shifted = backend.fft.fftshift(shift_source, axes=0)
assert backend.to_numpy(shifted).tolist() == np.fft.fftshift(
    shift_source,
    axes=0,
).tolist()
assert backend.to_numpy(backend.fft.ifftshift(shifted, axes=0)).tolist() == np.fft.ifftshift(
    np.fft.fftshift(shift_source, axes=0),
    axes=0,
).tolist()

shift_matrix = [[1, 2, 3], [4, 5, 6]]
assert backend.to_numpy(backend.fft.fftshift(shift_matrix, dim=0, axes=None)).tolist() == np.fft.fftshift(
    shift_matrix,
    axes=0,
).tolist()

try:
    backend.fft.fftn(matrix, axis=0, axes=(0, 1))
except TypeError:
    pass
else:
    raise AssertionError("fftn accepted conflicting axis aliases")

try:
    backend.fft.fftshift(shift_source, dim=0, axes=0)
except TypeError:
    pass
else:
    raise AssertionError("fftshift accepted conflicting dim/axes aliases")
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
