import importlib.util
import os
import subprocess
import sys

import pytest


@pytest.mark.backend_portable
def test_pytorch_qr_accepts_numpy_full_mode_alias():
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
import pyrecest._backend.pytorch.linalg as raw_pytorch_linalg
from pyrecest.backend import linalg

matrix = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]

for qr in (linalg.qr, raw_pytorch_linalg.qr):
    q_full, r_full = qr(matrix, mode="full")
    q_reduced, r_reduced = qr(matrix, mode="reduced")

    assert tuple(q_full.shape) == tuple(q_reduced.shape) == (3, 2)
    assert tuple(r_full.shape) == tuple(r_reduced.shape) == (2, 2)

    np.testing.assert_allclose(backend.to_numpy(q_full), backend.to_numpy(q_reduced))
    np.testing.assert_allclose(backend.to_numpy(r_full), backend.to_numpy(r_reduced))
    np.testing.assert_allclose(
        backend.to_numpy(q_full @ r_full),
        np.asarray(matrix),
        atol=1e-6,
    )
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
