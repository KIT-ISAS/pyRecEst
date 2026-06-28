import importlib.util
import os
import subprocess
import sys

import pytest


@pytest.mark.backend_portable
def test_pytorch_linalg_helpers_accept_array_like_inputs():
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
from pyrecest.backend import linalg

identity = [[1.0, 0.0], [0.0, 1.0]]
assert backend.to_numpy(linalg.inv(identity)).tolist() == identity
assert float(backend.to_numpy(linalg.det(identity))) == 1.0

cholesky = linalg.cholesky([[4.0, 0.0], [0.0, 9.0]])
assert backend.to_numpy(cholesky).tolist() == [[2.0, 0.0], [0.0, 3.0]]

assert backend.to_numpy(linalg.eigvalsh([[2.0, 0.0], [0.0, 3.0]])).tolist() == [2.0, 3.0]
assert backend.to_numpy(linalg.matrix_power([[1.0, 1.0], [0.0, 1.0]], 2)).tolist() == [[1.0, 2.0], [0.0, 1.0]]
assert backend.to_numpy(linalg.pinv([[1.0, 0.0], [0.0, 0.0]])).tolist() == [[1.0, 0.0], [0.0, 0.0]]
assert backend.to_numpy(linalg.expm([[0.0, 0.0], [0.0, 0.0]])).tolist() == identity
assert backend.to_numpy(linalg.block_diag([[1.0]], [[2.0]])).tolist() == [[1.0, 0.0], [0.0, 2.0]]
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
