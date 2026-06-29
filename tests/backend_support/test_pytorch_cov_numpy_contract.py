import importlib.util
import os
import subprocess
import sys

import pytest


@pytest.mark.backend_portable
def test_pytorch_cov_accepts_numpy_style_y_rowvar_and_dtype_contract():
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
from pyrecest._backend import pytorch as raw_pytorch

x = backend.array([[0.0, 0.0], [1.0, 2.0], [2.0, 4.0]], dtype=backend.float32)
y = backend.array([[0.0, 1.0], [3.0, 2.0], [6.0, 3.0]], dtype=backend.float64)
expected = backend.array(
    [
        [2.0 / 3.0, 4.0 / 3.0, 2.0, 2.0 / 3.0],
        [4.0 / 3.0, 8.0 / 3.0, 4.0, 4.0 / 3.0],
        [2.0, 4.0, 6.0, 2.0],
        [2.0 / 3.0, 4.0 / 3.0, 2.0, 2.0 / 3.0],
    ],
    dtype=backend.float64,
)
ddof_expected = backend.array(
    [[2.0 / 3.0, 4.0 / 3.0], [4.0 / 3.0, 8.0 / 3.0]],
    dtype=backend.float64,
)

for cov_func in (backend.cov, raw_pytorch.cov):
    result = cov_func(x, y=y, rowvar=False, bias=True, dtype=backend.float64)
    assert tuple(result.shape) == (4, 4)
    assert result.dtype == backend.float64
    assert bool(raw_pytorch.allclose(result, expected))

    ddof_result = cov_func(
        backend.array([[0.0, 1.0, 2.0], [0.0, 2.0, 4.0]]), ddof=0
    )
    assert bool(raw_pytorch.allclose(ddof_result, ddof_expected))
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
