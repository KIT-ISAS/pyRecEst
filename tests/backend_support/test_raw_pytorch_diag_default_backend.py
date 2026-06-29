import importlib.util
import os
import subprocess
import sys

import pytest


def test_raw_pytorch_diag_accepts_numpy_contract_under_numpy_backend():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    env = os.environ.copy()
    env["PYRECEST_BACKEND"] = "numpy"
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import pyrecest  # noqa: F401  # triggers raw-backend compatibility patches
import pyrecest.backend as backend
import pyrecest._backend.pytorch as raw_pytorch_backend

assert getattr(backend, "__backend_name__", None) == "numpy"

upper = raw_pytorch_backend.diag([1, 2, 3], k=1)
assert raw_pytorch_backend.to_numpy(upper).tolist() == [
    [0, 1, 0, 0],
    [0, 0, 2, 0],
    [0, 0, 0, 3],
    [0, 0, 0, 0],
]

matrix = [[0, 4, 0], [1, 0, 5], [0, 2, 0]]
lower = raw_pytorch_backend.diag(matrix, k=-1)
assert raw_pytorch_backend.to_numpy(lower).tolist() == [1, 2]
print("ok")
""",
        ],
        capture_output=True,
        env=env,
        text=True,
        timeout=30.0,
    )

    assert completed.returncode == 0, completed.stderr
    assert "ok" in completed.stdout
