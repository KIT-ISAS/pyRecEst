import importlib.util
import os
import subprocess
import sys

import pytest


@pytest.mark.backend_portable
def test_pytorch_reshape_accepts_array_like_values_and_numpy_style_shapes():
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
import pyrecest._backend.pytorch as pytorch_backend

for backend_module in (backend, pytorch_backend):
    matrix = backend_module.reshape([1, 2, 3, 4], (2, 2))
    assert tuple(matrix.shape) == (2, 2)
    assert matrix.detach().cpu().numpy().tolist() == [[1, 2], [3, 4]]

    vector = backend_module.reshape([[1, 2], [3, 4]], 4)
    assert tuple(vector.shape) == (4,)
    assert vector.detach().cpu().numpy().tolist() == [1, 2, 3, 4]

    inferred = backend_module.reshape([1, 2, 3, 4], backend.array([2, -1]))
    assert tuple(inferred.shape) == (2, 2)

    try:
        backend_module.reshape([1, 2], [2.0])
    except TypeError:
        pass
    else:
        raise AssertionError("reshape accepted non-integer dimensions")
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
