import importlib.util
import os
import subprocess
import sys

import pytest


@pytest.mark.backend_portable
def test_pytorch_broadcast_to_accepts_array_like_shapes():
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
from pyrecest._backend import pytorch as pytorch_backend

values = backend.array([1, 2, 3])

for broadcast_to, to_numpy in (
    (backend.broadcast_to, backend.to_numpy),
    (pytorch_backend.broadcast_to, pytorch_backend.to_numpy),
):
    result = broadcast_to(values, backend.array([2, 3]))
    assert tuple(result.shape) == (2, 3)
    assert to_numpy(result).tolist() == [[1, 2, 3], [1, 2, 3]]

    scalar_shape = broadcast_to(7, backend.array(3))
    assert tuple(scalar_shape.shape) == (3,)
    assert to_numpy(scalar_shape).tolist() == [7, 7, 7]

    empty_shape = broadcast_to(7, backend.array([]).to(dtype=backend.int64))
    assert tuple(empty_shape.shape) == ()
    assert to_numpy(empty_shape).tolist() == 7

    for bad_shape in (backend.array([-1]), backend.array([2.0, 3.0])):
        try:
            broadcast_to(values, bad_shape)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError(f"broadcast_to accepted invalid shape {bad_shape!r}")
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
