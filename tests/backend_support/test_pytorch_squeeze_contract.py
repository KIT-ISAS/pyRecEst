import importlib.util
import os
import subprocess
import sys

import pytest


@pytest.mark.backend_portable
def test_pytorch_squeeze_axis_rejects_non_singleton_axes_like_numpy():
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

values = backend.array([[[1, 2, 3]], [[4, 5, 6]]])

for squeeze_func, to_numpy in (
    (backend.squeeze, backend.to_numpy),
    (pytorch_backend.squeeze, pytorch_backend.to_numpy),
):
    squeezed = squeeze_func(values, axis=1)
    assert tuple(squeezed.shape) == (2, 3)
    assert to_numpy(squeezed).tolist() == [[1, 2, 3], [4, 5, 6]]

    unchanged = squeeze_func(values, axis=())
    assert tuple(unchanged.shape) == (2, 1, 3)
    assert to_numpy(unchanged).tolist() == [[[1, 2, 3]], [[4, 5, 6]]]

    try:
        squeeze_func(values, axis=0)
    except ValueError:
        pass
    else:
        raise AssertionError("squeeze accepted a non-singleton axis")

    try:
        squeeze_func(values, axis=(1, 1))
    except ValueError:
        pass
    else:
        raise AssertionError("squeeze accepted duplicate axes")

scalar = backend.array(5)
assert tuple(backend.squeeze(scalar, axis=0).shape) == ()
assert tuple(backend.squeeze(scalar, axis=-1).shape) == ()
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
