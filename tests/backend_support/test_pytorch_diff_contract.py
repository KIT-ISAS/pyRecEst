import importlib.util
import os
import subprocess
import sys

import pytest


def _backend_test_env(backend_name):
    env = os.environ.copy()
    env["PYRECEST_BACKEND"] = backend_name
    src_path = os.path.abspath("src")
    env["PYTHONPATH"] = (
        src_path
        if not env.get("PYTHONPATH")
        else os.pathsep.join([src_path, env["PYTHONPATH"]])
    )
    return env


@pytest.mark.backend_portable
def test_pytorch_diff_matches_numpy_axis_and_boundary_contract():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    code = """
import pyrecest
import pyrecest.backend as backend
from pyrecest._backend import pytorch as pytorch_backend

values = [[1, 3, 6], [10, 15, 21]]

for diff_func, to_numpy in (
    (backend.diff, backend.to_numpy),
    (pytorch_backend.diff, pytorch_backend.to_numpy),
):
    second = diff_func(values, n=2, axis=1)
    assert tuple(second.shape) == (2, 1)
    assert to_numpy(second).tolist() == [[1], [1]]

    prepended = diff_func(values, axis=1, prepend=0)
    assert tuple(prepended.shape) == (2, 3)
    assert to_numpy(prepended).tolist() == [[1, 2, 3], [10, 5, 6]]

    appended = diff_func(values, dim=0, append=[[20, 30, 42]])
    assert tuple(appended.shape) == (2, 3)
    assert to_numpy(appended).tolist() == [[9, 12, 15], [10, 15, 21]]

    out = backend.empty((2, 3), dtype=backend.int64)
    returned = diff_func(values, axis=1, prepend=0, out=out)
    assert returned is out
    assert to_numpy(out).tolist() == [[1, 2, 3], [10, 5, 6]]
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=_backend_test_env("pytorch"))


@pytest.mark.backend_portable
def test_raw_pytorch_diff_is_patched_with_numpy_public_backend():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    code = """
import pyrecest
import pyrecest.backend as backend
from pyrecest._backend import pytorch as pytorch_backend

assert getattr(backend, "__backend_name__", None) == "numpy"
values = [[1, 3, 6], [10, 15, 21]]

result = pytorch_backend.diff(values, axis=1, prepend=0)
assert tuple(result.shape) == (2, 3)
assert pytorch_backend.to_numpy(result).tolist() == [[1, 2, 3], [10, 5, 6]]

second = pytorch_backend.diff(values, n=2, axis=1)
assert tuple(second.shape) == (2, 1)
assert pytorch_backend.to_numpy(second).tolist() == [[1], [1]]
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=_backend_test_env("numpy"))
