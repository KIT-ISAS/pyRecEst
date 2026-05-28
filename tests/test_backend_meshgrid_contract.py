import importlib.util
import os
import subprocess
import sys

import pyrecest.backend as backend
import pytest


def _to_python(value):
    value = backend.to_numpy(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def test_meshgrid_accepts_numpy_style_array_like_axes():
    rows, cols = backend.meshgrid([0, 1], range(2), indexing="ij")

    assert _to_python(rows) == [[0, 0], [1, 1]]
    assert _to_python(cols) == [[0, 1], [0, 1]]


def test_meshgrid_accepts_scalar_axes():
    rows, cols = backend.meshgrid(1, [2, 3], indexing="ij")

    assert _to_python(rows) == [[1, 1]]
    assert _to_python(cols) == [[2, 3]]


@pytest.mark.backend_portable
def test_jax_meshgrid_accepts_numpy_style_axes():
    if importlib.util.find_spec("jax") is None:
        pytest.skip("jax is not installed")

    env = os.environ.copy()
    env["PYRECEST_BACKEND"] = "jax"
    src_path = os.path.abspath("src")
    env["PYTHONPATH"] = (
        src_path
        if not env.get("PYTHONPATH")
        else os.pathsep.join([src_path, env["PYTHONPATH"]])
    )

    code = """
import pyrecest.backend as backend
rows, cols = backend.meshgrid([0, 1], range(2), indexing='ij')
assert backend.to_numpy(rows).tolist() == [[0, 0], [1, 1]]
assert backend.to_numpy(cols).tolist() == [[0, 1], [0, 1]]
rows, cols = backend.meshgrid(1, [2, 3], indexing='ij')
assert backend.to_numpy(rows).tolist() == [[1, 1]]
assert backend.to_numpy(cols).tolist() == [[2, 3]]
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
