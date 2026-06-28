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


def test_logical_or_accepts_array_like_inputs():
    result = backend.logical_or([True, False, False], [False, False, True])

    assert _to_python(result) == [True, False, True]


@pytest.mark.backend_portable
def test_pytorch_logical_or_accepts_array_like_inputs():
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

result = backend.logical_or([True, False, False], [False, False, True])
assert backend.to_numpy(result).tolist() == [True, False, True]

tensor = backend.asarray([False, True, False])
mixed = backend.logical_or(tensor, [True, False, False])
assert backend.to_numpy(mixed).tolist() == [True, True, False]
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
