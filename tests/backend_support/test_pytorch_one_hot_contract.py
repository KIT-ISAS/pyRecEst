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
def test_pytorch_one_hot_accepts_integer_label_tensors():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    code = """
import pyrecest.backend as backend

labels = backend.asarray([0, 2], dtype=backend.int32)
result = backend.one_hot(labels, 3)

assert result.shape == (2, 3)
assert str(backend.to_numpy(result).dtype) == "uint8"
assert backend.to_numpy(result).tolist() == [[1, 0, 0], [0, 0, 1]]
"""
    subprocess.run(
        [sys.executable, "-c", code], check=True, env=_backend_test_env("pytorch")
    )


@pytest.mark.backend_portable
def test_raw_pytorch_one_hot_accepts_integer_label_tensors_with_numpy_backend():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    code = """
import pyrecest.backend as backend
import pyrecest._backend.pytorch as raw_pytorch

assert getattr(backend, "__backend_name__", None) == "numpy"

labels = raw_pytorch.asarray([0, 2], dtype=raw_pytorch.int32)
result = raw_pytorch.one_hot(labels, 3)

assert result.shape == (2, 3)
assert str(raw_pytorch.to_numpy(result).dtype) == "uint8"
assert raw_pytorch.to_numpy(result).tolist() == [[1, 0, 0], [0, 0, 1]]
"""
    subprocess.run(
        [sys.executable, "-c", code], check=True, env=_backend_test_env("numpy")
    )
