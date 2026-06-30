import importlib.util
import os
import subprocess
import sys

import pytest


def _run_like_creation_contract(backend_name):
    env = os.environ.copy()
    env["PYRECEST_BACKEND"] = backend_name
    src_path = os.path.abspath("src")
    env["PYTHONPATH"] = (
        src_path
        if not env.get("PYTHONPATH")
        else os.pathsep.join([src_path, env["PYTHONPATH"]])
    )

    code = """
import pyrecest.backend as backend
import pyrecest._backend.pytorch as raw_backend

modules = [raw_backend]
if backend.__backend_name__ == "pytorch":
    modules.append(backend)

for module in modules:
    zeros = module.zeros_like([1.0, 2.0])
    assert module.to_numpy(zeros).tolist() == [0.0, 0.0]

    ones = module.ones_like([1, 2, 3])
    assert module.to_numpy(ones).tolist() == [1, 1, 1]

    full = module.full_like([1, 2, 3], 7)
    assert module.to_numpy(full).tolist() == [7, 7, 7]

    empty = module.empty_like([[1, 2], [3, 4]])
    assert tuple(empty.shape) == (2, 2)
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


@pytest.mark.backend_portable
def test_raw_pytorch_like_creation_accepts_array_like_inputs_with_default_backend():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    _run_like_creation_contract("numpy")


@pytest.mark.backend_portable
def test_pytorch_like_creation_accepts_array_like_inputs_with_pytorch_backend():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    _run_like_creation_contract("pytorch")
