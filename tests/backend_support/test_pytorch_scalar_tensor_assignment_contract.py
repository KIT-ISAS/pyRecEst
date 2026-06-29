import importlib.util
import os
import subprocess
import sys

import pytest


@pytest.mark.backend_portable
def test_pytorch_assignment_accepts_scalar_tensor_indices():
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
import torch
import pyrecest.backend as backend
import pyrecest._backend.pytorch as pytorch_backend


def as_list(value):
    return backend.to_numpy(value).tolist()


original = backend.array([1.0, 2.0, 3.0])

assigned = backend.assignment(original, 9.0, torch.tensor(1))
added = backend.assignment_by_sum(original, 5.0, torch.tensor(2))
raw_assigned = pytorch_backend.assignment(original, -4.0, torch.tensor(0))
raw_added = pytorch_backend.assignment_by_sum(original, 10.0, torch.tensor(1))

assert as_list(assigned) == [1.0, 9.0, 3.0]
assert as_list(added) == [1.0, 2.0, 8.0]
assert as_list(raw_assigned) == [-4.0, 2.0, 3.0]
assert as_list(raw_added) == [1.0, 12.0, 3.0]
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
