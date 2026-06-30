import importlib.util
import os
import subprocess
import sys

import pytest


@pytest.mark.backend_portable
def test_pytorch_one_hot_accepts_integer_tensor_labels():
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
import pyrecest._backend.pytorch as raw_pytorch

labels = torch.tensor([0, 2], dtype=torch.int32)
expected = [[1, 0, 0], [0, 0, 1]]

result = backend.one_hot(labels, 3)
assert backend.to_numpy(result).tolist() == expected
assert str(backend.to_numpy(result).dtype) == "uint8"

raw_result = raw_pytorch.one_hot(torch.tensor([1], dtype=torch.int16), 3)
assert backend.to_numpy(raw_result).tolist() == [[0, 1, 0]]
assert str(backend.to_numpy(raw_result).dtype) == "uint8"
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
