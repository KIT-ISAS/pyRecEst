import importlib.util
import os
import subprocess
import sys

import pytest


@pytest.mark.backend_portable
def test_raw_pytorch_tile_matches_numpy_when_public_backend_is_numpy():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    env = os.environ.copy()
    env["PYRECEST_BACKEND"] = "numpy"
    src_path = os.path.abspath("src")
    env["PYTHONPATH"] = (
        src_path
        if not env.get("PYTHONPATH")
        else os.pathsep.join([src_path, env["PYTHONPATH"]])
    )

    code = """
import numpy as np
import numpy.testing as npt
import torch

import pyrecest.backend as public_backend
import pyrecest._backend.pytorch as raw_pytorch

assert public_backend.__backend_name__ == "numpy"

cases = [
    (np.arange(6).reshape(2, 3), 2),
    (np.arange(6).reshape(2, 3), (2,)),
    ([1, 2, 3], (2, 1, 2)),
    (np.array(5.0), (2, 3)),
    (np.arange(3), torch.tensor([2])),
    (np.arange(3), ()),
]

for values, reps in cases:
    numpy_reps = reps.detach().cpu().numpy() if torch.is_tensor(reps) else reps
    expected = np.tile(values, numpy_reps)
    result = raw_pytorch.tile(values, reps)
    npt.assert_array_equal(raw_pytorch.to_numpy(result), expected)
    assert tuple(result.shape) == expected.shape
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
