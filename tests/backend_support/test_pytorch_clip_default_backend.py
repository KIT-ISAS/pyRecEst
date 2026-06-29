import importlib.util
import os
import subprocess
import sys

import pytest


@pytest.mark.backend_portable
def test_raw_pytorch_clip_array_like_inputs_with_non_pytorch_public_backend():
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

import pyrecest  # noqa: F401
import pyrecest.backend as backend
import pyrecest._backend.pytorch as raw_pytorch

assert getattr(backend, "__backend_name__", None) == "numpy"

result = raw_pytorch.clip([-2.0, 0.5, 3.0], -1.0, 2.0)
assert raw_pytorch.is_array(result)
assert result.tolist() == [-1.0, 0.5, 2.0]

lower_bound_result = raw_pytorch.clip(
    [[-2.0, 0.5], [3.0, 4.0]], min=np.array([-1.0, 1.0])
)
assert raw_pytorch.is_array(lower_bound_result)
assert lower_bound_result.tolist() == [[-1.0, 1.0], [3.0, 4.0]]

out = raw_pytorch.zeros(3, dtype=raw_pytorch.float64)
returned = raw_pytorch.clip([-2.0, 0.5, 3.0], -1.0, 2.0, out=out)
assert returned is out
assert out.tolist() == [-1.0, 0.5, 2.0]

try:
    raw_pytorch.clip([1.0, 2.0])
except ValueError:
    pass
else:
    raise AssertionError("clip accepted missing bounds")

try:
    raw_pytorch.clip([1.0, 2.0], 0.0, min=0.0)
except TypeError:
    pass
else:
    raise AssertionError("clip accepted conflicting a_min/min aliases")
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
