import importlib.util
import os
import subprocess
import sys

import pytest


@pytest.mark.backend_portable
def test_pytorch_clip_accepts_array_like_inputs_and_numpy_bounds():
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
import pyrecest._backend.pytorch as pytorch_backend

values = [-2.0, 0.5, 3.0]

clipped = backend.clip(values, -1.0, 2.0)
assert backend.to_numpy(clipped).tolist() == [-1.0, 0.5, 2.0]

alias_clipped = backend.clip(values, min=-0.5, max=1.5)
assert backend.to_numpy(alias_clipped).tolist() == [-0.5, 0.5, 1.5]

array_bound_clipped = backend.clip(
    values,
    [-1.0, 0.0, 2.5],
    [0.0, 1.0, 2.75],
)
assert backend.to_numpy(array_bound_clipped).tolist() == [-1.0, 0.5, 2.75]

raw_clipped = pytorch_backend.clip(values, -1.0, 2.0)
assert backend.to_numpy(raw_clipped).tolist() == [-1.0, 0.5, 2.0]

out = backend.empty_like(clipped)
returned = backend.clip(values, -1.0, 2.0, out=out)
assert returned is out
assert backend.to_numpy(out).tolist() == [-1.0, 0.5, 2.0]

try:
    backend.clip(values, -1.0, 2.0, min=-1.0)
except TypeError:
    pass
else:
    raise AssertionError("clip accepted both a_min and min")
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
