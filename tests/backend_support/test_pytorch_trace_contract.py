import importlib.util
import os
import subprocess
import sys

import pytest


@pytest.mark.backend_portable
def test_pytorch_trace_accepts_numpy_signature_and_last_axis_default():
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

values = backend.array(
    [
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
    ],
    dtype=backend.float32,
)

# PyRecEst's backend trace contract uses the last two axes by default.
default_result = backend.trace(values)
assert backend.to_numpy(default_result).tolist() == [6.0, 18.0]

out = backend.zeros((2,), dtype=backend.float32)
result = backend.trace(
    values,
    offset=1,
    axis1=1,
    axis2=2,
    dtype=backend.float32,
    out=out,
)

assert result is out
assert backend.to_numpy(result).tolist() == [8.0, 20.0]
assert str(backend.to_numpy(result).dtype) == "float32"

raw_out = pytorch_backend.zeros((2,), dtype=pytorch_backend.float64)
raw_result = pytorch_backend.trace(
    [
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
    ],
    offset=1,
    axis1=1,
    axis2=2,
    dtype=pytorch_backend.float64,
    out=raw_out,
)
assert raw_result is raw_out
assert pytorch_backend.to_numpy(raw_result).tolist() == [8.0, 20.0]
assert str(pytorch_backend.to_numpy(raw_result).dtype) == "float64"
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
