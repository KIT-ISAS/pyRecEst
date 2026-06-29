import importlib.util
import os
import subprocess
import sys

import pytest


@pytest.mark.backend_portable
def test_pytorch_trace_matches_backend_contract_for_axes_offset_dtype_and_out():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    env = os.environ.copy()
    env["PYRECEST_BACKEND"] = "pytorch"
    src_path = os.path.abspath("src")
    env["PYTHONPATH"] = (
        src_path
        if not env.get("PYTHONPATH")
        else os.pathsep.join([src_path, env["PYTHONPATH"]])
    )

    code = """
import pyrecest  # noqa: F401  # triggers backend compatibility patches
import pyrecest.backend as backend
from pyrecest._backend import pytorch as pytorch_backend

values = backend.arange(24).reshape((2, 3, 4))

for trace_func, to_numpy in (
    (backend.trace, backend.to_numpy),
    (pytorch_backend.trace, pytorch_backend.to_numpy),
):
    default_result = trace_func(values)
    assert tuple(default_result.shape) == (2,)
    assert to_numpy(default_result).tolist() == [15, 51]

    offset_result = trace_func(values, offset=1, axis1=-2, axis2=-1)
    assert to_numpy(offset_result).tolist() == [18, 54]

    explicit_axes_result = trace_func(values, axis1=1, axis2=2)
    assert to_numpy(explicit_axes_result).tolist() == [15, 51]

    dtype_result = trace_func([[1, 2], [3, 4]], dtype=backend.float64)
    assert str(dtype_result.dtype).endswith("float64")
    assert float(dtype_result) == 5.0

    out = backend.zeros((2,))
    returned = trace_func(values, out=out)
    assert returned is out
    assert to_numpy(out).tolist() == [15, 51]

    bad_out = backend.zeros((1,))
    try:
        trace_func(values, out=bad_out)
    except RuntimeError:
        pass
    else:
        raise AssertionError("trace accepted incompatible out shape")
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
