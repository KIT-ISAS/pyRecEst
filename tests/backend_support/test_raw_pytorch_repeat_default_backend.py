import importlib.util
import os
import subprocess
import sys

import pytest


@pytest.mark.backend_portable
def test_raw_pytorch_repeat_accepts_array_like_with_numpy_backend():
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
import pyrecest  # noqa: F401
import pyrecest.backend as backend
import pyrecest._backend.pytorch as pytorch_backend

assert backend.__backend_name__ == "numpy"

flat_result = pytorch_backend.repeat([1, 2], 2)
assert pytorch_backend.to_numpy(flat_result).tolist() == [1, 1, 2, 2]

axis_result = pytorch_backend.repeat([[1, 2], [3, 4]], [1, 2], axis=0)
assert pytorch_backend.to_numpy(axis_result).tolist() == [[1, 2], [3, 4], [3, 4]]

dim_result = pytorch_backend.repeat([[1, 2], [3, 4]], pytorch_backend.array([2, 1]), dim=1)
assert pytorch_backend.to_numpy(dim_result).tolist() == [[1, 1, 2], [3, 3, 4]]

public_result = backend.repeat([1, 2], 2)
assert public_result.tolist() == [1, 1, 2, 2]
assert type(public_result).__module__.startswith("numpy")

try:
    pytorch_backend.repeat([1, 2], 2, axis=0, dim=1)
except TypeError:
    pass
else:
    raise AssertionError("raw repeat accepted conflicting axis and dim arguments")

try:
    pytorch_backend.repeat([1, 2], [[1, 0]])
except ValueError:
    pass
else:
    raise AssertionError("raw repeat accepted nested repeat counts")

try:
    pytorch_backend.repeat([1, 2], [1.5, 1])
except TypeError:
    pass
else:
    raise AssertionError("raw repeat accepted non-integer repeat counts")
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
