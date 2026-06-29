import importlib.util
import os
import subprocess
import sys

import pytest


@pytest.mark.backend_portable
def test_pytorch_stack_helpers_accept_array_like_sequences_with_numpy_public_backend():
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
import pyrecest.backend as public_backend
import pyrecest._backend.pytorch as stack_backend

assert public_backend.__backend_name__ == "numpy"
to_numpy = stack_backend.to_numpy

assert to_numpy(stack_backend.hstack(([1, 2], [3, 4]))).tolist() == [1, 2, 3, 4]
assert to_numpy(stack_backend.vstack(([1, 2], [3, 4]))).tolist() == [[1, 2], [3, 4]]
assert to_numpy(stack_backend.column_stack(([1, 2], [3, 4]))).tolist() == [[1, 3], [2, 4]]
assert to_numpy(stack_backend.dstack(([1, 2], [3, 4]))).tolist() == [[[1, 3], [2, 4]]]
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
