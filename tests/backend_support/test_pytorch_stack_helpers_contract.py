import importlib.util
import os
import subprocess
import sys

import pytest


@pytest.mark.backend_portable
def test_pytorch_stack_helpers_accept_array_like_sequences():
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

assert backend.to_numpy(backend.hstack(([1, 2], [3, 4]))).tolist() == [1, 2, 3, 4]
assert backend.to_numpy(backend.vstack(([1, 2], [3, 4]))).tolist() == [[1, 2], [3, 4]]
assert backend.to_numpy(backend.column_stack(([1, 2], [3, 4]))).tolist() == [[1, 3], [2, 4]]
assert backend.to_numpy(backend.dstack(([1, 2], [3, 4]))).tolist() == [[[1, 3], [2, 4]]]

assert backend.to_numpy(pytorch_backend.hstack(([1, 2], [3, 4]))).tolist() == [1, 2, 3, 4]
assert backend.to_numpy(pytorch_backend.vstack(([1, 2], [3, 4]))).tolist() == [[1, 2], [3, 4]]
assert backend.to_numpy(pytorch_backend.column_stack(([1, 2], [3, 4]))).tolist() == [[1, 3], [2, 4]]
assert backend.to_numpy(pytorch_backend.dstack(([1, 2], [3, 4]))).tolist() == [[[1, 3], [2, 4]]]

values = backend.array([[1, 2], [3, 4]])
assert backend.to_numpy(backend.hstack((values, [[5, 6], [7, 8]]))).tolist() == [[1, 2, 5, 6], [3, 4, 7, 8]]
assert backend.to_numpy(backend.column_stack((values, [[5, 6], [7, 8]]))).tolist() == [[1, 2, 5, 6], [3, 4, 7, 8]]
assert backend.to_numpy(pytorch_backend.hstack((values, [[5, 6], [7, 8]]))).tolist() == [[1, 2, 5, 6], [3, 4, 7, 8]]
assert backend.to_numpy(pytorch_backend.column_stack((values, [[5, 6], [7, 8]]))).tolist() == [[1, 2, 5, 6], [3, 4, 7, 8]]
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
