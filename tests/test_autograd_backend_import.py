import importlib.util
import os
import subprocess
import sys

import pytest


@pytest.mark.backend_portable
def test_autograd_backend_imports_and_exposes_contract_smoke():
    if importlib.util.find_spec("autograd") is None:
        pytest.skip("autograd is not installed")

    env = os.environ.copy()
    env["PYRECEST_BACKEND"] = "autograd"
    src_path = os.path.abspath("src")
    env["PYTHONPATH"] = (
        src_path
        if not env.get("PYTHONPATH")
        else os.pathsep.join([src_path, env["PYTHONPATH"]])
    )

    code = """
import pyrecest.backend as backend
assert backend.__backend_name__ == 'autograd'
assert backend.has_autodiff()
assert backend.diag(backend.array([1.0, 2.0])).shape == (2, 2)
assert backend.random.get_state() is not None
assert backend.fft.rfft(backend.array([1.0, 0.0])).shape == (2,)
assert backend.dot([1.0, 2.0], [3.0, 4.0]) == 11.0
assert backend.to_numpy(backend.outer([1.0, 2.0], [3.0, 4.0])).tolist() == [[3.0, 4.0], [6.0, 8.0]]
assert backend.to_numpy(backend.outer(2.0, backend.array([1.0, 2.0]))).tolist() == [2.0, 4.0]
_calls = []
def _add_one(row):
    _calls.append(None)
    return row + 1
_vmap_int = backend.vmap(_add_one)(
    backend.array([[1, 2], [3, 4]], dtype=backend.int64)
)
assert backend.to_numpy(_vmap_int).tolist() == [[2, 3], [4, 5]]
assert backend.to_numpy(_vmap_int).dtype.kind in ("i", "u")
assert len(_calls) == 2
_vmap_complex = backend.vmap(lambda row: row[0] + 1j * row[1])(
    backend.array([[1.0, 2.0], [3.0, 4.0]])
)
assert backend.to_numpy(_vmap_complex).tolist() == [1.0 + 2.0j, 3.0 + 4.0j]
assert backend.to_numpy(_vmap_complex).dtype.kind == "c"
from autograd import grad

def _outer_sum(x):
    return backend.sum(backend.outer(x, backend.array([2.0, 3.0])))
assert backend.to_numpy(grad(_outer_sum)(backend.array([1.0, 2.0]))).tolist() == [5.0, 5.0]

def _vmap_square_sum(x):
    return backend.sum(backend.vmap(lambda row: row * row)(x))
assert backend.to_numpy(grad(_vmap_square_sum)(backend.array([1.0, 2.0]))).tolist() == [2.0, 4.0]
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
