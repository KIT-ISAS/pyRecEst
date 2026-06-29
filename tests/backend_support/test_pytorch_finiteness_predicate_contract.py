import importlib.util
import os
import subprocess
import sys

import pytest


@pytest.mark.backend_portable
def test_pytorch_finiteness_predicates_accept_array_like_inputs():
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
import pyrecest._backend.pytorch as raw_backend

values = [1.0, float("inf"), float("nan")]
assert backend.to_numpy(backend.isfinite(values)).tolist() == [True, False, False]
assert backend.to_numpy(backend.isinf(values)).tolist() == [False, True, False]
assert backend.to_numpy(backend.isnan(values)).tolist() == [False, False, True]

complex_values = [1.0 + 0.0j, 1.0 + 2.0j]
assert backend.to_numpy(backend.isreal(complex_values)).tolist() == [True, False]

assert backend.to_numpy(raw_backend.isfinite(values)).tolist() == [True, False, False]
assert backend.to_numpy(raw_backend.isinf(values)).tolist() == [False, True, False]
assert backend.to_numpy(raw_backend.isnan(values)).tolist() == [False, False, True]
assert backend.to_numpy(raw_backend.isreal(complex_values)).tolist() == [True, False]
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
