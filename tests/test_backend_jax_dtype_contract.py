import importlib.util
import os
import subprocess
import sys

import pytest


@pytest.mark.backend_portable
def test_jax_set_default_dtype_sets_and_returns_dtype():
    if importlib.util.find_spec("jax") is None:
        pytest.skip("jax is not installed")

    env = os.environ.copy()
    env["PYRECEST_BACKEND"] = "jax"
    src_path = os.path.abspath("src")
    env["PYTHONPATH"] = (
        src_path
        if not env.get("PYTHONPATH")
        else os.pathsep.join([src_path, env["PYTHONPATH"]])
    )

    code = """
import pyrecest.backend as backend
result = backend.set_default_dtype('float32')
assert not callable(result)
assert str(result).endswith('float32')
result = backend.set_default_dtype('float64')
assert not callable(result)
assert str(result).endswith('float64')
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
