import importlib.util
import os
import subprocess
import sys

import pytest


@pytest.mark.backend_portable
def test_raw_jax_std_out_works_under_numpy_backend():
    if importlib.util.find_spec("jax") is None:
        pytest.skip("jax is not installed")

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
import pyrecest._backend.jax as raw_jax

values = raw_jax.asarray([[1.0, 3.0], [5.0, 7.0]])
out = raw_jax.zeros((2,), dtype=values.dtype)
returned = raw_jax.std(values, axis=0, out=out, ddof=0)

assert raw_jax.to_numpy(returned).tolist() == [2.0, 2.0]

bad_out = raw_jax.zeros((3,), dtype=values.dtype)
try:
    raw_jax.std(values, axis=0, out=bad_out, ddof=0)
except ValueError:
    pass
else:
    raise AssertionError("raw JAX std accepted incompatible out shape")
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
