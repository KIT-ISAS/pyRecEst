import importlib.util
import os
import subprocess
import sys

import pytest


def test_jax_dot_accepts_out_keyword_and_validates_shape():
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
import jax.numpy as jnp
import pyrecest.backend as backend
import pyrecest._backend.jax as raw_jax

scalar_out = jnp.zeros(())
scalar_result = backend.dot([1.0, 2.0], [3.0, 4.0], out=scalar_out)
assert scalar_result.shape == ()
assert float(scalar_result) == 11.0

vector_out = jnp.zeros((2,))
vector_result = backend.dot([[1.0, 2.0], [3.0, 4.0]], [5.0, 6.0], out=vector_out)
assert vector_result.shape == (2,)
assert backend.to_numpy(vector_result).tolist() == [17.0, 39.0]

raw_result = raw_jax.dot([1.0, 2.0], [3.0, 4.0], out=scalar_out)
assert float(raw_result) == 11.0

try:
    backend.dot([1.0, 2.0], [3.0, 4.0], out=jnp.zeros((2,)))
except ValueError as exc:
    assert "wrong shape" in str(exc)
else:
    raise AssertionError("JAX dot accepted an incompatible output shape")
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
