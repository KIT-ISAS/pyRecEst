import importlib.util
import os
import subprocess
import sys

import pytest


def _backend_subprocess_env(backend_name):
    env = os.environ.copy()
    env["PYRECEST_BACKEND"] = backend_name
    src_path = os.path.abspath("src")
    env["PYTHONPATH"] = (
        src_path
        if not env.get("PYTHONPATH")
        else os.pathsep.join([src_path, env["PYTHONPATH"]])
    )
    return env


@pytest.mark.backend_portable
def test_jax_take_honors_out_argument_for_public_backend():
    if importlib.util.find_spec("jax") is None:
        pytest.skip("jax is not installed")

    env = _backend_subprocess_env("jax")

    code = """
import numpy.testing as npt
import pyrecest.backend as backend

assert backend.__backend_name__ == "jax"

out = backend.zeros((2, 2), dtype=backend.int64)
result = backend.take(
    [[10, 20, 30], [40, 50, 60]],
    [[2, 0], [1, 1]],
    axis=1,
    out=out,
)

npt.assert_array_equal(backend.to_numpy(result), [[30, 10], [50, 50]])
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
