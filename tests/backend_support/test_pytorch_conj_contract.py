import importlib.util
import os
import subprocess
import sys

import pytest


def _run_backend_snippet(backend_name, code):
    env = os.environ.copy()
    env["PYRECEST_BACKEND"] = backend_name
    src_path = os.path.abspath("src")
    env["PYTHONPATH"] = (
        src_path
        if not env.get("PYTHONPATH")
        else os.pathsep.join([src_path, env["PYTHONPATH"]])
    )
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


@pytest.mark.backend_portable
def test_raw_pytorch_conj_accepts_array_like_inputs_with_numpy_public_backend():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    _run_backend_snippet(
        "numpy",
        """
import numpy as np

import pyrecest  # noqa: F401
import pyrecest.backend as backend
import pyrecest._backend.pytorch as raw_pytorch

assert getattr(backend, "__backend_name__", None) == "numpy"

result = raw_pytorch.conj([1.0 + 2.0j, 3.0 - 4.0j])
assert raw_pytorch.is_array(result)
assert result.tolist() == [1.0 - 2.0j, 3.0 + 4.0j]

torch_out = raw_pytorch.empty(2, dtype=raw_pytorch.complex128)
returned = raw_pytorch.conj([1.0 + 2.0j, 3.0 - 4.0j], out=torch_out)
assert returned is torch_out
assert torch_out.tolist() == [1.0 - 2.0j, 3.0 + 4.0j]

numpy_out = np.empty(2, dtype=np.complex128)
returned_numpy = raw_pytorch.conj([1.0 + 2.0j, 3.0 - 4.0j], out=numpy_out)
assert returned_numpy is numpy_out
assert numpy_out.tolist() == [1.0 - 2.0j, 3.0 + 4.0j]
""",
    )


@pytest.mark.backend_portable
def test_public_pytorch_conj_accepts_array_like_inputs():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    _run_backend_snippet(
        "pytorch",
        """
import pyrecest.backend as backend

assert getattr(backend, "__backend_name__", None) == "pytorch"

result = backend.conj([1.0 + 2.0j, 3.0 - 4.0j])
assert backend.is_array(result)
assert result.tolist() == [1.0 - 2.0j, 3.0 + 4.0j]

out = backend.empty(2, dtype=backend.complex128)
returned = backend.conj([1.0 + 2.0j, 3.0 - 4.0j], out=out)
assert returned is out
assert out.tolist() == [1.0 - 2.0j, 3.0 + 4.0j]
""",
    )
