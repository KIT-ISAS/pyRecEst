import importlib.util
import os
import subprocess
import sys

import pytest


def _run_with_backend(backend_name, code):
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
def test_pytorch_predicates_accept_array_like_inputs_and_out():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    code = """
import numpy as np
import torch

import pyrecest.backend as backend
import pyrecest._backend.pytorch as raw_pytorch

assert getattr(backend, "__backend_name__", None) == "pytorch"

values = [0.0, float("nan"), float("inf"), -float("inf")]
expected = {
    "isfinite": [True, False, False, False],
    "isinf": [False, False, True, True],
    "isnan": [False, True, False, False],
}

for module in (backend, raw_pytorch):
    for helper_name, expected_values in expected.items():
        result = getattr(module, helper_name)(values)
        assert module.is_array(result)
        assert module.to_numpy(result).tolist() == expected_values

complex_values = [1.0 + 0.0j, 2.0 + 3.0j]
for module in (backend, raw_pytorch):
    result = module.isreal(complex_values)
    assert module.is_array(result)
    assert module.to_numpy(result).tolist() == [True, False]

out = torch.empty(len(values), dtype=torch.bool)
returned = backend.isfinite(values, out=out)
assert returned is out
assert out.tolist() == expected["isfinite"]

np_out = np.empty(len(values), dtype=bool)
returned = raw_pytorch.isinf(values, out=np_out)
assert returned is np_out
assert np_out.tolist() == expected["isinf"]
"""
    _run_with_backend("pytorch", code)


@pytest.mark.backend_portable
def test_raw_pytorch_predicates_work_with_numpy_public_backend():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    code = """
import numpy as np
import torch

import pyrecest  # noqa: F401
import pyrecest.backend as backend
import pyrecest._backend.pytorch as raw_pytorch

assert getattr(backend, "__backend_name__", None) == "numpy"

values = [0.0, float("nan"), float("inf")]
finite = raw_pytorch.isfinite(values)
assert raw_pytorch.is_array(finite)
assert finite.tolist() == [True, False, False]

nan_values = raw_pytorch.isnan(values)
assert raw_pytorch.is_array(nan_values)
assert nan_values.tolist() == [False, True, False]

real = raw_pytorch.isreal([1.0 + 0.0j, 2.0 + 3.0j])
assert raw_pytorch.is_array(real)
assert real.tolist() == [True, False]

np_out = np.empty(len(values), dtype=bool)
returned = raw_pytorch.isinf(values, out=np_out)
assert returned is np_out
assert np_out.tolist() == [False, False, True]

torch_out = torch.empty(len(values), dtype=torch.bool)
returned = raw_pytorch.isfinite(values, out=torch_out)
assert returned is torch_out
assert torch_out.tolist() == [True, False, False]
"""
    _run_with_backend("numpy", code)
