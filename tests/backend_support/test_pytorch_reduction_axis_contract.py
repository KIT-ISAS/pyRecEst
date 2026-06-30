import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code


@pytest.mark.backend_portable
def test_pytorch_reductions_accept_scalar_array_axes():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    result = run_backend_code(
        "pytorch",
        """
import numpy as np
import numpy.testing as npt
import torch

import pyrecest.backend as backend

values = backend.reshape(backend.arange(6, dtype=backend.float64), (2, 3))

sum_result = backend.sum(values, axis=np.array(1), keepdims=True)
npt.assert_allclose(backend.to_numpy(sum_result), [[3.0], [12.0]])

mean_result = backend.mean(values, axis=torch.tensor(0), keepdims=True)
npt.assert_allclose(backend.to_numpy(mean_result), [[1.5, 2.5, 3.5]])

std_result = backend.std(values, axis=np.array(1), ddof=0)
npt.assert_allclose(
    backend.to_numpy(std_result),
    np.std(np.arange(6, dtype=float).reshape(2, 3), axis=1),
)

quantile_result = backend.quantile(values, 0.5, axis=np.array(1))
npt.assert_allclose(backend.to_numpy(quantile_result), [1.0, 4.0])

max_result = backend.max(values, axis=np.array(1))
npt.assert_allclose(backend.to_numpy(max_result), [2.0, 5.0])

argmax_result = backend.argmax(values, axis=np.array(1))
npt.assert_allclose(backend.to_numpy(argmax_result), [2, 2])

print("ok")
""",
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


@pytest.mark.backend_portable
def test_raw_pytorch_reductions_accept_scalar_array_axes_with_numpy_backend():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    result = run_backend_code(
        "numpy",
        """
import numpy as np
import numpy.testing as npt

import pyrecest.backend as public_backend
import pyrecest._backend.pytorch as raw_pytorch

assert public_backend.__backend_name__ == "numpy"

values = raw_pytorch.reshape(raw_pytorch.arange(6, dtype=raw_pytorch.float64), (2, 3))

npt.assert_allclose(
    raw_pytorch.to_numpy(raw_pytorch.sum(values, axis=np.array(1))),
    [3.0, 12.0],
)
npt.assert_allclose(
    raw_pytorch.to_numpy(raw_pytorch.mean(values, axis=np.array(0))),
    [1.5, 2.5, 3.5],
)
npt.assert_allclose(
    raw_pytorch.to_numpy(raw_pytorch.prod(values + 1, axis=np.array(1))),
    [6.0, 120.0],
)
npt.assert_allclose(
    raw_pytorch.to_numpy(raw_pytorch.argmin(values, axis=np.array(1))),
    [0, 0],
)

print("ok")
""",
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


@pytest.mark.backend_portable
def test_pytorch_reductions_reject_boolean_scalar_axes():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    result = run_backend_code(
        "pytorch",
        """
import numpy as np
import pytest

import pyrecest.backend as backend

values = backend.reshape(backend.arange(6), (2, 3))

for bad_axis in (True, np.array(True)):
    with pytest.raises(TypeError):
        backend.sum(values, axis=bad_axis)
    with pytest.raises(TypeError):
        backend.max(values, axis=bad_axis)
    with pytest.raises(TypeError):
        backend.argmax(values, axis=bad_axis)

print("ok")
""",
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
