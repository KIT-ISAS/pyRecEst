import pytest
from tests.support.backend_runner import run_backend_code


def test_common_torch_fallbacks_accept_negative_stride_numpy_inputs():
    pytest.importorskip("torch")

    code = """
import numpy as np
import torch

import pyrecest.backend as backend

values = np.arange(3.0)[::-1]
weights = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

outer_result = backend.outer(values, weights)
dot_result = backend.dot(values, weights)
matvec_result = backend.matvec(np.eye(3, dtype=float)[::-1], weights)

assert backend.__backend_name__ == "pytorch"
assert torch.is_tensor(outer_result)
assert backend.to_numpy(outer_result).tolist() == [
    [2.0, 4.0, 6.0],
    [1.0, 2.0, 3.0],
    [0.0, 0.0, 0.0],
]
assert float(dot_result) == 4.0
assert backend.to_numpy(matvec_result).tolist() == [3.0, 2.0, 1.0]
"""

    result = run_backend_code("pytorch", code)

    assert result.returncode == 0, result.stderr


def _alternate_device(torch):
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        torch.empty((), device="meta")
    except Exception as exc:  # pragma: no cover - depends on PyTorch build
        pytest.skip(f"no alternate PyTorch test device available: {exc}")
    return torch.device("meta")


def test_common_torch_promoted_pair_uses_existing_alternate_device():
    torch = pytest.importorskip("torch")

    import pyrecest._backend._common as common

    device = _alternate_device(torch)

    result = common.matvec(torch.eye(2), torch.ones(2, device=device))

    assert result.device.type == device.type
    assert tuple(result.shape) == (2,)
