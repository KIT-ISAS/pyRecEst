import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pyrecest._backend_submodules import _pytorch_repeat_counts  # noqa: E402


@pytest.mark.backend_portable
def test_pytorch_repeat_rejects_nested_numpy_repeat_counts_before_torch_call():
    with pytest.raises(ValueError, match="object too deep"):
        _pytorch_repeat_counts(
            [[1, 2]],
            numpy_module=np,
            torch_module=torch,
            device=torch.device("cpu"),
        )


@pytest.mark.backend_portable
def test_pytorch_repeat_rejects_nested_tensor_repeat_counts_before_torch_call():
    with pytest.raises(ValueError, match="object too deep"):
        _pytorch_repeat_counts(
            torch.tensor([[1, 2]], dtype=torch.long),
            numpy_module=np,
            torch_module=torch,
            device=torch.device("cpu"),
        )
