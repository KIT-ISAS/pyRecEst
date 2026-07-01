import importlib.util

import pytest


@pytest.mark.backend_portable
def test_pytorch_allow_complex_dtype_normalizes_string_alias():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    import torch
    from pyrecest._backend.pytorch._dtype import _allow_complex_dtype

    @_allow_complex_dtype
    def sample(dtype=None):
        del dtype
        return torch.ones(4)

    values = sample(dtype="torch.complex64")

    assert values.dtype == torch.complex64
    assert torch.all(values.imag == 1)


@pytest.mark.backend_portable
def test_pytorch_allow_complex_dtype_normalizes_numpy_alias():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    import numpy as np
    import torch
    from pyrecest._backend.pytorch._dtype import _allow_complex_dtype

    @_allow_complex_dtype
    def sample(dtype=None):
        del dtype
        return torch.ones(4)

    values = sample(dtype=np.complex128)

    assert values.dtype == torch.complex128
    assert torch.all(values.imag == 1)
