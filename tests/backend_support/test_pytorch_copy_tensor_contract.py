import numpy as np
import pytest


def test_pytorch_copy_returns_tensor_for_array_like_inputs():
    pytest.importorskip("torch")

    import pyrecest._backend.pytorch as pytorch_backend

    scalar_copy = pytorch_backend.copy(1.5)
    assert pytorch_backend.is_array(scalar_copy)
    assert tuple(scalar_copy.shape) == ()
    assert float(scalar_copy) == 1.5

    sequence_copy = pytorch_backend.copy([[1.0, 2.0], [3.0, 4.0]])
    assert pytorch_backend.is_array(sequence_copy)
    assert sequence_copy.tolist() == [[1.0, 2.0], [3.0, 4.0]]

    source = np.array([1.0, 2.0])
    array_copy = pytorch_backend.copy(source)
    source[0] = 99.0
    assert pytorch_backend.is_array(array_copy)
    assert array_copy.tolist() == [1.0, 2.0]
