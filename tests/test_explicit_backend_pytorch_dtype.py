import numpy as np
import pytest


def test_explicit_pytorch_backend_normalizes_numpy_default_dtype():
    torch = pytest.importorskip("torch")

    # Importing the public facade with the default backend initializes the shared
    # dtype state with NumPy dtype objects.  A concrete PyTorch backend obtained
    # afterwards must still pass torch.dtype instances to torch operations.
    import pyrecest.backend  # noqa: F401
    from pyrecest.backends import get_backend

    torch_backend = get_backend("pytorch")

    result = torch_backend.linspace(0, 1, 3)

    assert result.dtype == torch.float64
    np.testing.assert_allclose(torch_backend.to_numpy(result), [0.0, 0.5, 1.0])


def test_explicit_pytorch_backend_accepts_numpy_dtype_arguments():
    torch = pytest.importorskip("torch")

    from pyrecest.backends import get_backend

    torch_backend = get_backend("pytorch")

    result = torch_backend.array([1.0, 2.0], dtype=np.float64)

    assert result.dtype == torch.float64
    np.testing.assert_allclose(torch_backend.to_numpy(result), [1.0, 2.0])
