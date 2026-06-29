import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pyrecest._backend.pytorch import random  # noqa: E402


@pytest.mark.parametrize(
    ("dtype", "expected_dtype"),
    [
        (np.float32, torch.float32),
        (np.dtype("float64"), torch.float64),
    ],
)
def test_rand_accepts_numpy_dtype_aliases(dtype, expected_dtype):
    samples = random.rand(size=(2,), dtype=dtype)

    assert samples.shape == (2,)
    assert samples.dtype == expected_dtype


@pytest.mark.parametrize(
    ("dtype", "expected_dtype"),
    [
        (np.float32, torch.float32),
        (np.dtype("float64"), torch.float64),
    ],
)
def test_uniform_accepts_numpy_dtype_aliases(dtype, expected_dtype):
    samples = random.uniform(0.0, 1.0, size=(2,), dtype=dtype)

    assert samples.shape == (2,)
    assert samples.dtype == expected_dtype


@pytest.mark.parametrize(
    ("dtype", "expected_dtype"),
    [
        (np.int32, torch.int32),
        (np.dtype("int64"), torch.int64),
    ],
)
def test_randint_scalar_bounds_accepts_numpy_dtype_aliases(dtype, expected_dtype):
    samples = random.randint(0, 4, size=(2,), dtype=dtype)

    assert samples.shape == (2,)
    assert samples.dtype == expected_dtype
    assert torch.all(samples >= 0)
    assert torch.all(samples < 4)


def test_randint_array_bounds_accepts_numpy_dtype_alias():
    low = torch.tensor([0, 10])
    high = torch.tensor([4, 14])

    samples = random.randint(low, high, dtype=np.dtype("int32"))

    assert samples.shape == (2,)
    assert samples.dtype == torch.int32
    assert torch.all(samples >= low.to(dtype=torch.int32))
    assert torch.all(samples < high.to(dtype=torch.int32))
