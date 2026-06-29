import pytest

from tests.support.backend_runner import run_backend_code


@pytest.mark.backend_portable
def test_public_pytorch_isclose_accepts_equal_nan():
    pytest.importorskip("torch")

    result = run_backend_code(
        "pytorch",
        '''
import pyrecest.backend as backend

assert getattr(backend, "__backend_name__", None) == "pytorch"

nan = float("nan")
equal_nan = backend.isclose([nan, 1.0, 2.0], [nan, 1.0, 3.0], equal_nan=True)
not_equal_nan = backend.isclose([nan, 1.0, 2.0], [nan, 1.0, 3.0], equal_nan=False)

assert equal_nan.tolist() == [True, True, False]
assert not_equal_nan.tolist() == [False, True, False]
''',
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.backend_portable
def test_raw_pytorch_isclose_accepts_equal_nan_with_numpy_public_backend():
    pytest.importorskip("torch")

    result = run_backend_code(
        "numpy",
        '''
import pyrecest  # noqa: F401
import pyrecest.backend as backend
import pyrecest._backend.pytorch as raw_pytorch

assert getattr(backend, "__backend_name__", None) == "numpy"

nan = float("nan")
equal_nan = raw_pytorch.isclose([nan, 1.0, 2.0], [nan, 1.0, 3.0], equal_nan=True)
not_equal_nan = raw_pytorch.isclose([nan, 1.0, 2.0], [nan, 1.0, 3.0], equal_nan=False)

assert equal_nan.tolist() == [True, True, False]
assert not_equal_nan.tolist() == [False, True, False]
''',
    )

    assert result.returncode == 0, result.stderr
