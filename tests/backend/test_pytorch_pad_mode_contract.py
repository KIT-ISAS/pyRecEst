import pytest
from tests.support.backend_runner import run_backend_code


torch = pytest.importorskip("torch")

import pyrecest.backend_tools  # noqa: E402,F401
import pyrecest._backend.pytorch as pytorch_backend  # noqa: E402


def _to_list(value):
    return value.detach().cpu().tolist()


def test_raw_pytorch_pad_accepts_numpy_edge_and_wrap_mode_names():
    values = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)

    edge = pytorch_backend.pad(values, ((0, 0), (2, 1)), mode="edge")
    wrap = pytorch_backend.pad(values, ((0, 0), (1, 2)), mode="wrap")

    assert _to_list(edge) == [[1.0, 1.0, 1.0, 2.0, 3.0, 3.0]]
    assert _to_list(wrap) == [[3.0, 1.0, 2.0, 3.0, 1.0, 2.0]]


def test_public_pytorch_pad_accepts_numpy_edge_and_wrap_mode_names():
    result = run_backend_code(
        "pytorch",
        r'''
import pyrecest.backend as backend

values = backend.asarray([[1.0, 2.0, 3.0]])
edge = backend.pad(values, ((0, 0), (2, 1)), mode="edge")
wrap = backend.pad(values, ((0, 0), (1, 2)), mode="wrap")

assert backend.to_numpy(edge).tolist() == [[1.0, 1.0, 1.0, 2.0, 3.0, 3.0]]
assert backend.to_numpy(wrap).tolist() == [[3.0, 1.0, 2.0, 3.0, 1.0, 2.0]]
''',
    )

    assert result.returncode == 0, result.stderr
